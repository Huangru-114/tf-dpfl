"""
harness/pack_test.py  —  一卡多跑测试的预注册判定（D-047，先定后跑）

    python3 harness/pack_test.py [--json out.json]

读 `pilot/results/P1/PACK/PACK_K<K>__s<i>.metrics.json`（experiments/attack/hfl-mechanism/pack.sbatch
的产物）与 DET 第二轮的两个单跑参照（`pilot/results/P1/DET/DET__rep{1,2}__s42.metrics.json`）。

  **硬门槛（确定性）**：每个打包 run 的前 5 轮 `[Checksum]` 逐轮等于 DET 参照。
      参照从 DET 的 metrics 里读，不手抄；DET 两次自己必须先一致，否则参照作废（missing）。
  **有效性**：D-044 的闸（exit_code、client_failures），复用 pilot_a4.invalid_reasons。
  **加速比** = K × T_solo / max(T_pack)，T = round_time_total_s + bd_eval_total_s
      （一个 run 的训练 + 评估墙钟，不含启动与数据加载），T_solo = DET 两次单跑的均值。
      每个 run 的机时 = 单跑的 1 / 加速比。
  **采用规则**：checksum 全等且加速比 ≥ 1.5 → 该 K 可采用；可采用的 K 里取加速比最高的。
      没有可采用的 K → 不采用一卡多跑（照旧一卡一跑）。

纯标准库，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot_a4 import invalid_reasons   # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "experiments/attack/hfl-mechanism/pilot/results/P1"
PACK_DIR = RESULTS / "PACK"
DET_FILES = (RESULTS / "DET/DET__rep1__s42.metrics.json",
             RESULTS / "DET/DET__rep2__s42.metrics.json")

# ── 预注册（D-047；改动要在 DECISIONS 留记录）──────────────────────────────
MIN_SPEEDUP = 1.5
DET_ROUNDS = 5
EXPECTED_KS = (2, 3)


def run_seconds(m: dict):
    """训练 + 评估墙钟；任何一侧缺 → None（不当 0）。"""
    ts = m.get("timing_summary") or {}
    a, b = ts.get("round_time_total_s"), ts.get("bd_eval_total_s")
    return None if a is None or b is None else a + b


def _checksums(m: dict, k: int = DET_ROUNDS):
    cs = {c["round"]: c["global"] for c in m.get("checksums") or []}
    return [cs.get(r) for r in range(1, k + 1)]


def reference(det_runs) -> dict:
    """DET 两次单跑 → 参照 checksum 与 T_solo。两次不一致 / 缺 → verdict missing。"""
    if any(m is None for m in det_runs):
        return {"verdict": "missing", "reasons": ["DET 参照文件缺"]}
    a, b = (_checksums(m) for m in det_runs)
    if None in a or a != b:
        return {"verdict": "missing", "reasons": ["DET 两次的前 5 轮 checksum 不全或不一致 —— 参照作废"]}
    ts = [run_seconds(m) for m in det_runs]
    if None in ts:
        return {"verdict": "missing", "reasons": ["DET 参照缺计时"]}
    return {"verdict": "ok", "checksums": a, "t_solo_s": sum(ts) / len(ts)}


def judge_pack(k: int, runs: list, ref: dict) -> dict:
    """runs：这个 K 的 K 份 metrics（缺的是 None）。"""
    if ref.get("verdict") != "ok":
        return {"k": k, "verdict": "missing", "reasons": ref.get("reasons", [])}
    if len(runs) != k or any(m is None for m in runs):
        return {"k": k, "verdict": "missing",
                "reasons": [f"应有 {k} 份结果，实有 {sum(m is not None for m in runs)} 份"]}
    reasons = [f"s{i + 1}: {r}" for i, m in enumerate(runs) for r in invalid_reasons(m)]
    if reasons:
        return {"k": k, "verdict": "invalid", "reasons": reasons}
    diverged = [i + 1 for i, m in enumerate(runs) if _checksums(m) != ref["checksums"]]
    ts = [run_seconds(m) for m in runs]
    if None in ts:
        return {"k": k, "verdict": "missing", "reasons": ["打包 run 缺计时"]}
    t_pack = max(ts)
    speedup = k * ref["t_solo_s"] / t_pack
    out = {"k": k, "t_solo_s": round(ref["t_solo_s"], 1), "t_pack_max_s": round(t_pack, 1),
           "speedup": round(speedup, 3), "gpu_h_per_run_vs_solo": round(1 / speedup, 3),
           "diverged_slots": diverged}
    if diverged:
        out.update(verdict="diverged",
                   reasons=[f"槽位 {diverged} 的前 {DET_ROUNDS} 轮 checksum ≠ DET 参照 —— 同卡影响了确定性"])
    elif speedup < MIN_SPEEDUP:
        out.update(verdict="too-slow", reasons=[f"加速比 {speedup:.2f} < {MIN_SPEEDUP}"])
    else:
        out.update(verdict="eligible", reasons=[])
    return out


def judge_all(packs: dict, ref: dict) -> dict:
    """packs：{K: [metrics or None, ...]}。"""
    per_k = {k: judge_pack(k, packs.get(k, [None] * k), ref) for k in sorted(packs)}
    eligible = [r for r in per_k.values() if r["verdict"] == "eligible"]
    best = max(eligible, key=lambda r: r["speedup"])["k"] if eligible else None
    if best is not None:
        decision = f"采用 K={best}（每个 run 的机时约为单跑的 {per_k[best]['gpu_h_per_run_vs_solo']:.2f}）"
    elif any(r["verdict"] in ("missing", "invalid") for r in per_k.values()):
        decision = "结果不全或无效 —— 看 reasons，修好重交"
    else:
        decision = "不采用一卡多跑（照旧一卡一跑）"
    return {"reference": {k: v for k, v in ref.items() if k != "checksums"},
            "per_k": per_k, "adopt_k": best, "decision": decision}


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def load_packs(pack_dir: Path = PACK_DIR, ks=EXPECTED_KS) -> dict:
    return {k: [_load(pack_dir / f"PACK_K{k}__s{i}.metrics.json") for i in range(1, k + 1)]
            for k in ks}


def main(argv=None):
    ap = argparse.ArgumentParser(description="一卡多跑测试的预注册判定（D-047）")
    ap.add_argument("--json", help="把判定写到这个文件")
    a = ap.parse_args(argv)
    ref = reference([_load(p) for p in DET_FILES])
    res = judge_all(load_packs(), ref)
    for k, r in res["per_k"].items():
        gpu = _load(PACK_DIR / f"PACK_K{k}.gpu.json")
        extra = {x: r[x] for x in ("speedup", "gpu_h_per_run_vs_solo", "t_pack_max_s") if x in r}
        print(f"K={k}: {r['verdict']:<9} {json.dumps(extra, ensure_ascii=False)} "
              f"{'; '.join(r.get('reasons', []))}")
        if gpu:
            print(f"      GPU 利用率 均值 {gpu.get('util_mean')}% / 峰值 {gpu.get('util_max')}%，"
                  f"显存峰值 {gpu.get('mem_max_mib')} MiB，作业墙钟 {gpu.get('wall_s')} s")
    print(f"结论：{res['decision']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
