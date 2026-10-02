"""
harness/instrumentation_check.py  —  「新仪表没有改变任何已有的数」的验收（S9 / D-072）

    python3 harness/instrumentation_check.py <ref.metrics.json> <new.metrics.json> [--upto R]

比什么（只比第 ≤ R 轮；R 缺省 = 两边共有的最后一轮）：
  · checksums[]：每轮聚合后全局权重的哈希（训练轨迹）；
  · 改动前就有的数值字段：rounds[] / acc_rounds[] / per_edge_rounds / per_edge_acc_rounds 里
    **参照文件有的键**（新文件多出来的 schema 7 字段不比），去掉墙钟类字段（round_time、*_s）。
    日志打印的精度就是比较的精度（collect_metrics 从日志解析）。
不比：计时（每次都不同）、schema 7 的新字段、run 块（provenance / 配置路径本来就不同）。

为什么不用 check_reproducible.py：它比**全部**数值字段、含计时，且要求两边字段集合相同 ——
带仪表的新文件必然多出字段。

`--grid`（S5 / D-084）：参照 = 没开网格，新 = 开了统一评估网格的同一配置（S5P 组）。
  网格下非全量轮的 GM / EM 不评（null）—— 这些字段记为「网格下不评」单独计数，不算分歧；
  其余照旧逐位比。另报新文件的轻评估点数（light_rounds）。训练侧 checksum 必须逐轮相同。

S6a（D-085）：G1P 的开 / 关两格（coll-off = 参照、coll-on = 三个记录开关全开）直接比，不需要 --grid
（两边网格相同）。参照有 light_rounds[] 时，两边共有的轻评估点（按 effective_round 对齐）上
**参照有的列**也逐位比 —— 新开的 post-agg / 冻结列 / margin 列是新文件多出来的，不比。

用途（S9 的两次验收）：
  · DET：`--upto 5`，参照 = pilot/results/P1/DET/DET__rep1__s42.metrics.json（F-045 的 checksum）；
  · G8 vs G6(a) 同 seed：`--upto 30`（G8 在第 31 轮才停止投毒，之前配置只差停止轮与 n_rounds）。
**checksum 只看训练**：[Checksum] 在评估之前打印，评估数值另由上面的字段比对负责。
GPU 型号不同 → 结论是「无法判定」，不是失败（F-045 只证明了跨节点，没证明跨架构）。

退出码：0 = 逐位一致；1 = 有分歧（列出前几处）；2 = 没有可比的轮。纯标准库。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROW_LISTS = ("rounds", "acc_rounds")
EDGE_DICTS = ("per_edge_rounds", "per_edge_acc_rounds")
# --grid：网格下只在全量点评、其余轮是 null 的字段（server.CloudServer.run_round 的 gm_em_due）
GRID_SKIPPABLE = {("acc_rounds", "gm_acc"), ("acc_rounds", "em_acc"),
                  ("per_edge_acc_rounds", "em_acc")}


def _is_timing(key: str) -> bool:
    return key == "round_time" or key.endswith("_s")


def _checksums(m) -> dict:
    return {int(c["round"]): c["global"] for c in (m.get("checksums") or [])}


def compare(ref: dict, new: dict, upto: int | None = None, grid: bool = False) -> dict:
    ck_ref, ck_new = _checksums(ref), _checksums(new)
    common = sorted(set(ck_ref) & set(ck_new))
    if upto is None:
        upto = common[-1] if common else 0
    diffs, missing = [], []
    n_grid_skipped = 0

    def _skippable(where, k, v, got):
        return grid and (where, k) in GRID_SKIPPABLE and v is not None and got is None

    for r in range(1, upto + 1):
        if r not in ck_ref:
            continue
        if r not in ck_new:
            missing.append(f"checksums[round={r}]")
        elif ck_ref[r] != ck_new[r]:
            diffs.append((f"checksums[round={r}]", ck_ref[r], ck_new[r]))

    n_fields = 0
    for key in ROW_LISTS:
        new_by = {row.get("round"): row for row in (new.get(key) or [])}
        for row in ref.get(key) or []:
            r = row.get("round")
            if r is None or r > upto:
                continue
            other = new_by.get(r)
            if other is None:
                missing.append(f"{key}[round={r}]")
                continue
            for k, v in row.items():
                if _is_timing(k) or k == "round":
                    continue
                if _skippable(key, k, v, other.get(k)):
                    n_grid_skipped += 1
                    continue
                n_fields += 1
                if other.get(k) != v:
                    diffs.append((f"{key}[round={r}].{k}", v, other.get(k)))

    for key in EDGE_DICTS:
        ref_d, new_d = ref.get(key) or {}, new.get(key) or {}
        for rs, edges in ref_d.items():
            r = int(rs)
            if r > upto:
                continue
            other = {e.get("edge_id"): e for e in (new_d.get(rs) or new_d.get(r) or [])}
            for e in edges:
                o = other.get(e.get("edge_id"))
                if o is None:
                    missing.append(f"{key}[{r}][edge{e.get('edge_id')}]")
                    continue
                for k, v in e.items():
                    if _skippable(key, k, v, o.get(k)):
                        n_grid_skipped += 1
                        continue
                    n_fields += 1
                    if o.get(k) != v:
                        diffs.append((f"{key}[{r}][edge{e.get('edge_id')}].{k}", v, o.get(k)))

    # S6a：轻评估点（单独成表，按有效轮对齐）。参照有的列逐位比（计时列除外）；新文件多出的列不比。
    new_light = {row.get("effective_round"): row for row in (new.get("light_rounds") or [])}
    for row in ref.get("light_rounds") or []:
        r = row.get("round")
        if r is None or r > upto:
            continue
        eff = row.get("effective_round")
        other = new_light.get(eff)
        if other is None:
            missing.append(f"light_rounds[effective_round={eff}]")
            continue
        for k, v in row.items():
            if _is_timing(k) or k in ("round", "effective_round"):
                continue
            n_fields += 1
            if other.get(k) != v:
                diffs.append((f"light_rounds[effective_round={eff}].{k}", v, other.get(k)))

    n_ck = sum(1 for r in range(1, upto + 1) if r in ck_ref and r in ck_new)
    return {"upto": upto, "n_checksums": n_ck, "n_fields": n_fields,
            "diffs": diffs, "missing": missing, "n_grid_skipped": n_grid_skipped,
            "n_light": len(new.get("light_rounds") or [])}


def _timing_note(ref: dict, new: dict, upto: int) -> str | None:
    def mean(m):
        v = [t.get("asr_main_s") for t in (m.get("timing_rounds") or [])
             if t.get("round", 0) <= upto and t.get("asr_main_s") is not None]
        return sum(v) / len(v) if v else None
    a, b = mean(ref), mean(new)
    if a is None or b is None:
        return None
    return f"[TimingASR].main 均值：参照 {a:.2f}s → 新 {b:.2f}s（{(b - a) / a * 100:+.1f}%；只作参考，墙钟本来就有噪声）"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--upto", type=int, default=None)
    ap.add_argument("--grid", action="store_true",
                    help="新文件开了统一评估网格（S5）：非全量轮的 GM / EM 为 null 不算分歧")
    a = ap.parse_args(argv)
    ref = json.loads(Path(a.ref).read_text(encoding="utf-8"))
    new = json.loads(Path(a.new).read_text(encoding="utf-8"))
    res = compare(ref, new, a.upto, grid=a.grid)
    if res["n_checksums"] == 0:
        print(f"❌ 前 {res['upto']} 轮没有两边都有的 checksum —— 无从比较")
        return 2
    note = _timing_note(ref, new, res["upto"])
    if not res["diffs"] and not res["missing"]:
        print(f"✅ 前 {res['upto']} 轮逐位一致：{res['n_checksums']} 个 checksum、"
              f"{res['n_fields']} 个已有数值字段")
        if a.grid:
            print(f"   网格：{res['n_grid_skipped']} 个 GM / EM 字段网格下不评（不算分歧）；"
                  f"新文件 {res['n_light']} 个轻评估点")
        if note:
            print("   " + note)
        return 0
    print(f"❌ 前 {res['upto']} 轮不一致：{len(res['diffs'])} 处数值不同、"
          f"{len(res['missing'])} 处新文件缺失")
    for k, x, y in res["diffs"][:10]:
        print(f"   {k}: {x!r}  vs  {y!r}")
    for k in res["missing"][:5]:
        print(f"   缺失：{k}")
    ck = [k for k, _, _ in res["diffs"] if k.startswith("checksums")]
    if ck:
        print(f"   最早的 checksum 分歧：{ck[0]} → 训练轨迹变了（先确认 GPU 型号与核数相同，F-047）")
    elif res["diffs"]:
        print("   checksum 全同、评估数值不同 → 仪表改动了评估（多调了触发器 / 多耗了随机数？）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
