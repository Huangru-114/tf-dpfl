"""
harness/g1p_geometry.py  —  G1P（S6a 探路组，D-085）的**描述性**读数：不是判定，不下结论

    python3 harness/g1p_geometry.py <on.metrics.json> [<on2.metrics.json> …] [--json out.json]

G1P 要回答三件事里的两件靠这个脚本读（第三件「开 / 关逐位相同」是 `instrumentation_check.py`）：

  1. **几何分数分不分得开攻击者**（先于 3-D 的任何设计）：对 `update_geometry` 里每个上传的更新，
     以 −cos（越不像其余更新越可疑）和 norm（越大越可疑）为分数，算「恶意 vs 良性」的 AUROC（秩和，平局取一半）：
       · auroc_norm / auroc_neg_cos_global：池化全部更新（全局视角：每个 edge 轮里全体 edge 的更新一起比）；
       · auroc_neg_cos_edge：只用「同一个 (云轮, edge, edge 轮) 里恶意与良性都在」的更新（edge 视角能比的那部分）。
         集中布点（[10,0,0,0]）下 E0 全是恶意端 → edge 视角没有可比的组（n_mixed = 0 → None），这是结构而非 bug；
     都 ≈ 0.5 ⇒ 这种攻击在几何上藏得好（3-D 的 ΔAUROC ≈ 0 说的是这个，不是「edge 没用」，附录 D）。
  2. **Δ_jump**：云聚合后评估点的 pm_acc / 良性 ASR 减去上一个全量点（有效轮相同）。
  3. **冻结触发器列 vs 主列**（full 点）：同一个全量点上，冻结触发器的良性 ASR 减主列的 `local_benign_asr`
     —— 差 = 触发器在这一云轮内的漂移对 ASR 的贡献（受害 body 的变化不在内，因为两列测的是同一批受害 PM）。

**判读阈值没有预注册**：这是探路读数。要把它变成 3-D / 3-C 的判定，先在 FINDINGS 写规则，再写 `g1_verdict.py`
（D-081 的纪律：规则在数据之前）。空组 → None（陷阱 #13）。纯标准库 + numpy，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))
from analysis.functional_score import auroc            # noqa: E402  纯 numpy


def _updates(m: dict) -> list:
    """update_geometry 紧凑表 → 每个更新一条 {round, edge_id, edge_round, mal, norm, cos_edge, cos_global}。"""
    t = m.get("update_geometry") or {}
    cols = t.get("columns") or []
    out = []
    for row in t.get("rows") or []:
        d = dict(zip(cols, row))
        for i, cid in enumerate(d["cid"] or []):
            out.append({"round": d["round"], "edge_id": d["edge_id"], "edge_round": d["edge_round"],
                        "cid": cid, "mal": bool(d["mal"][i]), "norm": d["norm"][i],
                        "cos_edge": d["cos_edge"][i], "cos_global": d["cos_global"][i]})
    return out


def _auroc(ups, key, sign=1.0):
    pos = [sign * u[key] for u in ups if u["mal"] and u[key] is not None]
    neg = [sign * u[key] for u in ups if not u["mal"] and u[key] is not None]
    return auroc(pos, neg), len(pos), len(neg)


def geometry_readout(m: dict) -> dict:
    ups = _updates(m)
    # edge 视角：只留同一 (云轮, edge, edge 轮) 里恶意与良性都在的更新
    groups = {}
    for u in ups:
        groups.setdefault((u["round"], u["edge_id"], u["edge_round"]), []).append(u)
    mixed = [u for g in groups.values() if any(x["mal"] for x in g) and any(not x["mal"] for x in g)
             for u in g]
    a_norm, n_mal, n_ben = _auroc(ups, "norm")
    a_cg, *_ = _auroc(ups, "cos_global", -1.0)
    a_ce, nm_e, nb_e = _auroc(mixed, "cos_edge", -1.0)
    # 同一批更新上两种视角各自的 AUROC（只用 mixed 组里 cos_global 也有定义的更新）：等池的粗略对照
    a_cg_mixed, *_ = _auroc(mixed, "cos_global", -1.0)
    return {"n_updates": len(ups), "n_malicious": n_mal, "n_benign": n_ben,
            "auroc_norm": a_norm, "auroc_neg_cos_global": a_cg,
            "n_mixed_updates": len(mixed), "n_mixed_malicious": nm_e, "n_mixed_benign": nb_e,
            "auroc_neg_cos_edge": a_ce, "auroc_neg_cos_global_on_mixed": a_cg_mixed}


def jump_readout(m: dict) -> list:
    """云聚合后评估点 − 上一个全量点（同一有效轮）。"""
    pm = {r["round"]: r.get("pm_acc") for r in m.get("acc_rounds") or []}
    asr = {r["round"]: r.get("local_benign_asr") for r in m.get("rounds") or []}
    out = []
    for p in m.get("post_agg_rounds") or []:
        g = p["round"]
        prev_pm, prev_asr = pm.get(g - 1), asr.get(g - 1)
        out.append({"round": g, "effective_round": p.get("effective_round"),
                    "d_pm_acc": None if (p.get("pm_acc") is None or prev_pm is None)
                    else p["pm_acc"] - prev_pm,
                    "d_local_benign_asr": None if (p.get("local_benign_asr") is None
                                                  or prev_asr is None)
                    else p["local_benign_asr"] - prev_asr})
    return out


def frozen_readout(m: dict) -> list:
    """full 点：冻结触发器列 − 主列（良性端池化的 fresh-PM ASR）。"""
    main = {r["round"]: r.get("local_benign_asr") for r in m.get("rounds") or []}
    out = []
    for f in m.get("frozen_rounds") or []:
        if f.get("phase") != "full":
            continue
        a, b = f.get("local_benign"), main.get(f["round"])
        out.append({"round": f["round"], "frozen": a, "main": b,
                    "diff": None if (a is None or b is None) else a - b})
    return out


def readout(m: dict) -> dict:
    run = m.get("run") or {}
    return {"run_id": (run.get("provenance") or {}).get("run_id"),
            "malicious_per_edge": run.get("malicious_per_edge"),
            "update_geometry": run.get("update_geometry"), "post_agg_eval": run.get("post_agg_eval"),
            "frozen_trigger": run.get("frozen_trigger"),
            "exit_code": m.get("exit_code"), "client_failures": len(m.get("client_failures") or []),
            "geometry": geometry_readout(m), "jump": jump_readout(m), "frozen_vs_main": frozen_readout(m)}


def _f(v):
    return "n/a" if v is None else f"{v:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("metrics", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = {}
    for p in a.metrics:
        m = json.loads(Path(p).read_text(encoding="utf-8"))
        r = readout(m)
        res[r["run_id"] or Path(p).name] = r
        g = r["geometry"]
        print(f"== {r['run_id']}  布点 {r['malicious_per_edge']}  exit={r['exit_code']}  "
              f"client_failures={r['client_failures']}")
        print(f"   更新 {g['n_updates']}（恶意 {g['n_malicious']} / 良性 {g['n_benign']}）  "
              f"AUROC：norm {_f(g['auroc_norm'])} | −cos_global {_f(g['auroc_neg_cos_global'])} | "
              f"−cos_edge（混合组 {g['n_mixed_updates']} 个更新）{_f(g['auroc_neg_cos_edge'])} "
              f"（同一批上的 −cos_global {_f(g['auroc_neg_cos_global_on_mixed'])}）")
        j = [x["d_pm_acc"] for x in r["jump"] if x["d_pm_acc"] is not None]
        ja = [x["d_local_benign_asr"] for x in r["jump"] if x["d_local_benign_asr"] is not None]
        print(f"   Δ_jump：{len(r['jump'])} 点  pm_acc 均值 {_f(np.mean(j)) if j else 'n/a'} | "
              f"良性 ASR 均值 {_f(np.mean(ja)) if ja else 'n/a'}")
        fz = [x["diff"] for x in r["frozen_vs_main"] if x["diff"] is not None]
        print(f"   冻结 − 主列（full 点）：{len(fz)} 点  均值 {_f(np.mean(fz)) if fz else 'n/a'}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"[g1p] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
