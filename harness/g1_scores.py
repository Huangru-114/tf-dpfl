"""
harness/g1_scores.py  —  G1 的逐更新分数读数：几何 + 功能分数，edge 视角 vs 全局视角（S6b；D-087）

    python3 harness/g1_scores.py <metrics.json> [<metrics.json> …] [--json out.json] [--window 20] [--resamples 50]

**描述性读数，不是判定。** 判定规则要在数据之前写进 FINDINGS（N-007），判定脚本 `g1_verdict.py` 另写。
本脚本把 N-007 里的度量实现出来，供预注册之前对 G1P / 合成数据试跑、之后对 G1 回传读数。

每个上传更新有若干分数（越大越可疑）：
  s_ck                功能分数（原文 §7）：Δ_k = c_k(θ_before) − c_k(θ_i)，s = (max Δ − median Δ) / (MAD Δ + ε)
                      —— 来自 `ck_scores` / `ck_before`，只在评分点上有
  norm / norm_w / norm_s     ‖Δ‖：整个 body / 可训练权重部分 / BN 统计量部分（S6b 拆分；老日志没有后两者）
  neg_cos_global_w / neg_cos_edge_w   −cos（权重部分）：越不像其余更新越可疑

「视角」的定义（等池大小）：分数本身不分视角，差别在**拿谁当「正常」的参照来归一化**。
  · 每个更新属于一个窗口（`--window` 个有效轮，缺省 20，按 eff = (云轮−1)·R + edge 轮 分块）和一个 edge；
  · **edge 视角**：用本窗口内本 edge 的全部更新（池大小 m）求 median / MAD，对分数做稳健 z；
  · **全局视角（等池）**：从本窗口内**全部 edge** 的更新里随机抽 m 个（固定种子、重复 `--resamples` 次取平均）求 median / MAD；
  · 把全体更新的 z 池化后算恶意 vs 良性的 AUROC（秩和，平局取一半）；ΔAUROC = edge − 全局（等池）。
  另报 raw（不归一化）与「全局（整池）」。池里没有恶意或没有良性 → AUROC 为 None（陷阱 #13）。
集中布点（[10,0,0,0]）下 E0 全是恶意端：edge 视角在 E0 里没有良性对照，归一化会把攻击者 edge 的「整体偏高」抹掉 —— 这是结构，不是 bug。

纯标准库 + numpy，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))
from analysis.functional_score import auroc, update_score      # noqa: E402  纯 numpy

SCORES = ("s_ck", "norm", "norm_w", "norm_s", "neg_cos_global_w", "neg_cos_edge_w")
EPS = 1e-9


def _rz(x, ref):
    """稳健 z：(x − median(ref)) / (MAD(ref) + eps)。ref 为空 → None。"""
    ref = np.asarray(ref, dtype=np.float64)
    if ref.size == 0:
        return None
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med)))
    return (np.asarray(x, dtype=np.float64) - med) / (mad + EPS)


def collect_updates(m: dict) -> list:
    """metrics.json → 每个上传更新一条 {round, edge, er, eff, cid, mal, <分数…>}。缺的分数为 None。"""
    run = m.get("run") or {}
    R = int(run.get("edge_rounds") or 1)
    t = m.get("update_geometry") or {}
    cols = t.get("columns") or []
    by_key = {}
    for row in t.get("rows") or []:
        d = dict(zip(cols, row))
        for i, cid in enumerate(d.get("cid") or []):
            def at(name, i=i, d=d):
                v = d.get(name)
                return v[i] if isinstance(v, list) and i < len(v) else None
            nc = at("cos_global_w")
            ne = at("cos_edge_w")
            u = {"round": d["round"], "edge": d["edge_id"], "er": d["edge_round"],
                 "eff": (d["round"] - 1) * R + d["edge_round"], "cid": cid, "mal": bool(at("mal")),
                 "norm": at("norm"), "norm_w": at("norm_w"), "norm_s": at("norm_s"),
                 "neg_cos_global_w": None if nc is None else -nc,
                 "neg_cos_edge_w": None if ne is None else -ne, "s_ck": None}
            by_key[(d["round"], d["edge_id"], d["edge_round"], cid)] = u
    # 功能分数：ck_scores 与 ck_before 按 (云轮, edge, edge 轮) 配对
    cb = m.get("ck_before") or {}
    before = {}
    for row in cb.get("rows") or []:
        d = dict(zip(cb.get("columns") or [], row))
        before[(d["round"], d["edge_id"], d["edge_round"])] = d.get("c")
    cs = m.get("ck_scores") or {}
    for row in cs.get("rows") or []:
        d = dict(zip(cs.get("columns") or [], row))
        c0 = before.get((d["round"], d["edge_id"], d["edge_round"]))
        s, _ = update_score(c0, d["c"]) if (c0 is not None and d.get("c") is not None) else (None, None)
        key = (d["round"], d["edge_id"], d["edge_round"], d["cid"])
        u = by_key.get(key)
        if u is None:       # 没有几何记录（开关没开）：也留下功能分数
            u = {"round": d["round"], "edge": d["edge_id"], "er": d["edge_round"],
                 "eff": d["effective_round"], "cid": d["cid"], "mal": bool(d["mal"]),
                 **{k: None for k in SCORES}}
            by_key[key] = u
        u["s_ck"] = s
    return sorted(by_key.values(), key=lambda u: (u["eff"], u["edge"], u["cid"]))


def view_aurocs(updates: list, key: str, window: int = 20, resamples: int = 50, seed=0) -> dict:
    """某个分数在三种参照下的 AUROC：raw / edge 视角 / 全局视角（整池 与 等池）及 ΔAUROC（edge − 等池全局）。"""
    ups = [u for u in updates if u.get(key) is not None]
    out = {"n": len(ups), "n_malicious": sum(u["mal"] for u in ups),
           "raw": None, "edge": None, "global_full": None, "global_matched": None, "delta": None}
    if not ups:
        return out
    mal = np.array([u["mal"] for u in ups])
    x = np.array([u[key] for u in ups], dtype=np.float64)
    out["raw"] = auroc(x[mal], x[~mal])
    block = np.array([(u["eff"] - 1) // int(window) for u in ups])
    edge = np.array([u["edge"] for u in ups])
    z_edge, z_gfull = np.full(len(ups), np.nan), np.full(len(ups), np.nan)
    pools = {}                                   # (窗口, edge) → 下标
    for i in range(len(ups)):
        pools.setdefault((int(block[i]), int(edge[i])), []).append(i)
    for b in set(block.tolist()):
        gi = np.where(block == b)[0]
        z_gfull[gi] = _rz(x[gi], x[gi])
    for (b, e), idx in pools.items():
        idx = np.array(idx)
        z_edge[idx] = _rz(x[idx], x[idx])
    ok = ~np.isnan(z_edge)
    out["edge"] = auroc(z_edge[ok & mal], z_edge[ok & ~mal])
    out["global_full"] = auroc(z_gfull[ok & mal], z_gfull[ok & ~mal])
    # 等池全局：对每个 (窗口, edge) 池，从本窗口全部更新里抽同样大小的参照池
    rng = np.random.default_rng([0x91, int(seed)])
    vals = []
    for _ in range(int(resamples)):
        z = np.full(len(ups), np.nan)
        for (b, e), idx in pools.items():
            idx = np.array(idx)
            gi = np.where(block == b)[0]
            ref = rng.choice(gi, size=min(len(idx), len(gi)), replace=False)
            z[idx] = _rz(x[idx], x[ref])
        v = auroc(z[mal & ~np.isnan(z)], z[~mal & ~np.isnan(z)])
        if v is not None:
            vals.append(v)
    if vals:
        out["global_matched"] = float(np.mean(vals))
        if out["edge"] is not None:
            out["delta"] = float(out["edge"] - out["global_matched"])
    return out


def readout(m: dict, window: int = 20, resamples: int = 50) -> dict:
    run = m.get("run") or {}
    prov = run.get("provenance") or {}
    ups = collect_updates(m)
    seed = run.get("seed") or 0
    return {"run_id": prov.get("run_id"), "seed": seed,
            "malicious_per_edge": run.get("malicious_per_edge"), "edge_rounds": run.get("edge_rounds"),
            "partition": (run.get("data") or {}).get("partition"),
            "exit_code": m.get("exit_code"), "client_failures": len(m.get("client_failures") or []),
            "n_updates": len(ups), "window": window,
            "scores": {k: view_aurocs(ups, k, window, resamples, seed) for k in SCORES}}


def _f(v):
    return "  n/a" if v is None else f"{v:5.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("metrics", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--resamples", type=int, default=50)
    a = ap.parse_args(argv)
    res = {}
    for p in a.metrics:
        r = readout(json.loads(Path(p).read_text(encoding="utf-8")), a.window, a.resamples)
        res[r["run_id"] or Path(p).name] = r
        print(f"== {r['run_id']}  布点 {r['malicious_per_edge']}  R={r['edge_rounds']}  "
              f"exit={r['exit_code']}  client_failures={r['client_failures']}  更新 {r['n_updates']}")
        print("   分数                 n(恶意)     raw  edge视角 全局(整池) 全局(等池)    Δ(edge−等池)")
        for k, v in r["scores"].items():
            print(f"   {k:<20} {v['n']:>5}({v['n_malicious']:>3})  {_f(v['raw'])}  {_f(v['edge'])}    "
                  f"{_f(v['global_full'])}     {_f(v['global_matched'])}       {_f(v['delta'])}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"[g1] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
