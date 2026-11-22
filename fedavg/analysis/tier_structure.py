"""
analysis/tier_structure.py  —  补充实验 D0b（N-015 / D-135）：在三层个性化 run 的快照上核对结构

    cd fedavg && python3 -m analysis.tier_structure --run TBS__k1-snap__s42 --run TBS__k1-snap__s43 --run TBS__k1-snap__s44 \\
        --here ../experiments/defense/edge-native --dumps-root ../../tfdpfl-dumps --out ../experiments/defense/edge-native/analysis/d0_structure.json

为什么要拆成 D0a / D0b（PREREG-v2 §二）：快照是**上传前**的边缘模型（本云轮最后一次边缘聚合之后、下一次广播之前，
`BackdoorCloudServer._snapshot`），所以「广播后上云段四个边缘逐位相同」在快照上看不到（那一半由 D0a 在 CPU 小联邦上真跑核对，
`tests/test_three_tier_personalization.py::test_d0a_structure_after_one_cloud_round`）。快照上能核对、且只有真实 run 才能给出的是：

  ① 全局模型的上云段 = 四个边缘上云段的加权平均：对上云段的坐标做最小二乘，拟合 4 个权重；
     残差相对值 ≤ 1e-5、权重全为正、和为 1（±1e-4）→ 云聚合确实只聚合了上云段、而且是平均；
  ② 全局模型的边缘段在各快照轮逐位不变（永远是初值），且与每个边缘的边缘段都不同 → 边缘段不上云；
  ③ 四个边缘的边缘段两两不同 → 各边缘各训各的；
  ④ 客户端私有部分（分类头 + BN 统计量）两两不同，且私有索引不含边缘段的可训练权重。

任何一条不过 → 停下，6.1 需要重新审视（PREREG-v2 §二）。

比较逻辑是纯 numpy（`check_round` / `check_run` / `fit_mixture`，本地 `tests/test_tier_structure.py` 用合成数组测）；
只有 `index_sets`（按配置建模型、取索引）需要 TF。只读快照，不训练、不改任何东西。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

REL_RESIDUAL_MAX = 1e-5
WEIGHT_SUM_TOL = 1e-4
MAX_FIT_COORDS = 400_000          # 最小二乘只取上云段的前这么多个坐标（k = 1 时上云段约 123 万个元素）


# ══════════════════════════════════════════════════════════════════════════
# 读快照（纯 numpy；格式 = BackdoorCloudServer._snapshot）
# ══════════════════════════════════════════════════════════════════════════

def load_npz(path) -> dict:
    """→ {meta, global: [...], edges: {e: [...]}, private: {cid: (idx, vals)}}。"""
    z = np.load(path)
    meta = json.loads(str(z["meta_json"]))
    n_w = len([k for k in z.files if k.startswith("global_")])
    snap = {"meta": meta, "global": [z[f"global_{i:03d}"] for i in range(n_w)],
            "edges": {int(e): [z[f"edge{int(e)}_{i:03d}"] for i in range(n_w)] for e in meta["edge_ids"]},
            "private": {}}
    for cid in meta["client_ids"]:
        idx = [int(i) for i in z[f"client{int(cid)}_idx"]]
        snap["private"][int(cid)] = (idx, [z[f"client{int(cid)}_{i:03d}"] for i in range(len(idx))])
    return snap


# ══════════════════════════════════════════════════════════════════════════
# 核对（纯 numpy）
# ══════════════════════════════════════════════════════════════════════════

def _flat(ws, idx, limit=None) -> np.ndarray:
    v = np.concatenate([np.asarray(ws[i], np.float64).ravel() for i in idx]) if idx else np.zeros(0)
    return v[:limit] if limit else v


def fit_mixture(target: np.ndarray, parts: list) -> dict:
    """target ≈ Σ w_e · parts[e] 的最小二乘：→ {weights, rel_residual}。"""
    A = np.stack(parts, axis=1)
    w, *_ = np.linalg.lstsq(A, target, rcond=None)
    res = target - A @ w
    denom = float(np.linalg.norm(target)) or 1.0
    return {"weights": [float(x) for x in w], "rel_residual": float(np.linalg.norm(res)) / denom}


def _equal(a, b, idx) -> bool:
    return all(np.array_equal(a[i], b[i]) for i in idx)


def check_round(snap: dict, sets: dict) -> dict:
    """一个快照轮的 ①③④ + ② 的「与每个边缘不同」。sets = {edge, cloud, head, stat}（get_weights 索引）。"""
    edge_idx, cloud_idx, stat_idx = list(sets["edge"]), list(sets["cloud"]), set(sets["stat"])
    eids = sorted(snap["edges"])
    g = snap["global"]
    out = {}

    fit = fit_mixture(_flat(g, cloud_idx, MAX_FIT_COORDS),
                      [_flat(snap["edges"][e], cloud_idx, MAX_FIT_COORDS) for e in eids])
    w = fit["weights"]
    out["cloud_is_mixture"] = {
        **{k: (v if k != "weights" else [round(x, 6) for x in v]) for k, v in fit.items()},
        "pass": bool(fit["rel_residual"] <= REL_RESIDUAL_MAX and min(w) > 0 and abs(sum(w) - 1) <= WEIGHT_SUM_TOL)}

    diff_from_edges = {e: not _equal(g, snap["edges"][e], edge_idx) for e in eids}
    out["global_edge_seg_differs_from_every_edge"] = {"per_edge": diff_from_edges,
                                                      "pass": bool(edge_idx) and all(diff_from_edges.values())}

    pairs = {f"{a}-{b}": not _equal(snap["edges"][a], snap["edges"][b], edge_idx) for a, b in combinations(eids, 2)}
    out["edge_segs_pairwise_distinct"] = {"pairs": pairs, "pass": bool(edge_idx) and all(pairs.values())}

    trainable_edge = set(edge_idx) - stat_idx
    priv = {cid: dict(zip(idx, vals)) for cid, (idx, vals) in snap["private"].items() if idx}
    leaks = sorted(cid for cid, d in priv.items() if set(d) & trainable_edge)
    head_idx = list(sets["head"])
    stat_list = sorted(stat_idx)

    def _n_dup(idx):
        if not idx:                    # 没有可比的索引（如统计量共享时）→ 不判，记 None
            return None
        dup = 0
        for a, b in combinations(sorted(priv), 2):
            if all(i in priv[a] and i in priv[b] and np.array_equal(priv[a][i], priv[b][i]) for i in idx):
                dup += 1
        return dup

    n_dup_head = _n_dup(head_idx)
    n_dup_stat = _n_dup([i for i in stat_list if all(i in d for d in priv.values())])
    out["private_distinct"] = {"n_clients_with_private": len(priv), "n_pairs": len(priv) * (len(priv) - 1) // 2,
                               "duplicate_head_pairs": n_dup_head, "duplicate_stat_pairs": n_dup_stat,
                               "clients_with_edge_trainables_in_private": leaks,
                               "pass": len(priv) >= 2 and n_dup_head == 0 and n_dup_stat in (0, None) and not leaks}
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def _digest(ws, idx) -> str:
    h = hashlib.sha256()
    for i in idx:
        h.update(np.ascontiguousarray(ws[i]).tobytes())
    return h.hexdigest()[:12]


def check_run(snaps: dict, sets: dict) -> dict:
    """snaps = {round: snap}；② 的「全局边缘段各轮不变」跨轮核对，其余逐轮。"""
    rounds = sorted(snaps)
    per = {str(r): check_round(snaps[r], sets) for r in rounds}
    dig = {str(r): _digest(snaps[r]["global"], sets["edge"]) for r in rounds}
    frozen = {"digest_per_round": dig, "pass": bool(sets["edge"]) and len(set(dig.values())) == 1}
    return {"rounds": per, "global_edge_seg_frozen": frozen,
            "pass": frozen["pass"] and all(v["pass"] for v in per.values())}


# ══════════════════════════════════════════════════════════════════════════
# 索引（需要 TF）
# ══════════════════════════════════════════════════════════════════════════

def index_sets(config: dict) -> dict:
    """按配置建同一个模型，取 {edge, cloud, head, stat} 索引（= CloudServer / 客户端实际用的那一套）。"""
    from models.cnn import build_model, get_base_head_indices, get_bn_stat_indices
    from utils.tier_split import edge_shared_blocks
    d = config["data"]
    model = build_model(input_shape=(d["img_size"], d["img_size"], 3), num_classes=d["num_classes"],
                        arch=config["model"]["arch"])
    k = edge_shared_blocks(config)
    s = get_base_head_indices(model, d["num_classes"], k)
    return {"k": k, "edge": s["edge_weight_indices"], "cloud": s["cloud_weight_indices"],
            "head": s["head_weight_indices"], "stat": get_bn_stat_indices(model)}


def run_one(run_id: str, here: Path, dumps_root: Path, group: str) -> dict:
    import yaml
    config = yaml.safe_load((here / "configs" / f"{run_id}.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((here / "results/P2" / group / f"{run_id}.metrics.json").read_text(encoding="utf-8"))
    sets = index_sets(config)
    snaps, shas = {}, {}
    for s in metrics["dumps"]["snapshots"]:
        p = dumps_root / s["path"]
        sha = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        if s.get("sha") and sha != s["sha"]:
            raise RuntimeError(f"{p} 的 sha {sha} 与 manifest {s['sha']} 不一致")
        snaps[int(s["round"])] = load_npz(p)
        shas[str(s["round"])] = sha
    res = check_run(snaps, sets)
    res.update({"k": sets["k"], "n_edge_tensors": len(sets["edge"]), "n_cloud_tensors": len(sets["cloud"]),
                "snapshot_sha": shas})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D0b：三层个性化快照上的结构核查（N-015）")
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--group", default="TBS")
    ap.add_argument("--here", required=True, help="experiments/defense/edge-native")
    ap.add_argument("--dumps-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {"preregistered": "N-015 D0b（experiments/thesis/PREREG-v2.md §二）",
           "thresholds": {"rel_residual_max": REL_RESIDUAL_MAX, "weight_sum_tol": WEIGHT_SUM_TOL},
           "runs": {r: run_one(r, Path(a.here), Path(a.dumps_root), a.group) for r in a.run}}
    out["pass"] = all(v["pass"] for v in out["runs"].values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for r, v in out["runs"].items():
        print(f"[D0b] {r}: {'pass' if v['pass'] else 'FAIL'}")
    print(f"[D0b] 全部 {'pass' if out['pass'] else 'FAIL —— 停下，6.1 需要重新审视'} → {a.out}")
    return 0 if out["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
