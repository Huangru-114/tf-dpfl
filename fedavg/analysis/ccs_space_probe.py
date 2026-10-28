"""
analysis/ccs_space_probe.py  —  阶段三 SA-C：CCS 对抗样本「在哪个空间里做」的离线对比（零训练；D-109 Q1，用户 2026-10-28）

问题：官方代码在**标准化后**的输入空间里做 U(0,1) 贴片、eps = 0.5、clamp[0,1]（F-101 ②：clamp 会把比均值暗、
比 μ+σ 亮的像素整个夹过去）；原文没写这一层。把同样的三个数解释成 [0,1] 像素，对抗样本差多少？差到值得在 CCSF 里多跑一格吗？

做法：SNAP 的一个快照（缺省 s42 第 15 轮）→ `p0_snapshot.build_world` + `restore`（同 P0，不训练）→ 挑若干良性端，
每端取自己训练集的前 n 张、组出它的 fresh-PM（edge 模型 + 私有 head / BN 统计量），用 `client/ccs_mixin.CCSAttack`
（与在线训练**同一份代码**）在四个变体上造对抗样本：space ∈ {input, pixel} × steps ∈ {0（只有贴片）, 15}。
同一端的四个变体用同一条随机流（掩码与噪声逐位相同），只差空间与步数。

每个变体报告（端内取均值，再跨端取 fsum 均值）：
  clamp_frac      工作空间里原图落在 [0,1] 外的像素比例（= 第一次 clamp 会改动的比例；pixel 空间恒为 0）
  dpx_mean / dpx_linf / dpx_in_mask / dpx_out_mask   对抗样本与原图在 [0,1] 像素上的平均 |差|、逐样本 ‖·‖∞ 均值、掩码内 / 外的平均 |差|
  clean_acc / adv_acc   PM 在原图 / 对抗样本上的精度（推理模式）
  margin_adv      max-margin（z_second − z_y）在对抗样本上的均值
  kl              KL(p_clean ‖ p_adv)（训练损失里的那一项）
  feat_rel        ‖f_adv − f_clean‖ / ‖f_clean‖（f = head 的输入；SNAP 是 resnet10_torch → GAP 特征）
  mmd             MMD_rbf(f_clean, f_adv)（训练损失里的那一项，带宽同 ccs_spec）
  seconds_per_batch  造一批对抗样本的墙钟（第一次 trace 之后；估 CCSF 机时用）
回传只有 JSON（CLAUDE.md 协议）。**只报告，不进任何判定**；空间的取舍由用户定（D-109）。

用法（cwd = fedavg，经 cluster_env.sh 的容器；作业脚本 experiments/defense/edge-native/ccs_probe.sbatch）：
    python3 -m analysis.ccs_space_probe --config ../experiments/defense/edge-native/configs/SNAP__collocated__s42.yaml \\
        --metrics ../experiments/defense/edge-native/results/P2/SNAP/SNAP__collocated__s42.metrics.json \\
        --dumps-root ../../tfdpfl-dumps --round 15 --out ../experiments/defense/edge-native/analysis/ccs_space_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from analysis.p0_snapshot import build_world, load_snapshot, restore
from client.ccs_mixin import CCSAttack
from data.pixel_space import pixel_stats
from defense.ccs_spec import CCS_RNG_TAG, DEFAULTS, mmd_rbf
from hardening import spec
from utils.pm import compose_pm

PROBE_TAG = 0xB0B
VARIANTS = (("input", 15), ("pixel", 15), ("input", 0), ("pixel", 0))
SCHEMA = 1


def _mean(vals):
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return math.fsum(vals) / len(vals) if vals else None


def pick_benign(world, n_clients: int) -> list:
    """按 edge 轮流挑良性端（每 edge 按 client_id 升序），保证各 edge 都有；只挑有私有 head 的端。"""
    by_edge = {}
    for c in sorted(world.clients, key=lambda c: int(c.client_id)):
        if int(c.client_id) in world.malicious or c._head_weights is None:
            continue
        by_edge.setdefault(int(c.assigned_edge), []).append(c)
    out, k = [], 0
    while len(out) < n_clients and any(k < len(v) for v in by_edge.values()):
        for e in sorted(by_edge):
            if k < len(by_edge[e]) and len(out) < n_clients:
                out.append(by_edge[e][k])
        k += 1
    return out


def first_n(client, n: int):
    xs, ys = [], []
    for x, y in client.dataset:
        xs.append(np.asarray(x, np.float32))
        ys.append(np.asarray(y).reshape(-1).astype(np.int64))
        if sum(len(a) for a in xs) >= n:
            break
    return np.concatenate(xs)[:n], np.concatenate(ys)[:n]


def variant_stats(model, atk: CCSAttack, rng, x, y, mean, std) -> dict:
    mask, noise = atk.draw(rng, x.shape)
    yi = tf.constant(y.astype(np.int32))
    t0 = time.perf_counter()
    xa = atk.adv(tf.constant(x), yi, tf.constant(mask), tf.constant(noise)).numpy()
    sec = time.perf_counter() - t0
    a, b = atk.a.numpy(), atk.b.numpy()
    u = a * x + b
    fc, pc = (t.numpy() for t in atk.feat(x, training=False))
    fa, pa = (t.numpy() for t in atk.feat(xa, training=False))
    zc, za = atk.logits(fc).numpy(), atk.logits(fa).numpy()
    lc = zc - zc.max(1, keepdims=True)
    lc = lc - np.log(np.exp(lc).sum(1, keepdims=True))
    la = za - za.max(1, keepdims=True)
    la = la - np.log(np.exp(la).sum(1, keepdims=True))
    other = za.copy()
    other[np.arange(len(y)), y] = -np.inf
    px, pxa = x * std + mean, xa * std + mean
    d = np.abs(pxa - px)
    m = np.broadcast_to(mask > 0.5, d.shape)
    return {
        "clamp_frac": float(np.mean((u < 0.0) | (u > 1.0))),
        "dpx_mean": float(d.mean()),
        "dpx_linf": float(d.reshape(len(x), -1).max(1).mean()),
        "dpx_in_mask": float(d[m].mean()) if m.any() else None,
        "dpx_out_mask": float(d[~m].mean()) if (~m).any() else None,
        "clean_acc": float(np.mean(pc.argmax(1) == y)),
        "adv_acc": float(np.mean(pa.argmax(1) == y)),
        "margin_adv": float(np.mean(other.max(1) - za[np.arange(len(y)), y])),
        "kl": float(np.mean(np.sum(np.exp(lc) * (lc - la), axis=1))),
        "feat_rel": float(np.mean(np.linalg.norm(fa - fc, axis=1) / np.maximum(np.linalg.norm(fc, axis=1), 1e-12))),
        "mmd": float(mmd_rbf(fc, fa)),
        "seconds": sec,
    }


def run(args) -> dict:
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    manifest = {int(s["round"]): s for s in metrics["dumps"]["snapshots"]}
    t = int(args.round)
    if t not in manifest:
        raise FileNotFoundError(f"metrics.json 的快照 manifest 里没有第 {t} 轮（有 {sorted(manifest)}）")
    path = Path(args.dumps_root) / manifest[t]["path"]
    if manifest[t].get("sha") and spec.sha12(path) != manifest[t]["sha"]:
        raise RuntimeError(f"快照 {path} 的 sha 与 manifest（{manifest[t]['sha']}）不一致")
    world = build_world(args.config)
    snap = load_snapshot(path, world)
    restore(world, snap)
    mean, std = pixel_stats(world.config)
    mean, std = mean.reshape(1, 1, 1, -1), std.reshape(1, 1, 1, -1)
    st = random.getstate()
    try:
        from models.model_utils import clone_model
        model = clone_model(world.global_model)
    finally:
        random.setstate(st)
    p = dict(DEFAULTS)
    atks = {(sp, k): CCSAttack(model, {**p, "space": sp, "steps": k}, *pixel_stats(world.config))
            for sp, k in VARIANTS}
    clients = pick_benign(world, int(args.n_clients))
    rows = []
    for c in clients:
        idx, vals = c.private_state()
        model.set_weights(compose_pm(snap["edges"][int(c.assigned_edge)], vals, idx))
        x, y = first_n(c, int(args.n))
        row = {"client_id": int(c.client_id), "edge": int(c.assigned_edge), "n": int(len(x)), "variants": {}}
        for (sp, k), atk in atks.items():
            rng = np.random.default_rng([world.seed, int(c.client_id), CCS_RNG_TAG, PROBE_TAG])  # 四个变体同一组掩码 / 噪声
            row["variants"][f"{sp}-{k}"] = variant_stats(model, atk, rng, x, y, mean, std)
        rows.append(row)
        print(f"[ccs-probe] client {c.client_id} edge {c.assigned_edge}: "
              + " | ".join(f"{v}: adv_acc={r['adv_acc']:.3f} kl={r['kl']:.3f}" for v, r in row["variants"].items()))
    summary = {}
    for v in (f"{sp}-{k}" for sp, k in VARIANTS):
        keys = [k for k in (rows[0]["variants"][v] if rows else {}) if k != "seconds"]
        summary[v] = {k: _mean([r["variants"][v][k] for r in rows]) for k in keys}
        summary[v]["seconds_per_batch"] = _mean([r["variants"][v]["seconds"] for r in rows[1:]])
    return {"schema": SCHEMA, "run_id": (world.config.get("meta") or {}).get("run_id"), "round": t,
            "n_clients": len(rows), "n_per_client": int(args.n), "ccs_defaults": p,
            "note": "seconds_per_batch 不含第一个端（trace）；只报告，不进任何判定（D-109）",
            "summary": summary, "clients": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--dumps-root", required=True)
    ap.add_argument("--round", default="15")
    ap.add_argument("--n-clients", default="24")
    ap.add_argument("--n", default="32", help="每端取多少张（= 一批；与本仓库训练的 batch 32 相同 → 每批墙钟可直接估 CCSF 的开销；CCS 官方 batch 64）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = run(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    print(f"[ccs-probe] {a.out}（{time.perf_counter() - t0:.0f} s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
