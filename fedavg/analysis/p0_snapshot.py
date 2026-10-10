"""
analysis/p0_snapshot.py  —  阶段三 P0：SNAP 快照上的零训练离线反事实（SA0；PLAN §4 P0，判定规则 FINDINGS N-008）

问题：一次云聚合里，4 个 edge **都**在上传前用自己的 500 张干净图做对抗训练（AT）再 FedAvg，受害 edge 的灌入能砍掉多少（R_H），精度代价多少。

做法（快照 t ∈ {6, 15}；SNAP 的快照 = 第 t 云轮末：各 edge 的上传前模型 w_e、本轮云聚合结果 G、每端私有 head / BN 统计量、评估攻击者的生成器，F-092）：
  1. 按 main.py 的顺序重建这个 run 的世界（划分、客户端、edge、BackdoorCloudServer），**不训练**；从快照恢复全部状态；
  2. V0（N-008）：① FedAvg(w_e) 与快照 G 的 max|差| ≤ 1e-5；② 把每个 edge 设成 G，用 run 自己的云聚合后评估
     （`_light_acc` + `_light_asr(phase="post")`，同一随机数键）重算，与 run 记录的第 t+1 轮 `per_edge_post_agg_rounds` 比；
     ③ edge 设回 w_e，用全量点的主列评估（`_attacker_trigger` + `evaluate_hierarchical_asr`，同一随机数键）重算第 t 轮的 `per_edge_rounds`；
  3. 每个配置 H：4 个 edge 都加固（`hardening.core.EdgeHardener`）→ G′_H = FedAvg(H(w_e)) → 每个 edge 设成 G′_H → 同一个云聚合后评估；
  4. 附加读数（只报告）：阻尼、隔离攻击者 edge 的上界、触发器范数、CCS 聚类离线检测；冻结阶段另有 cloud 侧加固、干净集大小、
     只洗自己、加固前后 c_k、私有 head 均值当探针。

两个阶段（p0_configs.yaml 的 select / aux；N-008：配置清单在打开 s42 之前冻结）：
  screen  全部 32 个配置 + V0 + 基线 + 每快照附加读数 —— 只对 s42 collocated 跑，结果交 `harness/p0_verdict.py --select` 选前 3 名；
  frozen  冻结的前 3 名 + 同预算的 C-ft + 冻结阶段的附加读数 —— s42–44 × {collocated, distributed}。

用法（cwd = fedavg，经 cluster_env.sh 的容器；作业脚本 experiments/defense/edge-native/p0.sbatch）：
    python3 -m analysis.p0_snapshot --config ../experiments/defense/edge-native/configs/SNAP__collocated__s42.yaml \\
        --metrics ../experiments/defense/edge-native/results/P2/SNAP/SNAP__collocated__s42.metrics.json \\
        --dumps-root ../../tfdpfl-dumps --grid ../experiments/defense/edge-native/p0_configs.yaml \\
        --stage screen --out ../experiments/defense/edge-native/analysis/p0/SNAP__collocated__s42.screen.json
回传只有 JSON（每个配置 × 快照的受害 ASR / margin / pm_acc、V0 的数），不回传任何张量（CLAUDE.md 协议）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from analysis import bn_cluster
from analysis.ck_snapshot import _load_raw
from hardening import spec
from hardening.core import EdgeHardener

TAG = 0x5A0          # 本工具全部随机数的键（与训练、评估的键都不重叠）
SCHEMA = 1
# metrics.json 的 per_edge_post_agg_rounds 每行的列序 = harness/collect_metrics.LIGHT_EDGE_COLUMNS
# （harness 不在 fedavg 的 import 路径上 → 这里抄一份；守卫 tests/test_p0_snapshot.py 断言两份相同）
POST_EDGE_COLUMNS = ("edge_id", "edge_asr", "client_benign", "client_malicious", "pm_acc",
                     "em_acc", "margin_p50", "benign_asr_p90")


# ══════════════════════════════════════════════════════════════════════════
# 重建世界（main.run_experiment 到 BackdoorCloudServer 为止，不训练）
# ══════════════════════════════════════════════════════════════════════════

class World:
    pass


def build_world(config_path) -> World:
    """照 main.run_experiment 的顺序建出划分、客户端、edge 与 BackdoorCloudServer。不训练、不打开任何快照。"""
    import main as M
    from alignment import get_switch
    from models.cnn import build_model
    argv = sys.argv
    sys.argv = [argv[0], "--config", str(config_path)]          # load_config 会读命令行（main.py:289-299）
    try:
        config = M.load_config(str(config_path))
    finally:
        sys.argv = argv
    M.set_seed(config.get("seed", 42))
    if get_switch(config, "training.deterministic_ops"):
        tf.config.experimental.enable_op_determinism()
    x_train, y_train, x_test, y_test = _load_raw(config, M)
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    global_model = build_model(input_shape=(config["data"]["img_size"],) * 2 + (3,),
                               num_classes=config["data"]["num_classes"], arch=config["model"]["arch"],
                               rep_dim=int(config["model"].get("rep_dim", 64)))
    s3_out = {}
    clients, baked, _ = M.build_clients(x_all, y_all, global_model, config, s3_out=s3_out)
    edge_servers, _ = M.build_edge_servers(clients, global_model, config, precomputed_assignments=baked)
    if "clean_indices" not in s3_out:
        raise ValueError("这个配置不是 S3 划分：没有 edge 干净集（F-087）")
    for edge in edge_servers:
        edge.clean_indices = s3_out["clean_indices"][edge.edge_id]
        edge.set_test_dataset(M.merge_test_datasets(edge.clients, config["data"]["batch_size"]))
    g_test_ds = M.merge_test_datasets(edge_servers, config["data"]["batch_size"])
    bd_cfg = config.get("backdoor", {})
    malicious_ids = M.get_malicious_ids(bd_cfg)
    bd_trigger = M.build_trigger(bd_cfg, img_size=config["data"]["img_size"], config=config)
    eval_trigger = M.build_eval_trigger(bd_cfg, bd_trigger, clients, config)
    eval_attacker = M.select_eval_attacker(clients, malicious_ids, config.get("seed", 42))
    cloud = M.BackdoorCloudServer(global_model=global_model, edge_servers=edge_servers,
                                  test_dataset=g_test_ds, config=config, bd_cfg=bd_cfg,
                                  x_test=x_test, y_test=y_test, trigger_fn=eval_trigger,
                                  malicious_ids=malicious_ids, eval_attacker=eval_attacker)
    w = World()
    w.config, w.cloud, w.global_model = config, cloud, global_model
    w.edges = list(cloud.edge_servers)        # 与 run 的云聚合同序（FedAvg 的浮点求和顺序 → V0 ① 要逐位）
    w.clients, w.x_all, w.y_all = clients, x_all, y_all
    w.malicious = set(int(i) for i in malicious_ids)
    w.R = int(config["federation"].get("edge_rounds", 1) or 1)
    w.seed = int(config.get("seed", 42))
    mpe = list(config["backdoor"].get("malicious_per_edge") or [])
    w.victims = spec.victims(mpe) if mpe else []
    w.placement = "collocated" if sum(1 for m in mpe if int(m) > 0) == 1 else "distributed"
    w.clean = {int(e.edge_id): (np.asarray(x_all[np.asarray(e.clean_indices)], np.float32),
                                np.asarray(y_all[np.asarray(e.clean_indices)]).reshape(-1).astype(np.int64))
               for e in w.edges}
    return w


# ══════════════════════════════════════════════════════════════════════════
# 快照
# ══════════════════════════════════════════════════════════════════════════

def load_snapshot(path, world: World) -> dict:
    """读 npz → {meta, global, edges{e: [...]}, private{cid: (idx, vals)}, gen}；与重建的世界逐项核对，不一致就报错。"""
    z = np.load(path)
    meta = json.loads(str(z["meta_json"]))
    n_w = len(world.global_model.get_weights())
    snap = {"meta": meta,
            "global": [z[f"global_{i:03d}"] for i in range(n_w)],
            "edges": {int(e): [z[f"edge{int(e)}_{i:03d}"] for i in range(n_w)] for e in meta["edge_ids"]},
            "private": {}, "gen": None}
    for cid in meta["client_ids"]:
        idx = [int(i) for i in z[f"client{int(cid)}_idx"]]
        snap["private"][int(cid)] = (idx, [z[f"client{int(cid)}_{i:03d}"] for i in range(len(idx))])
    if meta.get("has_generator"):
        n_gen = len([k for k in z.files if k.startswith("gen_")])
        snap["gen"] = [z[f"gen_{i:03d}"] for i in range(n_gen)]
    errs = []
    got = {int(c.client_id): int(c.assigned_edge) for c in world.clients}
    want = {int(k): int(v) for k, v in zip(meta["client_ids"], meta["client_edge"])}
    if got != want:
        errs.append("重建的 client→edge 与快照的 client_edge 不一致（划分没复原）")
    if sorted(world.malicious) != sorted(int(i) for i in meta["malicious_ids"]):
        errs.append(f"恶意端不一致：重建 {sorted(world.malicious)} vs 快照 {meta['malicious_ids']}")
    att = world.cloud.eval_attacker
    if (None if att is None else int(att.client_id)) != meta.get("eval_attacker"):
        errs.append(f"评估攻击者不一致：重建 {None if att is None else att.client_id} vs 快照 {meta.get('eval_attacker')}")
    if meta.get("evaluated") and meta.get("edge_matches_eval") is not True:
        errs.append(f"快照的 edge 模型与该轮评估用的不一致（edge_matches_eval={meta.get('edge_matches_eval')}）")
    if errs:
        raise RuntimeError("快照与重建的世界对不上：\n  " + "\n  ".join(errs))
    return snap


def restore(world: World, snap: dict):
    """把快照状态装回：全局模型、每端私有 head / BN 统计量、评估攻击者的生成器。edge 模型由各评估自己设。"""
    world.global_model.set_weights(snap["global"])
    for c in world.clients:
        idx, vals = snap["private"][int(c.client_id)]
        if not idx:
            c._head_weights, c._stats = None, None
            continue
        head_idx, stat_idx = list(c._head_w_idx), list(c._stat_w_idx)
        if idx != head_idx + stat_idx:
            raise RuntimeError(f"client {c.client_id} 的私有索引与快照不一致：{idx} vs {head_idx + stat_idx}")
        c._head_weights = [np.array(v, copy=True) for v in vals[:len(head_idx)]]
        c._stats = [np.array(v, copy=True) for v in vals[len(head_idx):]] if stat_idx else None
    att = world.cloud.eval_attacker
    if snap["gen"] is not None and att is not None:
        att._atk_ensure_generator()
        att._atk_generator.set_weights(snap["gen"])


# ══════════════════════════════════════════════════════════════════════════
# 评估（全部调 run 自己的评估代码）
# ══════════════════════════════════════════════════════════════════════════

def set_edges(world: World, weights_by_edge: dict):
    for e in world.edges:
        e.model.set_weights(weights_by_edge[int(e.edge_id)])


def summarize(world: World, acc: dict, asr: dict) -> dict:
    """云聚合后点 / 轻评估点 → 本工具的读数：受害 edge 均值 V、池化良性 ASR、margin、pm_acc（池化 + 逐 edge）。"""
    pe = {int(d["edge_id"]): d for d in asr.get("per_edge") or []}
    de = {int(k): v for k, v in (asr.get("detail_edge") or {}).items()}
    per_edge = {}
    for e in world.edges:
        eid = int(e.edge_id)
        per_edge[eid] = {"client_benign": pe.get(eid, {}).get("client_benign"),
                         "client_malicious": pe.get(eid, {}).get("client_malicious"),
                         "pm_acc": (acc.get("per_edge") or {}).get(eid, {}).get("pm_acc"),
                         "margin_p50": de.get(eid, {}).get("margin_p50")}
    vict = [per_edge[e]["client_benign"] for e in world.victims]
    mv = [per_edge[e]["margin_p50"] for e in world.victims]
    return {"victim_asr": (spec.mean_or_none(vict) if world.victims and None not in vict else None),
            "margin_v": (spec.mean_or_none(mv) if world.victims and None not in mv else None),
            "pooled_benign": asr.get("local_asr_benign_mean"),
            "margin_pooled": (asr.get("detail") or {}).get("margin_p50"),
            "b0": per_edge.get(0, {}).get("client_benign"),
            "pm_acc": acc.get("pm_acc"), "per_edge": per_edge}


def eval_post(world: World, weights_by_edge: dict, t: int, phase: str = "post") -> dict:
    """把每个 edge 设成给定的模型，跑 run 自己的云聚合后评估（phase="post"，键 0x9057，eff = t·R）或轻评估点（"light"）。"""
    cloud = world.cloud
    set_edges(world, weights_by_edge)
    st = random.getstate()
    try:
        cloud.begin_pm_eval()
        acc = cloud._light_acc()
        asr = cloud._light_asr(int(t) + 1, 0, int(t) * world.R, phase=phase) or {}
    finally:
        cloud._edge_w_cache = None
        random.setstate(st)
    return summarize(world, acc, asr)


def eval_full_main(world: World, snap: dict, t: int) -> dict:
    """第 t 轮全量点的主列（`_backdoor_eval` 的同一调用：键 0xE7A1、先 global 再 edge 再各端）→ {edge: client_benign}。"""
    from attack.backdoor_eval import evaluate_hierarchical_asr
    cloud = world.cloud
    set_edges(world, snap["edges"])
    world.global_model.set_weights(snap["global"])
    st = random.getstate()
    try:
        cloud.begin_pm_eval()
        trig = cloud._attacker_trigger(int(t), 0, cloud.pm_kind)
        m = evaluate_hierarchical_asr(
            cloud.global_model, cloud.edge_servers, cloud._all_clients, cloud.test_dataset, trig,
            cloud.bd_target, cloud.malicious_ids, fallback_test_ds=cloud.test_dataset,
            local_model_fn=lambda c: cloud.main_pm(c), asr_max_samples=cloud.bd_asr_max,
            asr_columns=cloud.asr_columns, keep_probs=False, n_classes=cloud._n_classes)
    finally:
        cloud._edge_w_cache = None
        random.setstate(st)
    return {int(d["edge_id"]): d.get("client_benign") for d in m.get("per_edge") or []}


# ══════════════════════════════════════════════════════════════════════════
# run 记录的数
# ══════════════════════════════════════════════════════════════════════════

def recorded(metrics: dict, t: int) -> dict:
    """run 记录的：第 t+1 轮云聚合后点、第 t 轮全量点的逐 edge client_benign，与受害 edge 的 J_t。"""
    cols = list(POST_EDGE_COLUMNS)
    ei, ci = cols.index("edge_id"), cols.index("client_benign")
    rows = (metrics.get("per_edge_post_agg_rounds") or {}).get(str(int(t) + 1)) or []
    post = {int(r[ei]): r[ci] for r in rows if r and r[ei] is not None}
    full = {int(d["edge_id"]): d.get("client_benign")
            for d in (metrics.get("per_edge_rounds") or {}).get(str(int(t))) or []}
    return {"post": post, "full": full}


def victim_jump(rec: dict, victims: list):
    p = [rec["post"].get(e) for e in victims]
    f = [rec["full"].get(e) for e in victims]
    if not victims or None in p or None in f:
        return None
    return spec.mean_or_none(p) - spec.mean_or_none(f)


# ══════════════════════════════════════════════════════════════════════════
# 附加读数
# ══════════════════════════════════════════════════════════════════════════

def fedavg(world: World, weights_by_edge: dict, edges=None) -> list:
    from aggregation.fedavg import aggregate
    edges = [e for e in world.edges if edges is None or int(e.edge_id) in edges]
    return aggregate([(weights_by_edge[int(e.edge_id)], e.n_samples, 0.0, 0.0) for e in edges])


def trigger_norms(world: World, t: int, n: int) -> dict:
    """评估攻击者在 G 上的 ‖δ‖∞ / ‖ξ‖∞ / ‖δ+ξ‖∞（受害端留出样本的前 n 张；像素与模型输入两个空间；逐样本取 max 后报分位数）。"""
    from data.pixel_space import pixel_stats
    cloud, att = world.cloud, world.cloud.eval_attacker
    if att is None:
        return {}
    xs, ys = [], []
    pool = set(world.victims) or {int(e.edge_id) for e in world.edges}    # 集中布点：受害 edge；分散：全部 edge
    for c in world.clients:
        if int(c.client_id) in world.malicious or int(c.assigned_edge) not in pool:
            continue
        for xb, yb in c.test_dataset:
            xs.append(xb.numpy())
            ys.append(np.asarray(yb.numpy()).reshape(-1))
        if sum(len(a) for a in ys) >= n:
            break
    x = np.concatenate(xs)[:n].astype(np.float32)
    y = np.concatenate(ys)[:n]
    _, std = pixel_stats(world.config)
    st = random.getstate()
    try:
        cloud.begin_pm_eval()
        xi_model = cloud.pm_model(att, cloud.pm_kind, slot="p0_attacker")
        xi = att.eval_xi(xi_model, x, y, rng=np.random.default_rng([world.seed, TAG, int(t), 7]))
        delta = att.eval_delta(x)
    finally:
        cloud._edge_w_cache = None
        random.setstate(st)

    def q(d):
        inp = np.max(np.abs(d).reshape(len(d), -1), axis=1)
        px = np.max(np.abs(d * std).reshape(len(d), -1), axis=1)
        return {"input_max": float(inp.max()), "input_p50": float(np.median(inp)),
                "pixel_max": float(px.max()), "pixel_p50": float(np.median(px)),
                "pixel_max_x255": float(px.max() * 255.0)}
    return {"n": int(len(x)), "delta": q(delta), "xi": q(xi), "delta_plus_xi": q(delta + xi)}


def ccs_clu(world: World, snap: dict) -> dict:
    """每个 edge：有私有 BN 统计量的端 → HDBSCAN（bn_cluster.detect）→ 剔除对恶意端的 TPR / FPR。"""
    out = {}
    for e in world.edges:
        vec = {}
        for c in e.clients:
            idx, vals = snap["private"][int(c.client_id)]
            n_head = len(c._head_w_idx)
            if idx and len(vals) > n_head:
                vec[int(c.client_id)] = bn_cluster.stat_vector(vals[n_head:])
        r = bn_cluster.detect(vec)
        out[int(e.edge_id)] = {**{k: r[k] for k in ("status", "min_cluster_size", "rejected")},
                               **bn_cluster.rates(r["rejected"], world.malicious, list(vec))}
    return out


def mean_private_head(world: World, edge, snap: dict):
    """本 edge 各端私有 head 的均值（诊断用，语义 diff A2）：(W, b)；没有私有 head → None。"""
    Ws, bs = [], []
    for c in edge.clients:
        idx, vals = snap["private"][int(c.client_id)]
        if idx:
            Ws.append(np.asarray(vals[0], np.float64))
            bs.append(np.asarray(vals[1], np.float64))
    if not Ws:
        return None
    return np.mean(Ws, axis=0).astype(np.float32), np.mean(bs, axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════

def rng_for(world: World, t: int, edge_id: int, cfg_index: int, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([world.seed, TAG, int(t), int(edge_id), int(cfg_index), int(salt)])


def harden_all(world, hardener, snap, cfg, cfg_index, t, lr, *, n_clean=None, head_diag=False,
               only_edges=None, salt=0):
    """每个 edge 用自己的干净集加固上传前的 w_e。返回 ({edge: 新权重}, {edge: info})。"""
    new, infos = {}, {}
    for e in world.edges:
        eid = int(e.edge_id)
        if only_edges is not None and eid not in only_edges:
            new[eid] = snap["edges"][eid]
            continue
        x, y = world.clean[eid]
        rng = rng_for(world, t, eid, cfg_index, salt)
        if n_clean is not None and n_clean < len(y):
            sub = spec.stratified_subset(y, n_clean, rng)
            x, y = x[sub], y[sub]
        head = mean_private_head(world, e, snap) if head_diag else None
        new[eid], infos[eid] = hardener.harden(snap["edges"][eid], x, y, cfg, rng=rng, lr=lr, head_init=head)
    return new, infos


def eval_config(world, hardener, snap, cfg, cfg_index, t, lr, **kw) -> dict:
    t0 = time.perf_counter()
    new, infos = harden_all(world, hardener, snap, cfg, cfg_index, t, lr, **kw)
    G_h = fedavg(world, new)
    ev = eval_post(world, {int(e.edge_id): G_h for e in world.edges}, t)
    return {"eval": ev, "harden": {str(k): v for k, v in infos.items()},
            "seconds": round(time.perf_counter() - t0, 2)}


def run(args) -> dict:
    grid_spec = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))
    grid = spec.expand_grid(grid_spec["grid"])
    by_id = {c["id"]: c for c in grid}
    index_of = {c["id"]: i for i, c in enumerate(grid)}
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    manifest = {int(s["round"]): s for s in metrics["dumps"]["snapshots"]}
    snaps = {r: Path(args.dumps_root) / s["path"] for r, s in manifest.items()}
    rounds = [int(r) for r in args.rounds]
    missing = [r for r in rounds if r not in snaps or not snaps[r].exists()]
    if missing:
        raise FileNotFoundError(f"快照缺失：第 {missing} 轮（manifest {sorted(snaps)}，dumps-root {args.dumps_root}）")
    for r in rounds:                                   # 文件就是 run 写下的那一份（manifest 的 sha）
        if manifest[r].get("sha") and spec.sha12(snaps[r]) != manifest[r]["sha"]:
            raise RuntimeError(f"快照 {snaps[r]} 的 sha 与 metrics.json 的 manifest（{manifest[r]['sha']}）不一致")

    if args.stage == "screen":
        cfg_ids = [c["id"] for c in grid]
        frozen = None
    else:
        frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
        top = list(frozen["frozen"])
        cfg_ids = top + [c for c in dict.fromkeys(spec.matched_cft(by_id[i], grid) for i in top) if c]
    if args.only:
        cfg_ids = [i for i in cfg_ids if i in set(args.only)]
        unknown = set(args.only) - set(by_id)
        if unknown:
            raise ValueError(f"未知配置：{sorted(unknown)}")

    world = build_world(args.config)
    hardener = EdgeHardener(world.global_model, world.config, probe=grid_spec.get("probe"),
                            kstar=grid_spec.get("kstar"), batch=int(grid_spec.get("batch", 32)))
    aux = grid_spec.get("aux") or {}
    out = {"schema": SCHEMA, "run_id": (world.config.get("meta") or {}).get("run_id"),
           "seed": world.seed, "placement": world.placement, "victims": world.victims,
           "malicious": sorted(world.malicious), "stage": args.stage,
           "grid_sha": hashlib.sha256(Path(args.grid).read_bytes()).hexdigest()[:12],
           "frozen": None if frozen is None else frozen.get("frozen"),
           "frozen_sha": (None if frozen is None
                          else hashlib.sha256(Path(args.frozen).read_bytes()).hexdigest()[:12]),
           "configs": cfg_ids, "fb_clients_per_cloud_round": spec.fb_clients_per_cloud_round(world.config),
           "snapshots": {}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for t in rounds:
        snap = load_snapshot(snaps[t], world)
        restore(world, snap)
        lr = spec.lr_at(world.config, t)
        rec = recorded(metrics, t)
        S = {"lr": lr, "J": victim_jump(rec, world.victims), "configs": {}, "aux": {}}
        out["snapshots"][str(t)] = S
        # V0
        G_re = fedavg(world, snap["edges"])
        base = eval_post(world, {int(e.edge_id): snap["global"] for e in world.edges}, t)
        full = eval_full_main(world, snap, t)
        S["v0"] = spec.v0_check(spec.max_abs_diff(G_re, snap["global"]),
                                {e: v["client_benign"] for e, v in base["per_edge"].items()}, rec["post"],
                                full, rec["full"])
        S["base"] = base
        flush()
        if not S["v0"]["pass"] and not args.ignore_v0:
            print(f"[p0] 第 {t} 轮 V0 不过：{S['v0']['reasons']} —— 停（N-008：invalid）", file=sys.stderr)
            break
        # 每快照的附加读数（与配置无关）
        if aux.get("trigger_norms"):
            set_edges(world, {int(e.edge_id): snap["global"] for e in world.edges})
            S["aux"]["trigger_norms"] = trigger_norms(world, t, int(aux["trigger_norms"].get("n", 256)))
        if aux.get("ccs_clu"):
            S["aux"]["ccs_clu"] = ccs_clu(world, snap)
        for lam in aux.get("damping_lambda") or []:
            damp = {int(e.edge_id): [w + float(lam) * (g - w) for w, g in zip(snap["edges"][int(e.edge_id)],
                                                                              snap["global"])]
                    for e in world.edges}
            S["aux"][f"damping_{lam:g}"] = eval_post(world, damp, t)
        if aux.get("oracle_isolation") and world.placement == "collocated":
            G_iso = fedavg(world, snap["edges"], edges=set(world.victims))
            S["aux"]["oracle_isolation"] = eval_post(world, {int(e.edge_id): G_iso for e in world.edges}, t)
        flush()
        # 配置
        for cid in cfg_ids:
            S["configs"][cid] = eval_config(world, hardener, snap, by_id[cid], index_of[cid], t, lr)
            flush()
        # 冻结阶段的附加读数
        if args.stage == "frozen" and not args.only:
            S["aux"].update(frozen_aux(world, hardener, snap, frozen, by_id, index_of, t, lr, aux))
            flush()
    return out


def frozen_aux(world, hardener, snap, frozen, by_id, index_of, t, lr, aux) -> dict:
    top = list(frozen["frozen"])
    res = {}
    if aux.get("cloud_side") and top:
        xs = np.concatenate([world.clean[int(e.edge_id)][0] for e in world.edges])
        ys = np.concatenate([world.clean[int(e.edge_id)][1] for e in world.edges])
        res["cloud_side"] = {}
        for cid in top:
            Gc, info = hardener.harden(snap["global"], xs, ys, by_id[cid],
                                       rng=rng_for(world, t, 99, index_of[cid], 1), lr=lr)
            res["cloud_side"][cid] = {"eval": eval_post(world, {int(e.edge_id): Gc for e in world.edges}, t),
                                      "harden": info}
    if aux.get("n_clean") and top:
        res["n_clean"] = {}
        for n in aux["n_clean"].get("n", []):
            res["n_clean"][str(n)] = eval_config(world, hardener, snap, by_id[top[0]], index_of[top[0]], t, lr,
                                                 n_clean=int(n), salt=2)
    if aux.get("head_diag") and top:
        res["head_diag"] = {cid: eval_config(world, hardener, snap, by_id[cid], index_of[cid], t, lr,
                                             head_diag=True, salt=3) for cid in top}
    if aux.get("e0_own") and top:
        att_edges = [int(e.edge_id) for e in world.edges if int(e.edge_id) not in world.victims]
        base = eval_post(world, snap["edges"], t, phase="light")
        res["own"] = {"edges": att_edges, "base": base, "configs": {}}
        for cid in top:
            new, infos = harden_all(world, hardener, snap, by_id[cid], index_of[cid], t, lr,
                                    only_edges=set(att_edges), salt=4)
            res["own"]["configs"][cid] = {"eval": eval_post(world, new, t, phase="light"),
                                          "harden": {str(k): v for k, v in infos.items()}}
    if aux.get("ck_before_after") and top:
        n = int(aux["ck_before_after"].get("n", 100))
        res["ck"] = {}
        for cid in top:
            new, _ = harden_all(world, hardener, snap, by_id[cid], index_of[cid], t, lr, salt=5)
            per = {}
            for e in world.edges:
                eid = int(e.edge_id)
                x, y = world.clean[eid]
                att = spec.stratified_subset(y, n, rng_for(world, t, eid, index_of[cid], 6))
                st = random.getstate()
                try:
                    hardener.model.set_weights(snap["edges"][eid])
                    before = hardener.body_ck(x, y, x[att], y[att])
                    hardener.model.set_weights(new[eid])
                    after = hardener.body_ck(x, y, x[att], y[att])
                finally:
                    random.setstate(st)
                per[str(eid)] = {"before": before, "after": after}
            res["ck"][cid] = per
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--dumps-root", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--stage", choices=("screen", "frozen"), required=True)
    ap.add_argument("--frozen", help="frozen 阶段：harness/p0_verdict.py --select 写出的冻结清单")
    ap.add_argument("--rounds", nargs="+", default=["6", "15"])
    ap.add_argument("--only", nargs="+", help="只跑这几个配置（探路作业）；不跑冻结阶段的附加读数")
    ap.add_argument("--ignore-v0", action="store_true", help="V0 不过也继续（只用于排查，结果不进判定）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.stage == "frozen" and not a.frozen:
        ap.error("--stage frozen 需要 --frozen")
    t0 = time.perf_counter()
    res = run(a)
    print(f"[p0] {a.out}（{time.perf_counter() - t0:.0f} s）")
    return 0 if all(s.get("v0", {}).get("pass") for s in res["snapshots"].values()) else 3


if __name__ == "__main__":
    sys.exit(main())
