"""
analysis/d3_probes.py  —  阶段三 D3 离线探针（TF；只读集群上已有的快照，零训练）

规则：PREREG-v2 §五（预注册 N-015，`experiments/thesis/PREREG-v2.md` @ 25163a5）；实现取舍：DECISIONS D-138；
参数清单 `experiments/defense/edge-native/d3_configs.yaml`（内容 sha 写进输出 JSON，判定脚本核对）。
纯算术在 `analysis/d3_spec.py`（本地可测），判定在 `harness/d3_verdict.py`。

每个 run 一个作业：按 main.py 重建世界（`p0_snapshot.build_world`），对清单里这个 run 用到的每个快照轮 t：
  0. 重现闸：用 run 自己的评估代码重算第 t 轮全量点主列（`p0_snapshot.eval_full_main`，同一随机数键、同一顺序），
     四台边缘（含 E0）的良性 ASR 与 margin_p50 与 metrics.json 记录相差 ≤ reproduce_tol → 快照复原对了；
  P-a（SNAP 集中第 15 / 60 轮）重训攻击者所在边缘 E0：只训 stage4 / 只训 stem–stage3 / 训全部主干（P0 的 cft-train-high，
     `EdgeHardener` 换训练变量表）；每臂（含不重训的 none）把 E0 全部客户端的私有 BN 统计量在各自训练分片上**精确重估**；
     量 E0 良性端：主列 ASR（ξ 在当前攻击者 PM 上重求）、固定 ξ（none 臂的 ξ）的 ASR、margin、干净精度；
  P-b（TBS k1-snap 与 SNAP 集中第 60 轮）受害边缘良性端：ξ 在受害端自己的 fresh PM 上求（白盒）vs 在攻击者 PM 上求（主列），
     同一批探针图、同一个随机起点、同一个 δ；另报只 ξ（两种模型）、只 δ、无触发器；
  P-c（9 个 run × 第 6 / 15 / 60 轮）每台边缘的 500 张干净图加主列触发器前后，在边缘模型上一整批训练模式前向，
     各 BN 层输入逐通道矩的高斯 KL（加触发器 ‖ 干净），层均值 = kl_all（`defense/edge_ref._moments` + `edge_ref_spec.kl_scores`）。

**不改变任何外部状态**：全部随机数来自 `default_rng([seed, 0xD3, 探针, 轮, …])`；草稿模型（PM 槽、重估模型、KL 提取模型）在
`random.getstate()/setstate()` 围栏里创建（clone_model 推进 Python random，F-078）；P-a 改过的客户端统计量、边缘模型、
生成器权重（评估侧 δ 的训练模式前向会动它的 BN 运行统计量）都在探针结束时写回。不碰全局 np.random。

用法（cwd = fedavg，经 cluster_env.sh 的容器；作业脚本 experiments/defense/edge-native/d3.sbatch）：
    python3 -m analysis.d3_probes --run SNAP__collocated__s42 --here ../experiments/defense/edge-native \\
        --dumps-root ../../tfdpfl-dumps --out ../experiments/defense/edge-native/analysis/d3/SNAP__collocated__s42.json
回传只有这一个 JSON（逐 edge / 逐端的数），不回传任何张量（CLAUDE.md 协议）。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import yaml

from analysis import d3_spec as S
from analysis import p0_snapshot as P0
from attack.backdoor_eval import _acc_on_dataset, dataset_to_numpy
from attack.eval_detail import client_record
from defense import edge_ref_spec as ERS
from hardening import spec as HS
from models.model_utils import clone_model
from utils.pm import compose_pm

PB_COLUMNS = ("main", "whitebox", "xi_only_victim", "xi_only_attacker", "delta_only", "clean")


# ══════════════════════════════════════════════════════════════════════════
# 公共
# ══════════════════════════════════════════════════════════════════════════

def rng_for(world, probe: str, t: int, *rest) -> np.random.Generator:
    return np.random.default_rng([int(world.seed), S.TAG, S.PROBE_TAGS[probe], int(t)] + [int(r) for r in rest])


@contextlib.contextmanager
def fence(world):
    """探针前后：Python random 状态、生成器权重原样写回；PM 的边缘权重缓存清掉。"""
    st = random.getstate()
    att = world.cloud.eval_attacker
    gen = getattr(att, "_atk_generator", None) if att is not None else None
    gen_w = None if gen is None else gen.get_weights()
    try:
        yield
    finally:
        if gen_w is not None:
            gen.set_weights(gen_w)
        world.cloud._edge_w_cache = None
        random.setstate(st)


def scratch(world, name: str):
    """本工具专用的草稿模型（第一次用到才建，在 random 围栏里）。"""
    pool = world.__dict__.setdefault("_d3_scratch", {})
    if name not in pool:
        st = random.getstate()
        try:
            pool[name] = clone_model(world.global_model)
        finally:
            random.setstate(st)
    return pool[name]


def probe_set(world, c):
    """客户端的主列探针图：留出分片前 asr_max_samples 张、全部类别（four_way 的取法，`compute_asr_four_way`）。"""
    cache = world.__dict__.setdefault("_d3_probe", {})
    cid = int(c.client_id)
    if cid not in cache:
        X, Y = dataset_to_numpy(c.test_dataset, world.cloud.bd_asr_max)
        cache[cid] = (np.asarray(X, np.float32), np.asarray(Y).reshape(-1).astype(np.int64))
    return cache[cid]


def predict(model, x, batch: int = 256) -> np.ndarray:
    return np.concatenate([model(x[i:i + batch], training=False).numpy() for i in range(0, len(x), batch)])


def record(world, c, model, Y, xp, acc=None) -> dict:
    """一列触发器 → run 同款的逐端记录（`eval_detail.client_record`：过滤口径 ASR + 非目标类样本的 margin）。"""
    cloud = world.cloud
    p = predict(model, xp)
    rec = client_record(client_id=int(c.client_id), edge_id=int(c.assigned_edge),
                        malicious=int(c.client_id) in world.malicious, target=cloud.bd_target,
                        n_classes=cloud._n_classes, trig_probs=p, trig_pred=np.argmax(p, 1), y_probe=Y)
    rec["acc"] = acc
    return rec


def attacker_pm(world):
    cloud = world.cloud
    return cloud.pm_model(cloud.eval_attacker, cloud.pm_kind, slot="d3_attacker")


def victim_pm(world, c):
    return world.cloud.main_pm(c, slot="d3_victim")


# ══════════════════════════════════════════════════════════════════════════
# P-a：精确重估 + 子集重训
# ══════════════════════════════════════════════════════════════════════════

def bn_layers(model):
    return [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]


def precise_bn(model, x, batch: int) -> list:
    """
    D-138 ①：不打乱、不增强，逐批训练模式前向；第 k 批前把每个 BN 层的 momentum 设成 n_前 / (n_前 + n_k)
    （`d3_spec.precise_momenta`）→ 结束时 moving 统计量 = 各批 batch 统计量的样本数加权均值，旧值完全不留。
    方差口径就是 Keras 自己更新 moving_variance 的那个（同一段代码）。只改 BN 运行统计量；momentum 原样写回。
    返回 model.get_weights()。
    """
    bns = bn_layers(model)
    saved = [l.momentum for l in bns]
    spans = S.seq_batches(len(x), int(batch))
    try:
        for (s, e), m in zip(spans, S.precise_momenta([e - s for s, e in spans])):
            for l in bns:
                l.momentum = float(m)
            model(tf.convert_to_tensor(x[s:e], tf.float32), training=True)
    finally:
        for l, m in zip(bns, saved):
            l.momentum = m
    return model.get_weights()


def train_images(c) -> np.ndarray:
    images, _, idx = c._train_src
    return np.asarray(images[np.asarray(idx)], np.float32)


def reestimate(world, clients, edge_w, batch: int) -> dict:
    """每个客户端：[edge_w + 自己的 head + 自己的统计量] → 在自己训练分片上精确重估 → 写进 c._stats。返回记账。"""
    model = scratch(world, "d3_reest")
    done, skipped = [], []
    for c in clients:
        idx, vals = c.private_state()
        if not idx or getattr(c, "_stats", None) is None:
            skipped.append(int(c.client_id))            # 从未参训（PM = 边缘模型）或统计量共享：没有私有统计量可重估
            continue
        model.set_weights(compose_pm(edge_w, vals, idx))
        w = precise_bn(model, train_images(c), batch)
        c._stats = [np.array(w[i], copy=True) for i in c._stat_w_idx]
        done.append(int(c.client_id))
    return {"n": len(done), "skipped": skipped}


def owner_names(model, variables) -> list:
    own = {}
    for layer in model.layers:
        for v in layer.weights:
            own.setdefault(id(v), layer.name)
    return [own.get(id(v)) for v in variables]


def harden_subset(hardener, weights, x, y, base_cfg, prefixes, *, rng, lr):
    """只训名字以 prefixes 开头的层的主干可训练变量（None = 全部主干）；其余照 `EdgeHardener.harden`。"""
    full = list(hardener.body_vars)
    sel = S.select_by_prefix(owner_names(hardener.model, full), prefixes)
    if not sel:
        raise ValueError(f"前缀 {prefixes} 没有选中任何主干变量")
    hardener.body_vars = [full[j] for j in sel]
    try:
        new, info = hardener.harden(weights, x, y, base_cfg, rng=rng, lr=lr)
    finally:
        hardener.body_vars = full
    info["n_vars"] = len(sel)
    return new, info


def _e0_eval(world, clients, t, *, fixed_xi: dict, deltas: dict, set_fixed: bool):
    """E0 良性端的主列（ξ 在当前攻击者 PM 上重求）与固定 ξ 两列 + 干净精度。"""
    att = world.cloud.eval_attacker
    m_att = attacker_pm(world)
    main, fixed = [], []
    for c in clients:
        cid = int(c.client_id)
        X, Y = probe_set(world, c)
        if cid not in deltas:
            deltas[cid] = att.eval_delta(X)
        xi = att.eval_xi(m_att, X, Y, rng=rng_for(world, "pa", t, cid))
        if set_fixed:
            fixed_xi[cid] = xi
        m_v = victim_pm(world, c)
        acc = _acc_on_dataset(m_v, c.test_dataset)
        main.append(record(world, c, m_v, Y, X + xi + deltas[cid], acc))
        fixed.append(record(world, c, m_v, Y, X + fixed_xi[cid] + deltas[cid], acc))
    return main, fixed


def probe_pa(world, snap, t, d3cfg, hardener, base_cfg) -> dict:
    cloud, att = world.cloud, world.cloud.eval_attacker
    E = int(att.assigned_edge)
    if E in world.victims:
        raise RuntimeError(f"评估攻击者在受害 edge {E} 上？victims={world.victims}")
    e_clients = [c for c in world.clients if int(c.assigned_edge) == E]
    benign = [c for c in e_clients if int(c.client_id) not in world.malicious]
    batch = int(d3cfg["pa"]["reestimate"]["batch"])
    lr = HS.lr_at(world.config, t)
    lr_h = HS.harden_lr(base_cfg, lr)
    x_cl, y_cl = world.clean[E]
    orig = {int(c.client_id): (None if c._stats is None else [np.array(s, copy=True) for s in c._stats])
            for c in e_clients}
    t0 = time.perf_counter()
    weights, infos = {"none": snap["edges"][E]}, {"none": None}
    for i, (arm, prefixes) in enumerate(S.arm_prefixes(d3cfg).items()):
        weights[arm], infos[arm] = harden_subset(hardener, snap["edges"][E], x_cl, y_cl, base_cfg, prefixes,
                                                 rng=rng_for(world, "train", t, E, i), lr=lr_h)
    out = {"edge": E, "lr": lr, "lr_harden": lr_h, "n_benign": len(benign), "arms": {},
           "train_seconds": round(time.perf_counter() - t0, 2)}
    fixed_xi, deltas = {}, {}
    try:
        with fence(world):
            # 描述性：不重训、也不重估（原快照统计量）—— 看「重估」本身改了多少
            P0.set_edges(world, snap["edges"])
            cloud.begin_pm_eval()
            m0, _ = _e0_eval(world, benign, t, fixed_xi={}, deltas=deltas, set_fixed=True)
            out["orig_stats"] = S.edge_summary(m0, edges=[E])[str(E)]
            for arm in ["none"] + list(S.arm_prefixes(d3cfg)):
                info = infos[arm]
                if info is not None and info.get("diverged_at") is not None:
                    out["arms"][arm] = {"diverged": True, "harden": S.trim_info(info)}
                    continue
                w_by = dict(snap["edges"])
                w_by[E] = weights[arm]
                P0.set_edges(world, w_by)
                cloud.begin_pm_eval()
                ts = time.perf_counter()
                re_ = reestimate(world, e_clients, weights[arm], batch)
                main, fixed = _e0_eval(world, benign, t, fixed_xi=fixed_xi, deltas=deltas, set_fixed=(arm == "none"))
                out["arms"][arm] = {"diverged": False,
                                    "harden": None if info is None else {**S.trim_info(info), "n_vars": info.get("n_vars")},
                                    "reest": re_,
                                    "main": S.edge_summary(main, edges=[E])[str(E)],
                                    "fixed": S.edge_summary(fixed, edges=[E])[str(E)],
                                    "clients": S.client_rows(main),
                                    "seconds": round(time.perf_counter() - ts, 2)}
                for c in e_clients:                       # 下一臂从原统计量出发（精确重估本来就不留旧值；写回是为了别的探针）
                    c._stats = None if orig[int(c.client_id)] is None else [np.array(s, copy=True)
                                                                           for s in orig[int(c.client_id)]]
    finally:
        for c in e_clients:
            c._stats = orig[int(c.client_id)]
        P0.set_edges(world, snap["edges"])
    return out


# ══════════════════════════════════════════════════════════════════════════
# P-b：受害模型上白盒求 ξ
# ══════════════════════════════════════════════════════════════════════════

def pb_client(world, c, m_att, t) -> dict:
    """一个受害端的六列（同一批探针图、同一个随机起点 u、同一个 δ）→ {列: 逐端记录}。"""
    att = world.cloud.eval_attacker
    cid = int(c.client_id)
    X, Y = probe_set(world, c)
    delta = att.eval_delta(X)
    m_v = victim_pm(world, c)
    xi_a = att.eval_xi(m_att, X, Y, rng=rng_for(world, "pb", t, cid))
    xi_v = att.eval_xi(m_v, X, Y, rng=rng_for(world, "pb", t, cid))
    cols = {"main": X + xi_a + delta, "whitebox": X + xi_v + delta, "xi_only_victim": X + xi_v,
            "xi_only_attacker": X + xi_a, "delta_only": X + delta, "clean": X}
    return {k: record(world, c, m_v, Y, xp) for k, xp in cols.items()}


def probe_pb(world, snap, t, metrics) -> dict:
    victims = list(world.victims)
    if not victims:
        raise RuntimeError("P-b 需要集中布点（有受害 edge）")
    with fence(world):
        P0.set_edges(world, snap["edges"])
        world.cloud.begin_pm_eval()
        m_att = attacker_pm(world)
        recs = {k: [] for k in PB_COLUMNS}
        for c in world.clients:
            if int(c.assigned_edge) in victims and int(c.client_id) not in world.malicious:
                for k, r in pb_client(world, c, m_att, t).items():
                    recs[k].append(r)
    summ = {k: S.edge_summary(v, edges=victims) for k, v in recs.items()}
    rec = S.recorded_edges(metrics, t) if metrics is not None else {}
    rv = [(rec.get(e) or {}).get("client_benign") for e in victims]
    return {"victims": victims, "columns": summ,
            "V": {k: S.group_mean(summ[k], victims) for k in PB_COLUMNS},
            "margin_V": {k: S.group_mean(summ[k], victims, key=S.MARGIN_COL) for k in PB_COLUMNS},
            "recorded_V": None if not rv or None in rv else float(np.mean(rv)),
            "clients": {k: S.client_rows(v) for k, v in recs.items() if k in ("main", "whitebox")}}


# ══════════════════════════════════════════════════════════════════════════
# P-c：干净集加触发器前后的 BN 层输入矩
# ══════════════════════════════════════════════════════════════════════════

def moments_tool(world):
    """`defense/edge_ref.EdgeRefDefense` 的提取部分（子模型取每个 BN 层的输入，训练模式整批前向），独立使用。"""
    tool = world.__dict__.get("_d3_moments")
    if tool is None:
        from defense.edge_ref import EdgeRefDefense
        tool = EdgeRefDefense()
        tool.edge = SimpleNamespace(model=world.global_model)
        st = random.getstate()
        try:
            tool._build()
        finally:
            random.setstate(st)
        world._d3_moments = tool
    return tool


def probe_pc(world, snap, t) -> dict:
    att, target = world.cloud.eval_attacker, int(world.cloud.bd_target)
    tool = moments_tool(world)
    out = {}
    with fence(world):
        P0.set_edges(world, snap["edges"])
        world.cloud.begin_pm_eval()
        m_att = attacker_pm(world)
        for e in world.edges:
            eid = int(e.edge_id)
            x, y = world.clean[eid]
            xi = att.eval_xi(m_att, x, y, rng=rng_for(world, "pc", t, eid))
            xt = x + xi + att.eval_delta(x)
            mc = tool._moments(snap["edges"][eid], x)
            mt = tool._moments(snap["edges"][eid], xt)
            sc = ERS.kl_scores([ERS.subset_moments(*m) for m in mt], [ERS.subset_moments(*m) for m in mc], tool._layers)
            nt = np.flatnonzero(np.asarray(y) != target)
            sc_nt = ERS.kl_scores([ERS.subset_moments(m1, m2, nt) for m1, m2 in mt],
                                  [ERS.subset_moments(m1, m2, nt) for m1, m2 in mc], tool._layers)
            out[str(eid)] = {**sc, "kl_all_nontarget": sc_nt["kl_all"], "n": int(len(y)), "n_nontarget": int(len(nt))}
    return {"edges": out, "direction": "KL(triggered || clean)"}


# ══════════════════════════════════════════════════════════════════════════
# 一个 run
# ══════════════════════════════════════════════════════════════════════════

def make_hardener(world, d3cfg, here: Path, batch=None):
    grid_spec = yaml.safe_load((Path(here) / d3cfg["pa"]["grid"]).read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in HS.expand_grid(grid_spec["grid"])}
    from hardening.core import EdgeHardener
    h = EdgeHardener(world.global_model, world.config, probe=grid_spec.get("probe"), kstar=grid_spec.get("kstar"),
                     batch=int(batch if batch is not None else grid_spec.get("batch", 32)))
    return h, by_id[d3cfg["pa"]["base_config"]]


def probe_run(world, d3cfg, plan: dict, snaps: dict, metrics: dict, *, hardener=None, base_cfg=None,
              out: dict | None = None, flush=lambda: None) -> dict:
    """plan = {探针: [轮]}；snaps = {轮: 快照路径}。逐轮：载入 → 重现闸 → 各探针。返回 out（schema 1）。"""
    out = out if out is not None else {}
    out.setdefault("rounds", {})
    for t in S.rounds_needed(plan):
        snap = P0.load_snapshot(snaps[t], world)
        P0.restore(world, snap)
        R = {"snapshot": str(snaps[t])}
        out["rounds"][str(t)] = R
        ts = time.perf_counter()
        full = P0.eval_full_main(world, snap, t, detail=True)
        rec = S.recorded_edges(metrics, t)
        R["reproduce"] = S.reproduce_check(full, rec, float(d3cfg["reproduce_tol"]))
        R["recomputed"] = {str(e): v for e, v in full.items()}
        R["recorded"] = {str(e): v for e, v in rec.items()}
        R["reproduce_seconds"] = round(time.perf_counter() - ts, 2)
        flush()
        if t in plan.get("pa", []):
            ts = time.perf_counter()
            R["pa"] = probe_pa(world, snap, t, d3cfg, hardener, base_cfg)
            R["pa"]["seconds"] = round(time.perf_counter() - ts, 2)
            flush()
        if t in plan.get("pb", []):
            ts = time.perf_counter()
            R["pb"] = probe_pb(world, snap, t, metrics)
            R["pb"]["seconds"] = round(time.perf_counter() - ts, 2)
            flush()
        if t in plan.get("pc", []):
            ts = time.perf_counter()
            R["pc"] = probe_pc(world, snap, t)
            R["pc"]["seconds"] = round(time.perf_counter() - ts, 2)
            flush()
    return out


def run(args) -> dict:
    here = Path(args.here)
    d3cfg = S.load_config(Path(args.d3_config) if args.d3_config else here / "d3_configs.yaml")
    plan = S.plan_for(d3cfg, args.run)
    if not plan:
        raise ValueError(f"{args.run} 不在 d3_configs.yaml 的任何探针里")
    if args.rounds:
        keep = {int(r) for r in args.rounds}
        plan = {p: [t for t in rs if t in keep] for p, rs in plan.items()}
        plan = {p: rs for p, rs in plan.items() if rs}
    config_path = here / "configs" / f"{args.run}.yaml"
    metrics_path = here / "results/P2" / S.group_of(args.run) / f"{args.run}.metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = {int(s["round"]): s for s in metrics["dumps"]["snapshots"]}
    snaps = {r: Path(args.dumps_root) / s["path"] for r, s in manifest.items()}
    need = S.rounds_needed(plan)
    missing = [r for r in need if r not in snaps or not snaps[r].exists()]
    if missing:
        raise FileNotFoundError(f"快照缺失：第 {missing} 轮（manifest {sorted(snaps)}，dumps-root {args.dumps_root}）")
    for r in need:
        if manifest[r].get("sha") and HS.sha12(snaps[r]) != manifest[r]["sha"]:
            raise RuntimeError(f"快照 {snaps[r]} 的 sha 与 metrics.json 的 manifest（{manifest[r]['sha']}）不一致")

    world = P0.build_world(config_path)
    rid = (world.config.get("meta") or {}).get("run_id")
    if rid not in (None, args.run):
        raise RuntimeError(f"配置里的 run_id {rid!r} ≠ {args.run!r}")
    hardener, base_cfg = make_hardener(world, d3cfg, here) if "pa" in plan else (None, None)
    att = world.cloud.eval_attacker
    out = {"schema": S.SCHEMA, "tool": "d3_probes", "run_id": args.run, "seed": world.seed,
           "placement": world.placement, "victims": world.victims, "malicious": sorted(world.malicious),
           "eval_attacker": None if att is None else int(att.client_id),
           "attacker_edge": None if att is None else int(att.assigned_edge),
           "config_sha": S.content_sha(d3cfg), "plan": plan,
           "snapshot_sha": {str(r): manifest[r].get("sha") for r in need}, "rounds": {}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        out_path.write_text(json.dumps(S.jsonable(out), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    probe_run(world, d3cfg, plan, snaps, metrics, hardener=hardener, base_cfg=base_cfg, out=out, flush=flush)
    flush()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--run", required=True, help="run_id，如 SNAP__collocated__s42")
    ap.add_argument("--here", required=True, help="experiments/defense/edge-native 目录")
    ap.add_argument("--dumps-root", required=True)
    ap.add_argument("--d3-config", help="缺省 = <here>/d3_configs.yaml")
    ap.add_argument("--rounds", nargs="+", help="只跑这几轮（探路；结果不进判定）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = run(a)
    ok = all((R.get("reproduce") or {}).get("pass") for R in res["rounds"].values())
    print(f"[d3] {a.out}（{time.perf_counter() - t0:.0f} s）重现闸 {'全过' if ok else '有不过的轮'}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
