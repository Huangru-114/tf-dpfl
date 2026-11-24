"""
analysis/d3_pa2.py  —  P-a 追加探针（加大训练量；TF；只读集群上已有的 SNAP 快照，零训练）

起因：D3 的 P-a（FINDINGS F-134）三臂的清除比例全是 0 —— 沿用 P0 的 cft-train-high（lr = 当轮客户端 lr × 0.1、75 步、
临时线性探针头），权重几乎没动，探针没有分辨力。规则：预注册 **N-016**（本目录上级的 FINDINGS.md）、决定 **D-140**；
参数清单 `experiments/defense/edge-native/d3pa2_configs.yaml`（内容 sha 写进输出，判定脚本核对）。判定：`harness/d3pa2_verdict.py`。

两步：
  calibrate  SNAP 集中 s42 × 第 15 / 60 轮，只跑「训全部主干」，12 档训练量（固定学习率 × epoch × 分类头）都跑；
             判定脚本按规则选出能清除（f ≥ 0.5、精度降幅 ≤ 0.05）的最小一档 → 冻结清单 `analysis/d3pa2_frozen.json`；
  main       冻结的那一档 × 三臂（只训最后一块 / 只训前面几层 / 训全部主干）× s42–44 × 第 15 / 60 轮。

每个 (run, 轮)：先过重现闸（同 D3：`p0_snapshot.eval_full_main` 重算第 t 轮主列、四台边缘 ≤ 0.01）；
「不重训、只重估」的基线 + 每档 × 每臂：重训 E0 → E0 全部客户端在各自训练分片上精确重估 BN 统计量 →
E0 良性端主列（ξ 在当前攻击者 PM 上重求）/ 固定 ξ（基线的 ξ）两列的攻击成功率、margin、干净精度。
评估部分与 D3 P-a 相同（同一组随机数键 → 基线与 D3 的 none 臂一致，可以交叉核对）；训练部分只改三处：
学习率直接给（不随轮次、不乘 0.1）、epoch 数覆盖、分类头可选「随机重新初始化」。

**不改变任何外部状态**（同 D3：随机数只来自专用键；草稿模型在 random 围栏里建；客户端统计量、边缘模型、生成器权重都写回）。

用法（cwd = fedavg；作业脚本 experiments/defense/edge-native/d3pa2.sbatch）：
    python3 -m analysis.d3_pa2 --stage calibrate --run SNAP__collocated__s42 --here ../experiments/defense/edge-native \\
        --dumps-root ../../tfdpfl-dumps --out ../experiments/defense/edge-native/analysis/d3pa2/SNAP__collocated__s42.calibrate.json
    python3 -m analysis.d3_pa2 --stage main --run SNAP__collocated__s43 … --frozen ../experiments/defense/edge-native/analysis/d3pa2_frozen.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from analysis import d3_probes as D3
from analysis import d3_spec as S
from analysis import p0_snapshot as P0
from hardening import spec as HS

PA2_TAG = 0xA2


# ══════════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════════

def reinit_head(hardener, rng: np.random.Generator):
    """随机重新初始化的分类头：Glorot 均匀（limit = √(6 / (d + K))），偏置 0。d = 倒数第二层特征维数。"""
    d = int(hardener.feat.output_shape[-1])
    K = int(hardener.K)
    lim = float(np.sqrt(6.0 / (d + K)))
    W = rng.uniform(-lim, lim, size=(d, K)).astype(np.float32)
    return W, np.zeros((K,), np.float32)


def budget_cfg(base_cfg: dict, budget: dict) -> dict:
    """cft-train-high 的其余设置原样，只覆盖 epoch 数（学习率在调用时直接给）。"""
    c = dict(base_cfg)
    c["epochs"] = int(budget["epochs"])
    c["id"] = f"{base_cfg['id']}+{budget['id']}"
    return c


def train_arm(world, snap, t, E, hardener, base_cfg, budget, arm, prefixes):
    """一档 × 一臂：重训 E0 的上传前模型。随机数键 = [seed, 0xD3, 0xA2, 轮, 档号, 臂号]（标定与正式同一档可复现）。"""
    key = [int(world.seed), S.TAG, PA2_TAG, int(t), int(budget["index"]), int(S.PA2_ARM_INDEX[arm])]
    head = None
    if budget["head"] == "reinit":
        head = reinit_head(hardener, np.random.default_rng(key + [1]))
    x, y = world.clean[E]
    new, info = D3.harden_subset(hardener, snap["edges"][E], x, y, budget_cfg(base_cfg, budget), prefixes,
                                 rng=np.random.default_rng(key), lr=float(budget["lr"]), head_init=head)
    info["head"] = budget["head"]
    return new, info


# ══════════════════════════════════════════════════════════════════════════
# 一个快照轮
# ══════════════════════════════════════════════════════════════════════════

def run_round(world, snap, t, cfg2, hardener, base_cfg, budgets: list, arms: dict) -> dict:
    """
    budgets：要跑的训练量档（pa2_grid 的项）；arms：{臂名: 层名前缀 | None}。
    返回 {edge, lr_client, n_benign, baseline, budgets: {档 id: {budget, arms: {臂: {...}}}}}。
    """
    cloud, att = world.cloud, world.cloud.eval_attacker
    E = int(att.assigned_edge)
    if E in world.victims:
        raise RuntimeError(f"评估攻击者在受害 edge {E} 上？victims={world.victims}")
    e_clients = [c for c in world.clients if int(c.assigned_edge) == E]
    benign = [c for c in e_clients if int(c.client_id) not in world.malicious]
    batch = int(cfg2["reestimate"]["batch"])
    orig = {int(c.client_id): (None if c._stats is None else [np.array(s, copy=True) for s in c._stats])
            for c in e_clients}
    prev_edges = {int(e.edge_id): e.model.get_weights() for e in world.edges}   # 结束时原样写回（不只是设回快照）
    fixed_xi, deltas = {}, {}
    out = {"edge": E, "lr_client": HS.lr_at(world.config, t), "n_benign": len(benign), "budgets": {}}

    def restore_stats():
        for c in e_clients:
            o = orig[int(c.client_id)]
            c._stats = None if o is None else [np.array(s, copy=True) for s in o]

    def evaluate(w_E, set_fixed):
        w_by = dict(snap["edges"])
        w_by[E] = w_E
        P0.set_edges(world, w_by)
        cloud.begin_pm_eval()
        ts = time.perf_counter()
        re_ = D3.reestimate(world, e_clients, w_E, batch)
        main, fixed = D3._e0_eval(world, benign, t, fixed_xi=fixed_xi, deltas=deltas, set_fixed=set_fixed)
        restore_stats()
        return {"diverged": False, "reest": re_,
                "main": S.edge_summary(main, edges=[E])[str(E)],
                "fixed": S.edge_summary(fixed, edges=[E])[str(E)],
                "clients": S.client_rows(main), "eval_seconds": round(time.perf_counter() - ts, 2)}

    try:
        with D3.fence(world):
            out["baseline"] = evaluate(snap["edges"][E], set_fixed=True)       # 不重训、只重估（= D3 P-a 的 none 臂）
            for budget in budgets:
                row = {"budget": {k: budget[k] for k in ("id", "index", "lr", "epochs", "head")}, "arms": {}}
                out["budgets"][budget["id"]] = row
                for arm, prefixes in arms.items():
                    ts = time.perf_counter()
                    new, info = train_arm(world, snap, t, E, hardener, base_cfg, budget, arm, prefixes)
                    h = {**S.trim_info(info), "n_vars": info.get("n_vars"), "head": info.get("head"),
                         "train_seconds": round(time.perf_counter() - ts, 2)}
                    if info.get("diverged_at") is not None:
                        row["arms"][arm] = {"diverged": True, "harden": h}
                        continue
                    row["arms"][arm] = {**evaluate(new, set_fixed=False), "harden": h}
    finally:
        restore_stats()
        P0.set_edges(world, prev_edges)
    return out


def probe_run(world, cfg2, stage, rounds, snaps, metrics, hardener, base_cfg, budgets, arms, *, out=None,
              flush=lambda: None) -> dict:
    out = out if out is not None else {}
    out.setdefault("rounds", {})
    for t in rounds:
        snap = P0.load_snapshot(snaps[t], world)
        P0.restore(world, snap)
        ts = time.perf_counter()
        full = P0.eval_full_main(world, snap, t, detail=True)
        rec = S.recorded_edges(metrics, t)
        R = {"snapshot": str(snaps[t]),
             "reproduce": S.reproduce_check(full, rec, float(cfg2["reproduce_tol"])),
             "reproduce_seconds": round(time.perf_counter() - ts, 2)}
        out["rounds"][str(t)] = R
        flush()
        ts = time.perf_counter()
        R.update(run_round(world, snap, t, cfg2, hardener, base_cfg, budgets, arms))
        R["seconds"] = round(time.perf_counter() - ts, 2)
        flush()
    return out


# ══════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════

def make_hardener(world, cfg2: dict, here: Path, batch=None):
    """P0 网格里的 base_config（cft-train-high）+ 同一个 EdgeHardener（探针设置、批大小取网格文件的）。"""
    return D3.make_hardener(world, {"pa": {"grid": cfg2["grid"], "base_config": cfg2["base_config"]}}, Path(here),
                            batch=batch)


def plan(cfg2: dict, stage: str, run_id: str, frozen: dict | None):
    """→ (轮, 档列表, 臂 {名: 前缀})。"""
    arms_all = S.pa2_arm_prefixes(cfg2)
    if stage == "calibrate":
        if run_id != cfg2["calibrate"]["run"]:
            raise ValueError(f"标定只跑 {cfg2['calibrate']['run']}，不是 {run_id}")
        arm = cfg2["calibrate"]["arm"]
        return [int(r) for r in cfg2["calibrate"]["rounds"]], S.pa2_grid(cfg2), {arm: arms_all[arm]}
    if run_id not in cfg2["main"]["runs"]:
        raise ValueError(f"{run_id} 不在 main.runs 里")
    if not frozen or frozen.get("label") != "selected" or not frozen.get("selected"):
        raise ValueError(f"冻结清单没有选出训练量（label = {None if not frozen else frozen.get('label')!r}）→ 不跑正式阶段")
    if frozen.get("config_sha") != S.content_sha(cfg2):
        raise ValueError("冻结清单的参数清单 sha 与当前清单不一致")
    return [int(r) for r in cfg2["main"]["rounds"]], [S.pa2_budget(cfg2, frozen["selected"])], arms_all


def run(args) -> dict:
    here = Path(args.here)
    cfg2 = S.load_pa2_config(Path(args.config) if args.config else here / "d3pa2_configs.yaml")
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8")) if args.frozen else None
    rounds, budgets, arms = plan(cfg2, args.stage, args.run, frozen)
    metrics = json.loads((here / "results/P2" / S.group_of(args.run) / f"{args.run}.metrics.json").read_text(encoding="utf-8"))
    manifest = {int(s["round"]): s for s in metrics["dumps"]["snapshots"]}
    snaps = {r: Path(args.dumps_root) / s["path"] for r, s in manifest.items()}
    missing = [r for r in rounds if r not in snaps or not snaps[r].exists()]
    if missing:
        raise FileNotFoundError(f"快照缺失：第 {missing} 轮（manifest {sorted(snaps)}，dumps-root {args.dumps_root}）")
    for r in rounds:
        if manifest[r].get("sha") and HS.sha12(snaps[r]) != manifest[r]["sha"]:
            raise RuntimeError(f"快照 {snaps[r]} 的 sha 与 metrics.json 的 manifest（{manifest[r]['sha']}）不一致")

    world = P0.build_world(here / "configs" / f"{args.run}.yaml")
    rid = (world.config.get("meta") or {}).get("run_id")
    if rid not in (None, args.run):
        raise RuntimeError(f"配置里的 run_id {rid!r} ≠ {args.run!r}")
    hardener, base_cfg = make_hardener(world, cfg2, here)
    att = world.cloud.eval_attacker
    out = {"schema": S.SCHEMA, "tool": "d3_pa2", "stage": args.stage, "run_id": args.run, "seed": world.seed,
           "placement": world.placement, "victims": world.victims,
           "eval_attacker": None if att is None else int(att.client_id),
           "attacker_edge": None if att is None else int(att.assigned_edge),
           "config_sha": S.content_sha(cfg2),
           "frozen_sha": (None if not args.frozen else hashlib.sha256(Path(args.frozen).read_bytes()).hexdigest()[:12]),
           "frozen_selected": None if frozen is None else frozen.get("selected"),
           "budgets": [b["id"] for b in budgets], "arms": list(arms), "rounds_planned": rounds,
           "snapshot_sha": {str(r): manifest[r].get("sha") for r in rounds}, "rounds": {}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        out_path.write_text(json.dumps(S.jsonable(out), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    probe_run(world, cfg2, args.stage, rounds, snaps, metrics, hardener, base_cfg, budgets, arms, out=out, flush=flush)
    flush()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", choices=S.PA2_STAGES, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--here", required=True, help="experiments/defense/edge-native 目录")
    ap.add_argument("--dumps-root", required=True)
    ap.add_argument("--config", help="缺省 = <here>/d3pa2_configs.yaml")
    ap.add_argument("--frozen", help="main 阶段：harness/d3pa2_verdict.py select 写出的冻结清单")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.stage == "main" and not a.frozen:
        ap.error("--stage main 需要 --frozen")
    t0 = time.perf_counter()
    res = run(a)
    ok = all((R.get("reproduce") or {}).get("pass") for R in res["rounds"].values())
    print(f"[d3pa2] {a.out}（{time.perf_counter() - t0:.0f} s）重现闸 {'全过' if ok else '有不过的轮'}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
