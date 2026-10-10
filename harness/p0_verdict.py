"""
harness/p0_verdict.py  —  阶段三 P0 的选择与判定（FINDINGS N-008 P0 段，D-101 生效；**先 commit，再打开任何 P0 数据**）

    # ① s42 collocated 的 screen 结果回来后：选前 3 名并冻结（冻结清单 commit 之后才交 s43 / s44 与 distributed）
    python3 harness/p0_verdict.py select experiments/defense/edge-native/analysis/p0/SNAP__collocated__s42.screen.json \\
        --out experiments/defense/edge-native/analysis/p0_frozen.json
    # ② 全部回来后：判定
    python3 harness/p0_verdict.py judge experiments/defense/edge-native/analysis/p0 \\
        --frozen experiments/defense/edge-native/analysis/p0_frozen.json --out experiments/defense/edge-native/analysis/p0_verdict.json

输入是 `fedavg/analysis/p0_snapshot.py` 写的 JSON（每个 run × 阶段一个文件）。量（集中布点）：
  R_H = [A(G) − A(G′_H)] / J_t（`hardening.spec.r_h`）；A = 受害 edge 良性端 fresh-PM ASR 的三 edge 均值（离线重算）；
  J_t = run 记录的第 t+1 轮云聚合后点 − 第 t 轮全量点；J_t < 0.05 的快照不计。
  精度过滤（`spec.accuracy_ok`）：池化 fresh ΔMTA ≤ 0.02 且每个 edge ≤ 0.04；配置在某 seed 上「过滤通过」= 该 seed 计入的快照都通过。
规则（N-008）：
  invalid        任一用到的快照 V0 不过（离线 FedAvg ≠ 快照 G，或离线重算与 run 记录的 ASR 差 > 0.01）
  选择（只用 s42 collocated 的 screen）：过滤通过的 AT 配置按 min(R_H) 降序取前 3（并列 → 前向反向次数少 → id）
  go_online      冻结的前 3 名里有配置在三个 seed、全部计入的快照上 R_H ≥ 0.5 且三个 seed 都过滤通过 → 取名次最高的
  kill_pre       s42 上全部 AT 配置、以及冻结配置在 s43 / s44 上，计入的快照 R_H 都 < 0.2（A-every 不由 P0 判死）
  accuracy_bound s42 上没有过滤通过的配置在全部计入的快照上 R_H ≥ 0.5，但有过滤不通过的配置做到了 → 用户定
  inconclusive   其余（P1 照做，只作探路）
  insufficient   缺 run，或某 seed 没有计入的快照
另报（不进判定）：最佳配置的 R_H − 同预算 C-ft 的 R_H（逐 seed、逐快照）；分散布点的 D_H = 池化良性 ASR 之差；附加读数原样转抄。
纯标准库 + `fedavg/hardening/spec.py`（numpy），不 import TF。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fedavg"))
from hardening import spec      # noqa: E402

SEEDS = (42, 43, 44)
SELECT_SEED = 42
GRID = HERE.parent / "experiments/defense/edge-native/p0_configs.yaml"


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def load_grid(path=GRID):
    import yaml
    g = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return g, spec.expand_grid(g["grid"])


# ══════════════════════════════════════════════════════════════════════════
# 一个 run 的量
# ══════════════════════════════════════════════════════════════════════════

def snapshot_rows(res: dict, cid: str) -> dict:
    """{t: {r_h, acc_ok, d_pool, d_edge_max, usable, d_pool_benign}}；该快照没有这个配置 → 不出现。"""
    out = {}
    for t, S in (res.get("snapshots") or {}).items():
        c = (S.get("configs") or {}).get(cid)
        if c is None:
            continue
        base, hard = S["base"], c["eval"]
        J = S.get("J")
        rh = spec.r_h(base.get("victim_asr"), hard.get("victim_asr"), J)
        acc = spec.accuracy_ok(base, hard)
        db = (None if base.get("pooled_benign") is None or hard.get("pooled_benign") is None
              else base["pooled_benign"] - hard["pooled_benign"])
        out[int(t)] = {"r_h": rh, "usable": rh is not None, "acc_ok": acc["ok"],
                       "d_pool": acc["d_pool"], "d_edge_max": acc["d_edge_max"], "d_pooled_benign": db,
                       "J": J}
    return out


def v0_failures(res: dict) -> list:
    return [f"{res.get('run_id')} 第 {t} 轮：{'; '.join(S['v0']['reasons'])}"
            for t, S in (res.get("snapshots") or {}).items()
            if S.get("v0") is not None and not S["v0"].get("pass")]


def seed_summary(rows: dict) -> dict:
    """一个配置在一个 seed 上：计入的快照、min / max R_H、是否全部过滤通过。"""
    used = {t: r for t, r in rows.items() if r["usable"]}
    rh = [r["r_h"] for r in used.values()]
    return {"n_used": len(used), "rh_min": min(rh) if rh else None, "rh_max": max(rh) if rh else None,
            "acc_ok": bool(used) and all(r["acc_ok"] for r in used.values()),
            "per_snapshot": {str(t): {k: _r(v) if isinstance(v, float) else v for k, v in r.items()}
                             for t, r in sorted(rows.items())}}


# ══════════════════════════════════════════════════════════════════════════
# 选择
# ══════════════════════════════════════════════════════════════════════════

def select(screen: dict, grid: list, top: int = 3, n_clean: int = 500, batch: int = 32) -> dict:
    """s42 collocated 的 screen 结果 → 冻结的前 top 名（只在过滤通过的 AT 配置里选）。"""
    bad = v0_failures(screen)
    if bad:
        return {"frozen": [], "status": "invalid", "reasons": bad}
    if screen.get("stage") != "screen" or screen.get("seed") != SELECT_SEED or screen.get("placement") != "collocated":
        return {"frozen": [], "status": "wrong_input",
                "reasons": [f"选择只用 s{SELECT_SEED} collocated 的 screen：收到 {screen.get('run_id')} / {screen.get('stage')}"]}
    by_id = {c["id"]: c for c in grid}
    ranking = []
    for cid in screen.get("configs") or []:
        cfg = by_id[cid]
        if cfg["objective"] not in spec.AT_OBJECTIVES:
            continue
        s = seed_summary(snapshot_rows(screen, cid))
        ranking.append({"id": cid, "rh_min": s["rh_min"], "acc_ok": s["acc_ok"], "n_used": s["n_used"],
                        "fb": spec.fb_harden(cfg, n_clean, batch)})
    elig = [r for r in ranking if r["acc_ok"] and r["rh_min"] is not None]
    elig.sort(key=lambda r: (-r["rh_min"], r["fb"], r["id"]))
    frozen = [r["id"] for r in elig[:top]]
    return {"frozen": frozen, "status": "ok" if frozen else "none_passes_filter",
            "ranking": sorted(ranking, key=lambda r: (not (r["acc_ok"] and r["rh_min"] is not None),
                                                      -(r["rh_min"] if r["rh_min"] is not None else -1e9),
                                                      r["fb"], r["id"])),
            "screen_run": screen.get("run_id"), "grid_sha": screen.get("grid_sha")}


# ══════════════════════════════════════════════════════════════════════════
# 判定
# ══════════════════════════════════════════════════════════════════════════

def judge(results: list, frozen_doc: dict, grid: list) -> dict:
    """results：p0_snapshot 的全部 JSON；frozen_doc：select 的输出（已 commit）。"""
    by_id = {c["id"]: c for c in grid}
    frozen = list(frozen_doc.get("frozen") or [])
    col = {(r["seed"], r["stage"]): r for r in results if r.get("placement") == "collocated"}
    dist = [r for r in results if r.get("placement") == "distributed"]
    bad = sum((v0_failures(r) for r in results), [])
    out = {"frozen": frozen, "rule": {"R_GO": spec.R_GO, "R_KILL": spec.R_KILL, "JUMP_MIN": spec.JUMP_MIN,
                                      "MTA_POOL_MAX": spec.MTA_POOL_MAX, "MTA_EDGE_MAX": spec.MTA_EDGE_MAX},
           "preregistered": "FINDINGS N-008（D-101）"}
    if bad:
        return {**out, "verdict": "invalid", "reasons": bad}
    screen = col.get((SELECT_SEED, "screen"))
    missing = []
    if screen is None:
        missing.append(f"s{SELECT_SEED} collocated screen")

    def res_for(seed, cid):
        """seed 上某配置的结果：s42 用 screen（与 frozen 阶段同一组随机数键 → 同值），其余用 frozen 阶段。"""
        if seed == SELECT_SEED and screen is not None and cid in (screen.get("configs") or []):
            return screen
        return col.get((seed, "frozen"))

    per = {}
    for cid in frozen:
        per[cid] = {}
        for s in SEEDS:
            r = res_for(s, cid)
            if r is None or cid not in (r.get("configs") or []):
                missing.append(f"s{s} collocated {cid}")
                continue
            per[cid][s] = seed_summary(snapshot_rows(r, cid))
    out["frozen_per_seed"] = {cid: {f"s{s}": v for s, v in d.items()} for cid, d in per.items()}
    if missing:
        return {**out, "verdict": "insufficient", "missing": sorted(set(missing))}
    no_used = [f"s{s} {cid}" for cid, d in per.items() for s, v in d.items() if v["n_used"] == 0]
    if no_used:
        return {**out, "verdict": "insufficient", "reasons": [f"没有计入的快照（J_t < {spec.JUMP_MIN}）：{no_used}"]}

    go = [cid for cid in frozen
          if all(per[cid][s]["rh_min"] >= spec.R_GO and per[cid][s]["acc_ok"] for s in SEEDS)]
    s42_at = [cid for cid in screen.get("configs") or [] if by_id[cid]["objective"] in spec.AT_OBJECTIVES]
    s42 = {cid: seed_summary(snapshot_rows(screen, cid)) for cid in s42_at}
    kill = (all(v["rh_max"] is not None and v["rh_max"] < spec.R_KILL for v in s42.values())
            and all(per[cid][s]["rh_max"] < spec.R_KILL for cid in frozen for s in SEEDS if s != SELECT_SEED))
    good = [cid for cid, v in s42.items() if v["rh_min"] is not None and v["rh_min"] >= spec.R_GO]
    acc_bound = bool(good) and not any(s42[c]["acc_ok"] for c in good)
    if go:
        verdict, best = "go_online", go[0]
    elif kill:
        verdict, best = "kill_pre", (frozen[0] if frozen else None)
    elif acc_bound:
        verdict, best = "accuracy_bound", None
    else:
        verdict, best = "inconclusive", (frozen[0] if frozen else None)
    out.update({"verdict": verdict, "best": best, "go_candidates": go,
                "s42_meets_R_GO": sorted(good), "s42_meets_R_GO_and_filter": sorted(c for c in good if s42[c]["acc_ok"])})
    # 必报：最佳配置的 R_H − 同预算 C-ft 的 R_H（对抗部分的贡献）
    if best is not None:
        cft = spec.matched_cft(by_id[best], grid)
        attr = {}
        for s in SEEDS:
            r = res_for(s, best)
            rc = res_for(s, cft) if cft else None
            if r is None or rc is None or cft not in (rc.get("configs") or []):
                attr[f"s{s}"] = None
                continue
            a, b = snapshot_rows(r, best), snapshot_rows(rc, cft)
            attr[f"s{s}"] = {str(t): _r(a[t]["r_h"] - b[t]["r_h"])
                             for t in a if t in b and a[t]["usable"] and b[t]["usable"]}
        out["attribution_vs_cft"] = {"cft": cft, "per_seed": attr}
    # 只报告：分散布点的 D_H（池化良性 ASR 之差）
    out["distributed_D_H"] = {
        f"s{r['seed']}": {cid: {str(t): _r(v["d_pooled_benign"]) for t, v in snapshot_rows(r, cid).items()}
                          for cid in r.get("configs") or []}
        for r in sorted(dist, key=lambda r: r["seed"])}
    return out


# ══════════════════════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(description="阶段三 P0：选择与判定（N-008）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("select")
    a1.add_argument("screen")
    a1.add_argument("--grid", default=str(GRID))
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("judge")
    a2.add_argument("dir")
    a2.add_argument("--frozen", required=True)
    a2.add_argument("--grid", default=str(GRID))
    a2.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gspec, grid = load_grid(a.grid)
    if a.cmd == "select":
        screen = json.loads(Path(a.screen).read_text(encoding="utf-8"))
        sel = gspec.get("select") or {}
        res = select(screen, grid, top=int(sel.get("top", 3)), n_clean=int(gspec.get("n_clean", 500)),
                     batch=int(gspec.get("batch", 32)))
        res["screen_sha"] = hashlib.sha256(Path(a.screen).read_bytes()).hexdigest()[:12]
    else:
        results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(a.dir).glob("*.json"))]
        res = judge(results, json.loads(Path(a.frozen).read_text(encoding="utf-8")), grid)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: res.get(k) for k in ("verdict", "status", "frozen", "best") if k in res},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
