"""
harness/tail_clients.py  —  检查 4：衰减后剩下的长尾客户端是谁（**探索性，看过数据之后**）

    # 本地（只有 metrics.json：逐客户端值只有末个评估点 → 只能用末窗口代替）
    python3 harness/tail_clients.py --json out.json
    # 登录节点（有日志 + 划分画像时，按报告的窗口算）
    python3 harness/tail_clients.py --log G8:42=<log> … --profile G8:42=<client_profile.json> … --json out.json

问题：G8（HFL）第 51–60 云轮、G8F（flat）255–300 有效轮里，ASR 仍高的良性客户端有什么共同点 ——
特别是：是否集中在训练集里目标类（airplane，y_t = 0）占比高的客户端。

逐客户端 ASR 的来源（`[ClientEval]`，每个后门评估点每 edge 一行）：
  · --log 给了 → 窗口内每个评估点的值取均值（G8：第 51–60 云轮，10 点；G8F：有效轮 255–300，每 5 轮一点，10 点）；
  · 没给 → metrics.json 的 `client_final`（**只有末个评估点**：G8 第 70 云轮、G8F 第 350 有效轮），
    窗口改成同长度的末窗口（G8 第 61–70 云轮；G8F 301–350 有效轮），并在输出里标 `asr_source = final_point`。
每个良性客户端的协变量：
  edge、选中次数（观察窗口内的 edge 轮数）、最近一次被选中距窗口末的有效轮数、干净精度（同一来源的 acc）、
  yt_clean（干净非目标样本被判成 y_t 的比例 = PM 对目标类的偏向；只是代理，不是训练集占比）、
  训练集目标类占比（要 --profile：`fedavg/analysis/client_profile.py` 在登录节点 CPU 上重建划分；没有 → None）。
选端由 `participation_compare` 的规则重放（P2：余数按有效轮轮转），先与 metrics.json 的
`malicious_participation_by_client` 逐个核对，对不上就拒绝给选端相关的量。

统计：每个协变量与逐客户端 ASR 的 Spearman ρ（并列取平均秩）；尾部 = ASR > 0.5 ∪ ASR ≥ p90，
与其余良性端比较各协变量的均值 / 中位数。
另：G8 与 G8F 同 seed 用同一个划分 → 两边尾部集合的交集 vs 随机置换零分布（10000 次），逐客户端 ASR 的 Spearman。

纯标准库 + numpy（选端重放）。不 import TF，不改任何文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fedavg"))
from participation_compare import block_edges, quota            # noqa: E402
from utils.kvline import collect_kv, parse_list                  # noqa: E402

RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2"
SEEDS = (42, 43, 44)
TARGET = 0
THETA = 0.5
# 组 → (文件名模板, 报告窗口（有效轮，含两端）)
GROUPS = {"G8": ("G8/G8__a__s{s}", (255, 300)), "G8F": ("G8F/G8F__std__s{s}", (255, 300))}
COLS = (("ids", "client_id"), ("mal", "malicious"), ("asr", "asr"), ("acc", "acc"), ("yt", "yt_clean"))


# ══════════════════════════════════════════════════════════════════════════
# 逐客户端 ASR / acc
# ══════════════════════════════════════════════════════════════════════════

def client_rows_from_log(lines) -> dict:
    """{云轮: {client_id: {"edge", "malicious", "asr", "acc"}}}，来自全部 [ClientEval] 行。"""
    out = {}
    for d in collect_kv(lines, "[ClientEval]"):
        cols = {k: parse_list(d.get(src)) or [] for src, k in COLS}
        for i, cid in enumerate(cols["client_id"]):
            out.setdefault(int(d["round"]), {})[int(cid)] = {
                "edge": d.get("edge_id"), "malicious": bool(cols["malicious"][i]),
                "asr": cols["asr"][i], "acc": cols["acc"][i], "yt_clean": cols["yt_clean"][i]}
    return out


def client_rows_from_final(m: dict) -> dict:
    cf = m.get("client_final") or {}
    if not cf:
        return {}
    rows = {int(cid): {"edge": e, "malicious": bool(mal), "asr": a, "acc": c, "yt_clean": y}
            for cid, e, mal, a, c, y in zip(cf["client_id"], cf["edge_id"], cf["malicious"], cf["asr"], cf["acc"],
                                            cf["yt_clean"])}
    return {int(cf["round"]): rows}


def window_means(rows_by_round: dict, rounds) -> dict:
    """{client_id: {edge, malicious, asr, acc, n_points}}：窗口内有值的评估点取均值。"""
    acc = {}
    for g in rounds:
        for cid, r in (rows_by_round.get(g) or {}).items():
            a = acc.setdefault(cid, {"edge": r["edge"], "malicious": r["malicious"], "asr": [], "acc": [],
                                     "yt_clean": []})
            for k in ("asr", "acc", "yt_clean"):
                if r[k] is not None:
                    a[k].append(r[k])
    return {cid: {"edge": a["edge"], "malicious": a["malicious"], "n_points": len(a["asr"]),
                  "asr": float(np.mean(a["asr"])) if a["asr"] else None,
                  "acc": float(np.mean(a["acc"])) if a["acc"] else None,
                  "yt_clean": float(np.mean(a["yt_clean"])) if a["yt_clean"] else None} for cid, a in acc.items()}


# ══════════════════════════════════════════════════════════════════════════
# 选端重放
# ══════════════════════════════════════════════════════════════════════════

def replay_selection(seed: int, n_edges: int, R: int, n_rounds: int, n_clients: int = 100) -> dict:
    """{client_id: [被选中的有效轮]}（P2 规则；edge = block）。"""
    edges = block_edges(n_edges, n_clients)
    rngs = [np.random.default_rng([int(seed), e]) for e in range(n_edges)]
    sel = {c: [] for cl in edges for c in cl}
    for g in range(1, n_rounds + 1):
        for er in range(1, R + 1):
            t = (g - 1) * R + er
            for e, cl in enumerate(edges):
                for i in rngs[e].choice(len(cl), quota("P2", e, n_edges, len(cl), g, er, R), replace=False):
                    sel[cl[int(i)]].append(t)
    return sel


def selection_matches(sel: dict, m: dict, R: int) -> tuple:
    want = {int(k): sorted(v) for k, v in (m.get("malicious_participation_by_client") or {}).items()}
    got = {c: sorted({(t - 1) // R + 1 for t in sel.get(c, [])}) for c in want}
    ok = sum(1 for c in want if got[c] == want[c])
    return ok, len(want)


# ══════════════════════════════════════════════════════════════════════════
# 统计
# ══════════════════════════════════════════════════════════════════════════

def ranks(vals) -> list:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pts) < 3:
        return None
    rx, ry = ranks([a for a, _ in pts]), ranks([b for _, b in pts])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return None if sxx == 0 or syy == 0 else sxy / (sxx * syy) ** 0.5


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "mean": float(np.mean(vals)), "median": float(np.median(vals))}


VARS = ("target_frac", "yt_clean", "n_selected", "since_last", "acc", "edge")


def profile_run(m: dict, rows_by_round: dict, win_eff: tuple, profile: dict | None, source: str) -> dict:
    run = m["run"]
    R, n_rounds, E, seed = int(run["edge_rounds"]), int(run["n_rounds"]), int(run["n_edges"]), int(run["seed"])
    lo, hi = win_eff
    rounds = [g for g in sorted(rows_by_round) if lo <= g * R <= hi]
    sel = replay_selection(seed, E, R, n_rounds)
    ok, n_mal = selection_matches(sel, m, R)
    sel_ok = n_mal > 0 and ok == n_mal
    w = window_means(rows_by_round, rounds)
    prof = (profile or {}).get("clients") or {}
    clients = []
    for cid in sorted(w):
        r = w[cid]
        if r["malicious"] or r["asr"] is None:
            continue
        p = prof.get(str(cid))
        ts = [t for t in sel.get(cid, []) if lo <= t <= hi] if sel_ok else None
        before = [t for t in sel.get(cid, []) if t <= hi] if sel_ok else None
        clients.append({"client_id": cid, "edge": r["edge"], "asr": r["asr"], "acc": r["acc"],
                        "yt_clean": r["yt_clean"],
                        "n_points": r["n_points"],
                        "n_selected": None if ts is None else len(ts),
                        "since_last": None if not before else hi - max(before),
                        "target_frac": None if p is None else p["counts"][TARGET] / max(1, p["n_train"]),
                        "n_train": None if p is None else p["n_train"]})
    asr = [c["asr"] for c in clients]
    p90 = float(np.percentile(asr, 90)) if asr else None
    for c in clients:
        c["tail"] = bool(c["asr"] > THETA or (p90 is not None and c["asr"] >= p90))
    corr = {v: spearman([c[v] for c in clients], asr) for v in VARS if v != "edge"}
    tail = [c for c in clients if c["tail"]]
    rest = [c for c in clients if not c["tail"]]
    comp = {v: {"tail": _stats([c[v] for c in tail]), "rest": _stats([c[v] for c in rest])}
            for v in VARS if v != "edge"}
    edges = sorted({c["edge"] for c in clients})
    by_edge = {str(e): {"n": sum(1 for c in clients if c["edge"] == e),
                        "n_tail": sum(1 for c in tail if c["edge"] == e),
                        "n_gt50": sum(1 for c in clients if c["edge"] == e and c["asr"] > THETA)} for e in edges}
    return {"asr_source": source, "window_eff": [lo, hi], "rounds_used": rounds,
            "selection_replay": f"{ok}/{n_mal}", "selection_ok": sel_ok,
            "has_profile": bool(prof), "n_benign": len(clients), "p90": p90,
            "n_gt50": sum(1 for c in clients if c["asr"] > THETA), "n_tail": len(tail),
            "spearman": corr, "tail_vs_rest": comp, "by_edge": by_edge,
            "tail_clients": sorted(tail, key=lambda c: -c["asr"]), "clients": clients}


def tail_overlap(a: dict, b: dict, n_perm: int = 10000, seed: int = 0) -> dict:
    """同一 seed 的两个 run（同一划分）：尾部集合的交集大小 vs 随机置换的零分布；逐客户端 ASR 的 Spearman。"""
    ca = {c["client_id"]: c for c in a["clients"]}
    cb = {c["client_id"]: c for c in b["clients"]}
    ids = sorted(set(ca) & set(cb))
    ta = {i for i in ids if ca[i]["tail"]}
    tb = {i for i in ids if cb[i]["tail"]}
    k = len(ta & tb)
    rng = np.random.default_rng(seed)
    arr = np.array(ids)
    null = np.array([len(ta & set(rng.choice(arr, len(tb), replace=False).tolist())) for _ in range(n_perm)])
    return {"n_common": len(ids), "n_tail_a": len(ta), "n_tail_b": len(tb), "overlap": k,
            "overlap_ids": sorted(ta & tb), "null_mean": float(null.mean()),
            "p_ge": float((null >= k).mean()),
            "spearman_asr": spearman([ca[i]["asr"] for i in ids], [cb[i]["asr"] for i in ids])}


def _kv_args(items) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        g, s = k.split(":")
        out[(g, int(s))] = Path(v)
    return out


def analyse(results: Path = RESULTS, logs=None, profiles=None) -> dict:
    logs, profiles = logs or {}, profiles or {}
    out = {}
    for g, (stem, win) in GROUPS.items():
        for s in SEEDS:
            p = results / f"{stem.format(s=s)}.metrics.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text(encoding="utf-8"))
            if (g, s) in logs:
                rows = client_rows_from_log(logs[(g, s)].read_text(encoding="utf-8", errors="replace").splitlines())
                source, w = "log", win
            else:
                rows = client_rows_from_final(m)
                R = int(m["run"]["edge_rounds"])
                last = max(rows) * R
                source, w = "final_point", (last - (win[1] - win[0]), last)
            prof = (json.loads(profiles[(g, s)].read_text(encoding="utf-8")) if (g, s) in profiles else None)
            out[f"{g}:{s}"] = profile_run(m, rows, w, prof, source)
    cross = {str(s): tail_overlap(out[f"G8:{s}"], out[f"G8F:{s}"]) for s in SEEDS
             if f"G8:{s}" in out and f"G8F:{s}" in out}
    return {"purpose": "检查 4（探索性，看过数据之后）：衰减后的长尾良性客户端画像", "target": TARGET,
            "theta": THETA, "runs": out, "g8_vs_g8f": cross}


def _f(v, nd=3):
    return "None" if v is None else f"{v:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--log", nargs="*", help="组:seed=日志路径（含 [ClientEval] 行），如 G8:42=…/exp3v2_G8__a__s42.123.log")
    ap.add_argument("--profile", nargs="*", help="组:seed=client_profile.json")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = analyse(Path(a.results), _kv_args(a.log), _kv_args(a.profile))
    for k, r in res["runs"].items():
        sp = "  ".join(f"{v}={_f(c)}" for v, c in r["spearman"].items())
        print(f"{k:<8} src={r['asr_source']:<11} win={r['window_eff']} sel={r['selection_replay']} "
              f"benign={r['n_benign']} p90={_f(r['p90'])} >0.5={r['n_gt50']} tail={r['n_tail']} | ρ: {sp}")
        print(f"         by_edge {r['by_edge']}")
        for c in r["tail_clients"]:
            print(f"         tail c{c['client_id']:<3} e{c['edge']} asr={_f(c['asr'])} acc={_f(c['acc'])} "
                  f"sel={c['n_selected']} since_last={c['since_last']} yt_frac={_f(c['target_frac'])}")
    for s_, c in res["g8_vs_g8f"].items():
        print(f"G8 vs G8F s{s_}: tail overlap {c['overlap']} / {c['n_tail_a']}·{c['n_tail_b']} (null mean "
              f"{c['null_mean']:.2f}, p≥ {c['p_ge']:.4f}) ids {c['overlap_ids']}  ρ(asr) {_f(c['spearman_asr'])}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[tail_clients] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
