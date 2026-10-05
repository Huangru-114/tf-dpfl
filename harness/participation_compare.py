"""
harness/participation_compare.py  —  检查 5：旧实现（P0 / P1）与 P2 的参与配额（**探索性，看过数据之后**）

    python3 harness/participation_compare.py [--json out.json]

问题：实验 1 报告「Exp 3A/3B」那张图（Figure 10 = `hfl-propagation/plot_exp3.py` 的 `fig_timeseries_topology.png`）
里 HFL 各格与 flat 的良性 ASR 差别，有多少可能来自每个有效轮实际参与训练的攻击者 / 良性端次数不同。

三套配额（N = 100、client_fraction = 0.1，B = round(N·frac) = 10）：
  P0  每个 edge 各自 max(1, int(|clients_e|·frac))（陷阱 #12 修之前；2026-08 归档 `archive-pre-fix/`）
  P1  整数配额 `participation.edge_quota`，余数按**云轮号**轮转（`a00a959`；`hfl-propagation/results/`）
  P2  同一个整数配额，余数按**有效轮**轮转（D-036 登记行 D02）
抽样：每个 edge 一个 `default_rng([seed, edge_id])`，每个 edge 轮调一次 `rng.choice(|clients_e|, quota, replace=False)`
（`server/edge_server_base.py:select_clients`；每个 edge 的流独立 → 与 edge 的执行顺序无关）。

做两件事：
  1. 解析期望：每有效轮的攻击者参与次数 = Σ_e quota_e · m_e / |clients_e|，良性 = Σ_e quota_e − 攻击者；
  2. **复算并核对**：用上面的抽样规则逐 edge 轮重放选端，与 metrics.json 的 `malicious_participation_by_client`
     （每个恶意端「被选中过的云轮」集合）逐个比对。全部相同 → 重放的选端就是当时真实的选端，计数可信。
     client→edge 按 block（edge e = id ∈ [e·N/E, (e+1)·N/E)）。

纯标准库 + numpy（只用 default_rng）。不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "fedavg"))
from server.participation import edge_quota                      # noqa: E402

N_CLIENTS = 100
FRAC = 0.1

PROP = ROOT / "experiments/attack/hfl-propagation/results"
MECH = ROOT / "experiments/attack/hfl-mechanism/results/P2"

# 每个配置：(标签, 版本, metrics.json, n_edges, R)；R 与 n_rounds 从 run 块取，P0 文件没写 R → 用 400 / n_rounds
CASES = [
    ("4edge_collocated", "P0", PROP / "archive-pre-fix/4edge_collocated_seed{s}.metrics.json", 4),
    ("2edge_collocated", "P0", PROP / "archive-pre-fix/2edge_collocated_seed{s}.metrics.json", 2),
    ("flat", "P0", PROP / "archive-pre-fix/flat_baseline_seed{s}.metrics.json", 1),
    ("4edge_collocated", "P1", PROP / "4edge_collocated_seed{s}.metrics.json", 4),
    ("2edge_collocated", "P1", PROP / "2edge_collocated_seed{s}.metrics.json", 2),
    ("flat", "P1", PROP / "flat_baseline_seed{s}.metrics.json", 1),
    ("4edge_collocated", "P2", MECH / "G3/G3__C1__s{s}.metrics.json", 4),
    ("4edge_collocated", "P2", MECH / "G6/G6__a__s{s}.metrics.json", 4),
    ("flat", "P2", MECH / "G8F/G8F__std__s{s}.metrics.json", 1),
]
P0_BUDGET = 400                    # 2026-08 的 Exp 3：edge_rounds × n_rounds = 400（RESULTS.md）


def quota(version: str, edge_id: int, n_edges: int, n_local: int, g: int, er: int, R: int) -> int:
    if version == "P0":
        return max(1, int(n_local * FRAC))
    rnd = g if version == "P1" else (g - 1) * R + er
    return edge_quota(edge_id, rnd, n_clients=N_CLIENTS, n_edges=n_edges, client_fraction=FRAC,
                      n_local_clients=n_local)


def block_edges(n_edges: int, n_clients: int = N_CLIENTS) -> list:
    per = n_clients // n_edges
    return [list(range(e * per, (e + 1) * per)) for e in range(n_edges)]


def expected_per_eff(version: str, n_edges: int, R: int, mal: set, n_rounds: int) -> dict:
    """每有效轮期望参与次数（对全部有效轮取平均）：攻击者、良性、合计；以及攻击者所在 edge 的名额均值。"""
    edges = block_edges(n_edges)
    tot_a = tot_b = 0.0
    e_quota = {e: 0.0 for e in range(n_edges)}
    T = n_rounds * R
    for g in range(1, n_rounds + 1):
        for er in range(1, R + 1):
            for e, cl in enumerate(edges):
                q = quota(version, e, n_edges, len(cl), g, er, R)
                m = sum(1 for c in cl if c in mal)
                tot_a += q * m / len(cl)
                tot_b += q * (len(cl) - m) / len(cl)
                e_quota[e] += q
    return {"attacker": tot_a / T, "benign": tot_b / T, "total": (tot_a + tot_b) / T,
            "quota_per_edge": [e_quota[e] / T for e in range(n_edges)]}


def replay(version: str, seed: int, n_edges: int, R: int, n_rounds: int, mal: set) -> dict:
    """逐 edge 轮重放选端。返回 {"by_client": {恶意 id: 被选中过的云轮}, "attacker": [每有效轮次数], "benign": [...]}。"""
    edges = block_edges(n_edges)
    rngs = [np.random.default_rng([int(seed), e]) for e in range(n_edges)]
    by_client = {c: set() for c in mal}
    att, ben = [], []
    for g in range(1, n_rounds + 1):
        for er in range(1, R + 1):
            a = b = 0
            for e, cl in enumerate(edges):
                q = quota(version, e, n_edges, len(cl), g, er, R)
                for i in rngs[e].choice(len(cl), q, replace=False):
                    c = cl[int(i)]
                    if c in mal:
                        a += 1
                        by_client[c].add(g)
                    else:
                        b += 1
            att.append(a)
            ben.append(b)
    return {"by_client": {str(c): sorted(v) for c, v in sorted(by_client.items())}, "attacker": att, "benign": ben}


def case_result(label: str, version: str, path: Path, n_edges: int) -> dict | None:
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    run = m["run"]
    n_rounds = int(run["n_rounds"])
    R = int(run.get("edge_rounds") or (P0_BUDGET // n_rounds if version == "P0" else 1))
    seed = int(run.get("seed") or int(path.name.split("seed")[-1].split(".")[0]) if "seed" in path.name
               else run.get("seed"))
    mal = set(int(i) for i in run["malicious_ids"])
    rep = replay(version, seed, n_edges, R, n_rounds, mal)
    want = {str(k): sorted(v) for k, v in (m.get("malicious_participation_by_client") or {}).items()}
    match = sum(1 for c in want if want[c] == rep["by_client"].get(c))
    exp = expected_per_eff(version, n_edges, R, mal, n_rounds)
    mal_edges = [sum(1 for c in cl if c in mal) for cl in block_edges(n_edges)]
    return {"label": label, "version": version, "file": str(path.relative_to(ROOT)), "seed": seed, "R": R,
            "n_rounds": n_rounds, "n_edges": n_edges, "malicious_per_edge_block": mal_edges,
            "replay_matches": f"{match}/{len(want)}", "replay_ok": match == len(want) and len(want) > 0,
            "expected": exp,
            "replayed_mean": {"attacker": float(np.mean(rep["attacker"])), "benign": float(np.mean(rep["benign"]))}}


def analyse(seeds=(42, 43, 44)) -> dict:
    out = []
    for label, version, tmpl, E in CASES:
        for s in seeds:
            r = case_result(label, version, Path(str(tmpl).format(s=s)), E)
            if r is not None:
                out.append(r)
    return {"purpose": "检查 5（探索性，看过数据之后）：P0 / P1 / P2 的每有效轮参与次数", "cases": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = analyse()
    for r in res["cases"]:
        e, p = r["expected"], r["replayed_mean"]
        print(f"{r['version']} {r['label']:<17} {Path(r['file']).name:<36} R={r['R']:<2} mal/edge={r['malicious_per_edge_block']} "
              f"replay {r['replay_matches']:<6} exp att {e['attacker']:.3f} ben {e['benign']:.3f} tot {e['total']:.2f} "
              f"| replayed att {p['attacker']:.3f} ben {p['benign']:.3f} | quota/edge {[round(q, 2) for q in e['quota_per_edge']]}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[participation_compare] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
