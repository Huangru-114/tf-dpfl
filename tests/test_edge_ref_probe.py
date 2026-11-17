"""
tests/test_edge_ref_probe.py  —  N-013 判定脚本（harness/edge_ref_probe.py）在合成数据上的每个分支

  · 攻击者分数明显偏高 → feasible，选中的分数、阈值（开发 run 的 90% 分位点）、回放率都是手算值；
  · 没有可分性 → infeasible；
  · 门槛反向锚点：最佳四格最小 AUROC 恰好 0.75 → feasible，略低 → infeasible；
  · 选分数取「四格最小值最大」：一个分数某一格塌掉就不选它；并列取候选表靠前的；
  · 闸：缺格、开发 run 投毒比例不为 0、run 没开 edge_ref、攻击者从没上传、client_failures → invalid；
  · 两个划分（main / homo）分别判定、互不影响。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import edge_ref_probe as P                                   # noqa: E402

COLS = ["round", "edge_id", "effective_round", "edge_round", "cid", "n"] + list(P.CANDIDATES) + ["ccs_rej"]
MAL = [0, 1, 2]


def _metrics(part, tier, placement, rows, *, poison=0.2, edge_ref=True, mal=MAL, null_rows=None):
    run = {"edge_ref": edge_ref, "poison_ratio": poison, "malicious_ids": list(mal),
           "partition_profile": "fixed" if part == "homo" else None,
           "edge_shared_blocks": 1 if tier == "t1" else 0,
           "malicious_per_edge": [10, 0, 0, 0] if placement == "collocated" else [3, 3, 2, 2]}
    return {"exit_code": 0, "client_failures": [], "run": run,
            "edge_ref_scores": {"columns": COLS, "rows": rows},
            "edge_ref_null": {"columns": ["round", "edge_id", "effective_round", "edge_round", "n_clean",
                                          "ccs_pool", "ccs_status", "kl_all", "kl_stage4", "mix_res", "mix_res_base"],
                              "rows": null_rows or []}}


def _row(cid, scores: dict, rej=0):
    base = {c: 0.0 for c in P.CANDIDATES}
    base.update(scores)
    return [1, 0, 1, 1, cid, 375] + [base[c] for c in P.CANDIDATES] + [rej]


def _world(part="main", mal_score=lambda i: 10.0 + i, ben_score=lambda i: float(i % 5), score="kl_all",
           dev_vals=tuple(range(1, 11))):
    attack = []
    for t in P.TIERS:
        for p in P.PLACEMENTS:
            rows = [_row(c, {score: mal_score(i)}, rej=1) for i, c in enumerate(MAL)]
            rows += [_row(10 + i, {score: ben_score(i)}) for i in range(7)]
            attack.append(_metrics(part, t, p, rows))
    dev = [_metrics(part, t, "collocated", [_row(10 + i, {score: v}) for i, v in enumerate(dev_vals)], poison=0.0)
           for t in P.TIERS]
    return attack, dev


def test_separable_scores_are_feasible_with_hand_computed_threshold_and_replay():
    attack, dev = _world()
    res = P.assemble(attack, dev)
    r = res["main"]
    assert r["verdict"] == "feasible" and r["selected_score"] == "kl_all" and r["selected_min_auroc"] == 1.0
    assert r["threshold_dev_q90"] == {"t0": pytest.approx(np.quantile(np.arange(1, 11), 0.9)),
                                      "t1": pytest.approx(9.1)}
    rep = r["descriptive"]["replay_at_threshold"]["t0-collocated"]
    assert rep["attacker_pass_rate"] == 0.0 and rep["benign_reject_rate"] == 0.0     # 良性 0–4 < 9.1 < 攻击者 10–12
    assert r["descriptive"]["ccs_cache_majority"]["t0-collocated"] == {"hit_rate": 1.0, "false_reject_rate": 0.0}
    assert res["homo"]["verdict"] == "invalid"                                        # 没给 homo 的 run


def test_no_separation_is_infeasible():
    attack, dev = _world(mal_score=lambda i: float(i % 5), ben_score=lambda i: float(i % 5))
    assert P.assemble(attack, dev)["main"]["verdict"] == "infeasible"


def _world_n(mal_scores, ben_scores):
    attack = []
    for t in P.TIERS:
        for p in P.PLACEMENTS:
            rows = [_row(c, {"kl_all": v}) for c, v in zip(MAL, mal_scores)]
            rows += [_row(10 + i, {"kl_all": v}) for i, v in enumerate(ben_scores)]
            attack.append(_metrics("main", t, p, rows))
    dev = [_metrics("main", t, "collocated", [_row(10, {"kl_all": 1.0})], poison=0.0) for t in P.TIERS]
    return attack, dev


def test_threshold_reverse_anchor():
    """3 攻击者 × 4 良性 = 12 对：攻击者 (100, 100, 0.5) vs 良性 (0, 1, 2, 3) → U = 4 + 4 + 1 = 9 → AUROC 恰好 0.75。"""
    r = P.assemble(*_world_n([100.0, 100.0, 0.5], [0.0, 1.0, 2.0, 3.0]))["main"]
    assert r["selected_min_auroc"] == 0.75 and r["verdict"] == "feasible"
    r = P.assemble(*_world_n([100.0, 100.0, -1.0], [0.0, 1.0, 2.0, 3.0]))["main"]     # 8 / 12
    assert r["selected_min_auroc"] == pytest.approx(8 / 12, abs=1e-4) and r["verdict"] == "infeasible"
    assert P.AUROC_MIN == 0.75 and P.DEV_Q == 0.90


def test_selection_is_max_of_the_four_cell_minimum():
    attack, dev = _world(score="kl_all")
    # mix_res：三格完美、一格完全反向 → 四格最小 0 → 不该被选；kl_all 四格都是 1.0
    for m in attack:
        run = m["run"]
        bad = run["edge_shared_blocks"] == 1 and run["malicious_per_edge"][1] == 3
        for row in m["edge_ref_scores"]["rows"]:
            cid = row[4]
            hi = cid in MAL
            row[COLS.index("mix_res")] = (0.0 if hi else 50.0) if bad else (99.0 if hi else 0.0)
    r = P.assemble(attack, dev)["main"]
    assert r["min_auroc_by_score"]["mix_res"] == 0.0 and r["selected_score"] == "kl_all"


def test_ties_go_to_the_earlier_candidate():
    attack, dev = _world(score="kl_stage4")
    for m in attack:                                   # kl_stage2 与 kl_stage4 同样完美
        for row in m["edge_ref_scores"]["rows"]:
            row[COLS.index("kl_stage2")] = row[COLS.index("kl_stage4")]
    assert P.assemble(attack, dev)["main"]["selected_score"] == "kl_stage2"


@pytest.mark.parametrize("breaker,needle", [
    (lambda a, d: a.pop(0), "缺有攻击的格子"),
    (lambda a, d: d[0]["run"].update(poison_ratio=0.2), "开发 run 的 poison_ratio"),
    (lambda a, d: a[1]["run"].update(edge_ref=False), "edge_ref"),
    (lambda a, d: a[2]["run"].update(malicious_ids=[99]), "没上传过"),
    (lambda a, d: a[3].update(client_failures=[{"client_id": 4}]), "client_failures"),
    (lambda a, d: a.append(dict(a[0])), "多个 run"),
])
def test_gates(breaker, needle):
    attack, dev = _world()
    breaker(attack, dev)
    r = P.assemble(attack, dev)["main"]
    assert r["verdict"] == "invalid" and any(needle in x for x in r["reasons"]), r


def test_partitions_are_judged_separately():
    a1, d1 = _world("main")
    a2, d2 = _world("homo", mal_score=lambda i: float(i % 5), ben_score=lambda i: float(i % 5))
    res = P.assemble(a1 + a2, d1 + d2)
    assert res["main"]["verdict"] == "feasible" and res["homo"]["verdict"] == "infeasible"
    assert set(res["contrast"]["kl_all"]) == {"main", "homo"}


def test_null_fraction_above_threshold():
    nulls = [[1, 0, 1, 1, 500, 12, "ok", [0.0, 20.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
    attack, dev = _world()
    for m in attack:
        m["edge_ref_null"]["rows"] = nulls
    r = P.assemble(attack, dev)["main"]
    assert r["descriptive"]["null_fraction_above_threshold"] == {"t0": 0.5, "t1": 0.5}
