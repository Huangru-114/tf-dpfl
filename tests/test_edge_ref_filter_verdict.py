"""
tests/test_edge_ref_filter_verdict.py  —  N-014 判定脚本（harness/edge_ref_filter_verdict.py）在合成数据上的每个分支

  · 每格：protects / accuracy_bound / no_effect，含门槛的反向锚点（ΔASR 均值恰 0.15 / 最小恰 0.10 / ΔACC 恰 0.02）；
  · 组合：additive / redundant / partial；任一格 invalid → 组合 invalid；
  · 闸：缺种子、过滤臂不是 filter、没记阈值、对照是 filter、配置键不同、client_failures、攻击者没参与、t0 对照没植入（t1 不设此闸）、重复 run；
  · 在线漏网 / 误剔率与整轮全被剔的计数是手算值。
"""

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import edge_ref_filter_verdict as V                          # noqa: E402

COLS = ["round", "edge_id", "effective_round", "edge_round", "cid", "n", "mix_res", "admitted"]


def _m(part, tier, place, seed, P, MTA, *, mode, admitted_rows=None):
    run = {"seed": seed, "edge_ref": mode is not None, "edge_ref_mode": mode,
           "edge_ref_score": "mix_res" if mode == "filter" else None,
           "edge_ref_threshold": 0.12 if mode == "filter" else None,
           "partition_profile": "fixed" if part == "homo" else None,
           "edge_shared_blocks": 1 if tier == "t1" else 0,
           "malicious_per_edge": [10, 0, 0, 0] if place == "collocated" else [3, 3, 2, 2],
           "malicious_ids": [0, 1], "n_rounds": 60, "edge_rounds": 5, "n_edges": 4, "poison_ratio": 0.2,
           "partition": "designed"}
    return {"exit_code": 0, "client_failures": [], "n_malicious_participations": 7, "run": run,
            "rounds": [{"round": r, "local_benign_asr": P} for r in range(1, 61)],
            "acc_rounds": [{"round": r, "pm_acc": MTA} for r in range(1, 61)],
            "edge_ref_scores": {"columns": COLS, "rows": admitted_rows or []}}


def _world(dasr=(0.3, 0.3, 0.3), dacc=(0.0, 0.0, 0.0), pc=0.9, mc=0.86, cells=None, t1_ctrl=None, t1_filt=None):
    """每格：对照 P = pc，过滤 P = pc − dasr[i]；精度同理。t1_ctrl / t1_filt 可覆盖 t1 格的 P（组合测试用）。"""
    filt, ctrl = [], []
    for part in V.PARTS:
        for tier in V.TIERS:
            for place in V.PLACEMENTS:
                if cells and (part, tier, place) not in cells:
                    continue
                for i, s in enumerate(V.SEEDS):
                    p_c = t1_ctrl if (tier == "t1" and t1_ctrl is not None) else pc
                    p_f = t1_filt if (tier == "t1" and t1_filt is not None) else p_c - dasr[i]
                    ctrl.append(_m(part, tier, place, s, p_c, mc, mode="observe" if s != 42 or part != "main" or tier != "t0" else None))
                    filt.append(_m(part, tier, place, s, p_f, mc - dacc[i], mode="filter"))
    return filt, ctrl


def test_protects_and_hand_computed_quantities():
    res = V.assemble(*_world())
    c = res["cells"]["main-t0-collocated"]
    assert c["verdict"] == "protects" and c["dASR_mean"] == pytest.approx(0.3) and c["dACC_mean"] == 0.0
    assert c["per_seed"]["42"]["P_control"] == 0.9 and c["per_seed"]["42"]["P_filter"] == pytest.approx(0.6)
    assert all(r["verdict"] == "protects" for r in res["cells"].values())


@pytest.mark.parametrize("dasr,dacc,want", [
    ((0.15, 0.15, 0.15), (0.02, 0.02, 0.02), "protects"),           # 三个门槛都恰好压线
    ((0.20, 0.20, 0.10), (0.0, 0.0, 0.0), "protects"),              # 均值 0.167、最小恰 0.10
    ((0.25, 0.25, 0.09), (0.0, 0.0, 0.0), "no_effect"),             # 最小 0.09 < 0.10
    ((0.15, 0.15, 0.14), (0.0, 0.0, 0.0), "no_effect"),             # 均值 0.1467 < 0.15
    ((0.30, 0.30, 0.30), (0.02, 0.02, 0.03), "accuracy_bound"),     # ΔACC 均值 0.0233 > 0.02
])
def test_cell_thresholds(dasr, dacc, want):
    res = V.assemble(*_world(dasr=dasr, dacc=dacc))
    assert res["cells"]["homo-t1-distributed"]["verdict"] == want


def test_combination_labels():
    # t0：对照 0.9、过滤 0.6 → A_filt = 0.6；t1：对照（只开三层个性化）A_tier、过滤 + t1 = A_both
    res = V.assemble(*_world(t1_ctrl=0.5, t1_filt=0.38))       # best = min(0.5, 0.6) = 0.5；0.38 ≤ 0.40 → additive
    assert res["combination"]["main-collocated"]["verdict"] == "additive"
    res = V.assemble(*_world(t1_ctrl=0.5, t1_filt=0.47))       # |0.47 − 0.5| < 0.05 → redundant
    assert res["combination"]["main-collocated"]["verdict"] == "redundant"
    res = V.assemble(*_world(t1_ctrl=0.5, t1_filt=0.43))       # 介于两者之间 → partial
    assert res["combination"]["homo-distributed"]["verdict"] == "partial"


def test_t0_control_must_be_planted_but_t1_control_need_not():
    res = V.assemble(*_world(pc=0.45, dasr=(0.0, 0.0, 0.0)))
    assert res["cells"]["main-t0-collocated"]["verdict"] == "invalid_no_attack"
    assert res["cells"]["main-t1-collocated"]["verdict"] == "no_effect"      # t1 不设植入闸
    assert res["combination"]["main-collocated"]["verdict"] == "invalid"


def _break(fn, cell="main-t1-distributed"):
    f, c = _world()
    fn(f, c)
    return V.assemble(f, c)["cells"][cell]


@pytest.mark.parametrize("fn,needle", [
    (lambda f, c: f.pop(next(i for i, m in enumerate(f) if V.cell_of(m["run"]) == ("main", "t1", "distributed"))), "种子"),
    (lambda f, c: [m["run"].update(edge_ref_mode="observe") for m in f if V.cell_of(m["run"]) == ("main", "t1", "distributed")], "应为 filter"),
    (lambda f, c: [m["run"].update(edge_ref_threshold=None) for m in f if V.cell_of(m["run"]) == ("main", "t1", "distributed")], "threshold"),
    (lambda f, c: [m["run"].update(edge_ref_mode="filter") for m in c if V.cell_of(m["run"]) == ("main", "t1", "distributed")], "应为不设防"),
    (lambda f, c: [m["run"].update(n_rounds=30) for m in c if V.cell_of(m["run"]) == ("main", "t1", "distributed")], "n_rounds"),
    (lambda f, c: [m.update(client_failures=[{"client_id": 3}]) for m in f if V.cell_of(m["run"]) == ("main", "t1", "distributed")], "client_failures"),
    (lambda f, c: [m.update(n_malicious_participations=0) for m in f if V.cell_of(m["run"]) == ("main", "t1", "distributed")], "没参与"),
])
def test_gates(fn, needle):
    r = _break(fn)
    assert r["verdict"] == "invalid" and any(needle in x for x in r["reasons"]), r


def test_duplicate_runs_invalidate_everything():
    f, c = _world()
    f.append(copy.deepcopy(f[0]))
    res = V.assemble(f, c)
    assert all(r["verdict"] == "invalid" for r in res["cells"].values())


def test_online_rates_by_hand():
    rows = [[1, 0, 1, 1, 0, 375, 0.5, 0],   # 攻击者 0，被剔
            [1, 0, 1, 1, 1, 375, 0.1, 1],   # 攻击者 1，漏网
            [1, 0, 1, 1, 5, 375, 0.1, 1],   # 良性，接纳
            [1, 1, 1, 1, 6, 375, 0.2, 0],   # 良性，误剔；edge 1 这一轮只有它 → 整轮全被剔
            [2, 0, 6, 1, 7, 375, 0.1, 1]]
    m = _m("main", "t0", "collocated", 42, 0.5, 0.8, mode="filter", admitted_rows=rows)
    o = V.online_rates(m)
    assert o["attacker_pass"] == 0.5 and o["benign_reject"] == pytest.approx(1 / 3)
    assert o["n_attacker"] == 2 and o["n_benign"] == 3 and o["edge_rounds_all_rejected"] == 1


def test_thresholds_are_the_preregistered_values():
    assert (V.DASR_MEAN, V.DASR_MIN, V.DACC_MAX, V.ADD_MARGIN, V.RED_MARGIN, V.PLANT_MIN) == (0.15, 0.10, 0.02, 0.10, 0.05, 0.5)
    assert V.SEEDS == (42, 43, 44)
