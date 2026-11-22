"""
tests/test_supp_verdict.py  —  harness/supp_verdict.py（补充实验第一批的预注册判定，N-015 / D-135）

  · 合成数据覆盖 D1 / D2 / D4 的每个分支，以及门槛上的浮点容差（0.9 − 0.8 = 0.0999…98 算作 0.10）；
  · 阈值反向锚点：常量 = PREREG-v2 写定的数（改常量就是改预注册，这里红）；
  · 闸：缺文件 → insufficient；块数 / 布点 / 防御名 / 作用层不符、没跑满、非 P2、client_failures → invalid；
    D1 的闸 ②（k1-snap 与已有 B=1 逐位相同）—— checksum 差一个就 invalid；
  · 真实数据只钉现有 run 算得出的数：SNAP 集中 / ERP + ERC main-t1 集中的受害边缘攻击成功率与精度（current-focus〇·一的 0.996 → 0.281）。
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import supp_verdict as V                                    # noqa: E402

SEEDS = (42, 43, 44)


def _m(*, seed=42, k=0, mpe=(10, 0, 0, 0), defense="none", layers=None, edge=(0.9, 0.5, 0.5, 0.5),
       mta=0.86, P=0.6, rounds=60, template="p2", ck_salt="", edge_ref_mode=None):
    """合成 metrics.json：每台边缘的 client_benign 全程恒定 = edge[e]，精度恒定 = mta。"""
    run = {"seed": seed, "edge_shared_blocks": k, "malicious_per_edge": list(mpe), "defense": defense,
           "defense_layers": layers, "n_rounds": rounds, "alignment": {"template": template},
           "edge_ref_mode": edge_ref_mode,
           "tier_split": {"n_edge_params": 3_676_160, "n_params": 4_909_002} if k else None}
    return {"exit_code": 0, "client_failures": [], "errors": [], "n_malicious_participations": 25, "run": run,
            "rounds": [{"round": r, "local_benign_asr": P} for r in range(1, rounds + 1)],
            "acc_rounds": [{"round": r, "pm_acc": mta, "pm_acc_stale": mta + 0.03 if r % 2 else None}
                           for r in range(1, rounds + 1)],
            "per_edge_rounds": {str(r): [{"edge_id": e, "client_benign": edge[e]} for e in range(4)]
                                for r in range(1, rounds + 1)},
            "checksums": [{"round": r, "global": f"ck{seed}-{r}{ck_salt}"} for r in range(1, rounds + 1)]}


def _seeds(vals=None, **kw):
    """{seed: m}；vals = {seed: 覆盖参数}。"""
    vals = vals or {}
    return {s: _m(seed=s, **{**kw, **vals.get(s, {})}) for s in SEEDS}


# ── 阈值反向锚点 ─────────────────────────────────────────────────────────
def test_thresholds_are_the_preregistered_ones():
    assert (V.D1_MEANINGFUL, V.D1_FREE, V.D1_BETTER) == (0.02, 0.01, -0.01)
    assert (V.D2_CLEAN_PROTECTED, V.D2_CLEAN_UNPROTECTED, V.D2_E1_BREACHED, V.D2_E1_PROTECTED) == (0.30, 0.10, 0.20, 0.10)
    assert (V.D4_MTA, V.D4_VS_B1) == (0.02, 0.10)
    assert V.TOL == 1e-9 and V.SEEDS == SEEDS and V.N_ROUNDS == 60
    text = (ROOT / "experiments/thesis/PREREG-v2.md").read_text(encoding="utf-8")
    for s in ("D ≥ 0.02", "|D| < 0.01", "D ≤ −0.01", "Δ_clean ≥ 0.30", "Δ_clean ≤ 0.10", "Gap ≥ 0.20",
              "|Gap| < 0.10", "≥ 0.10 → `weaker_than_b1`"):
        assert s.replace("|", "\\|") in text or s in text, s


# ── 容差 ─────────────────────────────────────────────────────────────────
def test_tolerance_treats_values_within_1e9_as_the_threshold():
    x = 0.9 - 0.8                                  # 0.09999999999999998
    assert V._ge(x, 0.10) and not V._lt(x, 0.10)
    assert V._le(0.1 + 0.2 - 0.2, 0.10)
    assert not V._gt(1e-12, 0.0) and V._gt(1e-6, 0.0)


# ── D1 ───────────────────────────────────────────────────────────────────
def _d1(mta_k1, mta_k4, **extra):
    k1 = _seeds({s: {"mta": v} for s, v in zip(SEEDS, mta_k1)}, k=1)
    k4 = _seeds({s: {"mta": v} for s, v in zip(SEEDS, mta_k4)}, k=4)
    ref = _seeds({s: {"mta": v, "defense": "edge_ref", "edge_ref_mode": "observe"}
                  for s, v in zip(SEEDS, mta_k1)}, k=1)
    return V.judge_d1(k1=k1, k4=k4, k1_ref=ref, **extra)


@pytest.mark.parametrize("k1,k4,label", [
    ((0.87, 0.87, 0.87), (0.84, 0.85, 0.80), "b1_meaningful"),
    ((0.87, 0.87, 0.87), (0.865, 0.871, 0.8695), "isolation_free"),
    ((0.85, 0.85, 0.85), (0.86, 0.87, 0.90), "isolation_better"),
    ((0.87, 0.87, 0.87), (0.84, 0.865, 0.80), "inconclusive"),          # 一个种子落在两条阈值之间
])
def test_d1_branches(k1, k4, label):
    r = _d1(k1, k4)
    assert r["label"] == label, r
    assert r["gate2"]["s42"]["n_diffs"] == 0 and r["gate2"]["s42"]["n_checksums"] == 60


def test_d1_threshold_edge_uses_the_tolerance():
    """D = 0.87 − 0.85 在浮点里是 0.020000000000000018 / 0.01999999999999…，都算 ≥ 0.02。"""
    r = _d1((0.87, 0.9, 0.3), (0.85, 0.88, 0.28))
    assert r["label"] == "b1_meaningful", r["D_per_seed"]


def test_d1_gate2_one_checksum_off_is_invalid():
    k1 = _seeds(k=1)
    ref = _seeds({44: {"ck_salt": "x"}}, k=1, defense="edge_ref", edge_ref_mode="observe")
    r = V.judge_d1(k1=k1, k4=_seeds(k=4), k1_ref=ref)
    assert r["label"] == "invalid" and "gate2 s44" in r["invalid"] and "gate2 s42" not in r["invalid"]


def test_d1_missing_file_is_insufficient_and_wrong_k_is_invalid():
    k1 = _seeds(k=1)
    ref = _seeds(k=1, defense="edge_ref", edge_ref_mode="observe")
    r = V.judge_d1(k1=k1, k4={42: _m(k=4), 43: _m(seed=43, k=4)}, k1_ref=ref)
    assert r["label"] == "insufficient" and r["missing"] == ["k4 s44"]
    r = V.judge_d1(k1=k1, k4=_seeds({43: {"k": 2}}, k=4), k1_ref=ref)
    assert r["label"] == "invalid" and "k4 s43" in r["invalid"]


def test_d1_b1_reference_must_be_observe_not_filter():
    ref = _seeds(k=1, defense="edge_ref", edge_ref_mode="filter")
    r = V.judge_d1(k1=_seeds(k=1), k4=_seeds(k=4), k1_ref=ref)
    assert r["label"] == "invalid" and any("observe" in x for v in r["invalid"].values() for x in v)


def test_d1_report_and_monotonicity_flag():
    k2 = _seeds(k=2, edge=(0.9, 0.3, 0.3, 0.3))
    k4 = _seeds({44: {"edge": (0.9, 0.4, 0.4, 0.4)}}, k=4, edge=(0.9, 0.1, 0.1, 0.1))
    r = V.judge_d1(k1=_seeds(k=1), k4=k4, k2=k2, k0=_seeds(k=0, edge=(1.0, 1.0, 1.0, 1.0)),
                   k1_ref=_seeds(k=1, defense="edge_ref", edge_ref_mode="observe"))
    assert r["monotone_V_k4_le_k2_le_k1"] == {"s42": True, "s43": True, "s44": False}
    assert r["report"]["B=1"]["s42"]["cloud_fraction"] == round(1 - 3_676_160 / 4_909_002, 6)
    assert r["report"]["B=0"]["s42"]["cloud_fraction"] == 1.0
    assert r["report"]["B=1"]["s42"]["stale_acc"] == 0.89


# ── D2 ───────────────────────────────────────────────────────────────────
def _d2(k0_edges, k1_edges):
    k0 = _seeds({s: {"edge": e} for s, e in zip(SEEDS, k0_edges)}, k=0, mpe=(9, 1, 0, 0))
    k1 = _seeds({s: {"edge": e} for s, e in zip(SEEDS, k1_edges)}, k=1, mpe=(9, 1, 0, 0))
    return V.judge_d2(k0, k1)


def test_d2_protected_and_breached_is_the_trust_domain_reading():
    r = _d2([(1, 1, 1, 1)] * 3, [(1, 0.6, 0.3, 0.3), (1, 0.7, 0.2, 0.4), (1, 0.9, 0.25, 0.25)])
    assert r["label"] == {"clean": "clean_protected", "e1": "e1_breached"}
    assert "信任域" in r["reading"]
    assert r["gap_per_seed"]["s42"] == 0.3 and r["delta_clean_per_seed"]["s43"] == 0.7


def test_d2_protected_and_e1_also_protected():
    r = _d2([(1, 1, 1, 1)] * 3, [(1, 0.35, 0.3, 0.3)] * 3)
    assert r["label"] == {"clean": "clean_protected", "e1": "e1_also_protected"}
    assert "7.1" in r["reading"]


def test_d2_unprotected_overrides_the_reading():
    r = _d2([(1, 1, 1, 1)] * 3, [(1, 1, 0.95, 0.95)] * 3)
    assert r["label"]["clean"] == "clean_unprotected" and "只在一个边缘" in r["reading"]


def test_d2_between_thresholds_is_inconclusive():
    r = _d2([(1, 1, 1, 1)] * 3, [(1, 0.5, 0.75, 0.75), (1, 0.5, 0.3, 0.3), (1, 0.5, 0.3, 0.3)])
    assert r["label"] == {"clean": "inconclusive", "e1": "inconclusive"}


def test_d2_requires_the_boundary_placement():
    k0 = _seeds(k=0, mpe=(10, 0, 0, 0))
    r = V.judge_d2(k0, _seeds(k=1, mpe=(9, 1, 0, 0)))
    assert r["label"] == "invalid" and set(r["invalid"]) == {"k0 s42", "k0 s43", "k0 s44"}


# ── D4 ───────────────────────────────────────────────────────────────────
def _arm(layer, place, edge=(1, 0.5, 0.5, 0.5), P=0.5, mta=0.86, **kw):
    mpe = (10, 0, 0, 0) if place == "collocated" else (3, 3, 2, 2)
    return _seeds(k=0, mpe=mpe, defense="median", layers=[layer], edge=edge, P=P, mta=mta, **kw)


def _ctrl(place, edge=(1, 1, 1, 1), P=0.99, mta=0.86):
    mpe = (10, 0, 0, 0) if place == "collocated" else (3, 3, 2, 2)
    return _seeds(k=0, mpe=mpe, edge=edge, P=P, mta=mta)


def _b1(place, edge=(1, 0.3, 0.3, 0.3), P=0.95):
    mpe = (10, 0, 0, 0) if place == "collocated" else (3, 3, 2, 2)
    return _seeds(k=1, mpe=mpe, defense="edge_ref", edge_ref_mode="observe", edge=edge, P=P)


def test_d4_blocks_and_weaker_than_b1():
    c = V.judge_d4_cell(_arm("cloud", "collocated"), _ctrl("collocated"), _b1("collocated"), "cloud", "collocated")
    assert c["label"] == "blocks" and c["vs_b1"]["label"] == "weaker_than_b1"
    assert c["d_asr_per_seed"]["s42"] == 0.5 and c["vs_b1"]["diff_per_seed"]["s42"] == 0.2


def test_d4_accuracy_cost_over_002_is_not_blocks_mean_and_every_seed():
    arm = _arm("cloud", "collocated")
    arm[44]["acc_rounds"] = [dict(r, pm_acc=0.83) for r in arm[44]["acc_rounds"]]      # 这个种子降 0.03
    c = V.judge_d4_cell(arm, _ctrl("collocated"), _b1("collocated"), "cloud", "collocated")
    assert c["label"] == "not_blocks" and c["mta_mean_ok"] and not c["mta_every_seed_ok"]


def test_d4_asr_must_drop_in_every_seed():
    arm = _arm("edge", "collocated")
    arm[43]["per_edge_rounds"] = _ctrl("collocated")[43]["per_edge_rounds"]           # 这个种子没降
    c = V.judge_d4_cell(arm, _ctrl("collocated"), _b1("collocated"), "edge", "collocated")
    assert c["label"] == "not_blocks" and not c["asr_all_positive"]


def test_d4_distributed_uses_the_pooled_benign_asr():
    c = V.judge_d4_cell(_arm("edge", "distributed", edge=(1, 1, 1, 1), P=0.9), _ctrl("distributed", P=0.97),
                        _b1("distributed", P=0.95), "edge", "distributed")
    assert c["asr_quantity"].startswith("P") and c["d_asr_per_seed"]["s42"] == 0.07
    assert c["vs_b1"]["label"] == "comparable_to_b1"


@pytest.mark.parametrize("arm_v,v44,label", [(0.15, 0.18, "stronger_than_b1"),      # −0.15 / −0.15 / −0.12
                                             (0.35, 0.33, "comparable_to_b1"),       # 0.05 / 0.05 / 0.03
                                             (0.42, 0.33, "inconclusive")])          # 0.12 / 0.12 / 0.03
def test_d4_vs_b1_three_bands(arm_v, v44, label):
    arm = _arm("cloud", "collocated", edge=(1, arm_v, arm_v, arm_v))
    arm[44]["per_edge_rounds"] = _arm("cloud", "collocated", edge=(1, v44, v44, v44))[44]["per_edge_rounds"]
    c = V.judge_d4_cell(arm, _ctrl("collocated"), _b1("collocated"), "cloud", "collocated")
    assert c["vs_b1"]["label"] == label, c["vs_b1"]


def test_d4_layer_mismatch_and_missing_b1():
    arm = _arm("edge", "collocated")                     # 声称是云层臂，run 块却写着 edge
    c = V.judge_d4_cell(arm, _ctrl("collocated"), _b1("collocated"), "cloud", "collocated")
    assert c["label"] == "invalid" and any("作用层" in x for v in c["invalid"].values() for x in v)
    c = V.judge_d4_cell(_arm("edge", "collocated"), _ctrl("collocated"), {}, "edge", "collocated")
    assert c["label"] == "blocks" and c["vs_b1"]["label"] == "insufficient"


@pytest.mark.parametrize("mut,needle", [
    (lambda m: m.update(client_failures=[{"client_id": 3}]), "client_failures"),
    (lambda m: m.update(errors=["Traceback"]), "errors"),
    (lambda m: m["run"].update(n_rounds=59), "60"),
    (lambda m: m["run"]["alignment"].update(template="legacy"), "P2"),
    (lambda m: m.update(n_malicious_participations=0), "参与"),
])
def test_gate_rejects(mut, needle):
    m = _m()
    mut(m)
    assert any(needle in x for x in V.gate(m, k=0, mpe=[10, 0, 0, 0], defense="none"))
    assert V.gate(_m(), k=0, mpe=[10, 0, 0, 0], defense="none") == []


# ── 真实数据（只钉现有 run 算得出的数）────────────────────────────────────
def test_real_data_controls_reproduce_the_current_focus_numbers():
    res = V.run_all()
    d1 = res["d1"]
    assert d1["label"] == "insufficient" and d1["invalid"] == {}           # 新组还没回传；已有对照都过闸
    b0 = d1["report"]["B=0"]
    assert [b0[f"s{s}"]["V"] for s in SEEDS] == [0.995, 0.9965, 0.9963]
    col = res["d2"]["report"]["集中布点对照"]
    assert [col[f"s{s}"]["V_B1"] for s in SEEDS] == [0.3308, 0.1966, 0.3151]   # 均值 0.281（current-focus〇·一）
    for c in res["d4"]["cells"]:
        assert c["label"] == "insufficient" and c["missing"] == ["arm s42", "arm s43", "arm s44"]


def test_cli_writes_json(tmp_path):
    out = tmp_path / "v.json"
    assert V.main(["--only", "d2", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["d2"]["label"] == "insufficient"
