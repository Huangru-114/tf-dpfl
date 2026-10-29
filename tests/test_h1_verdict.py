"""
阶段三 H1 判定（`harness/h1_verdict.py`，规则 = PLAN §4 H1 + D-101；本批 D-111）。
对照 = 真实的 SNAP__collocated__s42–44 metrics.json；防御臂 = 它们的拷贝，按用例改动受害 / E0 ASR 与精度。纯 python。
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import h1_verdict as H                                       # noqa: E402

RES = ROOT / "experiments/defense/edge-native/results/P2"
SNAP = RES / "SNAP"
P1_ARM = RES / "P1/P1__col-pre__s42.metrics.json"
VICT = (1, 2, 3)


@pytest.fixture(scope="module")
def ctrls():
    paths = [SNAP / f"SNAP__collocated__s{s}.metrics.json" for s in (42, 43, 44)]
    if not all(p.exists() for p in paths):
        pytest.skip("SNAP 的 metrics.json 不在本地")
    return [json.loads(p.read_text()) for p in paths]


def _arm(ctrl, dv=0.0, db0=0.0, dmta=0.0, cfg=H.ARM_CFG):
    a = copy.deepcopy(ctrl)
    n = int(a["run"]["n_rounds"])
    a["run"]["edge_harden"] = cfg
    a["harden"] = [{"round": r, "edge": e, "diverged_at": None, "seconds": 1.0} for r in range(1, n + 1) for e in range(4)]
    ck1 = [c["global"] for c in ctrl["checksums"] if c["round"] == 1][0]
    a["harden_agg"] = [{"round": 1, "unhardened": ck1}]
    for rows in a["per_edge_rounds"].values():
        for d in rows:
            if d.get("client_benign") is None:
                continue
            if d["edge_id"] in VICT:
                d["client_benign"] -= dv
            elif d["edge_id"] == 0:
                d["client_benign"] -= db0
    for r in a["acc_rounds"]:
        if r.get("pm_acc") is not None:
            r["pm_acc"] -= dmta
    return a


def _pairs(ctrls, dvs, dmtas=None, db0s=None):
    dmtas = dmtas or (0,) * len(dvs)
    db0s = db0s or (0,) * len(dvs)
    assert len(ctrls) == len(dvs) == len(dmtas) == len(db0s)
    return [(_arm(c, dv, db, dm), c) for c, dv, dm, db in zip(ctrls, dvs, dmtas, db0s)]


def test_small_drop_below_the_ceiling_is_no_effect(ctrls):
    r = H.judge(_pairs(ctrls, (0.048,) * 3))           # 臂 V ≈ 0.947 < 0.95 → 不走天花板规则
    assert r["ceiling_rule"] is False and r["verdict"] == "no_effect"
    assert H.judge(_pairs(ctrls, (0.048, 0.048, 0.12)))["verdict"] == "partial"   # 最大 |Δ| ≥ 0.10（D-101）


def test_protects_needs_mean_and_min_seed(ctrls):
    assert H.judge(_pairs(ctrls, (0.2, 0.2, 0.2)))["verdict"] == "protects_victims"
    assert H.judge(_pairs(ctrls, (0.3, 0.25, 0.05)))["verdict"] == "partial"        # 最小 seed < 0.10
    assert H.judge(_pairs(ctrls, (0.16, 0.16, 0.16)))["verdict"] == "protects_victims"
    assert H.judge(_pairs(ctrls, (0.14, 0.14, 0.14)))["verdict"] == "partial"       # 反向锚点：均值 < 0.15


def test_accuracy_turns_protects_into_costly(ctrls):
    assert H.judge(_pairs(ctrls, (0.2,) * 3, dmtas=(0.03,) * 3))["verdict"] == "costly"
    assert H.judge(_pairs(ctrls, (0.2,) * 3, dmtas=(0.0, 0.0, 0.03)))["verdict"] == "costly"   # 单 seed > 0.025
    assert H.judge(_pairs(ctrls, (0.2,) * 3, dmtas=(0.015,) * 3))["verdict"] == "protects_victims"


def test_treats_e0_is_reported_separately(ctrls):
    r = H.judge(_pairs(ctrls, (0.2,) * 3, db0s=(0.2,) * 3))
    assert r["three_seed"]["treats_E0"] == "protects"
    assert H.judge(_pairs(ctrls, (0.2,) * 3))["three_seed"]["treats_E0"] == "no_effect"


def test_ceiling_is_left_to_the_user(ctrls):
    r = H.judge(_pairs(ctrls, (0.01,) * 3))           # 对照 0.995 → 臂 0.985，两边都 ≥ 0.95
    assert r["ceiling_rule"] is True and r["verdict"] == "ceiling_user_decides"


def test_fewer_than_three_pairs_is_insufficient(ctrls):
    assert H.judge(_pairs(ctrls[:2], (0.2, 0.2)))["verdict"] == "insufficient"


def test_gate_failure_is_invalid(ctrls):
    p = _pairs(ctrls, (0.2,) * 3)
    p[1] = (_arm(ctrls[1], 0.2, cfg="none"), ctrls[1])               # 开关没生效
    r = H.judge(p)
    assert r["verdict"] == "invalid" and any("s43" in x for x in r["reasons"])
    bad = _arm(ctrls[2], 0.2)
    bad["harden_agg"] = [{"round": 1, "unhardened": "deadbeef0000"}]  # 第一次加固之前就与对照不同
    p = _pairs(ctrls, (0.2,) * 3)
    p[2] = (bad, ctrls[2])
    assert H.judge(p)["verdict"] == "invalid"


def test_mismatched_seed_is_invalid(ctrls):
    p = _pairs(ctrls, (0.2,) * 3)
    p[0] = (_arm(ctrls[1], 0.2), ctrls[0])                            # s43 的臂配 s42 的对照
    assert H.judge(p)["verdict"] == "invalid"


def _reseed(m, s):
    m = copy.deepcopy(m)
    m["run"]["seed"] = s
    return m


def test_five_seeds_can_only_confirm_or_downgrade(ctrls):
    extra = [_reseed(ctrls[0], 45), _reseed(ctrls[0], 46)]
    up = _pairs(ctrls + extra, (0.2, 0.2, 0.2, 0.2, 0.2))
    r = H.judge(up)
    assert r["five_seed"]["dV"]["label"] == "protects" and r["verdict"] == "protects_victims"
    down = _pairs(ctrls + extra, (0.2, 0.2, 0.2, -0.3, -0.3))
    r = H.judge(down)
    assert r["three_seed"]["verdict"] == "protects_victims" and r["five_seed"]["dV"]["label"] != "protects"
    assert r["verdict"] == r["five_seed"]["dV"]["label"]              # 降级
    weak3 = _pairs(ctrls + extra, (0.3, 0.25, 0.05, 0.3, 0.3))        # 3 seed partial；5 seed 也不能升级
    assert H.judge(weak3)["verdict"] == "partial"


def test_bootstrap_ci_is_deterministic_and_degenerate_for_constants():
    assert H.bootstrap_ci([0.2] * 5) == (pytest.approx(0.2), pytest.approx(0.2))
    assert H.bootstrap_ci([0.1, 0.3, 0.2, 0.0, 0.4]) == H.bootstrap_ci([0.1, 0.3, 0.2, 0.0, 0.4])
    lo, hi = H.bootstrap_ci([0.1, 0.3, 0.2, 0.0, 0.4])
    assert 0.0 <= lo < 0.2 < hi <= 0.4


@pytest.mark.skipif(not P1_ARM.exists(), reason="P1 的 metrics.json 不在本地")
def test_real_s42_pair_matches_p1_verdict(ctrls):
    """真实锚点：s42 这一对的 ΔV / ΔMTA 与 p1_verdict（F-103）同一个数；只有一对 → insufficient。"""
    arm = json.loads(P1_ARM.read_text())
    r = H.judge([(arm, ctrls[0])])
    assert r["verdict"] == "insufficient"
    s = r["per_seed"][0]
    assert s["seed"] == 42 and round(s["dV"], 4) == 0.4761 and round(s["dMTA"], 4) == 0.0086
