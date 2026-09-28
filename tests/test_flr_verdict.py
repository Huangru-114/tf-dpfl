"""
tests/test_flr_verdict.py  —  floor 验证 pilot（FLR，D-061）

两件事分开测：
  1. 配对的前提：FLR 的配置与 G6(a) **只差 poison_ratio**（和 meta 里的组名 / run_id）；
  2. 预注册判定 `harness/flr_verdict.py` 的三档与边界（≤ LOW 可忽略、≥ HIGH 或 E0 差 ≥ GAP 不可忽略）。
纯 python + PyYAML，不 import TF。
"""

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import flr_verdict as F      # noqa: E402

CONFIGS = ROOT / "experiments/attack/hfl-mechanism/configs"
MAL = [6, 7, 9, 10, 15, 16, 17, 19, 20, 24]


# ── 1. 配对前提 ─────────────────────────────────────────────────────────────
def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


@pytest.mark.parametrize("seed", F.SEEDS)
def test_flr_config_differs_from_g6a_only_in_poison_ratio(seed):
    flr = yaml.safe_load((CONFIGS / f"FLR__g6a__s{seed}.yaml").read_text(encoding="utf-8"))
    g6a = yaml.safe_load((CONFIGS / f"G6__a__s{seed}.yaml").read_text(encoding="utf-8"))
    a, b = _flat(flr), _flat(g6a)
    diff = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert diff == {"backdoor.poison_ratio", "meta.group", "meta.run_id"}
    assert a["backdoor.poison_ratio"] == 0.0 and b["backdoor.poison_ratio"] == 0.2
    assert a["seed"] == b["seed"] == seed


# ── 2. 判定 ─────────────────────────────────────────────────────────────────
def _m(edge_floors, n=20, early=None, rho=0.0, mal=MAL):
    """合成 metrics：每个 edge 的 client_benign 末 10 点恒为 edge_floors[e]；前面的点为 early（缺省同值）。"""
    per, rounds = {}, []
    for i in range(1, n + 1):
        vals = edge_floors if (early is None or i > n - 10) else early
        per[str(i)] = [{"edge_id": e, "client_benign": v} for e, v in enumerate(vals)]
        rounds.append({"round": i, "local_benign_asr": sum(vals[1:]) / 3, "global_asr": vals[0]})
    return {"exit_code": 0, "client_failures": [], "rounds": rounds, "per_edge_rounds": per,
            "run": {"poison_ratio": rho, "n_edges": 4, "malicious_ids": list(mal)}}


def _pairs(flr, atk=None):
    return {s: (copy.deepcopy(flr), copy.deepcopy(atk)) for s in F.SEEDS}


def test_all_small_is_negligible_boundary_inclusive():
    assert F.judge(_pairs(_m([0.02, 0.01, 0.03, 0.02])))["overall"] == "negligible"
    assert F.judge(_pairs(_m([0.05, 0.05, 0.05, 0.05])))["overall"] == "negligible"   # ≤ LOW


def test_one_edge_at_high_is_not_negligible_boundary_inclusive():
    assert F.judge(_pairs(_m([0.02, 0.10, 0.02, 0.02])))["overall"] == "not_negligible"   # ≥ HIGH
    assert F.judge(_pairs(_m([0.02, 0.099, 0.02, 0.02])))["overall"] == "user_decides"


def test_e0_gap_alone_makes_it_not_negligible():
    res = F.judge(_pairs(_m([0.09, 0.02, 0.03, 0.04])))      # max < HIGH，但 E0 − 受害均值 = 0.06
    assert res["overall"] == "not_negligible"
    assert res["per_seed"][0]["gap_e0_vs_victims"] == pytest.approx(0.06)


def test_between_thresholds_without_gap_is_user_decides():
    assert F.judge(_pairs(_m([0.07, 0.07, 0.07, 0.07])))["overall"] == "user_decides"


def test_window_is_last_10_points_and_first_10_is_reported():
    """前面的点很高、末 10 点很低 → 判定只看末 10 点；首 10 点另报（floor 是否随训练变）。"""
    res = F.judge(_pairs(_m([0.01] * 4, early=[0.5] * 4)))
    assert res["overall"] == "negligible"
    assert res["per_seed"][0]["floor_edge_first10"] == [0.5] * 4


def test_excess_against_paired_g6a():
    res = F.judge(_pairs(_m([0.02, 0.01, 0.03, 0.02]), atk=_m([1.0, 0.70, 0.90, 0.94], rho=0.2)))
    a = res["per_seed"][0]["attack"]
    assert a["same_malicious_ids"] is True
    assert a["excess_edge"] == pytest.approx([0.98, 0.69, 0.87, 0.92])
    assert a["floor_over_attack"][1] == pytest.approx(0.01 / 0.70, abs=1e-4)


def test_mismatched_malicious_ids_are_flagged_not_hidden():
    res = F.judge(_pairs(_m([0.02] * 4), atk=_m([0.9] * 4, rho=0.2, mal=[1, 2, 3])))
    assert res["per_seed"][0]["attack"]["same_malicious_ids"] is False


def test_not_a_floor_run_is_invalid():
    """把攻击 run 误当成 floor（ρ≠0）→ invalid，而不是给出一个很大的 floor。"""
    res = F.judge(_pairs(_m([0.9] * 4, rho=0.2)))
    assert res["overall"] == "invalid"
    assert "poison_ratio" in res["per_seed"][0]["reasons"][0]


def test_swallowed_client_failure_is_invalid():
    bad = _m([0.02] * 4)
    bad["client_failures"] = [{"client_id": 3, "error": "x"}]
    assert F.judge(_pairs(bad))["overall"] == "invalid"


def test_missing_and_insufficient():
    assert F.judge({})["overall"] == "missing"
    part = {42: (_m([0.02] * 4), None)}
    assert F.judge(part)["overall"] == "insufficient"


def test_real_g6a_pairs_and_flr_not_yet_back():
    """真实 G6(a)（s42 / s43 已回）能被读成攻击臂；FLR 还没回来 → missing（不是报错）。"""
    pairs = F.load()
    assert all(pairs[s][0] is None for s in F.SEEDS)
    assert F.judge(pairs)["overall"] == "missing"
    atk = pairs[42][1]
    assert atk is not None
    res = F.judge_seed(_m([0.02] * 4, mal=atk["run"]["malicious_ids"]), atk, 42)
    assert res["attack"]["same_malicious_ids"] is True
    assert all(v is not None for v in res["attack"]["attack_edge"])
