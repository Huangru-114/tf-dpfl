"""
tests/test_verdicts.py  —  PLAN §3 预注册判定的执行器

3-A：T50(HFL)/T50(flat) 按 seed 配对，bootstrap CI 判方向；seed 不够 → insufficient，
不报方向。合成数据给出解析解；真实 P1 数据上必须全部是 insufficient（每格只有 1 个 seed）。
"""

import csv
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import verdicts as V                     # noqa: E402
import runs_table as T                   # noqa: E402


def test_bootstrap_ci_edge_cases_and_determinism():
    assert V.bootstrap_mean_ci([]) == (None, None, None)
    assert V.bootstrap_mean_ci([0.3]) == (0.3, None, None)       # 一个点给不出区间
    assert V.bootstrap_mean_ci([0.2] * 6) == (pytest.approx(0.2),) * 3
    a = V.bootstrap_mean_ci([0.1, 0.5, 0.2, 0.9, 0.4])
    assert a == V.bootstrap_mean_ci([0.1, 0.5, 0.2, 0.9, 0.4])   # 固定种子 → 可复现
    assert a[1] < a[0] < a[2]


def _rows(t_by_seed, state="crossed"):
    return [{"seed": str(s), "t0.5_local_benign_asr": str(t),
             "t0.5_local_benign_asr_state": state} for s, t in t_by_seed.items()]


FLAT = {s: 50.0 for s in range(5)}


def test_hfl_twice_as_slow_is_structural_delay():
    v = V.verdict_t50_ratio(_rows({s: 100.0 for s in range(5)}), _rows(FLAT))
    assert v["verdict"] == "structural_delay"
    assert v["ratio"]["mean"] == pytest.approx(2.0)
    assert v["n_paired"] == 5


def test_equal_speed_is_approx_equal():
    v = V.verdict_t50_ratio(_rows(FLAT), _rows(FLAT))
    assert v["verdict"] == "approx_equal"
    assert v["log_ratio"] == {"mean": 0.0, "lo": 0.0, "hi": 0.0}


def test_hfl_faster_is_its_own_verdict():
    v = V.verdict_t50_ratio(_rows({s: 25.0 for s in range(5)}), _rows(FLAT))
    assert v["verdict"] == "hfl_faster"


def test_noisy_ratio_is_inconclusive():
    hfl = {0: 40.0, 1: 70.0, 2: 45.0, 3: 65.0, 4: 50.0}
    v = V.verdict_t50_ratio(_rows(hfl), _rows(FLAT))
    assert v["verdict"] == "inconclusive"


def test_too_few_seeds_is_insufficient_and_reports_no_direction():
    v = V.verdict_t50_ratio(_rows({s: 100.0 for s in range(4)}), _rows(FLAT))
    assert v["verdict"] == "insufficient"
    assert v["ratio"] is None and v["log_ratio"] is None


def test_censored_seeds_are_counted_not_averaged():
    hfl = _rows({s: 100.0 for s in range(5)}) + []
    hfl[0]["t0.5_local_benign_asr_state"] = "censored"
    hfl[0]["t0.5_local_benign_asr"] = ""
    v = V.verdict_t50_ratio(hfl, _rows(FLAT))
    assert v["n_censored"] == 1 and v["n_paired"] == 4
    assert v["verdict"] == "insufficient"


def test_replicates_of_one_seed_are_geometric_mean():
    rows = _rows({0: 40.0}) + _rows({0: 90.0})
    assert V.per_seed_t(rows)["0"] == pytest.approx(math.sqrt(40.0 * 90.0))


def test_p1_every_comparison_is_insufficient(tmp_path):
    runs, series = T.build([ROOT / "experiments/attack/hfl-propagation/results"],
                           legacy_protocol="P1")
    T.write_tables(runs, series, tmp_path)
    with (tmp_path / "runs.csv").open() as f:
        rows = list(csv.DictReader(f))
    out = V.plan_3a(rows)
    # 10 个 10% 攻击者的 HFL 拓扑（8 个 3A/3B + 3c_R1 + 3c_R40）；攻击者比例格不能拿去和 flat 比
    assert len(out) == 10
    assert {v["verdict"] for v in out} == {"insufficient"}
    assert all(json_mpe_total(v["hfl"]["malicious_per_edge"]) == 10 for v in out)


def json_mpe_total(s):
    import json
    return sum(json.loads(s))
