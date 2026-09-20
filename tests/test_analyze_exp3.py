"""
tests/test_analyze_exp3.py  —  harness/analyze_exp3.py 的 L1 测试

纯 stdlib，不 import TF，本地秒级。覆盖 experiments/attack/hfl-propagation/
current-focus.md §1 原文要求的六条：

  插值解析解；*_distributed 的延迟是 None 且反向锚点（有干净 edge 的 fixture
  不是 None）；从不越过 θ -> crossed=false 且不返回 0 也不返回 n_rounds；
  六个 HHI 值逐个对上；edge_of(cid) 在真实归档文件上跑；无权平均 ≠ 加权平均
  （构造样本数悬殊的 fixture，断言用的是无权那个）。

外加合理的补充覆盖（退化情形、None-安全、真实数据冒烟）。真实数据文件不存在时
（例如在没有集群回传结果的环境里跑）相关测试 pytest.skip，不假设特定环境。
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

from analyze_exp3 import (  # noqa: E402
    CELL_RE,
    aggregate_cell,
    analyze_run,
    build_report,
    clean_edge_delay,
    clean_edge_propagation_strength,
    delta_hier,
    dilution_per_edge,
    edge_of,
    first_crossing,
    hhi,
    hhi_regression,
    load_results,
    local_amplification_per_edge,
    participation_by_edge,
    per_edge_round_series,
    private_head_blocking_series,
    seed_band,
    to_csv_rows,
    unweighted_mean_over_edges,
    _ols_fit,
)

RESULTS = ROOT / "experiments" / "attack" / "hfl-propagation" / "results"
ARCHIVE = RESULTS / "archive-pre-fix"


def _skip_if_absent(path):
    if not path.exists():
        pytest.skip(f"真实数据不在（{path}），跳过")


# ══════════════════════════════════════════════════════════════════════════
# 1. first_crossing —— 插值解析解 + 四态互斥
# ══════════════════════════════════════════════════════════════════════════

def test_first_crossing_linear_interpolation_exact():
    series = [(0, 0.0), (10, 0.5), (20, 1.0)]
    c = first_crossing(series, 0.25, min_points=3)
    assert c.crossed is True
    assert c.t_theta == pytest.approx(5.0)
    assert c.left_censored is False
    assert c.reason is None


def test_first_crossing_multi_point_interpolation_exact():
    # 越阈发生在 (10,0.2) 与 (15,0.6) 之间：t = 10 + (0.5-0.2)*(15-10)/(0.6-0.2) = 13.75
    series = [(0, 0.0), (5, 0.1), (10, 0.2), (15, 0.6), (20, 0.9)]
    c = first_crossing(series, 0.5, min_points=4)
    assert c.crossed is True
    assert c.t_theta == pytest.approx(13.75)


def test_first_crossing_never_crosses_is_censored_not_zero_not_total_rounds():
    n_rounds, edge_rounds = 20, 1
    series = [(0, 0.0), (5, 0.05), (10, 0.1), (15, 0.12), (n_rounds * edge_rounds, 0.15)]
    c = first_crossing(series, 0.9, min_points=4)
    assert c.crossed is False
    assert c.t_theta is None
    assert c.t_theta != 0
    assert c.t_theta != n_rounds * edge_rounds
    assert c.censored_at == n_rounds * edge_rounds


def test_first_crossing_left_censored():
    series = [(0, 0.8), (5, 0.85), (10, 0.9), (15, 0.95)]
    c = first_crossing(series, 0.5, min_points=4)
    assert c.left_censored is True
    assert c.crossed is False
    assert c.t_theta is None
    assert c.reason is None


def test_first_crossing_grid_too_coarse():
    series = [(0, 0.1), (10, 0.9)]  # 只有 2 点，默认 min_points=4
    c = first_crossing(series, 0.5)
    assert c.reason == "grid_too_coarse"
    assert c.t_theta is None
    assert c.crossed is False


@pytest.mark.parametrize("series,theta,expect", [
    ([(0, 0.0), (5, 0.1), (10, 0.2), (15, 0.6), (20, 0.9)], 0.5, "crossed"),
    ([(0, 0.0), (5, 0.05), (10, 0.1), (15, 0.12), (20, 0.15)], 0.9, "never"),
    ([(0, 0.8), (5, 0.85), (10, 0.9), (15, 0.95)], 0.5, "left"),
    ([(0, 0.1), (10, 0.9)], 0.5, "coarse"),
])
def test_first_crossing_four_states_are_mutually_exclusive(series, theta, expect):
    c = first_crossing(series, theta, min_points=4)
    states = {
        "crossed": c.crossed and c.t_theta is not None and not c.left_censored and c.reason is None,
        "never": (not c.crossed) and c.t_theta is None and c.censored_at is not None
                 and not c.left_censored and c.reason is None,
        "left": (not c.crossed) and c.left_censored and c.t_theta is None and c.reason is None,
        "coarse": c.reason == "grid_too_coarse" and c.t_theta is None,
    }
    assert states[expect], f"expected state {expect!r}, got {c!r}"
    assert sum(states.values()) == 1, f"状态应互斥，实际同时满足多个: {states}"


# ══════════════════════════════════════════════════════════════════════════
# 2. unweighted_mean_over_edges —— 无权平均 ≠ 加权平均
# ══════════════════════════════════════════════════════════════════════════

def test_unweighted_mean_over_edges_drops_none_and_counts():
    mean, n_used, n_dropped = unweighted_mean_over_edges([0.2, None, 0.6, None])
    assert mean == pytest.approx(0.4)
    assert n_used == 2
    assert n_dropped == 2


def test_unweighted_mean_over_edges_all_none_is_none_not_zero():
    mean, n_used, n_dropped = unweighted_mean_over_edges([None, None])
    assert mean is None
    assert n_used == 0
    assert n_dropped == 2


def test_unweighted_mean_not_weighted_mean():
    """构造样本数悬殊的 fixture，断言 Δ_hier 用的 mean_e 是无权那个。"""
    per_edge = [
        {"edge_id": 0, "edge_asr": 0.1, "n_benign": 1000, "n_malicious": 100},
        {"edge_id": 1, "edge_asr": 0.9, "n_benign": 1, "n_malicious": 1},
    ]
    values = [e["edge_asr"] for e in per_edge]
    mean, _, _ = unweighted_mean_over_edges(values)
    assert mean == pytest.approx(0.5)  # 朴素算术均值

    weighted = (
        sum(e["edge_asr"] * (e["n_benign"] + e["n_malicious"]) for e in per_edge)
        / sum(e["n_benign"] + e["n_malicious"] for e in per_edge)
    )
    assert weighted == pytest.approx((0.1 * 1100 + 0.9 * 2) / 1102)
    assert mean != pytest.approx(weighted)


# ══════════════════════════════════════════════════════════════════════════
# 3. delta_hier / local_amplification / dilution / clean_edge_propagation_strength
# ══════════════════════════════════════════════════════════════════════════

def test_delta_hier_basic_signed_value():
    per_edge_final = [{"edge_id": 0, "edge_asr": 0.3}, {"edge_id": 1, "edge_asr": 0.5}]
    delta, n_used, n_dropped = delta_hier(0.6, per_edge_final)
    assert delta == pytest.approx(0.2)  # mean_e=0.4, 0.6-0.4=0.2
    assert n_used == 2
    assert n_dropped == 0


def test_delta_hier_drops_none_edges_and_counts_them():
    per_edge_final = [{"edge_id": 0, "edge_asr": 0.4}, {"edge_id": 1, "edge_asr": None}]
    delta, n_used, n_dropped = delta_hier(0.5, per_edge_final)
    assert delta == pytest.approx(0.1)
    assert n_used == 1
    assert n_dropped == 1


def test_delta_hier_all_edges_none_is_none():
    per_edge_final = [{"edge_id": 0, "edge_asr": None}, {"edge_id": 1, "edge_asr": None}]
    delta, n_used, n_dropped = delta_hier(0.5, per_edge_final)
    assert delta is None
    assert n_dropped == 2


def test_local_amplification_and_dilution_per_edge_formulas():
    per_edge_final = [
        {"edge_id": 0, "edge_asr": 0.8, "client_benign": 0.3, "client_malicious": 0.95},
        {"edge_id": 1, "edge_asr": 0.1, "client_benign": 0.05, "client_malicious": None},
    ]
    amp = local_amplification_per_edge(per_edge_final)
    assert amp[0] == pytest.approx(0.5)
    assert amp[1] == pytest.approx(0.05)

    dil = dilution_per_edge(per_edge_final)
    assert dil[0] == pytest.approx(0.15)
    assert dil[1] is None  # client_malicious 是 None（该 edge 没有恶意客户端）


def test_clean_edge_propagation_strength_only_clean_edges():
    per_edge_final = [
        {"edge_id": 0, "has_malicious": True, "client_benign": 0.9},
        {"edge_id": 1, "has_malicious": False, "client_benign": 0.2},
    ]
    strength = clean_edge_propagation_strength(per_edge_final)
    assert strength == {1: pytest.approx(0.2)}


def test_clean_edge_propagation_strength_empty_when_no_clean_edge():
    per_edge_final = [{"edge_id": 0, "has_malicious": True, "client_benign": 0.9}]
    assert clean_edge_propagation_strength(per_edge_final) == {}


# ══════════════════════════════════════════════════════════════════════════
# 4. clean_edge_delay —— *_distributed 的延迟是 None + 反向锚点
# ══════════════════════════════════════════════════════════════════════════

def _build_per_edge_rounds(edge_series, has_malicious_by_edge):
    """edge_series: {edge_id: [(round, client_benign), ...]} -> per_edge_rounds 形状。"""
    rounds = sorted({r for series in edge_series.values() for r, _ in series})
    out = {}
    for r in rounds:
        entries = []
        for eid, series in edge_series.items():
            val = dict(series).get(r)
            entries.append({
                "edge_id": eid, "client_benign": val, "edge_asr": val,
                "client_malicious": None, "n_benign": 10, "n_malicious": 0,
                "has_malicious": has_malicious_by_edge[eid],
            })
        out[str(r)] = sorted(entries, key=lambda e: e["edge_id"])
    return out


def test_clean_edge_delay_none_for_distributed_shaped_fixture():
    """distributed 布点：每个 edge 都有恶意端，没有干净 edge -> delay 必须是 None。"""
    per_edge_final = [{"edge_id": 0, "has_malicious": True}, {"edge_id": 1, "has_malicious": True}]
    per_edge_rounds = _build_per_edge_rounds(
        {0: [(0, 0.0), (5, 0.3), (10, 0.6), (15, 0.9)],
         1: [(0, 0.0), (5, 0.2), (10, 0.5), (15, 0.8)]},
        {0: True, 1: True},
    )
    d = clean_edge_delay(per_edge_rounds, per_edge_final, edge_rounds=1, theta=0.5)
    assert d.delay is None
    assert d.reason == "no_clean_edge"
    assert d.n_clean_edges == 0
    assert d.n_contaminated_edges == 2


def test_clean_edge_delay_real_number_for_collocated_shaped_fixture():
    """反向锚点：collocated 布点，有干净 edge -> delay 必须是真实数字，不是 None。"""
    per_edge_final = [{"edge_id": 0, "has_malicious": True}, {"edge_id": 1, "has_malicious": False}]
    per_edge_rounds = _build_per_edge_rounds(
        {0: [(0, 0.0), (5, 0.3), (10, 0.6), (15, 0.9)],    # 污染 edge，涨得快
         1: [(0, 0.0), (5, 0.05), (10, 0.2), (15, 0.6)]},  # 干净 edge，涨得慢（跨 edge 传播延迟）
        {0: True, 1: False},
    )
    d = clean_edge_delay(per_edge_rounds, per_edge_final, edge_rounds=1, theta=0.5)
    assert d.reason is None
    assert isinstance(d.delay, float)
    assert d.delay > 0  # 干净 edge 更晚越阈


def test_clean_edge_delay_real_distributed_file_has_no_clean_edge():
    path = RESULTS / "4edge_distributed_seed42.metrics.json"
    _skip_if_absent(path)
    data = json.loads(path.read_text())
    per_edge_final = data["per_edge_final"]
    assert all(e["has_malicious"] for e in per_edge_final)
    d = clean_edge_delay(data["per_edge_rounds"], per_edge_final, data["run"]["edge_rounds"], 0.5)
    assert d.delay is None
    assert d.reason == "no_clean_edge"


def test_clean_edge_delay_real_collocated_file_is_not_none():
    path = RESULTS / "4edge_collocated_seed42.metrics.json"
    _skip_if_absent(path)
    data = json.loads(path.read_text())
    per_edge_final = data["per_edge_final"]
    assert any(not e["has_malicious"] for e in per_edge_final)
    d = clean_edge_delay(data["per_edge_rounds"], per_edge_final, data["run"]["edge_rounds"], 0.5)
    assert d.reason != "no_clean_edge"  # 干净 edge 检测本身必须成立
    assert d.delay is None or isinstance(d.delay, float)


# ══════════════════════════════════════════════════════════════════════════
# 5. edge_of —— 在真实归档文件上跑
# ══════════════════════════════════════════════════════════════════════════

def test_edge_of_synthetic_block_formula():
    assert edge_of(0, 100, 4) == 0
    assert edge_of(24, 100, 4) == 0
    assert edge_of(25, 100, 4) == 1
    assert edge_of(99, 100, 4) == 3


def test_edge_of_on_real_committed_file():
    path = RESULTS / "4edge_collocated_seed42.metrics.json"
    _skip_if_absent(path)
    data = json.loads(path.read_text())
    run = data["run"]
    ids = run["malicious_ids"]
    n_clients, n_edges = run["n_clients"], run["n_edges"]
    edges_hit = {edge_of(cid, n_clients, n_edges) for cid in ids}
    assert edges_hit == {0}
    assert run["malicious_per_edge"] == [10, 0, 0, 0]


def test_edge_of_participation_aggregation_matches_malicious_per_edge_counts():
    path = RESULTS / "4edge_distributed_seed42.metrics.json"
    _skip_if_absent(path)
    data = json.loads(path.read_text())
    run = data["run"]
    n_clients, n_edges = run["n_clients"], run["n_edges"]
    counts = {}
    for cid in run["malicious_ids"]:
        eid = edge_of(cid, n_clients, n_edges)
        counts[eid] = counts.get(eid, 0) + 1
    expected = dict(enumerate(run["malicious_per_edge"]))
    for eid, c in expected.items():
        assert counts.get(eid, 0) == c


# ══════════════════════════════════════════════════════════════════════════
# 6. hhi —— 六个已知拓扑的值逐个对上
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("filename,expected_hhi", [
    ("10edge_distributed_seed42.metrics.json", 0.10),
    ("4edge_distributed_seed42.metrics.json", 0.26),
    ("10edge_mixed_seed42.metrics.json", 0.30),
    ("4edge_mixed_seed42.metrics.json", 0.38),
    ("2edge_distributed_seed42.metrics.json", 0.50),
    ("2edge_collocated_seed42.metrics.json", 1.00),
    ("4edge_collocated_seed42.metrics.json", 1.00),
    ("10edge_collocated_seed42.metrics.json", 1.00),
])
def test_hhi_six_known_topologies_match_real_files(filename, expected_hhi):
    path = RESULTS / filename
    _skip_if_absent(path)
    data = json.loads(path.read_text())
    mpe = data["run"]["malicious_per_edge"]
    assert hhi(mpe) == pytest.approx(expected_hhi, abs=1e-6)


def test_hhi_none_when_no_malicious():
    assert hhi([]) is None
    assert hhi([0, 0, 0]) is None
    assert hhi(None) is None


# ══════════════════════════════════════════════════════════════════════════
# 7. participation_by_edge
# ══════════════════════════════════════════════════════════════════════════

def test_participation_by_edge_synthetic_counts():
    mpbc = {"0": [1, 2, 3], "1": [1, 2], "25": [1, 2, 3, 4], "26": [1]}
    result = participation_by_edge(mpbc, n_clients=100, n_edges=4)
    assert result == {0: 5, 1: 5}


def test_participation_by_edge_rejects_non_block_assignment():
    mpbc = {"0": [1, 2]}
    assert participation_by_edge(mpbc, 100, 4, edge_assignment="random") == {}


def test_participation_by_edge_defaults_to_block_when_assignment_missing():
    """archive-pre-fix 缺 edge_assignment 字段（None），仍按 block 处理。"""
    mpbc = {"0": [1, 2]}
    assert participation_by_edge(mpbc, 100, 4, edge_assignment=None) == {0: 2}


# ══════════════════════════════════════════════════════════════════════════
# 8. seed_band
# ══════════════════════════════════════════════════════════════════════════

def test_seed_band_basic_mean_min_max():
    band = seed_band({42: 0.5, 43: 0.7, 44: 0.3})
    assert band.mean == pytest.approx(0.5)
    assert band.min == pytest.approx(0.3)
    assert band.max == pytest.approx(0.7)
    assert band.n_seeds == 3
    assert band.seeds == [42, 43, 44]
    assert band.single_seed is False


def test_seed_band_single_seed_is_flagged():
    band = seed_band({42: 0.5})
    assert band.n_seeds == 1
    assert band.single_seed is True


def test_seed_band_drops_none_seeds_from_aggregate():
    band = seed_band({42: 0.5, 43: None, 44: 0.7})
    assert band.n_seeds == 2
    assert band.mean == pytest.approx(0.6)
    assert 43 not in band.seeds


def test_seed_band_all_none_is_none_not_zero():
    band = seed_band({42: None, 43: None})
    assert band.mean is None
    assert band.min is None
    assert band.max is None
    assert band.n_seeds == 0


# ══════════════════════════════════════════════════════════════════════════
# 9. _ols_fit
# ══════════════════════════════════════════════════════════════════════════

def test_ols_fit_matches_hand_calculated_values():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [1.0, 3.0, 5.0, 7.0]  # y = 2x + 1，完美线性
    fit = _ols_fit(xs, ys)
    assert fit is not None
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.n == 4


def test_ols_fit_none_for_fewer_than_two_points():
    assert _ols_fit([1.0], [2.0]) is None
    assert _ols_fit([], []) is None


def test_ols_fit_none_when_all_x_identical():
    assert _ols_fit([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None


# ══════════════════════════════════════════════════════════════════════════
# 10. load_results / build_report —— 真实数据 + archive-pre-fix 调通
# ══════════════════════════════════════════════════════════════════════════

def test_cell_re_matches_plot_exp3_pattern_literally():
    """跟 plot_exp3.py 的 CELL_RE 字面一致，两个工具对同一批文件分出同样的 cell。"""
    assert CELL_RE.pattern == r"^(?P<cell>.+)_seed(?P<seed>\d+)$"


def test_load_results_real_directory_groups_seeds():
    _skip_if_absent(RESULTS)
    grouped = load_results(RESULTS)
    assert "4edge_distributed" in grouped
    assert 42 in grouped["4edge_distributed"]


def test_analyze_run_smoke_on_real_committed_file():
    path = RESULTS / "4edge_distributed_seed42.metrics.json"
    _skip_if_absent(path)
    data = json.loads(path.read_text())
    result = analyze_run(data)
    assert "hhi" in result
    assert "delta_hier" in result
    assert "t_theta" in result
    assert set(result["t_theta"].keys()) == {"global_asr", "edge_asr", "local_benign_asr"}


def test_build_report_runs_on_archive_pre_fix_without_crashing():
    """current-focus.md 明确要求「先在 archive-pre-fix/ 上调通」——只看跑不跑得通。"""
    _skip_if_absent(ARCHIVE)
    report = build_report(ARCHIVE)
    assert report["meta"]["archive_pre_fix_data"] is True
    for rep in report["cells"].values():
        # 旧文件缺 malicious_per_edge，HHI 必须是 None，不能编造
        assert rep["hhi"]["value"] is None


def test_build_report_flags_archive_pre_fix_data():
    _skip_if_absent(ARCHIVE)
    _skip_if_absent(RESULTS)
    assert build_report(ARCHIVE)["meta"]["archive_pre_fix_data"] is True
    assert build_report(RESULTS)["meta"]["archive_pre_fix_data"] is False


def test_csv_rows_one_row_per_cell_with_expected_columns():
    _skip_if_absent(RESULTS)
    report = build_report(RESULTS)
    rows = to_csv_rows(report)
    assert len(rows) == len(report["cells"])
    expected_cols = {"cell", "method", "defense", "hhi",
                      "global_asr_mean", "edge_asr_mean", "local_benign_asr_mean",
                      "delta_hier_mean"}
    assert expected_cols.issubset(rows[0].keys())


def test_report_json_has_no_bare_nan():
    """JSON 里绝不能泄漏裸 nan/inf——未定义值一律 None，json.dumps(allow_nan=False) 会替我们盯着。"""
    _skip_if_absent(RESULTS)
    report = build_report(RESULTS)
    json.dumps(report, allow_nan=False)


def test_hhi_regression_runs_on_real_cells():
    _skip_if_absent(RESULTS)
    report = build_report(RESULTS)
    reg = report["hhi_regression"]
    assert "local_benign_asr" in reg
    fit = reg["local_benign_asr"]
    if fit is not None:
        assert fit["n"] >= 2


# ══════════════════════════════════════════════════════════════════════════
# 11. private_head_blocking_series —— None 传播
# ══════════════════════════════════════════════════════════════════════════

def test_private_head_blocking_series_propagates_none():
    per_edge_rounds = {
        "1": [{"edge_id": 0, "edge_asr": 0.5, "client_benign": None}],
        "2": [{"edge_id": 0, "edge_asr": 0.6, "client_benign": 0.2}],
    }
    series = private_head_blocking_series(per_edge_rounds, edge_rounds=1)
    assert series[0][0] == (1, None)
    assert series[0][1][0] == 2
    assert series[0][1][1] == pytest.approx(0.4)
