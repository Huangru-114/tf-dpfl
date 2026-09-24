"""
tests/test_runs_table.py  —  metrics.json → runs.csv / series.csv

锁住三件事：
  1. 终值是末 10 点均值（不是单个末轮，F-010）；
  2. 分组看 run 块里的**实际**因素，不看文件名 —— 真实数据锚点：
     def_median_flat_baseline 实际没开防御，它必须作为 flat_baseline 的一份重复出现（F-001/F-002）；
  3. 不同口径版本（P1 / P2）不许混进同一张表。

纯标准库，本地秒级。
"""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import runs_table as T                                          # noqa: E402
from analyze_exp3 import effective_round_series, first_crossing  # noqa: E402

P1_DIR = ROOT / "experiments" / "attack" / "hfl-propagation" / "results"


# ── 纯计算 ───────────────────────────────────────────────────────────────────
def test_last_k_mean_skips_none_and_uses_only_the_tail():
    vals = [0.0] * 5 + [None] + [1.0] * 10
    assert T.last_k_mean(vals) == (1.0, 10)
    assert T.last_k_mean([0.2, None, 0.4], k=10) == (pytest.approx(0.3), 2)
    assert T.last_k_mean([None, None]) == (None, 0)


def _metrics(lb, *, er=5, protocol=None, run_over=None, exit_code=0):
    rounds = [{"round": i + 1, "global_asr": v, "edge_asr": v, "local_benign_asr": v,
               "local_malicious_asr": 1.0, "same_edge_asr": None, "diff_edge_asr": None}
              for i, v in enumerate(lb)]
    run = {"seed": 42, "method": "hier_fedrep", "attack": "badpfl", "defense": "none",
           "n_clients": 100, "n_edges": 2, "edge_rounds": er, "client_fraction": 0.1,
           "poison_ratio": 0.2, "malicious_per_edge": [5, 5], "local_epochs": 5}
    run.update(run_over or {})
    if protocol:
        run["provenance"] = {"protocol": protocol, "run_id": None}
    return {"run": run, "rounds": rounds,
            "acc_rounds": [{"round": i + 1, "gm_acc": 0.5, "em_acc": 0.6,
                            "pm_acc": 0.7 if i % 2 else None} for i in range(len(lb))],
            "exit_code": exit_code}


def test_summary_uses_last_ten_points_not_the_last_round():
    lb = [0.1 * i for i in range(12)]          # 0.0 … 1.1，末轮 1.1，末 10 点均值 0.65
    row = T.summarize_run(_metrics(lb), name="x", source="x", legacy_protocol="P1")
    assert row["local_benign_asr_last10"] == pytest.approx(0.65)
    assert row["local_benign_asr_n"] == 10
    assert row["pm_acc_last10"] == pytest.approx(0.7) and row["pm_acc_n"] == 6
    assert row["protocol"] == "P1"


def test_t_theta_matches_analyze_exp3():
    m = _metrics([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    row = T.summarize_run(m, name="x", source="x")
    series = effective_round_series(m["rounds"], 5, "local_benign_asr")
    assert row["t0.5_local_benign_asr"] == first_crossing(series, 0.5).t_theta == 17.5
    assert row["t0.5_local_benign_asr_state"] == "crossed"
    assert row["t0.75_local_benign_asr"] == pytest.approx(23.75)


def test_censored_and_coarse_states_are_named():
    never = T.summarize_run(_metrics([0.1] * 6), name="a", source="a")
    assert never["t0.5_local_benign_asr"] is None
    assert never["t0.5_local_benign_asr_state"] == "censored"
    coarse = T.summarize_run(_metrics([0.1, 0.9]), name="b", source="b")
    assert coarse["t0.5_local_benign_asr_state"] == "grid_too_coarse"


def test_mixed_protocols_are_refused(tmp_path):
    (tmp_path / "a.metrics.json").write_text(json.dumps(_metrics([0.5] * 5, protocol="P1")))
    (tmp_path / "b.metrics.json").write_text(json.dumps(_metrics([0.5] * 5, protocol="P2")))
    with pytest.raises(T.MixedProtocolError):
        T.build([tmp_path])
    runs, _ = T.build([tmp_path], allow_mixed=True)
    assert sorted(r["protocol"] for r in runs) == ["P1", "P2"]


def test_archive_is_skipped_by_default(tmp_path):
    (tmp_path / "archive-pre-fix").mkdir()
    (tmp_path / "archive-pre-fix" / "old.metrics.json").write_text(json.dumps(_metrics([0.5] * 5)))
    (tmp_path / "new.metrics.json").write_text(json.dumps(_metrics([0.5] * 5)))
    runs, _ = T.build([tmp_path])
    assert [r["run"] for r in runs] == ["new"]


def test_series_rows_are_long_format_with_effective_rounds():
    m = _metrics([0.1, 0.2], er=5)
    m["per_edge_rounds"] = {"1": [{"edge_id": 0, "edge_asr": 0.3, "client_benign": 0.2,
                                   "client_malicious": None}]}
    rows = T.series_rows(m, "r")
    assert ("r", 5, 1, 0, "edge.client_benign", 0.2) in rows
    assert ("r", 10, 2, -1, "local_benign_asr", 0.2) in rows
    assert not any(m_ == "edge.client_malicious" for *_, m_, _ in rows)   # None 不写


# ── 真实数据（P1）─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def p1():
    runs, series = T.build([P1_DIR], legacy_protocol="P1")
    return {r["run"]: r for r in runs}, series


def test_def_flat_cells_group_with_flat_baseline_as_replicates(p1):
    runs, _ = p1
    names = ["def_median_flat_baseline_seed42", "def_multi_krum_flat_baseline_seed42",
             "flat_baseline_seed42"]
    keys = {runs[n]["factor_key"] for n in names}
    assert len(keys) == 1
    assert sorted(runs[n]["replicate"] for n in names) == [1, 2, 3]
    assert all(runs[n]["n_replicates"] == 3 for n in names)
    got = {n: round(runs[n]["local_benign_asr_last10"], 3) for n in names}
    assert got == {"def_median_flat_baseline_seed42": 0.696,
                   "def_multi_krum_flat_baseline_seed42": 0.685,
                   "flat_baseline_seed42": 0.755}


def test_3c_R5_is_the_same_cell_as_2edge_distributed(p1):
    """按实际因素分组才看得出来：3c_R5 与 2edge_distributed 的因素完全相同。"""
    runs, _ = p1
    names = ["2edge_distributed_seed42", "3c_R5_seed42",
             "def_median_2edge_distributed_seed42", "def_multi_krum_2edge_distributed_seed42"]
    assert len({runs[n]["factor_key"] for n in names}) == 1
    assert all(runs[n]["n_replicates"] == 4 for n in names)


def test_p1_table_writes_and_reads_back(p1, tmp_path):
    runs, series = p1
    T.write_tables(list(runs.values()), series, tmp_path)
    with (tmp_path / "runs.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 26
    assert {r["protocol"] for r in rows} == {"P1"}
    with (tmp_path / "series.csv").open() as f:
        assert f.readline().strip() == "run,r_eff,cloud_round,edge_id,metric,value"
