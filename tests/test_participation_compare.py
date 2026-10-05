"""
tests/test_participation_compare.py  —  harness/participation_compare.py（检查 5，探索性）。手算值，纯 numpy。
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import participation_compare as P                                  # noqa: E402

MAL = set(range(6, 16))                                            # 10 个恶意端全在 E0（block：0–24）


def test_quota_versions():
    # P0：int(25 · 0.1) = 2（陷阱 #12）；P1：云周期内不变；P2：每有效轮轮转
    assert [P.quota("P0", 0, 4, 25, 1, er, 5) for er in range(1, 6)] == [2] * 5
    assert [P.quota("P1", 0, 4, 25, 1, er, 5) for er in range(1, 6)] == [3] * 5
    assert [P.quota("P1", 0, 4, 25, 2, er, 5) for er in range(1, 6)] == [2] * 5
    assert [P.quota("P2", 0, 4, 25, 1, er, 5) for er in range(1, 6)] == [3, 2, 3, 2, 3]
    assert P.quota("P0", 0, 1, 100, 1, 1, 1) == P.quota("P2", 0, 1, 100, 1, 1, 1) == 10


def test_expected_counts_by_hand():
    e0 = P.expected_per_eff("P0", 4, 5, MAL, 2)
    assert e0["attacker"] == pytest.approx(2 * 10 / 25) and e0["total"] == pytest.approx(8)
    e2 = P.expected_per_eff("P2", 4, 5, MAL, 4)
    assert e2["attacker"] == pytest.approx(2.5 * 10 / 25) and e2["total"] == pytest.approx(10)
    assert P.expected_per_eff("P2", 1, 1, MAL, 3)["benign"] == pytest.approx(9)


def test_replay_reproduces_select_clients_stream():
    rep = P.replay("P0", 42, 4, 5, 1, MAL)
    rng = np.random.default_rng([42, 0])
    picked = [int(i) for _ in range(5) for i in rng.choice(25, 2, replace=False)]
    assert rep["attacker"] == [sum(1 for i in picked[2 * k:2 * k + 2] if i in MAL) for k in range(5)]
    assert sum(rep["attacker"]) + sum(rep["benign"]) == 5 * 8


def test_real_runs_replay_exactly():
    res = P.analyse(seeds=(42,))
    assert res["cases"] and all(c["replay_ok"] for c in res["cases"])
