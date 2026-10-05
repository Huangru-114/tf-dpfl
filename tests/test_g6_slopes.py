"""
tests/test_g6_slopes.py  —  harness/g6_slopes.py（检查 3，探索性）。手算值，纯标准库。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import g6_slopes as S                                              # noqa: E402


def _run(vals_by_round, R=5):
    """vals_by_round：{云轮: 受害 edge 的值}（三个受害 edge 同值，E0 = 1）。"""
    return {"run": {"edge_rounds": R, "n_rounds": max(vals_by_round)},
            "per_edge_rounds": {str(g): [{"edge_id": 0, "client_benign": 1.0}]
                                + [{"edge_id": e, "client_benign": v} for e in (1, 2, 3)]
                                for g, v in vals_by_round.items()}}


def test_slope_windows_are_in_effective_rounds():
    # 0.1 + 0.002·有效轮，R5、60 云轮 → 两个窗口的斜率都是 0.002 每有效轮，残差 0 → SE 0
    m = _run({g: 0.1 + 0.002 * 5 * g for g in range(1, 61)})
    q = S.run_quantities(m, None)
    assert q["n_50"] == 10 and q["n_100"] == 20
    assert abs(q["slope_50"] - 0.002) < 1e-12 and abs(q["slope_100"] - 0.002) < 1e-12
    assert q["se_50"] < 1e-9
    assert abs(q["end"] - (0.1 + 0.002 * 5 * 55.5)) < 1e-12 and q["floor"] is None


def test_end_minus_floor_and_paired_difference():
    flat = {g: 0.30 for g in range(1, 61)}
    up = {g: (0.30 if g <= 50 else 0.30 + 0.01 * (g - 50)) for g in range(1, 61)}
    flr = _run({g: 0.05 for g in range(1, 61)})
    runs = {}
    for s in S.SEEDS:
        runs[("a", s)] = _run(up)
        runs[("b", s)] = _run(flat)
        runs[("c", s)] = _run(flat)
    res = S.analyse(runs, {s: flr for s in S.SEEDS})
    c = res["arms"]["c"]
    assert abs(c["end_minus_floor"]["mean"] - 0.25) < 1e-12
    # a 臂末 10 点：每云轮 +0.01 = 每有效轮 +0.002；c 臂 0 → 差 −0.002，三个 seed 相同 → CI 退化成一点
    assert abs(c["slope_50_minus_a"]["mean"] + 0.002) < 1e-12
    assert abs(c["slope_50_minus_a"]["lo"] + 0.002) < 1e-12


def test_se_matches_hand_value():
    # 点 (0,0) (1,1) (2,0)：斜率 0，残差 (−1/3, 2/3, −1/3) → RSS 2/3，Sxx 2 → SE = √((2/3)/1/2) = √(1/3)
    assert abs(S.slope_se([(0, 0.0), (1, 1.0), (2, 0.0)]) - (1 / 3) ** 0.5) < 1e-12
