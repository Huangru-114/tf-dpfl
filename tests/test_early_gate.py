"""
tests/test_early_gate.py  —  harness/early_gate.py（检查 2，探索性）。手算值，纯标准库。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import early_gate as E                                            # noqa: E402


def _run(e0, victims, acc, R=5, mpe=(10, 0, 0, 0), light=None):
    """e0 / victims / acc：按云轮 1, 2, … 给值；light：{有效轮: (e0, victim)} 的轻评估点。"""
    per_edge = {str(g): [{"edge_id": 0, "client_benign": a}] + [{"edge_id": e, "client_benign": v}
                                                                 for e in (1, 2, 3)]
                for g, (a, v) in enumerate(zip(e0, victims), start=1)}
    m = {"run": {"edge_rounds": R, "malicious_per_edge": list(mpe)}, "per_edge_rounds": per_edge,
         "acc_rounds": [{"round": g, "pm_acc": x} for g, x in enumerate(acc, start=1)],
         "rounds": [{"round": g, "local_benign_asr": (a + 3 * v) / 4}
                    for g, (a, v) in enumerate(zip(e0, victims), start=1)]}
    if light:
        cols = list(E.LIGHT_COLS_DEFAULT)
        m["per_edge_light_columns"] = cols
        m["per_edge_light_rounds"] = {str(t): [[0, None, a, None, None, None, None, None]]
                                      + [[e, None, v, None, None, None, None, None] for e in (1, 2, 3)]
                                      for t, (a, v) in light.items()}
    return m


def test_cross_at_first_point_and_effective_round_axis():
    m = _run([0.9, 1.0], [0.1, 0.2], [0.45, 0.5], R=10, light={5: (0.8, 0.05)})
    tr = E.trajectory(m, None, 60)
    assert [p["eff"] for p in tr] == [5, 10, 20]
    d = E.describe(tr)
    assert d["gaps"] == [5, 10]
    assert d["shape"] == "crossed_at_first_point" and d["first_cross"]["eff"] == 5
    # 轻评估点没有 pm_acc → 取之前最近的点；第一个点之前没有 → None
    assert d["first_cross"]["pm_acc"] is None


def test_floor_then_takeoff_vs_rising():
    fl = _run([0.05, 0.05, 0.05, 0.05], [0.0] * 4, [0.4] * 4)
    hug = _run([0.10, 0.12, 0.7, 0.9], [0.0] * 4, [0.4, 0.5, 0.6, 0.7])
    d = E.describe(E.trajectory(hug, fl, 60))
    assert d["shape"] == "floor_then_takeoff"
    assert d["first_cross"] == {"eff": 15, "atk_benign": 0.7, "floor_atk": 0.05, "victim": 0.0,
                                "pm_acc": 0.6, "pm_acc_eff": 15}
    rise = _run([0.10, 0.30, 0.7, 0.9], [0.0] * 4, [0.4] * 4)
    assert E.describe(E.trajectory(rise, fl, 60))["shape"] == "rising_from_first_point"
    assert E.describe(E.trajectory(rise, None, 60))["shape"] == "crossed_later_no_floor"
    assert E.describe(E.trajectory(fl, fl, 60))["shape"] == "not_crossed"


def test_distributed_uses_pooled_benign_and_floor_follows_the_view():
    fl = _run([0.4, 0.4], [0.0, 0.0], [0.4, 0.4])                 # floor 自己名义上是集中布点
    m = _run([0.8, 0.8], [0.2, 0.2], [0.4, 0.4], mpe=(3, 3, 2, 2))
    tr = E.trajectory(m, fl, 60)
    assert tr[0]["atk_benign"] == (0.8 + 3 * 0.2) / 4 and tr[0]["victim"] is None
    assert tr[0]["floor_atk"] == 0.1                              # 池化视角：(0.4 + 0) / 4


def test_horizon_cuts():
    m = _run([0.1] * 20, [0.0] * 20, [0.5] * 20)
    assert E.trajectory(m, None, 60)[-1]["eff"] == 60
