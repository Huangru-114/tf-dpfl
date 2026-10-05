"""
tests/test_tail_clients.py  —  harness/tail_clients.py（检查 4，探索性）+ analysis/client_profile.label_profile。
手算值；纯 numpy，不需要 TF。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "fedavg"))

import tail_clients as T                                           # noqa: E402
from analysis.client_profile import label_profile                 # noqa: E402


def test_spearman_ties_and_sign():
    assert T.ranks([3, 1, 1, 2]) == [4.0, 1.5, 1.5, 3.0]
    assert T.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert T.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert T.spearman([1, None, 3], [1, 2, 3]) is None              # 有效点 < 3


def test_client_eval_log_rows_and_window_mean():
    lines = ["[ClientEval] Round 51 | edge0 | ids=1/2 | mal=0/1 | asr=0.6/0.9 | acc=0.8/0.7 | yt=0.1/0.0 | mmed=1/2 | n=5/5",
             "[ClientEval] Round 52 | edge0 | ids=1/2 | mal=0/1 | asr=0.2/na | acc=0.9/0.7 | yt=0.3/0.0 | mmed=1/2 | n=5/5",
             "[ClientEval] Round 70 | edge0 | ids=1/2 | mal=0/1 | asr=0.0/0.0 | acc=0.0/0.0 | yt=0.0/0.0 | mmed=1/2 | n=5/5"]
    rows = T.client_rows_from_log(lines)
    assert rows[51][2]["malicious"] and rows[52][2]["asr"] is None
    w = T.window_means(rows, [51, 52])
    assert w[1]["asr"] == pytest.approx(0.4) and w[1]["acc"] == pytest.approx(0.85) and w[1]["n_points"] == 2
    assert w[1]["yt_clean"] == pytest.approx(0.2) and w[2]["n_points"] == 1


def test_selection_replay_matches_real_g8():
    import json
    m = json.loads((T.RESULTS / "G8/G8__a__s42.metrics.json").read_text(encoding="utf-8"))
    sel = T.replay_selection(42, 4, 5, 70)
    ok, n = T.selection_matches(sel, m, 5)
    assert n == 10 and ok == 10
    assert sum(len(v) for v in sel.values()) == 70 * 5 * 10       # 每有效轮 10 个名额


def test_label_profile_counts():
    labels = np.array([0, 0, 1, 2, 0, 3])
    c = SimpleNamespace(client_id=7, assigned_edge=2, is_malicious=False, _train_src=(None, labels, [0, 1, 2, 5]))
    p = label_profile([c], None, 4)
    assert p["7"] == {"edge": 2, "n_train": 4, "counts": [2, 1, 0, 1], "malicious": False}


def test_overlap_null():
    def run(tails):
        return {"clients": [{"client_id": i, "asr": float(i in tails), "tail": i in tails} for i in range(20)]}
    o = T.tail_overlap(run({0, 1, 2}), run({0, 1, 2}), n_perm=2000)
    assert o["overlap"] == 3 and o["null_mean"] == pytest.approx(3 * 3 / 20, abs=0.05) and o["p_ge"] < 0.01
