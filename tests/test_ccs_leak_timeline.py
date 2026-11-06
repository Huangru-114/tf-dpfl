"""
`harness/ccs_leak_timeline.py`（F-118，描述性读数）：合成数据手算 + 真实种子 43 钉住 F-118 的数。纯 python。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import ccs_leak_timeline as L                                 # noqa: E402

RES = ROOT / "experiments" / "defense" / "edge-native" / "results" / "P2"


def _m(decisions, em=None, gasr=None, mal=(0, 1)):
    return {"run": {"malicious_ids": list(mal), "seed": 1, "ccs_at": False, "ccs_cluster": True},
            "ccs_decisions": decisions,
            "acc_rounds": [{"round": r, "em_acc": v} for r, v in (em or [])],
            "rounds": [{"round": r, "global_asr": v} for r, v in (gasr or [])]}


def test_leak_counts_by_hand():
    d = [{"cur": [0, 1, 5], "rejected": [0]},        # 放进 1（攻击者 1），良性 5 没剔
         {"cur": [0, 1, 5], "rejected": []},         # 放进 2
         {"cur": [1, 6], "rejected": [1, 6]}]        # 放进 0，良性 6 被剔
    res = L.leak(_m(d), block=2)
    assert (res["admitted"], res["selected"]) == (3, 5)
    assert res["admitted_per_block"] == [3, 0] and res["rounds_ge2"] == [2]
    assert res["by_attacker"] == {"0": "1/2", "1": "2/3"}
    assert res["benign_rejection"] == {"median": 1.0, "max": 1.0, "n_ge_half": 1, "n_benign": 2}


def test_history_replay_by_hand():
    # 攻击者 0：第 1、2 轮被剔 → 第 3 轮起视为剔除（出现 2 次、被剔 2/2）；第 3 轮原本放进 → 回放里不放进
    # 良性 5：第 1、2 轮被剔、第 3 轮原本放进 → 回放多剔 1 个
    d = [{"cur": [0, 5], "rejected": [0, 5]}, {"cur": [0, 5], "rejected": [0, 5]}, {"cur": [0, 5], "rejected": []}]
    res = L.history_replay(_m(d))
    assert res == {"min_seen": 2, "frac": 0.5, "attacker_admitted": 0, "extra_benign_rejected": 1, "benign_uploads": 3}
    # 只出现过 1 次不触发
    assert L.history_replay(_m(d[1:]))["attacker_admitted"] == 1


def test_no_cluster_rows_means_no_leak_section():
    assert L.leak(_m([])) is None


def test_stability_by_hand():
    em = [(5, 0.5), (10, 0.25), (15, 0.6), (20, 0.1)]
    s = L.stability(_m([], em=em))
    assert s == {"first_below": 10, "n_below": 2, "n_points": 4, "last10_mean": round((0.5 + 0.25 + 0.6 + 0.1) / 4, 4)}


_S43 = RES / "CCSDC5" / "CCSDC5__fedrep-bnshared-cluster-only__s43.metrics.json"
_B44 = RES / "CCSB3" / "CCSB3__fedrep-bnshared-ccs__s44.metrics.json"


@pytest.mark.skipif(not (_S43.exists() and _B44.exists()), reason="结果文件不在")
def test_real_numbers_match_f118():
    s43 = L.describe(json.loads(_S43.read_text(encoding="utf-8")))
    lk = s43["leak"]
    assert (lk["admitted"], lk["selected"]) == (67, 307)
    assert lk["admitted_per_block"] == [15, 3, 3, 2, 8, 11, 2, 6, 8, 1, 5, 3]
    assert lk["by_attacker"]["3"] == "15/38" and lk["history_replay"]["attacker_admitted"] == 29
    assert dict(s43["global_asr_every10"])[130] > 0.8 >= dict(s43["global_asr_every10"])[120] * 4
    b44 = L.describe(json.loads(_B44.read_text(encoding="utf-8")))
    assert b44["stability"]["first_below"] == 85 and b44["stability"]["n_below"] == 40
