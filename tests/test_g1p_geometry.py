"""
tests/test_g1p_geometry.py  —  G1P 的描述性读数（harness/g1p_geometry.py；S6a / D-085）。手算值，纯 numpy。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import g1p_geometry as GG                                        # noqa: E402

COLS = ["round", "edge_id", "edge_round", "cid", "mal", "norm", "cos_edge", "cos_global"]


def _m(rows, **kw):
    return {"update_geometry": {"columns": COLS, "rows": rows}, **kw}


def test_auroc_per_view_on_a_hand_built_case():
    rows = [
        # 云轮 1、edge 0、edge 轮 1：恶意 cid 0（cos −0.5）与良性 cid 1（cos +0.2）都在 → 混合组
        [1, 0, 1, [0, 1], [1, 0], [3.0, 1.0], [-0.5, 0.2], [-0.4, 0.1]],
        # edge 1：只有良性 → 不进混合组
        [1, 1, 1, [2, 3], [0, 0], [1.0, 1.2], [0.1, 0.3], [0.0, 0.2]],
    ]
    g = GG.geometry_readout(_m(rows))
    assert (g["n_updates"], g["n_malicious"], g["n_benign"]) == (4, 1, 3)
    assert g["auroc_norm"] == 1.0                                   # 3.0 比 1.0 / 1.0 / 1.2 都大
    assert g["auroc_neg_cos_global"] == 1.0                         # −(−0.4) = 0.4 > −0.1, 0, −0.2
    assert g["n_mixed_updates"] == 2 and (g["n_mixed_malicious"], g["n_mixed_benign"]) == (1, 1)
    assert g["auroc_neg_cos_edge"] == 1.0 and g["auroc_neg_cos_global_on_mixed"] == 1.0


def test_collocated_has_no_mixed_group_and_none_scores_are_skipped():
    rows = [[1, 0, 1, [0, 1], [1, 1], [1.0, 1.0], [None, None], [0.5, None]],      # E0 全是恶意端
            [1, 1, 1, [5], [0], [1.0], [None], [0.1]]]
    g = GG.geometry_readout(_m(rows))
    assert g["n_mixed_updates"] == 0 and g["auroc_neg_cos_edge"] is None
    # −cos_global：恶意 [−0.5]（一个 None 被跳过）vs 良性 [−0.1] → 恶意更小 → 0.0
    assert g["auroc_neg_cos_global"] == 0.0
    assert g["auroc_norm"] == 0.5                                   # 全平局


def test_empty_is_none_not_zero():
    g = GG.geometry_readout({})
    assert g["n_updates"] == 0 and g["auroc_norm"] is None and g["auroc_neg_cos_edge"] is None


def test_jump_and_frozen_readouts():
    m = {"acc_rounds": [{"round": 1, "pm_acc": 0.6}, {"round": 2, "pm_acc": 0.7}],
         "rounds": [{"round": 1, "local_benign_asr": 0.30}, {"round": 2, "local_benign_asr": 0.50}],
         "post_agg_rounds": [{"round": 2, "effective_round": 10, "pm_acc": 0.55,
                              "local_benign_asr": 0.40}],
         "frozen_rounds": [{"round": 1, "phase": "full", "local_benign": 0.25},
                           {"round": 2, "phase": "post", "local_benign": 0.40},
                           {"round": 2, "phase": "full", "local_benign": 0.45}]}
    j = GG.jump_readout(m)
    assert len(j) == 1 and j[0]["d_pm_acc"] == pytest.approx(-0.05)
    assert j[0]["d_local_benign_asr"] == pytest.approx(0.10)
    f = GG.frozen_readout(m)
    assert [(x["round"], round(x["diff"], 6)) for x in f] == [(1, -0.05), (2, -0.05)]   # post 不在其内
