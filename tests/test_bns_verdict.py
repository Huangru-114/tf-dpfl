"""
阶段三 BNS 判定（`harness/bns_verdict.py`，规则 = FINDINGS N-009；D-120）。合成数据覆盖每个分支；
真实数据只钉种子 42 的两份已入库结果（F-114 的数）。纯 python。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import bns_verdict as V                                       # noqa: E402

N = 30
EN = ROOT / "experiments" / "defense" / "edge-native"
PRIV42 = EN / "legacy" / "CCSF-fedrep" / "CCSF__off__s42.metrics.json"
SHARED42 = EN / "results" / "P2" / "CCSBOFF" / "CCSBOFF__fedrep-bnshared-off__s42.metrics.json"


def _run(P, arm, seed, mta=0.9):
    return {
        "exit_code": 0, "client_failures": [], "n_malicious_participations": 25,
        "run": {"n_rounds": N, "edge_rounds": 1, "n_edges": 1, "seed": seed, "partition": None,
                "malicious_per_edge": [10], "method": "hier_fedrep", "defense": "none", "ccs": False,
                "alignment": {"fedrep_bn_stats": arm}},
        "rounds": [{"round": r, "local_benign_asr": P} for r in range(1, N + 1)],
        "acc_rounds": [{"round": r, "pm_acc": mta} for r in range(1, N + 1)],
        "client_final": {"client_id": [0, 1, 2, 3], "malicious": [True, False, False, False],
                         "asr": [1.0, 0.05, 0.5, 0.95], "yt_clean": [0.9, 0.0, 0.2, 0.4]},
    }


def _arms(priv, shared, seeds=(42, 43, 44)):
    return ([_run(p, "private", s) for p, s in zip(priv, seeds)],
            [_run(p, "shared", s) for p, s in zip(shared, seeds)])


# 用二进制下精确的数，免得边界被浮点误差决定
@pytest.mark.parametrize("priv,shared,want", [
    ((0.875, 0.75, 0.875), (0.75, 0.625, 0.75), "real_effect"),        # d 全 0.125 > 0，均值 0.125
    ((0.875, 0.75, 0.875), (0.75, 0.75, 0.5), "not_established"),      # 一个种子 d = 0（要 > 0，严格）
    ((0.875, 0.75, 0.875), (0.75, 0.875, 0.5), "not_established"),     # 一个种子反向，均值 0.125 也不行
    ((0.875, 0.75, 0.875), (0.8125, 0.6875, 0.8125), "not_established"),  # d 全 0.0625 > 0，但均值 < 0.10
    ((0.5, 0.5, 0.5), (0.875, 0.875, 0.875), "not_established"),       # 整体反向
])
def test_labels(priv, shared, want):
    res = V.judge(*_arms(priv, shared))
    assert res["verdict"] == want, res


def test_mean_threshold_is_inclusive_as_written():
    # d = 0.125 / 0.125 / 0.0625 → 均值 0.3125 / 3 ≈ 0.104 ≥ 0.10；d = 0.0625 / 0.125 / 0.09375 → 均值 0.09375 < 0.10
    assert V.judge(*_arms((0.875,) * 3, (0.75, 0.75, 0.8125)))["verdict"] == "real_effect"
    assert V.judge(*_arms((0.875,) * 3, (0.8125, 0.75, 0.78125)))["verdict"] == "not_established"


def test_pairs_by_seed_not_by_argument_order():
    priv, shared = _arms((0.875, 0.75, 0.875), (0.75, 0.625, 0.75))
    res = V.judge(priv, list(reversed(shared)))
    assert res["verdict"] == "real_effect" and res["delta_P"] == {"42": 0.125, "43": 0.125, "44": 0.125}


def test_fewer_than_three_seeds_is_insufficient():
    res = V.judge(*_arms((0.875, 0.75), (0.75, 0.625), seeds=(42, 43)))
    assert res["verdict"] == "insufficient"


def test_missing_P_is_insufficient():
    priv, shared = _arms((0.875,) * 3, (0.75,) * 3)
    for r in shared[2]["rounds"]:
        r["local_benign_asr"] = None
    assert V.judge(priv, shared)["verdict"] == "insufficient"


@pytest.mark.parametrize("mutate,needle", [
    (lambda p, s: s[1].update(exit_code=1), "exit_code"),
    (lambda p, s: s[1].pop("exit_code"), "exit_code 缺失"),
    (lambda p, s: p[0].update(client_failures=[{"client_id": 3}]), "client_failures"),
    (lambda p, s: p[2].update(n_malicious_participations=0), "攻击者"),
    (lambda p, s: s[0]["run"].update(defense="ccs"), "开了防御"),
    (lambda p, s: s[0]["run"].update(ccs=True), "开了防御"),
    (lambda p, s: s[1]["run"]["alignment"].update(fedrep_bn_stats="private"), "开关没生效"),
    (lambda p, s: p[1]["run"]["alignment"].update(fedrep_bn_stats=None), "开关没生效"),
    (lambda p, s: s[2]["run"].update(n_rounds=200), "设置不同"),
    (lambda p, s: p[2]["run"].update(malicious_per_edge=[5]), "设置不同"),
    (lambda p, s: s[2]["run"].update(seed=45), "种子集合不同"),
    (lambda p, s: s[2]["run"].update(seed=43), "种子重复"),
])
def test_gates(mutate, needle):
    priv, shared = _arms((0.875, 0.75, 0.875), (0.75, 0.625, 0.75))
    mutate(priv, shared)
    res = V.judge(priv, shared)
    assert res["verdict"] == "invalid" and any(needle in r for r in res["reasons"]), res


def test_side_reports():
    res = V.judge(*_arms((0.875,) * 3, (0.75,) * 3))
    c = res["runs"]["private"]["42"]["clients"]
    assert c == {"n_benign": 3, "lt_0.1": 1, "gt_0.9": 1, "median": 0.5, "spearman_yt_clean": 1.0}
    assert res["delta_MTA"] == {"42": 0.0, "43": 0.0, "44": 0.0}


@pytest.mark.skipif(not (PRIV42.exists() and SHARED42.exists()), reason="种子 42 的结果文件不在")
def test_real_seed42_matches_f114():
    """反向锚点：真实的两份 s42 结果过闸、算出 F-114 的数（私有 0.926 / 共享 0.796；免疫端 6 / 12）。"""
    priv = json.loads(PRIV42.read_text(encoding="utf-8"))
    shared = json.loads(SHARED42.read_text(encoding="utf-8"))
    res = V.judge([priv], [shared])
    assert res["verdict"] == "insufficient" and res["reasons"] == ["只有 1 个种子，要 3 个"]   # 过了所有闸，只缺种子
    assert res["runs"]["private"]["42"]["P"] == 0.9262 and res["runs"]["shared"]["42"]["P"] == 0.7958
    assert res["runs"]["private"]["42"]["clients"]["lt_0.1"] == 6
    assert res["runs"]["shared"]["42"]["clients"]["lt_0.1"] == 12
    # 把两臂对调 → 开关核对必须拦下（反向锚点：闸真的在读 fedrep_bn_stats）
    bad = V.judge([shared], [priv])
    assert bad["verdict"] == "invalid" and sum("开关没生效" in r for r in bad["reasons"]) == 2
