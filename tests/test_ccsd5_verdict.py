"""
阶段三 CCS 拆分实验的多种子判定（`harness/ccsd5_verdict.py`，规则 = FINDINGS N-011；D-123）。
合成数据覆盖每个分支；真实数据钉「已有的种子 42–44 文件过闸、只缺种子」。纯 python。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import ccsd5_verdict as V5                                    # noqa: E402

N = 30
MAL = list(range(10))
RES = ROOT / "experiments" / "defense" / "edge-native" / "results" / "P2"


def _run(arm, seed, P, mta, em=0.85):
    at, cl = {"off": (None, None), "both": (True, True), "at": (True, False), "cluster": (False, True)}[arm]
    m = {
        "exit_code": 0, "client_failures": [], "n_malicious_participations": 25,
        "run": {"n_rounds": N, "edge_rounds": 1, "n_edges": 1, "seed": seed, "partition": None,
                "malicious_per_edge": [10], "malicious_ids": MAL, "method": "hier_fedrep",
                "alignment": {"fedrep_bn_stats": "shared"},
                "ccs": arm != "off", "ccs_at": at, "ccs_cluster": cl},
        "rounds": [{"round": r, "local_benign_asr": P, "local_benign_asr_unfiltered": P} for r in range(1, N + 1)],
        "acc_rounds": [{"round": r, "pm_acc": mta, "em_acc": em} for r in range(1, N + 1)],
        "ccs_decisions": [],
    }
    if cl:
        m["ccs_decisions"] = [{"edge": 0, "cur": [0, 1] + list(range(50, 58)), "n_pool": 10,
                               "status": "ok", "rejected": [0, 50]} for _ in range(N)]
    return m


H5, A3 = (42, 43, 44, 45, 46), (42, 43, 44)


def _set(cluster=None, at=None, both=None, at_em=None, off_P=0.875):
    """缺省：off P 0.875 / MTA 0.875；cluster 每种子 (0.125, 0.875)；both (0.125, 0.75)；at (0.125, 0.5)，at 的 EM 0.85。"""
    cluster = cluster or {s: (0.125, 0.875) for s in H5}
    at = at or {s: (0.125, 0.5) for s in A3}
    both = both or {s: (0.125, 0.75) for s in A3}
    at_em = at_em or {s: 0.85 for s in A3}
    return ([_run("off", s, off_P, 0.875) for s in H5],
            [_run("both", s, *both[s]) for s in A3],
            [_run("at", s, *at[s], em=at_em[s]) for s in A3],
            [_run("cluster", s, *cluster[s]) for s in cluster])


def test_cluster_alone_sufficient_and_attribution():
    res = V5.judge(*_set())
    assert res["headline"]["label"] == "cluster_alone_sufficient"
    assert res["headline"]["mean_delta_P"] == 0.75 and res["headline"]["ci95_delta_P"] == [0.75, 0.75]
    a = res["attribution"]
    assert a["cost"] == "adversarial_training" and a["defense"] == "both_components" and a["consistency"] == "consistent"
    assert res["verdict"] == "headline=cluster_alone_sufficient;attribution=cost=adversarial_training;defense=both_components;consistent"


def test_headline_mixed_when_fewer_than_four_seeds_meet():
    # 两个种子精度降 0.03125 > 0.02（不各自满足），其余 0 → 均值 0.0125 ≤ 0.02 但只有 3 个种子各自满足
    c = {s: (0.125, 0.875) for s in H5}
    c[45] = c[46] = (0.125, 0.84375)
    assert V5.judge(*_set(cluster=c))["headline"]["label"] == "mixed"
    c[44] = (0.125, 0.875)
    c[46] = (0.125, 0.875)                                   # 只有 45 一个不满足 → 4 个满足
    assert V5.judge(*_set(cluster=c))["headline"]["label"] == "cluster_alone_sufficient"


def test_headline_not_sufficient_when_mean_fails():
    c = {s: (0.125, 0.8125) for s in H5}                      # 精度降 0.0625
    assert V5.judge(*_set(cluster=c))["headline"]["label"] == "not_sufficient"
    c = {s: (0.5, 0.875) for s in H5}                         # 攻击成功率只降 0.375
    assert V5.judge(*_set(cluster=c))["headline"]["label"] == "not_sufficient"


def test_headline_needs_five_seeds():
    c = {s: (0.125, 0.875) for s in A3}
    off, both, at, cl = _set(cluster=c)
    res = V5.judge(off[:3], both, at, cl)
    assert res["headline"]["label"] == "insufficient" and res["attribution"]["label"] != "insufficient"


def test_attribution_needs_three_seeds_with_all_arms():
    off, both, at, cl = _set()
    res = V5.judge(off, both[:2], at, cl)
    assert res["attribution"]["label"] == "insufficient" and res["headline"]["label"] == "cluster_alone_sufficient"


def test_collapsed_at_arm_is_excluded_from_defense_attribution():
    # 对抗训练臂共享模型精度 0.125（< 0.2）：它的攻击成功率降幅不能算防御
    res = V5.judge(*_set(at_em={s: 0.125 for s in A3}))
    a = res["attribution"]
    assert a["at_collapsed_seeds"] == ["42", "43", "44"] and a["at_defense_confounded"] is True
    assert a["mean_delta_P"]["at"] is None and a["defense"] == "clustering"
    # 只有一个种子崩 → 用其余两个种子的均值
    res = V5.judge(*_set(at_em={42: 0.125, 43: 0.85, 44: 0.85}))
    assert res["attribution"]["at_collapsed_seeds"] == ["42"] and res["attribution"]["defense"] == "both_components"


def test_seed_dependent_attribution():
    at = {42: (0.125, 0.5), 43: (0.125, 0.5), 44: (0.125, 0.875)}   # 种子 44 上对抗训练不伤精度
    res = V5.judge(*_set(at=at))
    assert res["attribution"]["consistency"] == "seed_dependent"
    assert res["attribution"]["per_seed"]["44"]["cost"] != res["attribution"]["per_seed"]["42"]["cost"]


def test_pairs_by_seed_not_by_argument_order():
    off, both, at, cl = _set()
    assert V5.judge(list(reversed(off)), both, list(reversed(at)), list(reversed(cl)))["verdict"] == V5.judge(off, both, at, cl)["verdict"]


@pytest.mark.parametrize("mutate,needle", [
    (lambda o, b, a, c: c[3].update(exit_code=1), "种子 45"),
    (lambda o, b, a, c: c[4]["run"].update(ccs_at=True), "开关没生效"),
    (lambda o, b, a, c: a[1]["run"]["alignment"].update(fedrep_bn_stats="private"), "应为 'shared'"),
    (lambda o, b, a, c: o.pop(4), "没有不加防御的对照"),
    (lambda o, b, a, c: c.append(_run("cluster", 42, 0.125, 0.875)), "重复"),
    (lambda o, b, a, c: b[2]["run"].update(n_rounds=200), "设置不同"),
])
def test_gates(mutate, needle):
    off, both, at, cl = _set()
    mutate(off, both, at, cl)
    res = V5.judge(off, both, at, cl)
    assert res["verdict"] == "invalid" and any(needle in r for r in res["reasons"]), res


def test_off_must_plant_in_every_seed():
    res = V5.judge(*_set(off_P=0.375))
    assert res["verdict"] == "invalid_no_attack"


def test_ci95_uses_t_distribution():
    assert V5.ci95([0.5, 0.7]) == [round(0.6 - 12.706 * 0.1414213562 / 1.4142135624, 4),
                                   round(0.6 + 12.706 * 0.1414213562 / 1.4142135624, 4)]
    assert V5.ci95([0.5]) is None


_REAL = {"off": [RES / "CCSBOFF" / "CCSBOFF__fedrep-bnshared-off__s42.metrics.json",
                 RES / "BNS" / "BNS__fedrep-bnshared-off__s43.metrics.json",
                 RES / "BNS" / "BNS__fedrep-bnshared-off__s44.metrics.json"],
         "both": [RES / "CCSB" / "CCSB__fedrep-bnshared-ccs__s42.metrics.json"],
         "at": [RES / "CCSD" / "CCSD__fedrep-bnshared-at-only__s42.metrics.json"],
         "cluster": [RES / "CCSD" / "CCSD__fedrep-bnshared-cluster-only__s42.metrics.json"]}


@pytest.mark.skipif(not all(p.exists() for ps in _REAL.values() for p in ps), reason="已有结果文件不在")
def test_real_existing_runs_pass_the_gates_and_only_lack_seeds():
    """已有的种子 42 四臂 + BNS 的种子 43 / 44 对照：全部过闸，两部分都只缺种子；数与 F-117 一致；种子 42 上对抗训练臂判为崩坏。"""
    load = lambda ps: [json.loads(p.read_text(encoding="utf-8")) for p in ps]
    res = V5.judge(load(_REAL["off"]), load(_REAL["both"]), load(_REAL["at"]), load(_REAL["cluster"]))
    assert res["verdict"] == "headline=insufficient;attribution=insufficient", res
    assert res["headline"]["delta_P"]["42"] == 0.6143 and res["headline"]["delta_MTA"]["42"] == 0.0106
    assert res["runs"]["at"]["42"]["EM"] < V5.COLLAPSE_EM
