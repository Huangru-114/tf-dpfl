"""
阶段三 缓存池判定（`harness/ccsdp_verdict.py`，规则 = FINDINGS N-012；D-126）。合成数据覆盖每个分支；
真实数据钉「只开聚类（pool = round）的 run 冒充缓存池会被闸拦下」。纯 python。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import ccsdp_verdict as V                                     # noqa: E402

N = 30
MAL = list(range(10))
RES = ROOT / "experiments" / "defense" / "edge-native" / "results" / "P2"
SEEDS = (43, 46)


def _run(arm, seed, P, mta):
    m = {
        "exit_code": 0, "client_failures": [], "n_malicious_participations": 25,
        "run": {"n_rounds": N, "edge_rounds": 1, "n_edges": 1, "seed": seed, "partition": None,
                "malicious_per_edge": [10], "malicious_ids": MAL, "method": "hier_fedrep",
                "alignment": {"fedrep_bn_stats": "shared"}, "ccs": arm != "off",
                "ccs_at": None if arm == "off" else False, "ccs_cluster": None if arm == "off" else True,
                "ccs_pool": {"off": None, "cluster": "round", "cache": "edge_cache"}[arm]},
        "rounds": [{"round": r, "local_benign_asr": P, "local_benign_asr_unfiltered": P} for r in range(1, N + 1)],
        "acc_rounds": [{"round": r, "pm_acc": mta, "em_acc": 0.85} for r in range(1, N + 1)],
        "ccs_decisions": [],
    }
    if arm != "off":
        m["ccs_decisions"] = [{"edge": 0, "cur": [0, 1] + list(range(50, 58)), "n_pool": 10,
                               "status": "ok", "rejected": [0, 50]} for _ in range(N)]
    return m


def _arms(cache, off_P=0.875):
    """cache：{seed: (P, MTA)}；off P 0.875 / MTA 0.875；只开聚类 P 0.625 / MTA 0.875（失守）。"""
    return ([_run("off", s, off_P, 0.875) for s in SEEDS],
            [_run("cluster", s, 0.625, 0.875) for s in SEEDS],
            [_run("cache", s, *cache[s]) for s in SEEDS])


@pytest.mark.parametrize("cache,want", [
    ({43: (0.125, 0.875), 46: (0.125, 0.875)}, "fixes_failures"),
    ({43: (0.125, 0.875), 46: (0.625, 0.875)}, "partial"),
    ({43: (0.125, 0.8125), 46: (0.125, 0.875)}, "partial"),     # 种子 43 精度降 0.0625 > 0.02
    ({43: (0.5, 0.875), 46: (0.625, 0.875)}, "no_fix"),
])
def test_labels(cache, want):
    assert V.judge(*_arms(cache))["verdict"] == want


def test_thresholds_inclusive():
    # ΔP 恰好 0.5（0.875 − 0.375）、ΔMTA 0 → 满足
    assert V.judge(*_arms({43: (0.375, 0.875), 46: (0.375, 0.875)}))["verdict"] == "fixes_failures"


def test_side_reports():
    res = V.judge(*_arms({43: (0.125, 0.875), 46: (0.125, 0.875)}))
    assert res["delta"]["cluster"]["43"] == {"dP": 0.25, "dMTA": 0.0}
    r = res["runs"]["cache"]["43"]
    assert r["rejection"]["tpr"] == 0.5 and r["rejection"]["fpr"] == 0.125
    assert (r["attacker_admitted"], r["attacker_selected"]) == (30, 60)


def test_off_must_plant():
    assert V.judge(*_arms({43: (0.125, 0.875), 46: (0.125, 0.875)}, off_P=0.375))["verdict"] == "invalid_no_attack"


@pytest.mark.parametrize("mutate,needle", [
    (lambda o, c, k: k[0]["run"].update(ccs_pool="round"), "开关没生效"),
    (lambda o, c, k: c[1]["run"].update(ccs_pool="edge_cache"), "开关没生效"),
    (lambda o, c, k: k[1]["run"].update(ccs_at=True), "开关没生效"),
    (lambda o, c, k: o[0]["run"].update(ccs=True), "对照不该开"),
    (lambda o, c, k: k[0].update(exit_code=1), "exit_code"),
    (lambda o, c, k: k[0].pop("exit_code"), "exit_code 缺失"),
    (lambda o, c, k: k[1].update(ccs_decisions=[]), "[CCS] 行数"),
    (lambda o, c, k: k[1]["run"]["alignment"].update(fedrep_bn_stats="private"), "不是 shared"),
    (lambda o, c, k: k[1]["run"].update(seed=44), "种子集合不同"),
    (lambda o, c, k: c[0]["run"].update(n_rounds=200), "设置不同"),
    (lambda o, c, k: k.append(_run("cache", 43, 0.125, 0.875)), "重复"),
])
def test_gates(mutate, needle):
    off, cl, ca = _arms({43: (0.125, 0.875), 46: (0.125, 0.875)})
    mutate(off, cl, ca)
    res = V.judge(off, cl, ca)
    assert res["verdict"] == "invalid" and any(needle in r for r in res["reasons"]), res


def test_needs_two_seeds():
    off, cl, ca = _arms({43: (0.125, 0.875), 46: (0.125, 0.875)})
    res = V.judge(off[:1], cl[:1], ca[:1])
    assert res["verdict"] == "invalid" and any("至少要 2" in r for r in res["reasons"])


_OFF = [RES / "BNS" / "BNS__fedrep-bnshared-off__s43.metrics.json", RES / "CCSOFF5" / "CCSOFF5__fedrep-bnshared-off__s46.metrics.json"]
_CL = [RES / "CCSDC5" / f"CCSDC5__fedrep-bnshared-cluster-only__s{s}.metrics.json" for s in SEEDS]


@pytest.mark.skipif(not all(p.exists() for p in _OFF + _CL), reason="结果文件不在")
def test_real_round_pool_runs_cannot_pose_as_cache():
    """反向锚点：真实的只开聚类（pool = round）冒充缓存池那一臂 → 闸按 ccs_pool 拦下；对照与只开聚类本身过闸、数与 F-119 一致。"""
    load = lambda ps: [json.loads(p.read_text(encoding="utf-8")) for p in ps]
    off, cl = load(_OFF), load(_CL)
    res = V.judge(off, cl, cl)
    assert res["verdict"] == "invalid"
    assert sum("开关没生效" in r for r in res["reasons"]) == 2 and all(r.startswith("cache/") for r in res["reasons"])
    assert res["runs"]["cluster"]["43"]["P"] == 0.6846 and res["runs"]["cluster"]["46"]["P"] == 0.606
