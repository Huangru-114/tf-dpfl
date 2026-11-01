"""
阶段三 CCSD 判定（`harness/ccsd_verdict.py`，规则 = FINDINGS N-010 草案；D-121）。合成数据覆盖每个分支；
真实数据钉「把已有 run 放错位置会被闸拦下」。纯 python。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import ccsd_verdict as V                                      # noqa: E402

N = 30
MAL = list(range(10))
RES = ROOT / "experiments" / "defense" / "edge-native" / "results" / "P2"
OFF42 = RES / "CCSBOFF" / "CCSBOFF__fedrep-bnshared-off__s42.metrics.json"
BOTH42 = RES / "CCSB" / "CCSB__fedrep-bnshared-ccs__s42.metrics.json"


def _run(P, mta, arm):
    at, cl = {"off": (None, None), "both": (True, True), "at": (True, False), "cluster": (False, True)}[arm]
    m = {
        "exit_code": 0, "client_failures": [], "n_malicious_participations": 25,
        "run": {"n_rounds": N, "edge_rounds": 1, "n_edges": 1, "seed": 42, "partition": None,
                "malicious_per_edge": [10], "malicious_ids": MAL, "method": "hier_fedrep",
                "alignment": {"fedrep_bn_stats": "shared"},
                "ccs": arm != "off", "ccs_at": at, "ccs_cluster": cl},
        "rounds": [{"round": r, "local_benign_asr": P, "local_benign_asr_unfiltered": P} for r in range(1, N + 1)],
        "acc_rounds": [{"round": r, "pm_acc": mta} for r in range(1, N + 1)],
        "ccs_decisions": [],
    }
    if cl:
        m["ccs_decisions"] = [{"edge": 0, "cur": [0, 1] + list(range(50, 58)), "n_pool": 10,
                               "status": "ok", "rejected": [0, 50]} for _ in range(N)]
    return m


def _judge(at, cl, both=(0.125, 0.75), off=(0.875, 0.875)):
    """各臂传 (P, MTA)。缺省：off P 0.875 / MTA 0.875；both P 0.125 / MTA 0.75 → ΔP 0.75、ΔMTA 0.125；2/3 线 = 0.5 / 0.0833。"""
    return V.judge(_run(*off, "off"), _run(*both, "both"), _run(*at, "at"), _run(*cl, "cluster"))


# 用二进制下精确的数：ΔMTA = 0.875 − MTA
@pytest.mark.parametrize("mta_at,mta_cl,want", [
    (0.75, 0.75, "both_components"),           # 两臂都 0.125
    (0.75, 0.875, "adversarial_training"),     # 只有对抗训练那臂掉精度
    (0.875, 0.75, "clustering"),               # 只有聚类那臂
    (0.84375, 0.84375, "only_together"),       # 两臂都只掉 0.03125 < 0.0833
])
def test_cost_attribution(mta_at, mta_cl, want):
    res = _judge((0.125, mta_at), (0.125, mta_cl))
    assert res["cost"] == want, res


@pytest.mark.parametrize("p_at,p_cl,want", [
    (0.125, 0.125, "both_components"),
    (0.125, 0.875, "adversarial_training"),
    (0.875, 0.125, "clustering"),
    (0.625, 0.625, "only_together"),           # 两臂 ΔP 0.25 < 0.5
])
def test_defense_attribution(p_at, p_cl, want):
    res = _judge((p_at, 0.875), (p_cl, 0.875))
    assert res["defense"] == want, res
    assert res["verdict"] == f"cost={res['cost']};defense={want}"


def test_frac_threshold_is_inclusive():
    # both ΔP 0.75 → 线 0.5；at 臂 ΔP 恰好 0.5（含）→ 计入；0.46875 → 不计入
    assert _judge((0.375, 0.875), (0.875, 0.875))["defense"] == "adversarial_training"
    assert _judge((0.40625, 0.875), (0.875, 0.875))["defense"] == "only_together"


def test_nothing_to_attribute():
    res = _judge((0.125, 0.875), (0.125, 0.875), both=(0.125, 0.875))     # both 没掉精度
    assert res["cost"] == "no_cost"
    res = _judge((0.875, 0.875), (0.875, 0.875), both=(0.75, 0.75))        # both ΔP 0.125 < 0.2
    assert res["defense"] == "no_effect"


def test_standalone_labels():
    res = _judge((0.125, 0.875), (0.75, 0.75))
    assert res["standalone"] == {"both": "partial", "at": "meets_all", "cluster": "no_defense"}


def test_off_arm_must_plant():
    res = _judge((0.125, 0.75), (0.125, 0.75), off=(0.375, 0.875))
    assert res["verdict"] == "invalid_no_attack"


def test_side_reports():
    res = _judge((0.125, 0.75), (0.125, 0.75))
    assert res["rejection"]["cluster"]["tpr"] == 0.5 and res["rejection"]["cluster"]["fpr"] == 0.125
    assert res["runs"]["at"]["MTA_sd_last10"] == 0.0


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r["at"].update(exit_code=1), "exit_code"),
    (lambda r: r["cluster"].pop("exit_code"), "exit_code 缺失"),
    (lambda r: r["both"].update(client_failures=[{"client_id": 3}]), "client_failures"),
    (lambda r: r["off"].update(n_malicious_participations=0), "攻击者"),
    (lambda r: r["off"]["run"].update(ccs=True), "对照不该开"),
    (lambda r: r["at"]["run"].update(ccs_cluster=True), "开关没生效"),
    (lambda r: r["cluster"]["run"].update(ccs_at=True), "开关没生效"),
    (lambda r: r["both"]["run"].update(ccs_at=False), "开关没生效"),
    (lambda r: r["cluster"]["run"]["alignment"].update(fedrep_bn_stats="private"), "应为 'shared'"),
    (lambda r: r["cluster"].update(ccs_decisions=[]), "[CCS] 行数"),
    (lambda r: r["at"]["run"].update(seed=43), "设置不同"),
])
def test_gates(mutate, needle):
    runs = {"off": _run(0.875, 0.875, "off"), "both": _run(0.125, 0.75, "both"),
            "at": _run(0.125, 0.75, "at"), "cluster": _run(0.125, 0.75, "cluster")}
    mutate(runs)
    res = V.judge(runs["off"], runs["both"], runs["at"], runs["cluster"])
    assert res["verdict"] == "invalid" and any(needle in r for r in res["reasons"]), res


def test_at_only_arm_needs_no_ccs_lines():
    """只开对抗训练时服务端不聚类、不打 [CCS] 行（defense/ccs.py：cluster 关 → 只在有非有限上传时才打）→ 不能要求行数。"""
    runs = [_run(0.875, 0.875, "off"), _run(0.125, 0.75, "both"), _run(0.125, 0.75, "at"), _run(0.125, 0.75, "cluster")]
    assert runs[2]["ccs_decisions"] == []
    assert V.judge(*runs)["verdict"] != "invalid"


@pytest.mark.skipif(not (OFF42.exists() and BOTH42.exists()), reason="CCSBOFF / CCSB 的结果文件不在")
def test_real_runs_in_the_wrong_roles_are_rejected():
    """反向锚点：真实的 CCSB（两个组件都开）冒充单组件臂、CCSBOFF 冒充开防御的臂 —— 闸必须拦下。"""
    off = json.loads(OFF42.read_text(encoding="utf-8"))
    both = json.loads(BOTH42.read_text(encoding="utf-8"))
    res = V.judge(off, both, both, both)
    assert res["verdict"] == "invalid" and sum("开关没生效" in r for r in res["reasons"]) == 2
    res = V.judge(off, both, off, off)
    assert res["verdict"] == "invalid" and any("[CCS] 行数 0" in r for r in res["reasons"])
    assert res["runs"]["off"]["P"] == 0.7958 and res["runs"]["both"]["MTA"] == 0.7829     # F-115 的数
