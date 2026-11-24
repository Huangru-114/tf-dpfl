"""
harness/medf_verdict.py 的守卫（公平的边缘内中位数，预注册 N-017 / D-142）。

- 合成 metrics 覆盖每个标签分支与每一条闸（参与率、轮数、B=1、布点、防御名 / 作用层、每次边缘聚合 5 个上传）；
- 阈值反向锚点：改常量就是改预注册；
- 窗口 = 第 21–30 云轮、必须齐全；
- 超几何的解析值（计划里的表）；真实数据锚点：ERP main-t1-distributed s42 的窗口值与逐边缘值。

纯标准库，本地秒级。
"""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import medf_verdict as M                                  # noqa: E402

REAL = ROOT / "experiments/defense/edge-native/results/P2/ERP/ERP__main-t1-distributed__s42.metrics.json"


def _run(*, p=0.95, acc=0.82, defense="none", layers=None, n_rounds=30, cf=0.2, k=1,
         mpe=(3, 3, 2, 2), totals=None, p_by_round=None, acc_by_round=None):
    rounds = range(1, n_rounds + 1)
    pr = p_by_round or {}
    ar = acc_by_round or {}
    m = {
        "exit_code": 0, "client_failures": [], "errors": [], "n_malicious_participations": 40,
        "rounds": [{"round": t, "local_benign_asr": pr.get(t, p)} for t in rounds],
        "acc_rounds": [{"round": t, "pm_acc": ar.get(t, acc)} for t in rounds],
        "per_edge_rounds": {},
        "run": {"n_rounds": n_rounds, "alignment": {"template": "p2"}, "client_fraction": cf,
                "edge_shared_blocks": k, "malicious_per_edge": list(mpe), "defense": defense},
        "admitted": [],
    }
    if layers is not None:
        m["run"]["defense_layers"] = list(layers)
        tot = totals if totals is not None else [5] * (n_rounds * 5 * 4)
        m["admitted"] = [{"layer": f"edge{i % 4}", "defense": defense, "admitted": None, "total": t, "rejected": None}
                         for i, t in enumerate(tot)]
    return m


def _pair(p_ctrl=0.95, p_arm=0.50, acc_ctrl=0.82, acc_arm=0.81, **arm_kw):
    return (_run(p=p_ctrl, acc=acc_ctrl),
            _run(p=p_arm, acc=acc_arm, defense="median", layers=["edge"], **arm_kw))


# ── 阈值反向锚点 ────────────────────────────────────────────────────────────
def test_thresholds_are_the_preregistered_values():
    assert (M.CTRL_MIN_P, M.USEFUL_DP, M.MTA_MAX, M.TOL) == (0.80, 0.30, 0.02, 1e-9)
    assert M.WINDOW == tuple(range(21, 31)) and M.N_ROUNDS == 30
    assert (M.CLIENT_FRACTION, M.UPLOADS_PER_AGG, M.DISTRIBUTED) == (0.2, 5, [3, 3, 2, 2])
    assert (M.CTRL, M.ARM) == ("MEDF__ctrl-t1-distributed__s42", "MEDF__edge-t1-distributed__s42")


# ── 标签分支 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("p_arm,acc_arm,label", [
    (0.50, 0.81, "useful"),           # ΔP 0.45、ΔMTA 0.01
    (0.65, 0.81, "useful"),           # ΔP = 0.30（容差内算「等于」）
    (0.80, 0.81, "weak"),             # ΔP 0.15
    (0.95, 0.81, "no_effect"),        # ΔP 0
    (0.99, 0.81, "no_effect"),        # ΔP < 0
    (0.50, 0.80, "useful"),           # ΔMTA = 0.02（容差内算「等于」→ 不超门槛）→ 走 ΔP
])
def test_labels(p_arm, acc_arm, label):
    c, a = _pair(p_arm=p_arm, acc_arm=acc_arm)
    assert M.judge(c, a)["verdict"] == label


def test_accuracy_bound_wins_over_a_large_asr_drop():
    c, a = _pair(p_arm=0.10, acc_arm=0.70)                              # ΔP 0.85、ΔMTA 0.12
    res = M.judge(c, a)
    assert res["verdict"] == "accuracy_bound"
    assert res["delta_P"] == pytest.approx(0.85) and res["delta_MTA"] == pytest.approx(0.12)


def test_classify_tolerance_edges():
    assert M.classify(0.30 - 5e-10, 0.0) == "useful"
    assert M.classify(0.30 - 1e-6, 0.0) == "weak"
    assert M.classify(0.5, 0.02 + 5e-10) == "useful"
    assert M.classify(0.5, 0.02 + 1e-6) == "accuracy_bound"
    assert M.classify(1e-12, 0.0) == "no_effect"


def test_control_without_backdoor_is_invalid_no_attack():
    c, a = _pair(p_ctrl=0.79, p_arm=0.20)
    res = M.judge(c, a)
    assert res["verdict"] == "invalid_no_attack" and "delta_P" not in res


def test_window_is_rounds_21_to_30_and_must_be_complete():
    # 窗口外的值不影响结果
    c = _run(p=0.95, p_by_round={t: 0.0 for t in range(1, 21)})
    a = _run(p=0.50, defense="median", layers=["edge"], p_by_round={t: 1.0 for t in range(1, 21)})
    res = M.judge(c, a)
    assert res["quantities"]["ctrl"]["P"] == pytest.approx(0.95)
    assert res["quantities"]["median"]["P"] == pytest.approx(0.50)
    # 窗口内缺一个点 → invalid（不是用剩下 9 个点凑）
    c2 = _run(p=0.95, p_by_round={25: None})
    res2 = M.judge(c2, a)
    assert res2["verdict"] == "invalid" and "window" in res2["reasons"]


def test_missing_files_are_insufficient():
    c, _ = _pair()
    assert M.judge(c, None)["verdict"] == "insufficient"
    assert M.judge(None, None)["missing"] == [M.CTRL, M.ARM]


# ── 闸 ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kw,needle", [
    ({"cf": 0.1}, "client_fraction"),
    ({"n_rounds": 60}, "跑满"),
    ({"k": 0}, "edge_shared_blocks"),
    ({"mpe": (10, 0, 0, 0)}, "malicious_per_edge"),
    ({"totals": [5] * 599 + [3]}, "上传数"),
    ({"totals": []}, "没有边缘聚合"),
])
def test_gate_on_the_median_arm(kw, needle):
    a = _run(defense="median", layers=["edge"], **kw)
    reasons = M.gate(a, defense="median", layers=["edge"])
    assert any(needle in r for r in reasons), reasons


def test_gate_checks_defense_name_and_layers_and_crashes():
    c = _run()
    assert M.gate(c, defense="none") == []
    assert any("defense" in r for r in M.gate(c, defense="median"))
    a = _run(defense="median", layers=["cloud"])
    assert any("defense_layers" in r for r in M.gate(a, defense="median", layers=["edge"]))
    c["client_failures"] = [{"client_id": 7}]
    c["exit_code"] = 1
    assert len(M.gate(c, defense="none")) >= 2
    c2 = _run()
    c2["n_malicious_participations"] = 0
    assert any("攻击者" in r for r in M.gate(c2, defense="none"))
    c3 = _run()
    c3["run"]["alignment"]["template"] = "legacy"
    assert any("P2" in r for r in M.gate(c3, defense="none"))


def test_invalid_run_blocks_the_verdict():
    c, a = _pair()
    a["run"]["client_fraction"] = 0.1
    res = M.judge(c, a)
    assert res["verdict"] == "invalid" and M.ARM in res["reasons"]


# ── 只报告的量 ──────────────────────────────────────────────────────────────
def test_majority_probability_matches_the_plan_table():
    assert M.p_majority(25, 3, 5) == pytest.approx(0.004347826, abs=1e-9)    # 计划表：分散 3/25、m=5 → 0.004
    assert M.p_majority(25, 2, 5) == pytest.approx(0.0, abs=1e-12)
    assert M.p_majority(25, 10, 3) == pytest.approx(0.346, abs=5e-4)          # 集中 E0
    assert M.p_majority(25, 10, 10) == pytest.approx(0.337, abs=5e-4)
    assert M.p_majority(25, 3, 3) == pytest.approx(0.029, abs=5e-4)
    # 定义自洽：概率和 = 1
    tot = math.fsum(math.comb(10, k) * math.comb(15, 5 - k) for k in range(0, 6)) / math.comb(25, 5)
    assert tot == pytest.approx(1.0)


@pytest.mark.skipif(not REAL.exists(), reason="ERP 结果不在")
def test_real_data_anchor_erp_main_t1_distributed_s42():
    """本会话算过的读数（参与率 0.1、60 云轮的 run 的第 21–30 云轮），用来钉住窗口与逐边缘的取法。"""
    m = json.loads(REAL.read_text(encoding="utf-8"))
    assert M.pooled(m) == pytest.approx(0.8429, abs=5e-5)
    assert M.mta(m) == pytest.approx(0.81347, abs=5e-6)
    assert M.per_edge(m) == {0: 0.9005, 1: 0.9534, 2: 0.5761, 3: 0.9489}
    # 这个 run 本身不是 MEDF：闸必须拦下它（参与率、轮数、防御名）
    reasons = M.gate(m, defense="none")
    assert any("client_fraction" in r for r in reasons) and any("跑满" in r for r in reasons)


def test_cli_writes_insufficient_without_data(tmp_path):
    out = tmp_path / "v.json"
    assert M.main(["--results", str(tmp_path), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "insufficient"
