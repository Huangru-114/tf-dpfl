"""
阶段三 CCSF 判定（`harness/ccsf_verdict.py`，规则 = FINDINGS N-008 CCSF 段；实现 D-109）。合成数据覆盖每个分支。纯 python。
"""

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import ccsf_verdict as V                                      # noqa: E402

N = 30
MAL = list(range(10))


def _run(P, mta, ccs=False, P_unf=None):
    m = {
        "exit_code": 0, "client_failures": [], "n_malicious_participations": 25,
        "run": {"n_rounds": N, "edge_rounds": 1, "n_edges": 1, "seed": 42, "partition": "noniid",
                "malicious_per_edge": [10], "malicious_ids": MAL, "malicious_data_share": 0.0912,
                "ccs": ccs, "ccs_at": True if ccs else None, "ccs_cluster": True if ccs else None},
        "rounds": [{"round": r, "local_benign_asr": P, "margin_p50": 1.0,
                    "local_benign_asr_unfiltered": P if P_unf is None else P_unf} for r in range(1, N + 1)],
        "acc_rounds": [{"round": r, "pm_acc": mta} for r in range(1, N + 1)],
        "ccs_decisions": [],
    }
    if ccs:
        # 每轮 10 个上传：恶意 0、1 + 良性 50…57；剔除恶意 0 与良性 50 → TPR 0.5、FPR 1/8
        m["ccs_decisions"] = [{"edge": 0, "cur": [0, 1] + list(range(50, 58)), "n_pool": 10,
                               "status": "ok", "rejected": [0, 50]} for _ in range(N)]
    return m


@pytest.mark.parametrize("p_ccs,mta_ccs,want", [
    (0.20, 0.80, "ccs_reproduces"),        # ΔP 0.7、ΔMTA 0
    (0.20, 0.77, "partial_repro"),         # ΔP 够但 ΔMTA 0.03 > 0.02
    (0.85, 0.80, "ccs_not_reproduced"),    # ΔP 0.05 < 0.2
    (0.55, 0.80, "partial_repro"),         # ΔP 0.35
])
def test_labels(p_ccs, mta_ccs, want):
    res = V.judge(_run(0.9, 0.80), _run(p_ccs, mta_ccs, ccs=True))
    assert res["verdict"] == want, res


def test_threshold_edges_are_inclusive_as_written():
    # 用二进制下精确的数，免得边界被浮点误差决定：ΔP = 0.75 − 0.25 = 0.5（≥，含）
    assert V.judge(_run(0.75, 0.75), _run(0.25, 0.75, ccs=True))["verdict"] == "ccs_reproduces"
    assert V.judge(_run(0.75, 0.75), _run(0.25, 0.71875, ccs=True))["verdict"] == "partial_repro"  # ΔMTA 0.03125


def test_off_arm_must_plant_the_attack():
    res = V.judge(_run(0.45, 0.80), _run(0.10, 0.80, ccs=True))
    assert res["verdict"] == "invalid_no_attack"


def test_rejection_rates_and_side_reports():
    res = V.judge(_run(0.9, 0.80, P_unf=0.91), _run(0.2, 0.80, ccs=True, P_unf=0.28))
    rj = res["rejection"]
    assert rj["tpr"] == 0.5 and rj["fpr"] == 0.125 and rj["status"] == {"ok": N}
    assert res["delta"]["dP_unfiltered"] == pytest.approx(0.63)
    assert res["malicious_data_share"]["ccs"] == 0.0912


@pytest.mark.parametrize("mutate,needle", [
    (lambda o, c: c.update(exit_code=1), "exit_code"),
    (lambda o, c: c.update(client_failures=[{"client_id": 3}]), "client_failures"),
    (lambda o, c: o.update(n_malicious_participations=0), "攻击者"),
    (lambda o, c: c["run"].update(ccs=False), "开关没生效"),
    (lambda o, c: c["run"].update(ccs_cluster=False), "开关没生效"),
    (lambda o, c: o["run"].update(ccs=True), "对照不该开"),
    (lambda o, c: c["run"].update(seed=43), "run.seed"),
    (lambda o, c: c.update(ccs_decisions=c["ccs_decisions"][:-1]), "[CCS] 行数"),
])
def test_gates(mutate, needle):
    off, ccs = _run(0.9, 0.80), _run(0.2, 0.80, ccs=True)
    mutate(off, ccs)
    res = V.judge(off, ccs)
    assert res["verdict"] == "invalid" and any(needle in r for r in res["reasons"]), res


def test_missing_points_are_insufficient():
    ccs = _run(0.2, 0.80, ccs=True)
    for r in ccs["acc_rounds"]:
        r["pm_acc"] = None
    assert V.judge(_run(0.9, 0.80), ccs)["verdict"] == "insufficient"


def test_cli_writes_json(tmp_path):
    import json
    o, c = tmp_path / "off.json", tmp_path / "ccs.json"
    o.write_text(json.dumps(_run(0.9, 0.80)))
    c.write_text(json.dumps(_run(0.2, 0.80, ccs=True)))
    out = tmp_path / "v.json"
    assert V.main(["--off", str(o), "--ccs", str(c), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "ccs_reproduces"
