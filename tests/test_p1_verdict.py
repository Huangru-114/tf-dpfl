"""
阶段三 P1 判定（`harness/p1_verdict.py`，规则 = FINDINGS N-008 P1 段；闸 D-108）。
对照 = 真实的 SNAP__collocated__s42 metrics.json；防御臂 = 它的拷贝，按用例改动（加固记录、受害 ASR、精度、jump）。纯 python。
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import p1_verdict as V                                       # noqa: E402

CTRL = ROOT / "experiments/defense/edge-native/results/P2/SNAP/SNAP__collocated__s42.metrics.json"
VICT = (1, 2, 3)


@pytest.fixture(scope="module")
def ctrl():
    if not CTRL.exists():
        pytest.skip("SNAP 的 metrics.json 不在本地")
    return json.loads(CTRL.read_text())


def _arm(ctrl, dv=0.0, dmta=0.0, post_shift=0.0, cfg=V.ARM_CFG):
    a = copy.deepcopy(ctrl)
    n = int(a["run"]["n_rounds"])
    a["run"]["edge_harden"] = cfg
    a["harden"] = [{"round": r, "edge": e, "diverged_at": None, "seconds": 1.0} for r in range(1, n + 1) for e in range(4)]
    ck1 = [c["global"] for c in ctrl["checksums"] if c["round"] == 1][0]
    a["harden_agg"] = [{"round": 1, "unhardened": ck1}]
    for rows in a["per_edge_rounds"].values():
        for d in rows:
            if d["edge_id"] in VICT and d.get("client_benign") is not None:
                d["client_benign"] -= dv
    for r in a["acc_rounds"]:
        if r.get("pm_acc") is not None:
            r["pm_acc"] -= dmta
    if post_shift:
        cols = list(a["per_edge_light_columns"])
        ei, ci = cols.index("edge_id"), cols.index("client_benign")
        for rows in a["per_edge_post_agg_rounds"].values():
            for row in rows:
                if row[ei] in VICT and row[ci] is not None:
                    row[ci] -= post_shift
    return a


def test_identical_run_is_stop(ctrl):
    r = V.judge(_arm(ctrl), ctrl)
    assert r["verdict"] == "stop_col" and r["delta"]["dV"] == 0 and r["delta"]["jump_ratio"] == 1.0


def test_go_by_asr_needs_accuracy(ctrl):
    """对照 V = 0.995；臂 V 降 0.2 → 0.795 < 0.95 → 天花板规则不适用，按 ΔV ≥ 0.15 判 go；精度掉 0.03 → 不是 go。"""
    r = V.judge(_arm(ctrl, dv=0.2), ctrl)
    assert r["ceiling_rule"] is False and r["verdict"] == "go_main_col"
    r = V.judge(_arm(ctrl, dv=0.2, dmta=0.03), ctrl)
    assert r["verdict"] == "marginal" and r["mta_ok"] is False


def test_ceiling_rule_ignores_small_asr_drops(ctrl):
    """两边 V 都 ≥ 0.95 → ΔV 那一项换成 Δmargin_v ≥ 2：ΔV 0.03 + margin 不变 → 不是 go。"""
    r = V.judge(_arm(ctrl, dv=0.03), ctrl)
    assert r["ceiling_rule"] is True and r["verdict"] != "go_main_col"


def test_go_by_jump_halving(ctrl):
    q = V.quantities(ctrl)
    r = V.judge(_arm(ctrl, post_shift=0.6 * q["jump_half"]), ctrl)     # 每次云聚合后点压低 0.6 × jump → jump_half 降 60%
    assert r["delta"]["jump_ratio"] == pytest.approx(0.4, abs=1e-6) and r["verdict"] == "go_main_col"
    r = V.judge(_arm(ctrl, post_shift=0.6 * q["jump_half"], dmta=0.021), ctrl)
    assert r["verdict"] == "marginal"                              # 精度过不了 → 不是 go，也不满足 stop


def test_ceiling_rule_uses_margin(ctrl):
    a = _arm(ctrl)
    cols = list(a["per_edge_detail_columns"])
    ei, mi = cols.index("edge_id"), cols.index("margin_p50")
    for rows in a["per_edge_detail_rounds"].values():
        for row in rows:
            if row[ei] in VICT and row[mi] is not None:
                row[mi] -= 2.5
    r = V.judge(a, ctrl)
    assert r["ceiling_rule"] is True and r["delta"]["dmargin_v"] == pytest.approx(2.5) and r["verdict"] == "go_main_col"


@pytest.mark.parametrize("breaker,needle", [
    (lambda a: a["harden"].pop(), "[Harden] 行数"),
    (lambda a: a["harden_agg"].__setitem__(0, {"round": 1, "unhardened": "0" * 12}), "第一次加固之前"),
    (lambda a: a["run"].__setitem__("edge_harden", "none"), "run.edge_harden"),
    (lambda a: a.__setitem__("client_failures", [{"client_id": 3}]), "client_failures"),
    (lambda a: a["run"].__setitem__("seed", 43), "run.seed"),
])
def test_gates(ctrl, breaker, needle):
    a = _arm(ctrl)
    breaker(a)
    r = V.judge(a, ctrl)
    assert r["verdict"] == "invalid" and any(needle in x for x in r["reasons"])


def test_cli(ctrl, tmp_path):
    p = tmp_path / "arm.json"
    p.write_text(json.dumps(_arm(ctrl)))
    assert V.main(["--arm", str(p), "--control", str(CTRL), "--out", str(tmp_path / "v.json")]) == 0
    assert json.loads((tmp_path / "v.json").read_text())["verdict"] == "stop_col"
