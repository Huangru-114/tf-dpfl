"""
tests/test_instrumentation_check.py  —  S9：「新仪表没有改变任何已有的数」的验收工具

  · 合成：计时字段、schema 7 新字段不比；checksum 或已有评估数值变了 → 失败；只比 ≤ R 轮；
  · 真实数据的正向锚点：F-045 的 DET 两次 run、F-050 的 G7 std s42 vs pilot D029 2edge s42；
  · 反向锚点：G6 a 与 b（第 1 轮起训练就不同）必须判不一致。
"""

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import instrumentation_check as IC                              # noqa: E402

MECH = ROOT / "experiments/attack/hfl-mechanism"


def _m(n=3):
    return {
        "checksums": [{"round": r, "global": f"abc{r}"} for r in range(1, n + 1)],
        "rounds": [{"round": r, "global_asr": 0.1 * r, "local_benign_asr": 0.2 * r}
                   for r in range(1, n + 1)],
        "acc_rounds": [{"round": r, "pm_acc": 0.5, "round_time": 10.0 + r, "acc_pm_s": 1.0}
                       for r in range(1, n + 1)],
        "per_edge_rounds": {str(r): [{"edge_id": 0, "client_benign": 0.3}]
                            for r in range(1, n + 1)},
        "per_edge_acc_rounds": {},
    }


def test_identical_and_extra_fields_and_timing_are_fine():
    ref, new = _m(), _m()
    for row in new["rounds"]:
        row["margin_p50"] = 1.23                             # schema 7 新字段
    for row in new["acc_rounds"]:
        row["round_time"] += 5.0                             # 墙钟
    new["dumps"] = {"logits": {"count": 3}}
    res = IC.compare(ref, new)
    assert res["diffs"] == [] and res["missing"] == [] and res["n_checksums"] == 3
    assert IC.main([_w(ref, "a"), _w(new, "b")]) == 0


def test_checksum_or_eval_difference_fails():
    ref, new = _m(), _m()
    new["checksums"][1]["global"] = "zzz"
    assert IC.compare(ref, new)["diffs"][0][0] == "checksums[round=2]"
    new = _m()
    new["rounds"][0]["local_benign_asr"] = 0.99
    assert [k for k, _, _ in IC.compare(ref, new)["diffs"]] == ["rounds[round=1].local_benign_asr"]
    new = _m()
    new["per_edge_rounds"]["2"][0]["client_benign"] = 0.0
    assert IC.compare(ref, new)["diffs"]
    assert IC.main([_w(ref, "a"), _w(new, "b")]) == 1


def test_upto_limits_the_comparison():
    ref, new = _m(5), _m(5)
    new["checksums"][4]["global"] = "later"                  # 第 5 轮不同（例如停止投毒之后）
    new["rounds"][4]["global_asr"] = 0.0
    assert IC.compare(ref, new, upto=4)["diffs"] == []
    assert IC.compare(ref, new, upto=5)["diffs"]


def test_missing_rows_in_new_are_reported():
    ref, new = _m(), _m(2)
    res = IC.compare(ref, new, upto=3)
    assert "checksums[round=3]" in res["missing"]


# ── 真实数据的锚点 ─────────────────────────────────────────────────────────────
def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def test_real_det_pair_is_identical():
    """F-045：DET rep1 / rep2 在不同节点上前 5 轮逐位相同。"""
    d = MECH / "pilot/results/P1/DET"
    res = IC.compare(_load(d / "DET__rep1__s42.metrics.json"),
                     _load(d / "DET__rep2__s42.metrics.json"), upto=5)
    assert res["n_checksums"] == 5 and res["diffs"] == [] and res["missing"] == []


def test_real_cross_commit_pair_is_identical():
    """F-050：G7 std s42（726d9d5）与 pilot D029 2edge s42（299afe6）全部指标逐位相同。"""
    res = IC.compare(_load(MECH / "pilot/results/P1/D029/D029__2edge_distributed__s42.metrics.json"),
                     _load(MECH / "results/P2/G7/G7__std__s42.metrics.json"))
    assert res["n_checksums"] > 30 and res["diffs"] == []


def test_real_reverse_anchor_g6_a_vs_b_differs():
    res = IC.compare(_load(MECH / "results/P2/G6/G6__a__s42.metrics.json"),
                     _load(MECH / "results/P2/G6/G6__b__s42.metrics.json"), upto=30)
    assert any(k.startswith("checksums") for k, _, _ in res["diffs"])


def _w(m, name, _tmp={}):
    import tempfile
    d = _tmp.setdefault("d", Path(tempfile.mkdtemp()))
    p = d / f"{name}.json"
    p.write_text(json.dumps(copy.deepcopy(m)))
    return str(p)
