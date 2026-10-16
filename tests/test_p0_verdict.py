"""
阶段三 P0 的选择与判定（`harness/p0_verdict.py`，规则 = FINDINGS N-008 P0 段，D-101）。
合成的 `analysis/p0_snapshot.py` 输出覆盖每个分支；阈值的反向锚点。纯 python + numpy。
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import p0_verdict as V                                       # noqa: E402

pytest.importorskip("yaml")
GSPEC, GRID = V.load_grid()
AT = [c["id"] for c in GRID if c["objective"] != "cft"]
TOP = ["pgd-e8-frozen-low", "trades-e8-b6-frozen-low", "tgt-e4-frozen-low"]


def _ev(v, pm=0.86, pooled=None):
    return {"victim_asr": v, "pooled_benign": pooled if pooled is not None else v, "pm_acc": pm,
            "per_edge": {e: {"pm_acc": pm} for e in range(4)}}


def _snap(J, rh_by_cfg, base_v=0.5, d_pm=None, v0=True):
    """rh_by_cfg：{cfg: R_H}；受害 ASR = base_v − R_H·J（R_H 由此精确反推）。d_pm：{cfg: 精度损失}。"""
    d_pm = d_pm or {}
    return {"J": J, "lr": 0.07,
            "v0": {"pass": v0, "reasons": [] if v0 else ["x"]},
            "base": _ev(base_v),
            "configs": {c: {"eval": _ev(base_v - r * J, pm=0.86 - d_pm.get(c, 0.0))}
                        for c, r in rh_by_cfg.items()},
            "aux": {}}


def _res(seed, stage, rh, placement="collocated", d_pm=None, J=(0.25, 0.30), v0=True):
    cfgs = list(rh)
    return {"schema": 1, "run_id": f"SNAP__{placement}__s{seed}", "seed": seed, "placement": placement,
            "stage": stage, "configs": cfgs, "grid_sha": "x",
            "snapshots": {"6": _snap(J[0], rh, d_pm=d_pm, v0=v0), "15": _snap(J[1], rh, d_pm=d_pm, v0=v0)}}


def _screen(rh_default=0.1, override=None, d_pm=None, **kw):
    rh = {c: rh_default for c in [c["id"] for c in GRID]}
    rh.update(override or {})
    return _res(42, "screen", rh, d_pm=d_pm, **kw)


def _frozen(seed, rh, **kw):
    by = {c["id"]: c for c in GRID}
    cft = {V.spec.matched_cft(by[c], GRID) for c in rh}
    full = dict(rh)
    full.update({c: 0.0 for c in cft})
    return _res(seed, "frozen", full, **kw)


def _doc(screen):
    return V.select(screen, GRID)


# ── 选择 ─────────────────────────────────────────────────────────────────────
def test_select_ranks_filter_passing_at_configs_by_min_rh():
    sc = _screen(0.1, {"pgd-e8-frozen-low": 0.9, "trades-e8-b6-frozen-low": 0.8, "tgt-e4-frozen-low": 0.7,
                       "sau-e51-frozen-high": 0.95, "cft-frozen-low": 0.99},
                 d_pm={"sau-e51-frozen-high": 0.05})               # 精度不过 → 不进选择；C-ft 永远不进
    d = _doc(sc)
    assert d["status"] == "ok" and d["frozen"] == TOP


def test_select_tie_break_prefers_the_cheaper_config():
    sc = _screen(0.1, {"pgd-e8-frozen-high": 0.6, "pgd-e8-frozen-low": 0.6})
    assert _doc(sc)["frozen"][:2] == ["pgd-e8-frozen-low", "pgd-e8-frozen-high"]


def test_select_refuses_wrong_input_and_v0_failure():
    assert V.select(_res(43, "screen", {c: 0.1 for c in AT}), GRID)["status"] == "wrong_input"
    assert V.select(_screen(0.1, v0=False), GRID)["status"] == "invalid"


# ── 判定 ─────────────────────────────────────────────────────────────────────
def _all(screen, frozen_rh, dist=True):
    out = [screen] + [_frozen(s, frozen_rh[s]) for s in (43, 44)]
    if dist:
        out += [_frozen(s, {c: 0.0 for c in TOP}, placement="distributed") for s in (42, 43, 44)]
    return out


def test_go_online():
    sc = _screen(0.1, {"pgd-e8-frozen-low": 0.9, "trades-e8-b6-frozen-low": 0.8, "tgt-e4-frozen-low": 0.7})
    doc = _doc(sc)
    res = V.judge(_all(sc, {43: {c: 0.6 for c in TOP}, 44: {c: 0.55 for c in TOP}}), doc, GRID)
    assert res["verdict"] == "go_online" and res["best"] == "pgd-e8-frozen-low"
    assert res["attribution_vs_cft"]["cft"] == "cft-frozen-low"
    assert res["attribution_vs_cft"]["per_seed"]["s43"]["6"] == pytest.approx(0.6)
    assert set(res["distributed_D_H"]) == {"s42", "s43", "s44"}


def test_go_online_needs_every_seed_reverse_anchor():
    sc = _screen(0.1, {"pgd-e8-frozen-low": 0.9, "trades-e8-b6-frozen-low": 0.8, "tgt-e4-frozen-low": 0.7})
    doc = _doc(sc)
    rh44 = {c: 0.55 for c in TOP}
    rh44["pgd-e8-frozen-low"] = 0.49                               # 一个 seed 差一点 → 换下一名
    res = V.judge(_all(sc, {43: {c: 0.6 for c in TOP}, 44: rh44}), doc, GRID)
    assert res["verdict"] == "go_online" and res["best"] == "trades-e8-b6-frozen-low"
    res = V.judge(_all(sc, {43: {c: 0.6 for c in TOP}, 44: {c: 0.49 for c in TOP}}), doc, GRID)
    assert res["verdict"] == "inconclusive"


def test_kill_pre():
    sc = _screen(0.15)
    doc = _doc(sc)
    res = V.judge(_all(sc, {43: {c: 0.1 for c in doc["frozen"]}, 44: {c: 0.19 for c in doc["frozen"]}}), doc, GRID)
    assert res["verdict"] == "kill_pre"
    res = V.judge(_all(sc, {43: {c: 0.1 for c in doc["frozen"]}, 44: {c: 0.2 for c in doc["frozen"]}}), doc, GRID)
    assert res["verdict"] == "inconclusive"                       # 0.2 不算 < 0.2


def test_accuracy_bound():
    sc = _screen(0.1, {"sau-e51-frozen-high": 0.9}, d_pm={"sau-e51-frozen-high": 0.03})
    doc = _doc(sc)
    res = V.judge(_all(sc, {43: {c: 0.3 for c in doc["frozen"]}, 44: {c: 0.3 for c in doc["frozen"]}}), doc, GRID)
    assert res["verdict"] == "accuracy_bound" and res["s42_meets_R_GO"] == ["sau-e51-frozen-high"]


def test_snapshot_with_small_jump_is_excluded():
    sc = _screen(0.1, {"pgd-e8-frozen-low": 0.9, "trades-e8-b6-frozen-low": 0.8, "tgt-e4-frozen-low": 0.7})
    doc = _doc(sc)
    f43 = _frozen(43, {c: 0.6 for c in TOP})
    f43["snapshots"]["15"]["J"] = 0.04                            # 不计：只看第 6 轮
    for c in TOP:
        f43["snapshots"]["15"]["configs"][c]["eval"]["victim_asr"] = 0.5   # R_H 会是 0，但不计
    res = V.judge([sc, f43, _frozen(44, {c: 0.6 for c in TOP})], doc, GRID)
    assert res["verdict"] == "go_online"
    assert res["frozen_per_seed"]["pgd-e8-frozen-low"]["s43"]["n_used"] == 1


def test_invalid_and_insufficient():
    sc = _screen(0.1, {"pgd-e8-frozen-low": 0.9, "trades-e8-b6-frozen-low": 0.8, "tgt-e4-frozen-low": 0.7})
    doc = _doc(sc)
    res = V.judge([sc, _frozen(43, {c: 0.6 for c in TOP}, v0=False), _frozen(44, {c: 0.6 for c in TOP})], doc, GRID)
    assert res["verdict"] == "invalid"
    res = V.judge([sc, _frozen(43, {c: 0.6 for c in TOP})], doc, GRID)
    assert res["verdict"] == "insufficient" and any("s44" in m for m in res["missing"])


def test_rule_constants_are_n008():
    assert (V.spec.R_GO, V.spec.R_KILL, V.spec.JUMP_MIN, V.spec.MTA_POOL_MAX, V.spec.MTA_EDGE_MAX) == \
        (0.5, 0.2, 0.05, 0.02, 0.04)


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def test_cli_round_trip(tmp_path):
    sc = _screen(0.1, {"pgd-e8-frozen-low": 0.9, "trades-e8-b6-frozen-low": 0.8, "tgt-e4-frozen-low": 0.7})
    gsha = _sha(V.GRID)
    sc["grid_sha"] = gsha
    d = tmp_path / "p0"
    d.mkdir()
    (d / "s42.screen.json").write_text(json.dumps(sc))
    assert V.main(["select", str(d / "s42.screen.json"), "--out", str(tmp_path / "frozen.json")]) == 0
    doc = json.loads((tmp_path / "frozen.json").read_text())
    assert doc["frozen"] == TOP and len(doc["screen_sha"]) == 12 and doc["grid_sha"] == gsha
    fsha = _sha(tmp_path / "frozen.json")
    for s, rh in ((43, 0.6), (44, 0.6)):
        r = _frozen(s, {c: rh for c in TOP})
        r.update(grid_sha=gsha, frozen_sha=fsha)
        (d / f"s{s}.frozen.json").write_text(json.dumps(r))
    assert V.main(["judge", str(d), "--frozen", str(tmp_path / "frozen.json"), "--out", str(tmp_path / "v.json")]) == 0
    assert json.loads((tmp_path / "v.json").read_text())["verdict"] == "go_online"


# ── 第二轮审查（D-105）：两快照不同、缺数据、grid_sha、分散布点的 V0、附加读数 ────────
def _top_screen():
    return _screen(0.1, {"pgd-e8-frozen-low": 0.9, "trades-e8-b6-frozen-low": 0.8, "tgt-e4-frozen-low": 0.7})


def test_rh_uses_the_min_over_both_snapshots():
    sc = _top_screen()
    for c, v in (("pgd-e8-frozen-low", 0.3), ("trades-e8-b6-frozen-low", 0.75)):   # 第 15 轮单独压低 pgd
        sc["snapshots"]["15"]["configs"][c]["eval"]["victim_asr"] = 0.5 - v * 0.30
    doc = _doc(sc)
    assert doc["frozen"][0] == "trades-e8-b6-frozen-low"          # pgd 的 min = 0.3，掉到 tgt（0.7）之后
    assert doc["frozen"] == ["trades-e8-b6-frozen-low", "tgt-e4-frozen-low", "pgd-e8-frozen-low"]


def test_filter_needs_both_snapshots_even_when_one_is_not_counted():
    """N-008：「过滤通过」= 两个快照都通过；J_t < 0.05 只决定 R_H 计不计，不让那个快照免检。"""
    sc = _top_screen()
    sc["snapshots"]["15"]["J"] = 0.04                              # 第 15 轮不计 R_H
    for e in sc["snapshots"]["15"]["configs"]["pgd-e8-frozen-low"]["eval"]["per_edge"].values():
        e["pm_acc"] = 0.80                                          # 但它的精度掉了 0.06
    sc["snapshots"]["15"]["configs"]["pgd-e8-frozen-low"]["eval"]["pm_acc"] = 0.80
    doc = _doc(sc)
    assert "pgd-e8-frozen-low" not in doc["frozen"]               # 旧实现只看计入的快照 → 会把它选上


def test_missing_snapshot_or_config_is_incomplete_not_silently_excluded():
    sc = _top_screen()
    del sc["snapshots"]["15"]["configs"]["pgd-e8-frozen-low"]
    d = _doc(sc)
    assert d["status"] == "incomplete" and any("pgd-e8-frozen-low" in r for r in d["reasons"])
    sc = _top_screen()
    del sc["snapshots"]["15"]
    assert _doc(sc)["status"] == "incomplete"
    sc = _top_screen()
    sc["snapshots"]["6"]["configs"]["tgt-e4-frozen-low"]["eval"]["victim_asr"] = None
    assert _doc(sc)["status"] == "incomplete"
    # judge：s43 少一个快照 → insufficient（不是用剩下的那个快照判 go_online）
    sc = _top_screen()
    doc = _doc(sc)
    f43 = _frozen(43, {c: 0.6 for c in TOP})
    del f43["snapshots"]["15"]
    res = V.judge([sc, f43, _frozen(44, {c: 0.6 for c in TOP})], doc, GRID)
    assert res["verdict"] == "insufficient" and any("第 15 轮" in m for m in res["missing"])


def test_grid_sha_mismatch_is_invalid():
    sc = _top_screen()
    doc = _doc(sc)
    f43 = _frozen(43, {c: 0.6 for c in TOP})
    f43["grid_sha"] = "y"
    res = V.judge([sc, f43, _frozen(44, {c: 0.6 for c in TOP})], doc, GRID)
    assert res["verdict"] == "invalid" and any("grid_sha" in r for r in res["reasons"])


def test_distributed_v0_failure_only_marks_its_d_h():
    sc = _top_screen()
    doc = _doc(sc)
    runs = _all(sc, {43: {c: 0.6 for c in TOP}, 44: {c: 0.55 for c in TOP}})
    bad = [r for r in runs if r["placement"] == "distributed" and r["seed"] == 43][0]
    bad["snapshots"]["6"]["v0"] = {"pass": False, "reasons": ["x"]}
    res = V.judge(runs, doc, GRID)
    assert res["verdict"] == "go_online"
    assert "invalid" in res["distributed_D_H"]["s43"] and "invalid" not in res["distributed_D_H"]["s42"]


def test_aux_is_copied_into_the_verdict():
    sc = _top_screen()
    sc["snapshots"]["6"]["aux"] = {"trigger_norms": {"n": 8}, "bn_private_clu": {"0": {"tpr": 1.0}}}
    doc = _doc(sc)
    res = V.judge(_all(sc, {43: {c: 0.6 for c in TOP}, 44: {c: 0.55 for c in TOP}}), doc, GRID)
    assert res["aux"]["collocated/s42/screen"]["6"]["trigger_norms"] == {"n": 8}


def test_sau_attribution_uses_the_sau_matched_cft():
    sc = _screen(0.1, {"sau-e8-frozen-low": 0.9, "trades-e8-b6-frozen-low": 0.8, "tgt-e4-frozen-low": 0.7})
    doc = _doc(sc)
    top = doc["frozen"]
    res = V.judge([sc] + [_frozen(s, {c: 0.6 for c in top}) for s in (43, 44)], doc, GRID)
    assert res["verdict"] == "go_online" and res["best"] == "sau-e8-frozen-low"
    assert res["attribution_vs_cft"]["cft"] == "cft-sau-frozen-low"


# ── 第三轮审查：冻结清单自检、V0 的单边 None ───────────────────────────────────
def test_judge_rejects_a_frozen_list_that_is_not_the_preregistered_top3():
    sc = _screen(0.1, {"pgd-e4-frozen-low": 0.9, "pgd-e4-train-low": 0.9, "pgd-e8-frozen-low": 0.9,
                       "pgd-e8-train-low": 0.9, "tgt-e4-frozen-low": 0.6})
    doc = _doc(sc)
    assert doc["status"] == "ok" and "tgt-e4-frozen-low" not in doc["frozen"]
    fake = dict(doc, frozen=["tgt-e4-frozen-low"])                 # 第 5 名被手改进冻结清单
    runs = [sc] + [_frozen(s, {"tgt-e4-frozen-low": 0.6}) for s in (43, 44)]
    assert V.judge(runs, fake, GRID)["verdict"] == "invalid"
    bad = {"frozen": [], "status": "wrong_input", "grid_sha": "x"}   # 失败的 select 产物 → 不能悄悄判 inconclusive
    assert V.judge(runs, bad, GRID)["verdict"] == "invalid"


def test_judge_checks_grid_and_frozen_file_shas():
    sc = _top_screen()
    doc = _doc(sc)
    runs = _all(sc, {43: {c: 0.6 for c in TOP}, 44: {c: 0.55 for c in TOP}})
    assert V.judge(runs, doc, GRID, grid_sha="x")["verdict"] == "go_online"
    assert V.judge(runs, doc, GRID, grid_sha="other")["verdict"] == "invalid"
    for r in runs:
        r["frozen_sha"] = "f1"
    assert V.judge(runs, doc, GRID, frozen_sha="f1")["verdict"] == "go_online"
    runs[1]["frozen_sha"] = "f0"
    assert V.judge(runs, doc, GRID, frozen_sha="f1")["verdict"] == "invalid"
    assert V.judge(runs, {k: v for k, v in doc.items() if k != "grid_sha"}, GRID)["verdict"] == "invalid"


def test_v0_one_sided_none_fails_even_without_victims():
    f = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
    r = V.spec.v0_check(0.0, {0: 0.5, 1: None, 2: 0.5, 3: 0.5}, {0: 0.5, 1: 0.9, 2: 0.5, 3: 0.5}, f, f, edges=None)
    assert not r["pass"]


# ── 第三轮审查：go / kill / accuracy_bound 的每个条件各有一个能分辨它的用例 ────────────
def _set_rh(res, t, cid, rh):
    S = res["snapshots"][str(t)]
    S["configs"][cid]["eval"]["victim_asr"] = 0.5 - rh * S["J"]


def test_go_needs_rh_on_both_snapshots_of_every_seed():
    sc = _top_screen()
    f43 = _frozen(43, {c: 0.6 for c in TOP})
    for c in TOP:
        _set_rh(f43, 15, c, 0.4)                                    # s43 第 15 轮掉到 0.4
    res = V.judge([sc, f43, _frozen(44, {c: 0.6 for c in TOP})], _doc(sc), GRID)
    assert res["verdict"] == "inconclusive"


def test_go_needs_the_filter_on_every_seed():
    sc = _top_screen()
    f44 = _frozen(44, {c: 0.6 for c in TOP}, d_pm={c: 0.03 for c in TOP})
    res = V.judge([sc, _frozen(43, {c: 0.6 for c in TOP}), f44], _doc(sc), GRID)
    assert res["verdict"] == "inconclusive"


def test_kill_needs_every_s42_at_config_including_unfrozen_ones():
    sc = _screen(0.15, {"pgd-e4-train-high": 0.25}, d_pm={"pgd-e4-train-high": 0.05})   # 过滤不过 → 不会被冻结
    doc = _doc(sc)
    assert "pgd-e4-train-high" not in doc["frozen"]
    low = {c: 0.1 for c in doc["frozen"]}
    assert V.judge(_all(sc, {43: low, 44: low}), doc, GRID)["verdict"] == "inconclusive"


def test_kill_uses_the_max_over_snapshots():
    sc = _screen(0.1)
    _set_rh(sc, 15, "sau-e8-train-low", 0.3)                        # 0.1@6 / 0.3@15 → 不是「两个快照都 < 0.2」
    doc = _doc(sc)
    low = {c: 0.1 for c in doc["frozen"]}
    assert V.judge(_all(sc, {43: low, 44: low}), doc, GRID)["verdict"] == "inconclusive"


def test_accuracy_bound_only_when_no_filter_passing_config_reaches_r_go():
    sc = _screen(0.1, {"pgd-e8-frozen-low": 0.9, "sau-e51-frozen-high": 0.9},
                 d_pm={"sau-e51-frozen-high": 0.03})
    doc = _doc(sc)
    low = {c: 0.3 for c in doc["frozen"]}
    res = V.judge(_all(sc, {43: low, 44: low}), doc, GRID)
    assert res["verdict"] == "inconclusive" and res["s42_meets_R_GO_and_filter"] == ["pgd-e8-frozen-low"]


def test_distributed_d_h_value_and_sign_and_incomplete():
    sc = _top_screen()
    doc = _doc(sc)
    runs = _all(sc, {43: {c: 0.6 for c in TOP}, 44: {c: 0.55 for c in TOP}})
    d42 = [r for r in runs if r["placement"] == "distributed" and r["seed"] == 42][0]
    for t in ("6", "15"):
        d42["snapshots"][t]["base"]["pooled_benign"] = 0.9
        d42["snapshots"][t]["configs"]["pgd-e8-frozen-low"]["eval"]["pooled_benign"] = 0.6
    d43 = [r for r in runs if r["placement"] == "distributed" and r["seed"] == 43][0]
    del d43["snapshots"]["15"]["configs"]["tgt-e4-frozen-low"]
    res = V.judge(runs, doc, GRID)
    assert res["distributed_D_H"]["s42"]["pgd-e8-frozen-low"] == {"6": 0.3, "15": 0.3}
    assert res["distributed_D_H"]["s43"]["incomplete"]
    assert res["verdict"] == "go_online"                            # 分散布点只报告


def test_kill_boundary_is_exact_not_float_rounding():
    """R_H 恰好 = 0.2（A(G) 0.75、A(G′) 0.5、J 1.25 → 0.25 / 1.25 在浮点里就是 0.2）→ 不算 < 0.2。"""
    sc = _screen(0.15)
    doc = _doc(sc)

    def exact(seed):
        r = _frozen(seed, {c: 0.0 for c in doc["frozen"]})
        for t in ("6", "15"):
            S = r["snapshots"][t]
            S["J"], S["base"]["victim_asr"] = 1.25, 0.75
            for c in doc["frozen"]:
                S["configs"][c]["eval"]["victim_asr"] = 0.5
        return r
    rows = V.snapshot_rows(exact(43), doc["frozen"][0])
    assert rows[6]["r_h"] == 0.2 and rows[15]["r_h"] == 0.2
    assert V.judge([sc, exact(43), exact(44)], doc, GRID)["verdict"] == "inconclusive"


def test_diverged_config_makes_the_screen_incomplete():
    sc = _top_screen()
    c = sc["snapshots"]["6"]["configs"]["pgd-e8-frozen-low"]
    c["eval"], c["diverged"] = None, {"0": 3}
    d = _doc(sc)
    assert d["status"] == "incomplete" and any("发散" in r for r in d["reasons"])
