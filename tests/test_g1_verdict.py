"""
tests/test_g1_verdict.py  —  harness/g1_verdict.py（G1 / G1R5 的预注册判定，N-007）。

合成数据覆盖每一个判定分支；真实数据**只用 seed 42 / 43**（脚本写好之前只看过它们，F-083），
钉住两件事：3-C 的 r_down / Δ_jump / 末周期与 F-083 的手算表相同；3-D 的 AUROC 与已入库的
analysis/g1_scores.json 同源（逐位）。seed 44 与 G1R5 的判定输出在脚本 commit 之后才生成（F-085）。
"""

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "fedavg"))

import g1_verdict as V                                           # noqa: E402
from analysis.functional_score import tpr_at_fpr                 # noqa: E402

REGISTRY = ROOT / "experiments/attack/hfl-mechanism/registry.yaml"
BASE = ROOT / "experiments/attack/hfl-mechanism/base.yaml"
SCORES_JSON = ROOT / "experiments/attack/hfl-mechanism/analysis/g1_scores.json"
LIGHT = list(V.LIGHT_EDGE_COLUMNS)


# ══════════════════════════════════════════════════════════════════════════
# 合成 run
# ══════════════════════════════════════════════════════════════════════════

def _run_block(part, place, R, n, seed):
    want_part, cond = V.PARTITIONS[part]
    return {"partition": want_part, "partition_condition": cond, "malicious_per_edge": V.PLACEMENTS[place],
            "edge_rounds": R, "n_rounds": n, "seed": seed, "update_geometry": True, "post_agg_eval": True,
            "update_ck": True, "provenance": {"run_id": V.g1_run_id(part, place, R, seed)}}


def _sawtooth_run(part="C1", R=10, n=None, seed=42, *, post=0.9, full=0.7, e0=1.0, late=None):
    """集中布点：受害 edge 在每个云轮的云聚合后 = post、云轮末 = full（E0 = e0，不该进量）。
    late：后半程周期改成这个 (post, full)，用来证明后半程不计。"""
    n = n or (V.G1_ROUNDS.get(R) or V.G1R5_ROUNDS)
    per_edge, post_agg = {}, {}
    for g in range(1, n + 1):
        p, f = (late if (late and g > n // 2) else (post, full))
        per_edge[str(g)] = ([{"edge_id": 0, "client_benign": e0}]
                            + [{"edge_id": e, "client_benign": f} for e in V.VICTIMS])
        if g >= 2:
            post_agg[str(g)] = [[e, 0.0, (e0 if e == 0 else p), None, 0.5, 0.5, 0.0, 0.0] for e in range(4)]
    return {"exit_code": 0, "client_failures": [], "n_malicious_participations": n,
            "run": _run_block(part, "collocated", R, n, seed), "per_edge_rounds": per_edge,
            "per_edge_post_agg_rounds": post_agg, "per_edge_light_columns": LIGHT}


def _readout(delta, gm, edge=None, raw=0.8):
    """judge_3d_one 吃的最小读数：一个分数的 delta / global_matched。"""
    if edge is None and delta is not None and gm is not None:
        edge = gm + delta
    sc = {"delta": delta, "global_matched": gm, "edge": edge, "raw": raw}
    return {"reasons": [], "readout": {"scores": {"norm_w": sc}}}


def _grp(deltas, gms):
    return {s: _readout(d, g) for s, d, g in zip(V.SEEDS, deltas, gms)}


# ══════════════════════════════════════════════════════════════════════════
# 原语
# ══════════════════════════════════════════════════════════════════════════

def test_tpr_at_fpr_hand_values():
    neg = list(range(20))                       # 0 … 19；⌊0.05·20⌋ = 1 个假阳 → 阈值 = 第 2 大 = 18
    assert tpr_at_fpr([19, 18.5, 17, 25], neg) == pytest.approx(3 / 4)    # 只有 > 18 的判阳
    assert tpr_at_fpr([0.5], [0.0] * 10) == 1.0                            # ⌊0.5⌋ = 0 → 阈值 = 最大的负类
    assert tpr_at_fpr([0.0], [0.0] * 10) == 0.0                            # 平局判阴（保守）
    assert tpr_at_fpr([], neg) is None and tpr_at_fpr([1.0], []) is None
    # 实际 FPR 永远 ≤ 5%
    r = np.random.default_rng(0)
    for n in (7, 20, 41, 100):
        b = r.normal(size=n)
        t = np.sort(b)[::-1][int(np.floor(0.05 * n))]
        assert (b > t).mean() <= 0.05


def test_direction_thresholds():
    assert V.direction([0.06, 0.07, 0.051]) == "edge_better"
    assert V.direction([0.049, -0.049, 0.0]) == "no_edge_gain"
    assert V.direction([0.06, 0.07, 0.04]) == "mixed"
    assert V.direction([-0.06, -0.07, -0.08]) == "mixed"        # 变差不是 edge_better 也不是 no_gain
    assert V.direction([0.05, 0.05, 0.05]) == "mixed"           # 恰好 0.05：两边都是严格不等号


# ══════════════════════════════════════════════════════════════════════════
# 3-D 判定分支
# ══════════════════════════════════════════════════════════════════════════

NULL_OK = ([0.01, -0.02, 0.0], [0.6, 0.6, 0.6])


def test_3d_edge_better_and_no_gain_and_mixed():
    rnd = _grp(*NULL_OK)
    assert V.judge_3d_one(_grp([0.08, 0.1, 0.06], [0.8, 0.75, 0.9]), rnd, "norm_w")["verdict"] == "edge_better"
    assert V.judge_3d_one(_grp([0.01, -0.03, 0.02], [0.8, 0.75, 0.9]), rnd, "norm_w")["verdict"] == "no_edge_gain"
    assert V.judge_3d_one(_grp([0.01, 0.08, 0.02], [0.8, 0.75, 0.9]), rnd, "norm_w")["verdict"] == "mixed"


def test_3d_undetectable_when_global_auroc_below_0_7():
    v = V.judge_3d_one(_grp([0.01, 0.0, 0.02], [0.55, 0.6, 0.65]), _grp(*NULL_OK), "norm_w")
    assert v["verdict"] == "undetectable" and v["direction_if_detectable"] == "no_edge_gain"


def test_3d_gate3_threshold_is_0_7_inclusive():
    assert V.judge_3d_one(_grp([0.0, 0.0, 0.0], [0.69, 0.69, 0.69]), _grp(*NULL_OK), "norm_w")["verdict"] == "undetectable"
    assert V.judge_3d_one(_grp([0.0, 0.0, 0.0], [0.7, 0.71, 0.72]), _grp(*NULL_OK), "norm_w")["verdict"] == "no_edge_gain"


def test_3d_gate3_two_readings_disagree():
    # 均值 0.72 ≥ 0.7，但有一个 seed 0.64 < 0.7 → 两种读法结论不同
    v = V.judge_3d_one(_grp([0.01, 0.0, 0.02], [0.64, 0.76, 0.76]), _grp(*NULL_OK), "norm_w")
    assert v["verdict"] == "gate3_reading_matters"
    assert v["gate3_mean_ok"] is True and v["gate3_every_seed_ok"] is False
    assert "no_edge_gain" in v["reasons"][0] and "undetectable" in v["reasons"][0]


def test_3d_null_control_gate_makes_it_invalid():
    rnd = _grp([0.06, -0.05, 0.05], [0.6, 0.6, 0.6])           # |Δ| 均值 0.0533 ≥ 0.05
    v = V.judge_3d_one(_grp([0.08, 0.1, 0.06], [0.8, 0.8, 0.8]), rnd, "norm_w")
    assert v["verdict"] == "invalid" and "闸 ①" in v["reasons"][0]
    assert v["null_abs_delta_mean"] == pytest.approx(0.0533, abs=1e-4)


def test_3d_invalid_run_blocks_the_verdict_and_missing_is_insufficient():
    c1 = _grp([0.08, 0.1, 0.06], [0.8, 0.8, 0.8])
    c1[43] = {"reasons": ["client_failures 非空"], "readout": None}
    v = V.judge_3d_one(c1, _grp(*NULL_OK), "norm_w")
    assert v["verdict"] == "invalid" and "C1/s43" in v["reasons"][0]
    c1 = _grp([0.08, 0.1, 0.06], [0.8, 0.8, 0.8])
    c1[44] = None
    assert V.judge_3d_one(c1, _grp(*NULL_OK), "norm_w")["verdict"] == "insufficient"
    assert V.judge_3d_one({s: None for s in V.SEEDS}, {s: None for s in V.SEEDS}, "norm_w")["verdict"] == "missing"


def test_3d_undefined_auroc_is_invalid_not_a_direction():
    v = V.judge_3d_one(_grp([None, 0.1, 0.06], [0.8, 0.8, 0.8]), _grp(*NULL_OK), "norm_w")
    assert v["verdict"] == "invalid"


def test_3d_bootstrap_ci_lower_bound_is_the_smallest_seed():
    v = V.judge_3d_one(_grp([0.08, 0.1, 0.06], [0.8, 0.8, 0.8]), _grp(*NULL_OK), "norm_w")
    assert v["delta_C1_ci"][0] == pytest.approx(0.06) and v["delta_C1_ci"][1] == pytest.approx(0.1)


# ══════════════════════════════════════════════════════════════════════════
# 3-C 判定分支
# ══════════════════════════════════════════════════════════════════════════

def test_sawtooth_quantities_by_hand():
    m = _sawtooth_run(R=10, post=0.9, full=0.7)
    q = V.sawtooth(m)
    assert q["g_range"] == [2, 15] and q["n_g_used"] == 14
    assert q["r_down"] == pytest.approx(0.2 / 10)                # (0.9 − 0.7) / R
    assert q["d_jump"] == pytest.approx(0.2)                     # 0.9 − 上一云轮末 0.7
    assert q["wash_per_cycle"] == pytest.approx(0.2)


def test_attacker_edge_is_not_in_the_quantity():
    a = V.sawtooth(_sawtooth_run(e0=1.0))
    b = V.sawtooth(_sawtooth_run(e0=0.0))
    assert a["r_down"] == b["r_down"] and a["d_jump"] == b["d_jump"]


def test_second_half_cycles_are_ignored():
    a = V.sawtooth(_sawtooth_run(R=20, post=0.9, full=0.7))
    b = V.sawtooth(_sawtooth_run(R=20, post=0.9, full=0.7, late=(0.0, 1.0)))
    assert a["r_down"] == b["r_down"] and a["g_range"] == [2, 7]


def _cell(r_downs, R=10, part="C1"):
    """三个 seed，r_down 依次为给定值（post − full = r_down × R）。"""
    return {s: _sawtooth_run(part=part, R=R, seed=s, post=0.5 + rd * R, full=0.5)
            for s, rd in zip(V.SEEDS, r_downs)}


@pytest.mark.parametrize("rds,want", [
    ((0.006, 0.02, 0.005), "self_cleaning"),
    ((0.0009, 0.0, -0.01), "no_cleaning"),
    ((0.006, 0.0049, 0.02), "user_decides"),
    ((0.002, 0.003, 0.004), "user_decides"),
])
def test_3c_branches(rds, want):
    v = V.judge_3c_cell(_cell(rds), "C1", 10, 30)
    assert v["verdict"] == want
    assert [v["per_seed"][f"s{s}"]["r_down"] for s in V.SEEDS] == pytest.approx(list(rds), abs=1e-9)


def test_3c_invalid_and_missing():
    runs = _cell((0.01, 0.01, 0.01))
    runs[43]["client_failures"] = [{"client_id": 3}]
    assert V.judge_3c_cell(runs, "C1", 10, 30)["verdict"] == "invalid"
    runs = _cell((0.01, 0.01, 0.01))
    del runs[44]["per_edge_post_agg_rounds"]["5"]                  # 前半程缺一个云聚合后点
    v = V.judge_3c_cell(runs, "C1", 10, 30)
    assert v["verdict"] == "invalid" and "缺点" in " ".join(v["reasons"])
    runs = _cell((0.01, 0.01, 0.01))
    runs[42]["per_edge_light_columns"] = LIGHT[::-1]               # 列序变了 → 不能按位置读
    assert V.judge_3c_cell(runs, "C1", 10, 30)["verdict"] == "invalid"
    runs = _cell((0.01, 0.01, 0.01))
    runs[44] = None
    assert V.judge_3c_cell(runs, "C1", 10, 30)["verdict"] == "insufficient"
    assert V.judge_3c_cell({s: None for s in V.SEEDS}, "C1", 10, 30)["verdict"] == "missing"


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r.update(malicious_per_edge=[3, 3, 2, 2]), "malicious_per_edge"),
    (lambda r: r.update(update_ck=False), "update_ck"),
    (lambda r: r.update(partition="equal_random"), "partition"),
    (lambda r: r.update(edge_rounds=5), "edge_rounds"),
])
def test_run_reasons_catch_switches_and_factors_that_did_not_take_effect(mutate, needle):
    m = _sawtooth_run()
    mutate(m["run"])
    assert any(needle in r for r in V.run_reasons(m, "C1", "collocated", 10, 30))


def test_run_reasons_attacker_must_participate():
    m = _sawtooth_run()
    m["n_malicious_participations"] = 0
    assert any("攻击者" in r for r in V.run_reasons(m, "C1", "collocated", 10, 30))
    assert V.run_reasons(None, "C1", "collocated", 10, 30) == ["缺"]


# ══════════════════════════════════════════════════════════════════════════
# 端到端接线（合成几何记录）
# ══════════════════════════════════════════════════════════════════════════

GEO = ["round", "edge_id", "edge_round", "cid", "mal", "norm", "cos_edge", "cos_global",
       "norm_w", "norm_s", "cos_edge_w", "cos_global_w"]


def _geo_run(part, place, R, seed, offsets, bump=3.0, n_per=10, rounds=4):
    """分散布点：每个 edge 一半更新恶意（norm_w 高 bump）；edge 之间有基线偏移 offsets。"""
    r = np.random.default_rng(seed)
    n = V.G1_ROUNDS[R]
    rows, cid = [], 0
    for g in range(1, rounds + 1):
        for e, off in enumerate(offsets):
            ups = []
            for j in range(n_per):
                mal = place == "distributed" and j % 2 == 0
                ups.append((cid, mal, off + (bump if mal else 0.0) + r.normal(0, 0.5)))
                cid += 1
            z = [0.0] * n_per
            rows.append([g, e, 1, [u[0] for u in ups], [int(u[1]) for u in ups], [u[2] for u in ups],
                         z, z, [u[2] for u in ups], z, z, z])
    m = _sawtooth_run(part=part, R=R, seed=seed)
    m["run"] = _run_block(part, place, R, n, seed)
    m["update_geometry"] = {"columns": GEO, "rows": rows}
    return m


def test_judge_end_to_end_on_synthetic_runs():
    g1 = {}
    for part in V.PARTITIONS:
        for place in V.PLACEMENTS:
            for R in V.G1_ROUNDS:
                for s in V.SEEDS:
                    # C1：edge 基线不同 → edge 视角更好；random：edge 可交换 → 零对照 Δ ≈ 0
                    off = [0, 2, 4, 6] if part == "C1" else [0, 0, 0, 0]
                    g1[(part, place, R, s)] = _geo_run(part, place, R, s, off)
    g1r5 = {s: _sawtooth_run(R=5, seed=s) for s in V.SEEDS}
    res = V.judge(g1, g1r5, window=10, resamples=10)
    assert res["overall"]["3D"]["R10"]["norm_w"] == "edge_better"
    assert res["overall"]["3D"]["R20"]["norm_w"] == "edge_better"
    assert res["overall"]["3D"]["R10"]["s_ck"] == "invalid"          # 合成数据没有 c_k → 无定义 → 不判方向
    assert res["overall"]["3C"] == {c: "self_cleaning" for c in res["3C"]}
    assert res["3C"]["C1_collocated_R5"]["bridge"] is True
    assert not res["invalid_runs"] and not res["missing_runs"]
    assert set(res["cells"]) == {f"{p}_{pl}_R{R}" for p in V.PARTITIONS for pl in V.PLACEMENTS for R in V.G1_ROUNDS}
    json.dumps(res)                                                   # 可序列化


# ══════════════════════════════════════════════════════════════════════════
# 与登记表 / base.yaml 交叉核对
# ══════════════════════════════════════════════════════════════════════════

def test_cells_match_the_registry():
    yaml = pytest.importorskip("yaml")
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["groups"]
    g1, g1r5 = reg["G1"], reg["G1R5"]
    assert sorted(g1["seeds"]) == list(V.SEEDS) == sorted(g1r5["seeds"])
    f = g1["factors"]
    parts = {x["label"]: x["set"] for x in f["partition"]}
    assert set(parts) == set(V.PARTITIONS)
    assert parts["random"]["federation.partition"] == V.PARTITIONS["random"][0]
    assert (parts["C1"]["federation.partition"], parts["C1"]["federation.design.condition"]) == V.PARTITIONS["C1"]
    assert {x["label"]: x["set"]["backdoor.malicious_per_edge"] for x in f["placement"]} == V.PLACEMENTS
    assert {int(x["label"][1:]): x["set"]["federation.n_rounds"] for x in f["r_edge"]} == V.G1_ROUNDS
    s5 = g1r5["set"]
    assert (s5["federation.edge_rounds"], s5["federation.n_rounds"]) == (V.G1R5_R, V.G1R5_ROUNDS)
    assert s5["backdoor.malicious_per_edge"] == V.PLACEMENTS["collocated"]
    for grp in (g1, g1r5):
        assert "backdoor.target_label" not in grp["set"]
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    assert base["backdoor"]["target_label"] == V.TARGET


# ══════════════════════════════════════════════════════════════════════════
# 真实数据（只用 seed 42 / 43）
# ══════════════════════════════════════════════════════════════════════════

REAL = (ROOT / "experiments/attack/hfl-mechanism/results/P2/G1").is_dir()


@pytest.fixture(scope="module")
def real_42_43():
    if not REAL:
        pytest.skip("没有 G1 结果")
    return V.load(seeds=(42, 43))


# F-083 的手算表（受害 edge E1–E3、g = 2 … ⌊n/2⌋；Δ_jump C1·R10·s43 F-083 写 0.31，脚本 0.3047，见 F-085）
F083_R_DOWN = {("C1", 10): (0.0217, 0.0252), ("C1", 20): (0.0068, 0.0065),
               ("random", 10): (0.0200, 0.0190), ("random", 20): (0.0064, 0.0053)}
F083_LAST = {("C1", 10): ((1.00, 0.98), (1.00, 0.85)), ("C1", 20): ((0.99, 0.84), (0.97, 0.78)),
             ("random", 10): ((1.00, 0.96), (1.00, 0.73)), ("random", 20): ((0.98, 0.82), (1.00, 0.93))}


def test_real_sawtooth_reproduces_f083(real_42_43):
    g1, _ = real_42_43
    for (part, R), want in F083_R_DOWN.items():
        for s, rd, last in zip((42, 43), want, F083_LAST[(part, R)]):
            m = g1[(part, "collocated", R, s)]
            assert V.run_reasons(m, part, "collocated", R, V.G1_ROUNDS[R]) == []
            q = V.sawtooth(m)
            assert round(q["r_down"], 4) == pytest.approx(rd, abs=1e-9)
            assert q["n_g_used"] == q["n_g_expected"]
            assert (round(q["last_cycle"]["post"], 2), round(q["last_cycle"]["full"], 2)) == last


def test_real_3d_auroc_is_the_same_as_the_committed_g1_scores(real_42_43):
    """判定读数与 g1_scores.json（F-083 的原样输出）逐位相同：两条路径共用 view_arrays。只比 2 个 run 省时。"""
    committed = json.loads(SCORES_JSON.read_text(encoding="utf-8"))
    g1, _ = real_42_43
    for key in [("C1", "distributed", 10, 42), ("random", "collocated", 20, 43)]:
        rid = V.g1_run_id(*key)
        rd = V.score_readouts(g1[key])
        for score, ref in committed[rid]["scores"].items():
            got = rd["scores"][score]
            for f in ("n", "n_malicious", "raw", "edge", "global_full", "global_matched", "delta"):
                assert got[f] == ref[f], (rid, score, f)
            for f in ("tpr5_raw", "tpr5_edge", "tpr5_global_full", "tpr5_global_matched"):
                assert got[f] is None or 0.0 <= got[f] <= 1.0
        ks = rd["kstar"]
        assert ks["malicious"]["n"] > 0 and ks["benign"]["n"] > 0


def test_real_judge_with_two_seeds_is_insufficient_not_a_direction(real_42_43):
    g1, _ = real_42_43
    v3c = V.judge_3c(g1, {})                    # G1R5 在脚本 commit 之前不打开
    assert all(v["verdict"] in ("insufficient", "missing") for v in v3c.values())
