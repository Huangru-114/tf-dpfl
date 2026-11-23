"""
阶段三 D3：`harness/d3_verdict.py`（PREREG-v2 §五 / N-015；D-138）。纯标准库 + d3_spec，本地秒级。

写于任何 D3 数据之前：合成的探针 JSON 覆盖每个标签分支与每道闸；后门强度 S 与底线读真实的已入库 metrics.json。
数据回来之后再加「入库的 d3_verdict.json == 脚本输出」的锚点。
"""

import copy
import json
import math
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "fedavg"))
import d3_verdict as V                                       # noqa: E402
from analysis import d3_spec as S                            # noqa: E402

EN = ROOT / "experiments/defense/edge-native"
PREREG = ROOT / "experiments/thesis/PREREG-v2.md"
REAL_CFG = S.load_config(EN / "d3_configs.yaml")


# ── 合成数据 ─────────────────────────────────────────────────────────────────
def _cfg(tmp_path, n_perm=1000):
    """真实清单，只把置换次数降到 1000（测试快）；sha 跟着清单走。"""
    c = copy.deepcopy(REAL_CFG)
    c["pc"]["permutation"]["n"] = n_perm
    p = tmp_path / "d3_configs.yaml"
    p.write_text(yaml.safe_dump(c, allow_unicode=True), encoding="utf-8")
    return p, c


def _rep(ok=True):
    return {"pass": ok, "tol": 0.01, "max_abs_diff": 0.0 if ok else 0.5, "per_edge": {},
            "reasons": [] if ok else ["edge 0 的 client_benign 差 0.5 > 0.01"]}


def _arm(asr, acc=0.8, fixed=None, diverged=False, margin=1.0):
    return {"diverged": diverged, "main": {"asr": asr, "acc": acc, "margin_p50": margin, "n": 15},
            "fixed": {"asr": asr if fixed is None else fixed, "acc": acc, "margin_p50": margin, "n": 15}}


def _pa(after=0.3, before=1.0, acc_after=0.79, diverged=False, ctrl=0.9):
    return {"lr": 0.05, "lr_harden": 0.005, "edge": 0,
            "arms": {"none": _arm(before), "stage4": _arm(after, acc=acc_after, diverged=diverged),
                     "stem_stage3": _arm(ctrl), "body": _arm(0.2, acc=0.7)}}


def _pb(main=0.3, wb=0.75):
    return {"victims": [1, 2, 3], "recorded_V": main,
            "V": {"main": main, "whitebox": wb, "xi_only_victim": 0.1, "xi_only_attacker": 0.05,
                  "delta_only": 0.2, "clean": 0.01}}


def _pc(metrics, t, fn, nonfinite=False):
    rec = S.recorded_edges(metrics, t)
    edges = {}
    for e in range(4):
        d = fn(rec[e]["margin_p50"], t, e)
        edges[str(e)] = {"kl_all": float("nan") if nonfinite and e == 0 else d, "kl_all_nontarget": d * 0.9,
                         "kl_stem": d * 0.1, "kl_stage1": d, "kl_stage2": d, "kl_stage3": d, "kl_stage4": d,
                         "n": 500, "n_nontarget": 450}
    return {"edges": edges}


def _metrics(run):
    return json.loads((EN / f"results/P2/{S.group_of(run)}/{run}.metrics.json").read_text(encoding="utf-8"))


def write_all(tmp_path, cfg, *, pa=None, pb=None, pb_ref=None, pc_fn=None, rep_fail=(), drop=(),
              nonfinite=(), sha=None):
    """为清单里的每个 run 写一个探针 JSON。pa / pb：{run: kwargs}；pc_fn(S, t, e) → Δ。"""
    d3 = tmp_path / "d3"
    d3.mkdir(exist_ok=True)
    pc_fn = pc_fn or (lambda s, t, e: 0.1 + 0.01 * s)
    for run in S.all_runs(cfg):
        if run in drop:
            continue
        plan = S.plan_for(cfg, run)
        m = _metrics(run)
        j = {"schema": S.SCHEMA, "run_id": run, "seed": S.seed_of(run),
             "config_sha": S.content_sha(cfg) if sha is None else sha, "plan": plan, "rounds": {}}
        for t in S.rounds_needed(plan):
            R = {"reproduce": _rep((run, t) not in rep_fail)}
            if t in plan.get("pa", []):
                R["pa"] = _pa(**(pa or {}).get((run, t), (pa or {}).get(run, {})))
            if t in plan.get("pb", []):
                kw = (pb or {}).get(run) if run in cfg["pb"]["runs"] else (pb_ref or {}).get(run)
                R["pb"] = _pb(**(kw or {}))
            if t in plan.get("pc", []):
                R["pc"] = _pc(m, t, pc_fn, nonfinite=(run, t) in nonfinite)
            j["rounds"][str(t)] = R
        (d3 / f"{run}.json").write_text(json.dumps(j), encoding="utf-8")
    return d3


def judge(tmp_path, cfg_path, d3):
    return V.run_all(d3_dir=d3, config=cfg_path)


# ── 阈值反向锚点：与预注册原文逐字对得上 ─────────────────────────────────────
def test_thresholds_are_the_preregistered_ones():
    txt = PREREG.read_text(encoding="utf-8")
    sec = txt[txt.index("## 五、D3 离线探针"):txt.index("## 六、")]
    assert (V.PA_MAIN, V.PA_EARLY, V.PA_ACC_GATE) == (0.5, 0.25, 0.05)
    assert "三个种子 f 都 ≥ 0.5" in sec and "三个种子 f 都 < 0.25" in sec and "干净精度降幅 ≤ 0.05" in sec
    assert (V.PB_GAP, V.PB_WB, V.PB_SAME) == (0.30, 0.5, 0.10)
    assert "白盒 − 主列 ≥ 0.30 且白盒 ≥ 0.5" in sec and "|白盒 − 主列| < 0.10" in sec
    assert (V.PC_RHO, V.PC_P, V.PC_NONE) == (0.5, 0.01, 0.2)
    assert "合并 Spearman ≥ 0.5、三个种子内各自 > 0、置换检验 p < 0.01" in sec and "合并 Spearman ≤ 0.2" in sec
    assert V.FLOOR_WIN == (51, 60) and V.TOL == 1e-9
    assert "25163a5" in V.PREREG


def test_floors_are_the_g0_c1_e0_last10_values():
    """PREREG-v2 §五：底线_E0 = G0-C1 同种子 E0 末 10 点（0.097 / 0.063 / 0.073）。"""
    got = [V.floor_e0(V.G0_RESULTS, s) for s in V.SEEDS]
    assert [round(v, 4) for v in got] == [0.0974, 0.0633, 0.0731]


def test_tolerance_comparators():
    assert V._ge(0.5 - 1e-12, 0.5) and not V._ge(0.5 - 1e-6, 0.5)
    assert V._lt(0.25 - 1e-6, 0.25) and not V._lt(0.25 - 1e-12, 0.25)
    assert V._le(0.9 - 0.8, 0.10)                          # 0.0999…98 不应判成 < 0.10 的反面（D-133）


# ── 统计 ────────────────────────────────────────────────────────────────────
def test_ranks_and_spearman():
    assert V.ranks([3.0, 1.0, 2.0, 2.0]) == [4.0, 1.0, 2.5, 2.5]
    assert V.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert V.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert V.spearman([1, 1, 1], [1, 2, 3]) is None
    r = np.random.default_rng(0)
    a, b = r.normal(size=50), r.normal(size=50)
    # 与平均秩 Pearson 的 numpy 写法一致
    ra = np.argsort(np.argsort(a)) + 1.0
    rb = np.argsort(np.argsort(b)) + 1.0
    assert V.spearman(list(a), list(b)) == pytest.approx(np.corrcoef(ra, rb)[0, 1], abs=1e-12)


def test_permutation_p_is_small_for_perfect_order_and_large_for_reversed():
    s = list(range(40))
    rho, p = V.perm_p(s, [x * 2.0 for x in s], n=2000, seed=1)
    assert rho == pytest.approx(1.0) and p == pytest.approx(1 / 2001)
    rho, p = V.perm_p(s, [-x for x in s], n=2000, seed=1)
    assert rho == pytest.approx(-1.0) and p == pytest.approx(1.0)
    # 可复现
    assert V.perm_p(s, [x % 7 for x in s], n=500, seed=3) == V.perm_p(s, [x % 7 for x in s], n=500, seed=3)


def test_stratified_permutation_removes_a_pure_group_effect():
    """S 与 Δ 只靠「组」相关（组内无关）：自由置换显著、按组分层置换不显著 —— 轮次混杂的样子。"""
    r = np.random.default_rng(5)
    groups = [g for g in range(3) for _ in range(30)]
    s = [g * 10 + float(r.normal()) for g in groups]
    d = [g * 5 + float(r.normal()) for g in groups]
    _, p_free = V.perm_p(s, d, n=2000, seed=2)
    _, p_strat = V.perm_p(s, d, n=2000, seed=2, groups=groups)
    assert p_free < 0.01 and p_strat > 0.05


# ── 缺数据 / 清单 sha ────────────────────────────────────────────────────────
def test_no_files_is_insufficient_everywhere(tmp_path):
    p, _ = _cfg(tmp_path)
    res = judge(tmp_path, p, tmp_path / "nothing")
    assert (res["pa"]["label"], res["pb"]["label"], res["pc"]["label"]) == ("insufficient",) * 3


def test_a_missing_run_makes_its_probes_insufficient(tmp_path):
    p, c = _cfg(tmp_path)
    d3 = write_all(tmp_path, c, drop=("SNAP__distributed__s43",))
    res = judge(tmp_path, p, d3)
    assert res["pc"]["label"] == "insufficient"
    assert res["pa"]["label"] != "insufficient" and res["pb"]["label"] != "insufficient"


def test_json_written_under_another_config_is_invalid(tmp_path):
    p, c = _cfg(tmp_path)
    d3 = write_all(tmp_path, c, sha="0000000000000000")
    res = judge(tmp_path, p, d3)
    assert res["pa"]["label"] == res["pb"]["label"] == res["pc"]["label"] == "invalid"
    assert "参数清单 sha" in res["problems"][0]


# ── P-a ─────────────────────────────────────────────────────────────────────
def test_pa_mainly_last_block(tmp_path):
    p, c = _cfg(tmp_path)
    res = judge(tmp_path, p, write_all(tmp_path, c, pa={r: {"after": 0.3} for r in c["pa"]["runs"]}))
    pa = res["pa"]
    assert pa["label"] == "mainly_last_block" and len(pa["rows"]) == 6
    r0 = pa["rows"][0]
    assert r0["f"] == pytest.approx((1.0 - 0.3) / (1.0 - 0.0974), abs=1e-4)
    assert pa["per_round"] == {"15": "mainly_last_block", "60": "mainly_last_block"}
    assert set(r0["controls"]) == {"stem_stage3", "body"}


def test_pa_mainly_earlier_layers(tmp_path):
    p, c = _cfg(tmp_path)
    res = judge(tmp_path, p, write_all(tmp_path, c, pa={r: {"after": 0.95} for r in c["pa"]["runs"]}))
    assert res["pa"]["label"] == "mainly_earlier_layers"


def test_pa_rounds_disagreeing_is_inconclusive(tmp_path):
    """第 15 轮 f 大、第 60 轮 f 小 → 6 个值不全满足 → inconclusive；逐轮标签照报。"""
    p, c = _cfg(tmp_path)
    pa = {}
    for r in c["pa"]["runs"]:
        pa[(r, 15)] = {"after": 0.3}
        pa[(r, 60)] = {"after": 0.95}
    res = judge(tmp_path, p, write_all(tmp_path, c, pa=pa))
    assert res["pa"]["label"] == "inconclusive"
    assert res["pa"]["per_round"] == {"15": "mainly_last_block", "60": "mainly_earlier_layers"}


def test_pa_one_seed_in_between_is_inconclusive(tmp_path):
    p, c = _cfg(tmp_path)
    pa = {r: {"after": 0.3} for r in c["pa"]["runs"]}
    pa["SNAP__collocated__s44"] = {"after": 0.6}                # f ≈ 0.43：落在两条阈值之间
    assert judge(tmp_path, p, write_all(tmp_path, c, pa=pa))["pa"]["label"] == "inconclusive"


@pytest.mark.parametrize("kw,needle", [
    ({"acc_after": 0.74}, "干净精度降"),                       # 0.80 − 0.74 = 0.06 > 0.05
    ({"diverged": True}, "发散"),
])
def test_pa_gates(tmp_path, kw, needle):
    p, c = _cfg(tmp_path)
    pa = {r: {"after": 0.3} for r in c["pa"]["runs"]}
    pa["SNAP__collocated__s43"] = {"after": 0.3, **kw}
    res = judge(tmp_path, p, write_all(tmp_path, c, pa=pa))
    assert res["pa"]["label"] == "invalid" and any(needle in x for x in res["pa"]["reasons"])


def test_pa_accuracy_gate_boundary_passes(tmp_path):
    p, c = _cfg(tmp_path)
    pa = {r: {"after": 0.3, "acc_after": 0.75} for r in c["pa"]["runs"]}   # 降 0.05（浮点 0.0500…04）→ 通过
    assert judge(tmp_path, p, write_all(tmp_path, c, pa=pa))["pa"]["label"] == "mainly_last_block"


def test_pa_reproduce_gate(tmp_path):
    p, c = _cfg(tmp_path)
    res = judge(tmp_path, p, write_all(tmp_path, c, rep_fail={("SNAP__collocated__s42", 60)}))
    assert res["pa"]["label"] == "invalid" and res["pb"]["label"] != "invalid"   # P-b 不用 SNAP 第 60 轮判定（只作参照）
    assert res["pc"]["label"] == "invalid"


def test_pa_baseline_at_floor_is_insufficient(tmp_path):
    p, c = _cfg(tmp_path)
    pa = {r: {"after": 0.05, "before": 0.05} for r in c["pa"]["runs"]}
    res = judge(tmp_path, p, write_all(tmp_path, c, pa=pa))
    assert res["pa"]["label"] == "insufficient" and res["pa"]["rows"][0]["f"] is None


# ── P-b ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("main,wb,label", [
    (0.30, 0.75, "trigger_not_transferable"),
    (0.30, 0.35, "not_transmitted"),
    (0.20, 0.45, "inconclusive"),                              # 差 0.25：两条阈值之间
    (0.10, 0.45, "inconclusive"),                              # 差 0.35 但白盒 < 0.5
])
def test_pb_labels(tmp_path, main, wb, label):
    p, c = _cfg(tmp_path)
    res = judge(tmp_path, p, write_all(tmp_path, c, pb={r: {"main": main, "wb": wb} for r in c["pb"]["runs"]}))
    assert res["pb"]["label"] == label
    assert len(res["pb"]["reference"]) == 3 and res["pb"]["rows"][0]["gap"] == pytest.approx(wb - main, abs=1e-4)


def test_pb_needs_all_three_seeds(tmp_path):
    p, c = _cfg(tmp_path)
    pb = {r: {"main": 0.3, "wb": 0.75} for r in c["pb"]["runs"]}
    pb["TBS__k1-snap__s44"] = {"main": 0.3, "wb": 0.35}
    assert judge(tmp_path, p, write_all(tmp_path, c, pb=pb))["pb"]["label"] == "inconclusive"


def test_pb_reproduce_gate(tmp_path):
    p, c = _cfg(tmp_path)
    res = judge(tmp_path, p, write_all(tmp_path, c, rep_fail={("TBS__k1-snap__s43", 60)}))
    assert res["pb"]["label"] == "invalid"


# ── P-c ─────────────────────────────────────────────────────────────────────
def test_pc_units_come_from_the_real_metrics(tmp_path):
    p, c = _cfg(tmp_path)
    units, missing, invalid = V.pc_units(c, {r: json.loads(f.read_text()) for r, f in
                                             ((r, write_all(tmp_path, c) / f"{r}.json") for r in S.all_runs(c))},
                                         {r: _metrics(r) for r in c["pc"]["runs"]})
    assert len(units) == 108 and not missing and not invalid
    u = next(x for x in units if x["run"] == "SNAP__collocated__s42" and x["round"] == 60 and x["edge"] == 0)
    assert u["S"] == 10.0426                                   # metrics.json 的 per_edge_detail_rounds["60"][0]


def test_pc_feedback_moderate_with_the_real_permutation_count(tmp_path):
    """Δ 随 S 单调 → 合并 ρ = 1、每个种子内 1、p = 1 / 10001 → feedback_moderate（用真实清单的 n = 10000）。"""
    p, c = _cfg(tmp_path, n_perm=10000)
    res = judge(tmp_path, p, write_all(tmp_path, c, pc_fn=lambda s, t, e: math.exp(0.1 * s)))
    pc = res["pc"]
    assert pc["label"] == "feedback_moderate" and pc["n_units"] == 108
    assert pc["pooled_spearman"] == pytest.approx(1.0) and pc["perm_p"] == pytest.approx(1 / 10001, abs=1e-6)
    assert set(pc["per_seed_spearman"]) == {"42", "43", "44"}
    assert pc["descriptive"]["nontarget_spearman"] == pytest.approx(1.0)
    assert set(pc["descriptive"]["within_round_spearman"]) == {"6", "15", "60"}


def test_pc_not_supported(tmp_path):
    p, c = _cfg(tmp_path)
    res = judge(tmp_path, p, write_all(tmp_path, c, pc_fn=lambda s, t, e: -s))
    assert res["pc"]["label"] == "not_supported" and res["pc"]["pooled_spearman"] == pytest.approx(-1.0)


def test_pc_one_seed_negative_is_not_feedback_moderate(tmp_path):
    p, c = _cfg(tmp_path)
    res = judge(tmp_path, p, write_all(tmp_path, c, pc_fn=None))
    assert res["pc"]["label"] == "feedback_moderate"
    # 种子 44 的三个 run 上 Δ 反号（pc_fn 看不到种子 → 直接改写 JSON）
    d3 = write_all(tmp_path, c)
    for run in c["pc"]["runs"]:
        if run.endswith("s44"):
            f = d3 / f"{run}.json"
            j = json.loads(f.read_text())
            for R in j["rounds"].values():
                for ed in (R.get("pc") or {}).get("edges", {}).values():
                    ed["kl_all"] = -ed["kl_all"]
            f.write_text(json.dumps(j))
    res = judge(tmp_path, p, d3)
    assert res["pc"]["per_seed_spearman"]["44"] < 0
    assert res["pc"]["label"] != "feedback_moderate"


def test_pc_round_only_effect_shows_in_the_descriptive_readouts(tmp_path):
    """Δ 只随轮次变（组内常数）：判定看合并 ρ，描述性读数里每轮内部的 ρ 无定义 —— 轮次混杂一眼看得出。"""
    p, c = _cfg(tmp_path)
    res = judge(tmp_path, p, write_all(tmp_path, c, pc_fn=lambda s, t, e: float(t)))
    d = res["pc"]["descriptive"]
    assert all(v is None for v in d["within_round_spearman"].values())
    assert res["pc"]["pooled_spearman"] > 0.5


def test_pc_gates(tmp_path):
    p, c = _cfg(tmp_path)
    res = judge(tmp_path, p, write_all(tmp_path, c, nonfinite={("TBS__k1-snap__s42", 6)}))
    assert res["pc"]["label"] == "invalid" and "不是有限数" in res["pc"]["reasons"][0]
    res = judge(tmp_path, p, write_all(tmp_path, c, rep_fail={("SNAP__distributed__s42", 15)}))
    assert res["pc"]["label"] == "invalid" and res["pa"]["label"] != "invalid"


# ── CLI ─────────────────────────────────────────────────────────────────────
def test_cli_writes_the_verdict(tmp_path, capsys):
    p, c = _cfg(tmp_path)
    d3 = write_all(tmp_path, c, pa={r: {"after": 0.3} for r in c["pa"]["runs"]})
    out = tmp_path / "v.json"
    assert V.main(["--d3", str(d3), "--config", str(p), "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["pa"]["label"] == "mainly_last_block" and res["config_sha"] == S.content_sha(c)
    assert "[d3] pa: mainly_last_block" in capsys.readouterr().out


def test_verdict_does_not_import_tensorflow():
    src = (ROOT / "harness/d3_verdict.py").read_text(encoding="utf-8")
    assert "tensorflow" not in src.split('"""', 2)[2]
