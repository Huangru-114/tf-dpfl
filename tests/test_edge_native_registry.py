"""
阶段三（edge 原生防御）D0：登记表 / 提交脚本 / 功效分析的守卫（2026-10-09）。

- SNAP 的配置与 G1R5 只差记录 / 快照开关（PLAN §3.1「D-base」；SNAP-collocated 要与 G1R5 逐位相同，F-084 / D-073）；
- CLAUDE.md 要求补 `set:` 时核对的三件：布点长度 = n_edges、有效轮 = 300、评估网格一致；
- configs/ 与 registry.yaml 同步（改了登记表没重新 materialize → sha 对不上）；
- submit.sh 只读本目录的 INDEX、复用 hfl-mechanism 的作业脚本、保留审计门槛；
- harness/d0_power.py：合成数据的解析值 + 真实数据的锚点（G1R5 在快照轮 6 / 15 的 Δ_jump）+ 入库的 d0_power.json 可复现。

纯标准库 + PyYAML，本地秒级。
"""

import json
import math
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "fedavg"))

import registry as R                                       # noqa: E402
import d0_power as P                                       # noqa: E402

HERE = ROOT / "experiments/defense/edge-native"
REG = HERE / "registry.yaml"
MECH = ROOT / "experiments/attack/hfl-mechanism"
G1R5_RESULTS = MECH / "results/P2/G1R5"

RECORDING_KEYS = {"evaluation.update_geometry", "evaluation.update_ck", "evaluation.update_ck_every",
                  "evaluation.update_ck_n", "evaluation.update_ck_steps",
                  "evaluation.frozen_trigger", "evaluation.snapshot_rounds"}


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def _diff(a, b):
    fa, fb = _flat(a), _flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


@pytest.fixture(scope="module")
def reg():
    return R.Registry(REG)


# ── 登记表 ───────────────────────────────────────────────────────────────────
def test_snap_has_six_runs_and_only_snap_is_unblocked(reg):
    runs = reg.runs()
    snap = [r["run_id"] for r in runs if r["group"] == "SNAP"]
    assert snap == [f"SNAP__{p}__s{s}" for p in ("collocated", "distributed") for s in (42, 43, 44)]
    assert reg.unmet_requires("SNAP") == []
    assert "SA1" in reg.unmet_requires("CAL")
    assert reg.unmet_requires("CCSF") == []                             # D-098 划分；SA-C 代码；空间 = input（D-110）
    assert reg.unmet_requires("SNAP5") == ["h-main"]                    # D-099：H1 / H3 真要跑才放行
    assert "P0" in reg.raw["offline"] and "P0" not in reg.groups      # 离线组不产生训练 run


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_snap_col_differs_from_g1r5_only_in_recording_switches(reg, seed):
    """SNAP-collocated 的有效性闸是「[Checksum] 逐轮 = G1R5」：配置上只许差记录 / 快照开关与 meta。"""
    run = next(r for r in reg.runs() if r["run_id"] == f"SNAP__collocated__s{seed}")
    snap = reg.declared_config(run)
    mech = R.Registry(MECH / "registry.yaml")
    g1r5 = mech.declared_config(next(r for r in mech.runs()
                                     if r["run_id"] == f"G1R5__C1-collocated-R5__s{seed}"))
    d = _diff(snap, g1r5)
    assert d - {"meta.group", "meta.run_id", "meta.study"} == RECORDING_KEYS
    assert snap["evaluation"]["frozen_trigger"] is True and snap["evaluation"]["post_agg_eval"] is True
    assert "update_geometry" not in snap["evaluation"] and "update_ck" not in snap["evaluation"]


def test_snap_dist_differs_from_col_only_in_placement(reg):
    by = {r["run_id"]: r for r in reg.runs()}
    a = reg.declared_config(by["SNAP__collocated__s42"])
    b = reg.declared_config(by["SNAP__distributed__s42"])
    assert _diff(a, b) == {"backdoor.malicious_per_edge", "meta.run_id"}


def test_three_checks_for_every_declared_set(reg):
    """CLAUDE.md：布点长度 = n_edges；有效轮 = 300（stopping 关）；各格评估网格（有效轮）一致。"""
    grids = set()
    for run in reg.runs():
        cfg = reg.declared_config(run)
        fed, ev = cfg["federation"], cfg["evaluation"]
        assert len(cfg["backdoor"]["malicious_per_edge"]) == fed["n_edges"], run["run_id"]
        # CCSA（D-113 的消融）刻意只跑 30 轮：第 10 轮主任务就已分开（F-109），对照是 CCSF / 旧 FedRep 版的前 30 轮
        want = 30 if run["group"] in ("CCSA", "CCSR30", "CCSB30") else 300   # 30 轮的诊断组：只看前 30 轮主任务
        if run["group"] == "ATKP":                    # D-141 的 GPU 探路：4 云轮（只看能不能跑、确定性、投没投毒）
            want = 20
        if run["group"] == "MEDF":                    # D-142：公平的边缘内中位数，30 云轮（分散布点第 25–30 云轮后门已近饱和）
            want = 150
        assert fed["n_rounds"] * fed["edge_rounds"] == want, run["run_id"]
        assert cfg.get("stopping") is None, run["run_id"]
        grids.add(ev["eval_grid"])
    assert grids == {5}


def test_snapshot_rounds_is_a_string_that_covers_p0(reg):
    from utils.dumps import MAX_SNAPSHOTS, parse_rounds
    run = next(r for r in reg.runs() if r["group"] == "SNAP")
    cfg = reg.declared_config(run)
    s = cfg["evaluation"]["snapshot_rounds"]
    assert isinstance(s, str)                         # 列表在 [设定4] 里往返会变形（D-073）
    rounds = parse_rounds(s)
    assert set(P.SNAPSHOT_ROUNDS) <= set(rounds) and len(rounds) <= MAX_SNAPSHOTS
    assert max(rounds) <= cfg["federation"]["n_rounds"]


def test_ccsf_off_cell_is_the_string_off_not_a_yaml_boolean(reg):
    assert [c["cell"] for c in reg.cells("CCSF")] == ["off", "ccs-full"]


def test_ccsf_uses_the_old_noniid_partition_d098(reg):
    """D-098：CCSF 主实验 = CCS 原文的 Dir 0.5 = base 的旧 noniid（逐类 Dirichlet、客户端不等大，F-090）。"""
    cfg = reg.declared_config(next(r for r in reg.runs() if r["group"] == "CCSF"))
    assert cfg["federation"]["partition"] == "noniid"
    assert cfg["federation"]["n_edges"] == 1 and cfg["federation"]["edge_rounds"] == 1


def test_snap5_is_snap_with_seeds_45_46(reg):
    """D-099：5 seed 的对照臂与 SNAP 逐字同配置，只差 seed 与 meta。"""
    by = {r["run_id"]: r for r in reg.runs()}
    assert sorted(r for r in by if r.startswith("SNAP5__")) == sorted(
        f"SNAP5__{p}__s{s}" for p in ("collocated", "distributed") for s in (45, 46))
    for p in ("collocated", "distributed"):
        a = reg.declared_config(by[f"SNAP__{p}__s42"])
        b = reg.declared_config(by[f"SNAP5__{p}__s45"])
        assert _diff(a, b) == {"seed", "meta.group", "meta.run_id"}


def test_materialized_configs_are_in_sync_with_the_registry(reg):
    """改了 registry.yaml 而没重新 materialize（不带 --group）→ 这里红。"""
    index = R.read_index(reg)
    eligible = [r for r in reg.runs()
                if not [t for t in reg.unmet_requires(r["group"]) if not t.startswith("audit")]]
    assert sorted(index) == sorted(r["run_id"] for r in eligible)
    for run in eligible:
        path = reg.config_path(run)
        on_disk = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert on_disk == reg.declared_config(run), run["run_id"]
        assert index[run["run_id"]]["config_sha"] == R.config_sha(on_disk)


# ── 提交脚本 ─────────────────────────────────────────────────────────────────
def test_submit_reads_only_its_own_index_and_reuses_the_mechanism_jobs():
    src = (HERE / "submit.sh").read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert 'INDEX="$HERE/configs/INDEX.tsv"' in code
    assert 'JOB="$MECH/cell.sbatch"' in code and 'PACK_JOB="$MECH/pack.sbatch"' in code
    assert 'AUDIT="$MECH/AUDIT.md"' in code and "gate_closed" in code      # P2 → 审计门槛（D-006）
    run = "\n".join(ln for ln in code.splitlines() if not ln.lstrip().startswith("echo"))
    assert not re.search(r"\bpython3?\b|\$PY\b", run)                       # 登录节点：纯 bash（echo 的提示除外）
    assert "--defense" not in code                                          # 陷阱 #19


# ── 功效分析 ─────────────────────────────────────────────────────────────────
def _synthetic_run(victim, e0, pooled, pm, margin_v=None, n=12, placement=(10, 0, 0, 0)):
    per_edge = {str(r): [{"edge_id": 0, "client_benign": e0}]
                + [{"edge_id": e, "client_benign": victim + 0.01 * (e - 2)} for e in (1, 2, 3)]
                for r in range(1, n + 1)}
    out = {"exit_code": 0, "client_failures": [],
           "run": {"malicious_per_edge": list(placement), "n_rounds": n},
           "rounds": [{"round": r, "local_benign_asr": pooled, "margin_p50": 1.5} for r in range(1, n + 1)],
           "acc_rounds": [{"round": r, "pm_acc": pm} for r in range(1, n + 1)],
           "per_edge_rounds": per_edge}
    if margin_v is not None:
        out["per_edge_detail_columns"] = ["edge_id", "margin_p50", "benign_asr_p90", "yt_clean_benign"]
        out["per_edge_detail_rounds"] = {str(r): [[e, margin_v + e, None, None] for e in range(4)]
                                         for r in range(1, n + 1)}
    return out


def test_run_quantities_on_synthetic_values():
    q = P.run_quantities(_synthetic_run(0.5, 0.9, 0.6, 0.8, margin_v=1.0))
    assert q["V"] == pytest.approx(0.5) and q["B0"] == pytest.approx(0.9)
    assert q["P"] == pytest.approx(0.6) and q["MTA"] == pytest.approx(0.8)
    assert q["margin_v"] == pytest.approx(1.0 + 2.0)          # 受害 edge 1 / 2 / 3 → 2、3、4 的均值
    q = P.run_quantities(_synthetic_run(0.5, 0.9, 0.6, 0.8, placement=(3, 3, 2, 2)))
    assert q["V"] is None and q["B0"] is None and q["P"] == pytest.approx(0.6)   # V / B0 只对集中布点


def test_pairs_and_sd_are_paired_by_seed():
    runs = {lab: {s: None for s in P.SEEDS} for lab in P.CONFIGS}
    ref, trt = "G6 a (旧划分·集中·R5)", "G6 b"
    for i, s in enumerate(P.SEEDS):
        runs[ref][s] = _synthetic_run(0.9, 0.9, 0.9, 0.80)
        runs[trt][s] = _synthetic_run(0.9 - 0.1 * (i + 1), 0.9, 0.9, 0.80 - 0.01)
    res = P.analyse(runs, draws=200)
    v = res["pairs"]["G6 a − b（3-E k=1，真实干预）"]["quantities"]["V"]
    assert v["mean"] == pytest.approx(0.2) and v["sd"] == pytest.approx(0.1)
    assert res["pairs"]["G6 a − b（3-E k=1，真实干预）"]["quantities"]["MTA"]["mean"] == pytest.approx(0.01)


def test_h1_no_effect_rule_d101_vs_v1():
    """D-101：no_effect 从「三个 |Δ| 都 < 0.05」改为「|均值| < 0.05 且最大 |Δ| < 0.10」；protects 不变。"""
    assert P.classify_h1([0.06, -0.02, 0.0], "v1") == "partial"           # 一个 seed 0.06 → 旧规则判不出
    assert P.classify_h1([0.06, -0.02, 0.0], "d101") == "no_effect"
    assert P.classify_h1([0.12, -0.10, 0.0], "d101") == "partial"         # 最大 |Δ| 0.12 ≥ 0.10
    assert P.classify_h1([0.09, 0.03, 0.04], "d101") == "partial"         # 均值 0.0533 ≥ 0.05
    assert P.classify_h1([0.20, 0.15, 0.12], "d101") == "protects" == P.classify_h1([0.20, 0.15, 0.12], "v1")
    assert P.classify_h1([0.30, 0.25, 0.05], "d101") == "partial"         # 最小 seed < 0.10
    with pytest.raises(ValueError):
        P.classify_h1([0.0, 0.0, 0.0], "v2")


def test_registry_has_d0_prereg_after_user_confirmation(reg):
    """N-008 用户确认（2026-10-10，D-101）→ d0-prereg 放行；P0 只剩 SA0。"""
    assert "d0-prereg" in reg.available
    assert set(reg.raw["offline"]["P0"]["requires"]) - reg.available == {"SA0"}


def test_operating_characteristics_degenerate_cases():
    assert P.oc_asr_rule(0.20, 0.0, draws=50)["protects"] == 1.0
    assert P.oc_asr_rule(0.0, 0.0, draws=50)["no_effect"] == 1.0
    assert P.oc_asr_rule(0.12, 0.0, draws=50)["partial"] == 1.0     # 均值 < 0.15
    assert P.oc_mta_rule(0.0, 0.0, draws=50) == 1.0 and P.oc_mta_rule(0.03, 0.0, draws=50) == 0.0
    assert P.oc_asr_rule(0.0, 0.0, draws=50, rule="d101")["no_effect"] == 1.0


@pytest.mark.skipif(not G1R5_RESULTS.exists(), reason="没有 G1R5 结果")
def test_p0_denominators_on_the_real_g1r5_runs():
    """真实锚点：G1R5（= SNAP-collocated 预期逐位相同）在快照轮 6 / 15 的灌入量都 ≥ 0.05 → P0 的 6 个 col 快照都计入。"""
    expect = {42: (0.2309, 0.4226), 43: (0.2497, 0.2244), 44: (0.336, 0.146)}
    for s, (j6, j15) in expect.items():
        m = json.loads((G1R5_RESULTS / f"G1R5__C1-collocated-R5__s{s}.metrics.json").read_text())
        assert round(P.victim_jump(m, 6), 4) == j6 and round(P.victim_jump(m, 15), 4) == j15
        assert min(j6, j15) >= P.JUMP_MIN


@pytest.mark.skipif(not G1R5_RESULTS.exists(), reason="没有 G1R5 结果")
def test_committed_power_json_reproduces(tmp_path):
    out = tmp_path / "p.json"
    assert P.main(["--json", str(out)]) == 0
    committed = json.loads((HERE / "analysis/d0_power.json").read_text(encoding="utf-8"))
    assert json.loads(out.read_text(encoding="utf-8")) == committed


def test_power_script_sums_with_fsum():
    """F-089：内置 sum 的浮点结果随 Python 版本变 → 均值一律 math.fsum。"""
    src = (ROOT / "harness/d0_power.py").read_text(encoding="utf-8")
    assert "from runs_table import last_k_mean" not in src
    for ln in src.splitlines():
        if re.search(r"(?<![\w.])sum\(", ln):
            assert re.search(r"sum\(1 for", ln), ln               # 只许整数计数
    assert math.fsum([0.1] * 10) == 1.0


# ── D-111：H1（A-pre 补 seed）与 P1 dist-pre ────────────────────────────────────
def test_h1_differs_from_p1_col_pre_only_in_seed(reg):
    """H1 = P1__col-pre 同配置补 s43 / s44（s42 复用 P1）。"""
    by = {r["run_id"]: r for r in reg.runs()}
    assert sorted(r for r in by if r.startswith("H1__")) == ["H1__col-pre__s43", "H1__col-pre__s44"]
    a = reg.declared_config(by["P1__col-pre__s42"])
    b = reg.declared_config(by["H1__col-pre__s43"])
    assert _diff(a, b) == {"seed", "meta.group", "meta.run_id"}
    assert reg.unmet_requires("H1") == []


@pytest.mark.parametrize("cell,snap", [("col-pre", "collocated"), ("dist-pre", "distributed")])
def test_p1_cells_differ_from_their_snap_control_only_in_the_hardening_switch(reg, cell, snap):
    """P1 的闸（第 1 云轮 [HardenAgg] = 对照 [Checksum]）要求：与同布点的 SNAP 只差加固开关与快照开关（快照只是记录）。"""
    by = {r["run_id"]: r for r in reg.runs()}
    a = reg.declared_config(by[f"P1__{cell}__s42"])
    b = reg.declared_config(by[f"SNAP__{snap}__s42"])
    assert _diff(a, b) == {"defense.edge_harden", "evaluation.snapshot_rounds", "meta.group", "meta.run_id"}


# ── D-113 / D-114：CCSF 改在 FedPer + 官方超参数下；CCSA 只差超参数 ───────────────
def test_ccsf_is_fedper_with_the_official_hyperparameters(reg):
    """官方 config/defaults.yaml：local_epoch 2、batch 64、lr 0.1、无 momentum（lr 常数 = lr_decay 1.0）。"""
    for r in (x for x in reg.runs() if x["group"] == "CCSF"):
        cfg = reg.declared_config(r)
        t = cfg["training"]
        assert t["drift_correction"] == "hier_fedper", r["run_id"]
        assert (t["local_epochs"], cfg["data"]["batch_size"], t["learning_rate"], t["lr_decay"]) == (2, 64, 0.1, 1.0)


def test_ccsa_differs_from_ccsf_only_in_hyperparameters_and_length(reg):
    by = {r["run_id"]: r for r in reg.runs()}
    for ccsf, ccsa in (("CCSF__off__s42", "CCSA__fedper-ours-off__s42"),
                       ("CCSF__ccs-full__s42", "CCSA__fedper-ours-ccs__s42")):
        d = _diff(reg.declared_config(by[ccsf]), reg.declared_config(by[ccsa]))
        assert d == {"training.local_epochs", "data.batch_size", "training.lr_decay", "federation.n_rounds",
                     "meta.group", "meta.run_id"}, d


def test_ccsl_is_ccsa_off_with_300_rounds(reg):
    """D-115：CCSL 只把 CCSA 的无防御格子拉长到 300 轮（剂量反应的检验）。"""
    by = {r["run_id"]: r for r in reg.runs()}
    d = _diff(reg.declared_config(by["CCSA__fedper-ours-off__s42"]), reg.declared_config(by["CCSL__fedper-ours-off__s42"]))
    assert d == {"federation.n_rounds", "meta.group", "meta.run_id"}, d
    assert reg.declared_config(by["CCSL__fedper-ours-off__s42"])["federation"]["n_rounds"] == 300


def test_ccsr30_is_fedrep_with_projection_trained_in_both_phases(reg):
    """D-116：CCSR30 = FedRep（base 的方法与训练量）+ 投影网络两段都训练 + CCS，30 轮；300 轮那组等放行。"""
    by = {r["run_id"]: r for r in reg.runs()}
    a = reg.declared_config(by["CCSF__ccs-full__s42"])
    b = reg.declared_config(by["CCSR30__fedrep-projboth-ccs__s42"])
    assert b["training"]["drift_correction"] == "hier_fedrep" and b["training"]["fedrep_proj_phase"] == "both"
    assert _diff(a, b) == {"training.drift_correction", "training.local_epochs", "data.batch_size", "training.lr_decay",
                           "training.fedrep_proj_phase", "federation.n_rounds", "meta.group", "meta.run_id"}
    assert "CCSR" not in reg.groups                                 # D-118：前提被否定，300 轮那组已撤


def test_group_protocol_override_only_touches_its_own_meta(reg):
    """D-117：组可以声明 protocol（如 P2-ablation）；不写的组照旧是整张表的 P2（逐字节同以前）。"""
    by = {r["run_id"]: r for r in reg.runs()}
    ablation = {"CCSB30", "CCSBOFF", "CCSB", "BNS", "CCSD",                    # 统计量共享 = 刻意偏离对齐模板的组（D-117 / D-118 / D-120 / D-121）
                "CCSOFF5", "CCSB3", "CCSDA3", "CCSDC5",                    # D-123：拆分实验补种子
                "CCSDP"}                                                   # D-126：缓存池探路
    assert {g for g, d in reg.groups.items() if d.get("protocol")} == ablation
    assert reg.declared_config(by["CCSB30__fedrep-bnshared-ccs__s42"])["meta"]["protocol"] == "P2-ablation"
    others = [r for r in reg.runs() if r["group"] not in ablation]
    assert others and all(reg.declared_config(r)["meta"]["protocol"] == "P2" for r in others)


def test_ccsb30_differs_from_ccsr30_only_in_the_bn_and_projection_switches(reg):
    """D-117：CCSB30 = 旧 FedRep 版 CCS + 统计量共享（投影网络照原写法），30 轮；与 CCSR30 只差这两个开关与口径。"""
    by = {r["run_id"]: r for r in reg.runs()}
    a = reg.declared_config(by["CCSR30__fedrep-projboth-ccs__s42"])
    b = reg.declared_config(by["CCSB30__fedrep-bnshared-ccs__s42"])
    assert b["training"]["fedrep_bn_stats"] == "shared" and "fedrep_proj_phase" not in b["training"]
    assert _diff(a, b) == {"training.fedrep_bn_stats", "training.fedrep_proj_phase", "meta.protocol",
                           "meta.group", "meta.run_id"}


def test_ccsb_off_first_then_ccs_after_the_attack_is_confirmed(reg):
    """D-118：先只交不加防御（CCSBOFF，检验后门能否植入）；开 CCS（CCSB）挂 ccsb-attack-ok。
    CCSB 的配置 = CCSB30 拉长到 300 轮；与 CCSBOFF 只差防御设置。"""
    assert reg.unmet_requires("CCSBOFF") == [] and reg.unmet_requires("CCSB") == []   # D-119：CCSBOFF 通过条件 1 后放行
    by = {r["run_id"]: r for r in reg.runs()}
    off = reg.declared_config(by["CCSBOFF__fedrep-bnshared-off__s42"])
    ccs = reg.declared_config(by["CCSB__fedrep-bnshared-ccs__s42"])
    short = reg.declared_config(by["CCSB30__fedrep-bnshared-ccs__s42"])
    assert _diff(short, ccs) == {"federation.n_rounds", "meta.group", "meta.run_id"}
    assert {k for k in _diff(off, ccs) if not k.startswith("defense.")} == {"meta.group", "meta.run_id"}
    assert off["meta"]["protocol"] == ccs["meta"]["protocol"] == "P2-ablation"
    assert off["training"]["fedrep_bn_stats"] == "shared" and off["federation"]["n_rounds"] == 300


def test_bns_arms_differ_only_in_the_bn_switch_and_shared_arm_equals_ccsboff(reg):
    """D-120 / N-009：BNS 共享格 = CCSBOFF 换种子；私有格与共享格只差 fedrep_bn_stats（私有 = 模板值）；
    种子 42 不重跑（用已有的两份结果）；四个 run 都已放行。"""
    by = {r["run_id"]: r for r in reg.runs()}
    assert sorted(r["run_id"] for r in reg.runs() if r["group"] == "BNS") == [
        "BNS__fedrep-bnprivate-off__s43", "BNS__fedrep-bnprivate-off__s44",
        "BNS__fedrep-bnshared-off__s43", "BNS__fedrep-bnshared-off__s44"]
    assert reg.unmet_requires("BNS") == []
    off42 = reg.declared_config(by["CCSBOFF__fedrep-bnshared-off__s42"])
    for s in (43, 44):
        sh = reg.declared_config(by[f"BNS__fedrep-bnshared-off__s{s}"])
        pr = reg.declared_config(by[f"BNS__fedrep-bnprivate-off__s{s}"])
        assert _diff(off42, sh) == {"seed", "meta.group", "meta.run_id"}, _diff(off42, sh)
        assert _diff(sh, pr) == {"training.fedrep_bn_stats", "meta.run_id"}, _diff(sh, pr)
        assert sh["training"]["fedrep_bn_stats"] == "shared" and pr["training"]["fedrep_bn_stats"] == "private"
        assert sh["seed"] == pr["seed"] == s


def test_ccsd_arms_each_differ_from_ccsb_in_exactly_one_ccs_switch(reg):
    """D-121 / N-010：CCS 拆分 —— 只开对抗训练 / 只开聚类剔除，与 CCSB（两个都开）各只差一个开关；
    N-010 用户确认前挂 ccsd-prereg（D-122 已确认、已放行）。"""
    by = {r["run_id"]: r for r in reg.runs()}
    assert sorted(r["run_id"] for r in reg.runs() if r["group"] == "CCSD") == [
        "CCSD__fedrep-bnshared-at-only__s42", "CCSD__fedrep-bnshared-cluster-only__s42"]
    both = reg.declared_config(by["CCSB__fedrep-bnshared-ccs__s42"])
    at = reg.declared_config(by["CCSD__fedrep-bnshared-at-only__s42"])
    cl = reg.declared_config(by["CCSD__fedrep-bnshared-cluster-only__s42"])
    assert _diff(both, at) == {"defense.ccs.cluster", "meta.group", "meta.run_id"}
    assert _diff(both, cl) == {"defense.ccs.at", "meta.group", "meta.run_id"}
    assert (at["defense"]["ccs"]["at"], at["defense"]["ccs"]["cluster"]) == (True, False)
    assert (cl["defense"]["ccs"]["at"], cl["defense"]["ccs"]["cluster"]) == (False, True)
    assert "ccsd-prereg" in reg.groups["CCSD"]["requires"]                  # 判读规则确认之前不能交（确认后把 token 加进 available）


def test_ccs_seed_extension_matches_the_seed42_configs(reg):
    """D-123 / N-011：分层补种子 —— 每个新 run 与同臂的种子 42 配置只差 seed 与 meta；种子分配照计划
    （不加防御补 45 / 46、只开聚类补 43–46、两个都开与只开对抗训练补 43 / 44）；判读规则确认前挂 ccsd5-prereg。"""
    by = {r["run_id"]: r for r in reg.runs()}
    ref = {"CCSOFF5": ("CCSBOFF__fedrep-bnshared-off__s42", [45, 46]),
           "CCSB3": ("CCSB__fedrep-bnshared-ccs__s42", [43, 44]),
           "CCSDA3": ("CCSD__fedrep-bnshared-at-only__s42", [43, 44]),
           "CCSDC5": ("CCSD__fedrep-bnshared-cluster-only__s42", [43, 44, 45, 46])}
    for g, (rid, seeds) in ref.items():
        runs = [r for r in reg.runs() if r["group"] == g]
        assert sorted(reg.declared_config(r)["seed"] for r in runs) == seeds, g
        base = reg.declared_config(by[rid])
        for r in runs:
            assert _diff(base, reg.declared_config(r)) == {"seed", "meta.group", "meta.run_id"}, r["run_id"]
        assert "ccsd5-prereg" in reg.groups[g]["requires"]


def test_ccsdp_differs_from_cluster_only_only_in_the_pool(reg):
    """D-126 / N-012：缓存池探路 = 只开聚类（CCSDC5）同种子，只把 defense.ccs.pool 从 round 换成 edge_cache；
    种子 = 只开聚类失守的 43、46；判读规则确认前挂 ccsdp-prereg（D-127 已确认、已放行）。"""
    by = {r["run_id"]: r for r in reg.runs()}
    runs = sorted(r["run_id"] for r in reg.runs() if r["group"] == "CCSDP")
    assert runs == ["CCSDP__fedrep-bnshared-cluster-cache__s43", "CCSDP__fedrep-bnshared-cluster-cache__s46"]
    for s in (43, 46):
        a = reg.declared_config(by[f"CCSDC5__fedrep-bnshared-cluster-only__s{s}"])
        b = reg.declared_config(by[f"CCSDP__fedrep-bnshared-cluster-cache__s{s}"])
        assert _diff(a, b) == {"defense.ccs.pool", "meta.group", "meta.run_id"}
        assert a["defense"]["ccs"]["pool"] == "round" and b["defense"]["ccs"]["pool"] == "edge_cache"
        assert (b["defense"]["ccs"]["at"], b["defense"]["ccs"]["cluster"]) == (False, True)
    assert "ccsdp-prereg" in reg.groups["CCSDP"]["requires"]


# ── 方向 2 第一段（D-130 / D-131）：ERP / ERPB ─────────────────────────────────
HOMO = {"federation.design.client_profile", "federation.design.profile_lambda"}
META = {"meta.group", "meta.run_id"}


@pytest.mark.parametrize("snap", ["collocated", "distributed"])
def test_erp_main_t0_differs_from_snap_only_in_the_defense_block(reg, snap):
    """「记录不改训练」的闸（ERP main-t0 的 [Checksum] = SNAP 同种子）要求：只差 defense 块与快照开关（快照只是记录）。"""
    by = {r["run_id"]: r for r in reg.runs()}
    a = reg.declared_config(by[f"ERP__main-t0-{snap}__s42"])
    b = reg.declared_config(by[f"SNAP__{snap}__s42"])
    assert _diff(a, b) == {"defense.name", "defense.edge_ref.mode", "evaluation.snapshot_rounds"} | META


@pytest.mark.parametrize("part", ["main", "homo"])
@pytest.mark.parametrize("snap", ["collocated", "distributed"])
def test_erp_t1_differs_from_t0_only_in_the_tier_switch(reg, part, snap):
    by = {r["run_id"]: r for r in reg.runs()}
    a = reg.declared_config(by[f"ERP__{part}-t0-{snap}__s42"])
    b = reg.declared_config(by[f"ERP__{part}-t1-{snap}__s42"])
    assert _diff(a, b) == {"federation.edge_shared_blocks", "meta.run_id"}


@pytest.mark.parametrize("tier", ["t0", "t1"])
@pytest.mark.parametrize("snap", ["collocated", "distributed"])
def test_erp_homo_differs_from_main_only_in_the_client_profile(reg, tier, snap):
    by = {r["run_id"]: r for r in reg.runs()}
    a = reg.declared_config(by[f"ERP__main-{tier}-{snap}__s42"])
    b = reg.declared_config(by[f"ERP__homo-{tier}-{snap}__s42"])
    assert _diff(a, b) == HOMO | {"meta.run_id"}
    assert b["federation"]["design"]["profile_lambda"] == 0.4


@pytest.mark.parametrize("cell", ["main-t0", "main-t1", "homo-t0", "homo-t1"])
def test_erpb_differs_from_the_collocated_erp_cell_only_in_poison_ratio_and_seed(reg, cell):
    by = {r["run_id"]: r for r in reg.runs()}
    a = reg.declared_config(by[f"ERP__{cell}-collocated__s42"])
    b = reg.declared_config(by[f"ERPB__{cell}__s41"])
    assert _diff(a, b) == {"backdoor.poison_ratio", "seed"} | META
    assert b["backdoor"]["poison_ratio"] == 0.0 and b["seed"] == 41


def test_erp_groups_are_released_after_the_n013_confirmation(reg):
    assert reg.unmet_requires("ERP") == [] and reg.unmet_requires("ERPB") == []
    assert sum(r["group"] == "ERP" for r in reg.runs()) == 8
    assert sum(r["group"] == "ERPB" for r in reg.runs()) == 4



# ── 方向 2 第二段（N-014 / D-132）：ERF / ERC ─────────────────────────────────
PROBE = HERE / "analysis/edge_ref_probe.json"


def test_erf_scores_and_thresholds_are_exactly_the_n013_selection(reg):
    """手抄错一位就是另一个实验：ERF 每格的 score / threshold 必须等于 edge_ref_probe.json 的选择。"""
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    runs = [r for r in reg.runs() if r["group"] == "ERF"]
    assert len(runs) == 24
    for r in runs:
        part, tier, _ = r["cell"].split("-")
        er = reg.declared_config(r)["defense"]["edge_ref"]
        assert er["mode"] == "filter"
        assert er["score"] == probe[part]["selected_score"], r["run_id"]
        assert er["threshold"] == probe[part]["threshold_dev_q90"][tier], r["run_id"]


@pytest.mark.parametrize("part", ["main", "homo"])
@pytest.mark.parametrize("tier", ["t0", "t1"])
@pytest.mark.parametrize("snap", ["collocated", "distributed"])
def test_erf_differs_from_the_erp_cell_only_in_the_filter_switches(reg, part, tier, snap):
    by = {r["run_id"]: r for r in reg.runs()}
    a = reg.declared_config(by[f"ERP__{part}-{tier}-{snap}__s42"])
    b = reg.declared_config(by[f"ERF__{part}-{tier}-{snap}__s42"])
    assert _diff(a, b) == {"defense.edge_ref.mode", "defense.edge_ref.score", "defense.edge_ref.threshold"} | META


def test_erc_is_erp_with_seeds_43_44(reg):
    by = {r["run_id"]: r for r in reg.runs()}
    erc = sorted(r for r in by if r.startswith("ERC__"))
    assert len(erc) == 12 and not any("main-t0" in r for r in erc)        # main-t0 的对照是 SNAP
    for rid in erc:
        cell, seed = rid.split("__")[1], rid.split("__")[2]
        a = reg.declared_config(by[f"ERP__{cell}__s42"])
        b = reg.declared_config(by[rid])
        assert _diff(a, b) == {"seed"} | META and seed in ("s43", "s44")


def test_every_erf_cell_has_a_three_seed_control(reg):
    """N-014 的配对：main-t0 → SNAP s42–44；其余 → ERP s42 + ERC s43 / s44。"""
    ids = {r["run_id"] for r in reg.runs()}
    for r in reg.runs():
        if r["group"] != "ERF":
            continue
        cell, seed = r["cell"], r["seed"]
        part, tier, place = cell.split("-")
        if (part, tier) == ("main", "t0"):
            assert f"SNAP__{place}__s{seed}" in ids
        else:
            assert (f"ERP__{cell}__s42" if seed == 42 else f"ERC__{cell}__s{seed}") in ids


def test_erf_erc_are_released(reg):
    assert reg.unmet_requires("ERF") == [] and reg.unmet_requires("ERC") == []


# ── 论文收尾：补充实验第一批（N-015 / D-135）：TBS / TBH / MED ──────────────────────────────
def _cfg(reg, rid):
    return reg.declared_config(next(r for r in reg.runs() if r["run_id"] == rid))


def _b1_ref(cell_place, seed):
    """B=1 的已有对照：s42 = ERP（observe），s43 / s44 = ERC（observe）。"""
    return f"ERP__main-t1-{cell_place}__s42" if seed == 42 else f"ERC__main-t1-{cell_place}__s{seed}"


def test_supplement_groups_are_released_with_the_planned_sizes(reg):
    for g, n in (("TBS", 9), ("TBH", 6), ("MED", 12)):
        assert reg.unmet_requires(g) == [], g
        assert sum(r["group"] == g for r in reg.runs()) == n, g
    assert "supp-prereg" in reg.groups["MED"]["requires"] and "tier-0to4" in reg.groups["TBS"]["requires"]


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_tbs_k1_snap_differs_from_the_existing_b1_run_only_in_defense_and_snapshots(reg, seed):
    """闸 ②（N-015）：TBS k1-snap 的 [Checksum] 要与 ERP / ERC main-t1-collocated 同种子逐位相同 →
    配置上只许差「observe 的 defense 块」与快照开关（两者都只记录、不改训练）。"""
    a = _cfg(reg, f"TBS__k1-snap__s{seed}")
    b = _cfg(reg, _b1_ref("collocated", seed))
    assert _diff(a, b) == {"defense.name", "defense.edge_ref.mode", "evaluation.snapshot_rounds"} | META
    assert a["defense"]["name"] == "none" and a["evaluation"]["snapshot_rounds"] == "6/15/60"


@pytest.mark.parametrize("cell,k", [("k1-snap", 1), ("k2", 2), ("k4", 4)])
def test_tbs_differs_from_snap_collocated_only_in_the_block_count(reg, cell, k):
    a = _cfg(reg, f"TBS__{cell}__s42")
    b = _cfg(reg, "SNAP__collocated__s42")
    extra = set() if cell == "k1-snap" else {"evaluation.snapshot_rounds"}      # SNAP 也存 6/15/60
    assert _diff(a, b) == {"federation.edge_shared_blocks"} | extra | META
    assert a["federation"]["edge_shared_blocks"] == k


def test_tbh_is_the_threat_boundary_placement(reg):
    a, b = _cfg(reg, "TBH__k0__s42"), _cfg(reg, "TBH__k1__s42")
    snap = _cfg(reg, "SNAP__collocated__s42")
    assert a["backdoor"]["malicious_per_edge"] == [9, 1, 0, 0]
    assert _diff(a, snap) == {"backdoor.malicious_per_edge", "evaluation.snapshot_rounds"} | META
    assert _diff(a, b) == {"federation.edge_shared_blocks", "meta.run_id"}


@pytest.mark.parametrize("layer", ["edge", "cloud"])
@pytest.mark.parametrize("place", ["collocated", "distributed"])
def test_med_differs_from_snap_only_in_the_median_switch(reg, layer, place):
    a = _cfg(reg, f"MED__{layer}-{place}__s42")
    b = _cfg(reg, f"SNAP__{place}__s42")
    assert _diff(a, b) == {"defense.name", "defense.layers", "evaluation.snapshot_rounds"} | META
    assert a["defense"]["name"] == "median" and a["defense"]["layers"] == [layer]
    assert a["federation"].get("edge_shared_blocks") in (None, 0)        # 两臂都在 B=0 上（云层防御与 k > 0 互斥）


@pytest.mark.parametrize("place", ["collocated", "distributed"])
def test_med_edge_and_cloud_arms_differ_only_in_the_layer(reg, place):
    a, b = _cfg(reg, f"MED__edge-{place}__s42"), _cfg(reg, f"MED__cloud-{place}__s42")
    assert _diff(a, b) == {"defense.layers", "meta.run_id"}


@pytest.mark.parametrize("group", ["TBS", "TBH", "MED"])
def test_supplement_seeds_differ_only_in_seed(reg, group):
    by = {r["run_id"]: r for r in reg.runs()}
    for cell in {r["cell"] for r in reg.runs() if r["group"] == group}:
        a = reg.declared_config(by[f"{group}__{cell}__s42"])
        for s in (43, 44):
            assert _diff(a, reg.declared_config(by[f"{group}__{cell}__s{s}"])) == {"seed", "meta.run_id"}


def test_every_supplement_cell_has_its_paired_controls(reg):
    """N-015 的配对都在登记表里：TBS → SNAP 集中（B=0）+ ERP / ERC main-t1 集中（闸 ②）；
    MED → SNAP 同布点（B=0）+ ERP / ERC main-t1 同布点（与 B=1 比）；TBH 自带 B=0 / B=1。"""
    ids = {r["run_id"] for r in reg.runs()}
    for r in reg.runs():
        s = r["seed"]
        if r["group"] == "TBS":
            assert f"SNAP__collocated__s{s}" in ids and _b1_ref("collocated", s) in ids
        elif r["group"] == "MED":
            place = r["cell"].split("-", 1)[1]
            assert f"SNAP__{place}__s{s}" in ids and _b1_ref(place, s) in ids
        elif r["group"] == "TBH":
            assert {f"TBH__k0__s{s}", f"TBH__k1__s{s}"} <= ids


def test_supplement_configs_pass_startup_validation():
    """27 个新配置都过 config_validate（k = 4、中位数 × 网格 / 云聚合后评估 / 冻结触发器都不被拒绝）。"""
    from config_validate import validate_config
    for p in sorted((HERE / "configs").glob("*.yaml")):
        if p.name.split("__")[0] in ("TBS", "TBH", "MED"):
            validate_config(yaml.safe_load(p.read_text(encoding="utf-8")))


# ── 公平的边缘内中位数（N-017 / D-142）：MEDF ─────────────────────────────────
MEDF_SETTING = {"federation.client_fraction", "federation.n_rounds",
                "evaluation.post_agg_eval", "evaluation.frozen_trigger"}


def test_medf_has_two_runs_and_waits_for_the_user_confirmed_prereg(reg):
    runs = sorted(r["run_id"] for r in reg.runs() if r["group"] == "MEDF")
    assert runs == ["MEDF__ctrl-t1-distributed__s42", "MEDF__edge-t1-distributed__s42"]
    assert "medf-prereg" in reg.groups["MEDF"]["requires"]


def test_medf_ctrl_differs_from_the_b1_distributed_run_only_in_participation_length_and_recording(reg):
    """不设防对照 = ERP main-t1-distributed（B=1、分散、observe）换成 defense none，再改参与率 / 轮数 / 两个记录开关。"""
    a = _cfg(reg, "MEDF__ctrl-t1-distributed__s42")
    b = _cfg(reg, "ERP__main-t1-distributed__s42")
    assert _diff(a, b) == {"defense.name", "defense.edge_ref.mode"} | MEDF_SETTING | META
    f = a["federation"]
    assert (f["client_fraction"], f["n_rounds"], f["edge_shared_blocks"]) == (0.2, 30, 1)
    assert a["backdoor"]["malicious_per_edge"] == [3, 3, 2, 2] and a["defense"]["name"] == "none"
    assert a["evaluation"]["post_agg_eval"] is False and a["evaluation"]["frozen_trigger"] is False
    # 每台边缘每个边缘轮 5 个上传（0.2 × 25 端），与判定脚本的闸同一个数
    assert round(f["client_fraction"] * f["n_clients"] / f["n_edges"]) == 5


def test_medf_median_arm_differs_from_its_control_only_in_the_defense(reg):
    a = _cfg(reg, "MEDF__edge-t1-distributed__s42")
    b = _cfg(reg, "MEDF__ctrl-t1-distributed__s42")
    assert _diff(a, b) == {"defense.name", "defense.layers", "meta.run_id"}
    assert a["defense"]["name"] == "median" and a["defense"]["layers"] == ["edge"]


def test_medf_configs_pass_startup_validation(reg):
    """B=1 + 边缘层中位数（§4e 只拒绝云层防御）、参与率 0.2、30 云轮都不被拒绝。"""
    from config_validate import validate_config
    for r in reg.runs():
        if r["group"] == "MEDF":
            validate_config(reg.declared_config(r))


def test_medf_is_released_after_the_n017_confirmation(reg):
    assert reg.unmet_requires("MEDF") == []
