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
    assert reg.unmet_requires("CCSF") == ["SA-C"]                       # 划分已定（D-098）
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
        assert fed["n_rounds"] * fed["edge_rounds"] == 300, run["run_id"]
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


def test_operating_characteristics_degenerate_cases():
    assert P.oc_asr_rule(0.20, 0.0, draws=50)["protects"] == 1.0
    assert P.oc_asr_rule(0.0, 0.0, draws=50)["no_effect"] == 1.0
    assert P.oc_asr_rule(0.12, 0.0, draws=50)["partial"] == 1.0     # 均值 < 0.15
    assert P.oc_mta_rule(0.0, 0.0, draws=50) == 1.0 and P.oc_mta_rule(0.03, 0.0, draws=50) == 0.0


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
