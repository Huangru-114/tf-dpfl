"""
tests/test_registry.py  —  登记表的展开 / materialize / 审计门槛 / 提交脚本

登记表是「声明」，harness/status.py 用它去对「实际」（metrics.json）。
这里锁住声明本身：run 个数与 id 精确、materialize 出的 config_sha 与 main.py
读同一个文件算出的相同、审计门槛读得出来、提交脚本不带覆盖参数。

纯标准库 + pyyaml + bash，本地秒级。
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import registry as R                               # noqa: E402
from utils.provenance import config_sha            # noqa: E402

MECH = ROOT / "experiments" / "attack" / "hfl-mechanism"
V2 = MECH / "registry.yaml"
V1 = ROOT / "experiments" / "attack" / "hfl-propagation" / "registry" / "v1.yaml"


# ── 新方案登记表（PLAN.md §4）─────────────────────────────────────────────────
def test_v2_group_sizes_match_plan():
    reg = R.Registry(V2)
    runs = reg.runs()
    sizes = {}
    for r in runs:
        sizes[r["group"]] = sizes.get(r["group"], 0) + 1
    assert sizes == {"G0": 15, "G1": 24, "G2": 55, "G3": 24, "G4": 12, "G5": 15, "G6": 9,   # G3：S3 补 C1（D-062）
                     "FLR": 3,                   # FLR：floor 验证 pilot（D-061）
                     "G7": 6,                    # G7：A4 登记（D-025 预处理对比）
                     "G8": 3, "G6D": 3,          # S9（D-075）：G8 衰减 + G6D 分散布点探针
                     "G5AB": 8, "G8F": 3,        # S4（D-079 / D-077）：生成器语义 A/B 对比 + G8 的 flat 对照
                     "S5P": 4,                   # S5（D-084）：GPU 上开 / 关网格的 checksum 探路
                     "G1P": 3}                   # S6a（D-085）：GPU 上开 / 关记录开关的 checksum 探路 + 几何 AUROC
    assert len({r["run_id"] for r in runs}) == len(runs) == 187


def test_v2_run_ids_and_factor_settings():
    runs = {r["run_id"]: r for r in R.Registry(V2).runs()}
    r = runs["G1__random_collocated_R10__s42"]
    assert r["set"] == {"federation.n_edges": 4, "federation.edge_rounds": 10,
                        # S3（D-065）：random = 等大小版
                        "federation.partition": "equal_random",
                        "federation.design.n_per_client": 500, "federation.design.clean_per_edge": 500,
                        "federation.design.alpha_client": 0.5,
                        # S6a（D-085）：布点 / 轮数（300 有效轮）/ 关停止判据 / 网格
                        "backdoor.malicious_per_edge": [10, 0, 0, 0], "federation.n_rounds": 30,
                        "stopping": None, "evaluation.eval_grid": 5}
    assert runs["G4__p1.0_fedavg__s44"]["set"] == {
        "backdoor.poison_ratio": 1.0, "training.drift_correction": "hierfedavg"}
    # D-047（2026-09-27）：G2 flat 补上布点、轮数与评估间隔（原来只有前两项 → 布点 [5,5]、截断在 60 轮）
    assert runs["G2__flat__s46"]["set"] == {
        "federation.n_edges": 1, "federation.edge_rounds": 1, "federation.n_rounds": 300,
        "backdoor.malicious_per_edge": [10],
        "backdoor.eval_interval": 5, "evaluation.eval_interval": 5,
        # S3（D-065）：G2 与 G1-random 同一套等大小数据
        "federation.partition": "equal_random",
        "federation.design.n_per_client": 500, "federation.design.clean_per_edge": 500,
        "federation.design.alpha_client": 0.5,
        # S5（D-084）：统一评估网格（有效轮）
        "evaluation.eval_grid": 5}


def test_v2_base_is_decided_in_a4_and_carries_the_p2_template():
    """A4 定了 base：P1 锚点格去掉死配置 + overlays 叠上「P2 对齐」模板。"""
    import alignment as AL
    from config_validate import validate_config
    reg = R.Registry(V2)
    assert reg.base == "base.yaml"
    assert [Path(o).name for o in reg.overlays] == ["alignment_p2.yaml"]
    for run in (r for r in reg.runs() if r["group"] == "G7"):
        cfg = reg.declared_config(run)
        assert cfg["meta"]["protocol"] == "P2" and AL.template_state(cfg) == "p2"
        assert "label_smoothing" not in cfg["training"]               # A16 / D-028
        assert cfg["backdoor"]["target_label"] == 0                    # A22 / D-028
        assert cfg["backdoor"]["malicious_strategy"] == "badpfl"       # cell.sbatch 不传 CLI
        validate_config(cfg)                                           # P2 核对模板：通过
    off = next(r for r in reg.runs() if r["run_id"] == "G7__official__s42")
    cfg = reg.declared_config(off)
    assert cfg["data"]["normalize"] is False and cfg["data"]["augment"] is False


def test_base_null_is_still_refused_with_a_pointer_to_a4(tmp_path):
    """base 没定时 materialize 拒绝，报错里指向 A4（harness/registry.py 的文案）。"""
    d = tmp_path
    (d / "reg.yaml").write_text(yaml.safe_dump({
        "study": "t", "protocol": "P2", "base": None,
        "groups": {"G": {"seeds": [1], "cells": [{"cell": "c"}]}}}))
    reg = R.Registry(d / "reg.yaml")
    with pytest.raises(R.RegistryError, match="A4"):
        reg.declared_config(reg.runs()[0])


def test_label_smoothing_is_read_nowhere_in_fedavg():
    """A16：将来谁实现了 label_smoothing，这条先红 —— 旧配置里的 0.1 不会悄悄生效。"""
    pat = re.compile(r"""["']label_smoothing["']""")
    hits = [p for p in (ROOT / "fedavg").rglob("*.py")
            if pat.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert hits == []


def test_v2_every_group_is_gated_by_the_audit():
    reg = R.Registry(V2)
    for g in reg.groups:
        assert "audit" in reg.groups[g]["requires"], g


# ── 旧方案登记表 ─────────────────────────────────────────────────────────────
def test_v1_registry_covers_every_old_config_exactly_once():
    reg = R.Registry(V1)
    cfgs = sorted(Path(r["config"]).name for r in reg.runs())
    on_disk = sorted(p.name for p in (ROOT / "experiments/attack/hfl-propagation").glob("*.yaml"))
    assert cfgs == on_disk and len(cfgs) == 25


def test_v1_registry_is_not_picked_up_as_a_cell():
    """run_exp3.sh 用 `ls *.yaml`；登记表放在顶层会被当成第 26 格提交。"""
    assert V1.parent.name == "registry"
    assert not (ROOT / "experiments/attack/hfl-propagation/registry.yaml").exists()


# ── 审计门槛 ─────────────────────────────────────────────────────────────────
def test_real_audit_parses_and_is_closed():
    rows = R.audit_rows(MECH / "AUDIT.md")
    assert set(rows.values()) <= set(R.AUDIT_STATUSES)
    # 加行时要同步改这里 —— 故意的：AUDIT 的行只增不删，行数变化应当是一次有意识的提交。
    assert {f"A{i:02d}" for i in range(1, 30)} | {f"D{i:02d}" for i in range(1, 7)} == set(rows)
    assert rows["A19"] == "done"
    assert rows["A04"] == "deviate"                # D-017：对齐论文 Eq.7，偏离官方代码
    # A2 会话（2026-09-25）：训练协议保留调过参的现状（deviate）
    assert {k: rows[k] for k in ("A07", "A09", "A10", "A11", "A12", "A13", "A23")} == \
        dict.fromkeys(("A07", "A09", "A10", "A11", "A12", "A13", "A23"), "deviate")
    assert rows["A20"] == "done"
    # A3 会话（2026-09-25/26）：D-035 / D-037
    assert rows["A17"] == "done" and rows["A18"] == "done" and rows["A21"] == "deviate"
    assert {k: rows[k] for k in ("D03", "D04", "D05", "D06")} == \
        dict.fromkeys(("D03", "D04", "D05", "D06"), "deviate")
    # A4 会话（2026-09-26）：按拍板实现为开关 + L1 → done
    a4_done = ("A01", "A02", "A03", "A05", "A06", "A14", "A16", "A22", "A24",
               "A27", "A28", "A29", "D01", "D02")
    assert {k: rows[k] for k in a4_done} == dict.fromkeys(a4_done, "done")
    # A4 收口（2026-09-27，pilot `2853433`）：D-029 pass → A08 deviate、A25 done；
    # DET pass → A15 done；D-031 different → 用户维持 head_first（D-045）→ A26 deviate
    assert rows["A15"] == "done" and rows["A25"] == "done"
    assert rows["A08"] == "deviate" and rows["A26"] == "deviate"
    assert R.audit_open_rows(MECH / "AUDIT.md") == []


def _audit(tmp_path, rows):
    p = tmp_path / "AUDIT.md"
    p.write_text("# audit\n| ID | 项 | 状态 |\n|---|---|---|\n"
                 + "".join(f"| {k} | x | `{v}` |\n" for k, v in rows.items()))
    return p


def test_gate_opens_only_when_every_row_is_done_or_deviate(tmp_path):
    assert R.audit_open_rows(_audit(tmp_path, {"A01": "done", "D01": "deviate"})) == []
    assert R.audit_open_rows(_audit(tmp_path, {"A01": "done", "A02": "align"})) == ["A02"]


def test_audit_row_with_two_status_cells_is_an_error(tmp_path):
    p = tmp_path / "AUDIT.md"
    p.write_text("| A01 | x | `open` | `done` |\n")
    with pytest.raises(R.RegistryError, match="恰好"):
        R.audit_rows(p)


def test_audit_without_rows_is_an_error_not_an_open_gate(tmp_path):
    p = tmp_path / "AUDIT.md"
    p.write_text("# 表格式被改坏了\n")
    with pytest.raises(R.RegistryError):
        R.audit_rows(p)


# ── materialize（合成登记表）──────────────────────────────────────────────────
def _synthetic(tmp_path, audit_rows=None, available=("S3",)):
    d = tmp_path / "study"
    d.mkdir()
    (d / "base.yaml").write_text(yaml.safe_dump({
        "seed": 1, "federation": {"n_edges": 2, "edge_rounds": 5, "n_clients": 100,
                                  "client_fraction": 0.1},
        "training": {"drift_correction": "hier_fedrep", "local_epochs": 5},
        "backdoor": {"malicious_strategy": "badpfl", "poison_ratio": 0.2,
                     "malicious_per_edge": [5, 5]},
        "defense": {"name": "none"}}))
    _audit(d, audit_rows or {"A01": "done"})
    (d / "registry.yaml").write_text(yaml.safe_dump({
        "study": "demo", "protocol": "P2", "layout": "nested", "base": "base.yaml",
        "audit": "AUDIT.md", "available": list(available),
        "groups": {
            "GA": {"requires": ["audit"], "seeds": [42, 43],
                   "factors": {"e": [{"label": "e2"},
                                     {"label": "e4", "set": {"federation.n_edges": 4}}]}},
            "GB": {"requires": ["audit", "S9"], "seeds": [42],
                   "cells": [{"cell": "x"}]},
        }}, sort_keys=False))
    return d


def test_materialize_writes_configs_whose_sha_main_py_would_reproduce(tmp_path):
    d = _synthetic(tmp_path)
    reg = R.Registry(d / "registry.yaml")
    rows = R.materialize(reg)
    assert [r["run_id"] for r in rows] == ["GA__e2__s42", "GA__e2__s43",
                                           "GA__e4__s42", "GA__e4__s43"]   # GB 缺 S9 → 不生成
    for r in rows:
        cfg_file = reg.configs_dir / f"{r['run_id']}.yaml"
        text = cfg_file.read_text()
        assert text.startswith("# GENERATED")
        cfg = yaml.safe_load(text)                 # 与 main.load_config 的读法相同
        assert config_sha(cfg) == r["config_sha"]
        assert cfg["seed"] == r["seed"]
        assert cfg["meta"] == {"study": "demo", "group": "GA", "run_id": r["run_id"],
                               "protocol": "P2"}
    e4 = yaml.safe_load((reg.configs_dir / "GA__e4__s42.yaml").read_text())
    assert e4["federation"]["n_edges"] == 4 and e4["federation"]["edge_rounds"] == 5
    idx = R.read_index(reg)
    assert set(idx) == {r["run_id"] for r in rows}
    assert idx["GA__e2__s42"]["metrics"].endswith("results/P2/GA/GA__e2__s42.metrics.json")


def test_materialize_with_nothing_eligible_errors_and_writes_nothing(tmp_path):
    d = _synthetic(tmp_path, available=())         # GA 只要 audit，但我们只要 GB（缺 S9）
    reg = R.Registry(d / "registry.yaml")
    with pytest.raises(R.RegistryError, match="GB 缺 S9"):
        R.materialize(reg, groups=["GB"])
    assert not reg.configs_dir.exists()


def test_materialize_real_v2_registry_only_unblocked_groups_are_generable(tmp_path, monkeypatch):
    """A4 定了 base 之后：不依赖功能会话的 G7、S8 之后的 G6、只改配置的 FLR（D-061）、S3 之后的 G3
    能生成配置（审计门槛只在 submit.sh 拦）；其余组缺 S4–S6，一个都不生成。写到临时目录，不在仓库里留 INDEX。
    S5 之后（D-084）：S5P 可生成；G2 仍被 `g2-scale` 挡着（规模未定），G1 仍缺 S6。"""
    reg = R.Registry(V2)
    monkeypatch.setattr(reg, "configs_dir", tmp_path / "configs")
    rows = R.materialize(reg)
    assert sorted(r["group"] for r in rows) == (["FLR"] * 3 + ["G0"] * 15 + ["G1P"] * 3 + ["G3"] * 24
                                               + ["G5"] * 15 + ["G5AB"] * 8 + ["G6"] * 9
                                               + ["G6D"] * 3 + ["G7"] * 6 + ["G8"] * 3 + ["G8F"] * 3
                                               + ["S5P"] * 4)
    with pytest.raises(R.RegistryError, match="不写 INDEX.tsv"):
        R.materialize(reg, groups=["G1", "G2"])


def test_g2_is_gated_on_its_scale_and_g1_on_s6():
    """S5 进了 available 之后，G2 只剩 `g2-scale`（用户定规模后才放行，D-083 / D-084）；G1 只剩 S6。"""
    reg = R.Registry(V2)
    assert "S5" in reg.available
    assert reg.unmet_requires("G2") == ["g2-scale"]
    assert reg.unmet_requires("G1") == ["S6"]
    assert reg.unmet_requires("S5P") == []
    assert "S6a" in reg.available and reg.unmet_requires("G1P") == []     # S6a：探路组放行，G1 仍缺 S6（= S6b）


def test_eligible_group_with_undecided_base_errors(tmp_path):
    d = _synthetic(tmp_path)
    text = (d / "registry.yaml").read_text().replace("base: base.yaml", "base: null")
    (d / "registry.yaml").write_text(text)
    reg = R.Registry(d / "registry.yaml")
    with pytest.raises(R.RegistryError, match="A4"):
        R.materialize(reg)
    assert not reg.configs_dir.exists()


def test_slug_rejects_double_underscore():
    with pytest.raises(R.RegistryError):
        R._slug("a__b")
    assert R._slug("e2-R10") == "e2-R10"


# ── 提交脚本 ─────────────────────────────────────────────────────────────────
def _main_py_invocation(path: Path) -> str:
    """把 `$PY main.py \\ ...` 这条（可能跨行的）命令拼成一行。"""
    lines = path.read_text().splitlines()
    for i, ln in enumerate(lines):
        if re.search(r"\$PY main\.py", ln) and not ln.lstrip().startswith("#"):
            cmd = ln
            while cmd.rstrip().endswith("\\"):
                i += 1
                cmd = cmd.rstrip()[:-1] + " " + lines[i]
            return cmd
    raise AssertionError(f"{path.name} 里找不到 $PY main.py")


def test_cell_sbatch_passes_no_overrides():
    cmd = _main_py_invocation(MECH / "cell.sbatch")
    flags = re.findall(r"--[\w-]+", cmd)
    assert flags == ["--config"], f"cell.sbatch 只许传 --config，实际：{flags}"


# 已知写死 `--defense none` 的旧脚本：D-007 决定先记录、另开会话（S1b）修。
KNOWN_HARDCODED_DEFENSE = {"exp3_cell.sbatch", "calib_cell.sbatch"}


def test_no_new_job_script_hardcodes_a_defense():
    """反向锚点：exp3_cell.sbatch 确实被抓到（F-001）；其余脚本只能传变量。"""
    scripts = [p for p in list(ROOT.glob("*.sh")) + list(ROOT.glob("experiments/**/*.sbatch"))
               + list(ROOT.glob("experiments/**/*.sh"))]
    hits = set()
    for p in scripts:
        for ln in p.read_text().splitlines():
            if ln.lstrip().startswith("#"):
                continue
            if re.search(r"--defense\s+(?![\"$])\S+", ln):
                hits.add(p.name)
    assert "exp3_cell.sbatch" in hits              # 守卫确实抓得到
    assert hits == KNOWN_HARDCODED_DEFENSE, (
        f"又有脚本在命令行写死防御（会静默盖掉 yaml，F-001）：{hits - KNOWN_HARDCODED_DEFENSE}")


@pytest.fixture
def mech_copy(tmp_path):
    d = _synthetic(tmp_path)
    shutil.copy(MECH / "submit.sh", d / "submit.sh")
    shutil.copy(MECH / "submit_lib.sh", d / "submit_lib.sh")
    reg = R.Registry(d / "registry.yaml")
    rows = R.materialize(reg)
    return d, reg, rows


def _submit(d, *args, **env):
    e = {**os.environ, "TFDPFL_MECH_DIR": str(d), "TFDPFL_ROOT": str(R.ROOT),
         "PATH": "/usr/bin:/bin", **env}             # PATH 里没有 sbatch：真去提交就会报错
    return subprocess.run(["bash", str(d / "submit.sh"), *args], capture_output=True,
                          text=True, env=e)


def _write_metrics(reg, run_id, sha, rc=0):
    p = R.ROOT / R.read_index(reg)[run_id]["metrics"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f'{{"exit_code": {rc}, "run": {{"provenance": {{"config_sha": "{sha}"}}}}}}')


def test_submit_refuses_while_audit_is_open(tmp_path):
    d = _synthetic(tmp_path, audit_rows={"A01": "done", "A02": "open"})
    shutil.copy(MECH / "submit.sh", d / "submit.sh")
    shutil.copy(MECH / "submit_lib.sh", d / "submit_lib.sh")
    R.materialize(R.Registry(d / "registry.yaml"))
    out = _submit(d)
    assert out.returncode == 0, out.stderr
    assert "1 行未关闭" in out.stdout and "本次入队=0" in out.stdout
    assert "sbatch: command not found" not in out.stderr


def test_submit_done_requires_matching_config_sha(mech_copy):
    d, reg, rows = mech_copy
    sha = {r["run_id"]: r["config_sha"] for r in rows}
    _write_metrics(reg, "GA__e2__s42", sha["GA__e2__s42"])           # 完成
    _write_metrics(reg, "GA__e2__s43", "000000000000")               # 配置已变 → stale（D-053：不算完成）
    _write_metrics(reg, "GA__e4__s42", sha["GA__e4__s42"], rc=1)     # 跑挂了
    out = _submit(d, "--status")
    lines = dict(ln.split()[::-1] for ln in out.stdout.splitlines()
                 if ln.startswith(("  done", "  todo", "  stale")))
    assert lines == {"GA__e2__s42": "done", "GA__e2__s43": "stale",
                     "GA__e4__s42": "todo", "GA__e4__s43": "todo"}


def test_submit_run_groups_filter(mech_copy):
    d, _, _ = mech_copy
    out = _submit(d, "--dry-run", RUN_GROUPS="NOPE")
    assert "run 总数=0" in out.stdout
    out = _submit(d, "--dry-run", RUN_GROUPS="GA")
    assert "run 总数=4" in out.stdout


# ══════════════════════════════════════════════════════════════════════════
# D-047：G2 的声明配置要能跑（布点 / 轮数 / 评估网格）；G4 搁置
# ══════════════════════════════════════════════════════════════════════════
def _g2_configs():
    reg = R.Registry(MECH / "registry.yaml")
    return [(run["cell"], reg.declared_config(run)) for run in reg.runs() if run["group"] == "G2"]


def test_g2_placement_matches_the_number_of_edges():
    """G2 的 set 曾经只写 n_edges / edge_rounds —— 所有格子都继承了 [5, 5]。"""
    for cell, c in _g2_configs():
        mpe = c["backdoor"]["malicious_per_edge"]
        assert len(mpe) == c["federation"]["n_edges"], (cell, mpe)
        assert sum(mpe) == 10, (cell, mpe)


def test_g2_round_budget_reaches_the_cap():
    """n_rounds 一律 60 时，flat（R=1）会被截断在 60 有效轮 —— 低于地板 150。"""
    for cell, c in _g2_configs():
        f = c["federation"]
        assert f["n_rounds"] * f["edge_rounds"] >= c["stopping"]["cap_effective"], cell


def test_g2_flat_is_evaluated_on_the_same_grid_as_r5():
    """flat 的评估间隔与旧 P1 flat 一样取 5 个云轮（= 5 有效轮），否则 flat 每轮都评、开销 5 倍。"""
    grid = {cell: c["federation"]["edge_rounds"] * c["backdoor"]["eval_interval"]
            for cell, c in _g2_configs()}
    assert grid["flat"] == grid["e2-R5"] == grid["e4-R5"] == grid["e10-R5"] == 5


def test_every_g2_cell_is_on_the_same_effective_round_grid():
    """S5（D-084）：全部 11 格开 eval_grid 5，两个 eval_interval = lcm(5, R) / R →
    全量 + 轻评估点都在 5 的倍数有效轮上（R2 不再每 2 有效轮全量评一次，R20 不再每 20 才一点）。"""
    from math import lcm
    cells = _g2_configs()
    assert len({cell for cell, _ in cells}) == 11
    for cell, c in cells:
        R_ = c["federation"]["edge_rounds"]
        assert c["evaluation"]["eval_grid"] == 5, cell
        assert c["backdoor"]["eval_interval"] == c["evaluation"]["eval_interval"] == lcm(5, R_) // R_, cell


def test_g4_is_parked_until_3_2_is_reformulated():
    reg = R.Registry(MECH / "registry.yaml")
    assert "reformulate-3.2" in reg.unmet_requires("G4")


# ── S4（2026-09-29）：G0 固定长度、G5 的窗口、G5AB / G8F ─────────────────────────
def test_g0_is_fixed_length_like_flr():
    """ρ=0 时「越过阈值」永远不成立 → 自适应判据必然跑满，cap_reached 是假的删失（D-078）。"""
    for r in R.Registry(V2).runs():
        if r["group"] == "G0":
            assert r["set"]["stopping"] is None and r["set"]["federation.n_rounds"] == 60, r["run_id"]


def test_g5_windows_follow_t0_and_use_the_schedule_g5ab_chose():
    """G5 挂 `g5-schedule`（D-079），G5AB 判 insensitive 后放行、用 A = window（D-080）。"""
    reg = R.Registry(V2)
    assert "g5-schedule" in reg.groups["G5"]["requires"] and "g5-schedule" in reg.available
    cells = {r["cell"]: r["set"] for r in reg.runs() if r["group"] == "G5" and r["seed"] == 42}
    assert sorted(cells, key=lambda c: int(c[1:])) == ["t20", "t60", "t100", "t140", "t180"]
    for cell, s in cells.items():
        t0 = int(cell[1:])
        assert s["backdoor.attack_start_round"] == t0 // 5 + 1
        assert s["backdoor.attack_stop_round"] == s["backdoor.attack_start_round"] + 4
        assert s["federation.n_rounds"] == (t0 + 75) // 5
        assert s["stopping"] is None and s["backdoor.malicious_per_edge"] == [10, 0, 0, 0]
        assert s["federation.partition"] == "equal_random"
        assert s["backdoor.generator_schedule"] == "window"


def test_g5ab_is_g5_at_two_t0_with_both_schedules():
    runs = [r for r in R.Registry(V2).runs() if r["group"] == "G5AB"]
    assert {r["seed"] for r in runs} == {42, 43}
    g5 = {r["cell"]: r["set"] for r in R.Registry(V2).runs() if r["group"] == "G5" and r["seed"] == 42}
    for r in runs:
        t0, arm = r["cell"].split("-")
        s, g = dict(r["set"]), dict(g5[t0])
        assert s.pop("backdoor.generator_schedule") == {"A": "window", "B": "always"}[arm]
        assert g.pop("backdoor.generator_schedule") == "window"
        assert s == g, r["run_id"]

