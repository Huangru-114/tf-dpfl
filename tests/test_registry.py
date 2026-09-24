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
    assert sizes == {"G0": 15, "G1": 24, "G2": 55, "G3": 21, "G4": 12, "G5": 15, "G6": 9}
    assert len({r["run_id"] for r in runs}) == len(runs) == 151


def test_v2_run_ids_and_factor_settings():
    runs = {r["run_id"]: r for r in R.Registry(V2).runs()}
    r = runs["G1__random_collocated_R10__s42"]
    assert r["set"] == {"federation.n_edges": 4, "federation.edge_rounds": 10}
    assert runs["G4__p1.0_fedavg__s44"]["set"] == {
        "backdoor.poison_ratio": 1.0, "training.drift_correction": "hierfedavg"}
    assert runs["G2__flat__s46"]["set"] == {"federation.n_edges": 1,
                                            "federation.edge_rounds": 1}


def test_v2_base_is_not_decided_yet_so_config_cannot_be_generated():
    reg = R.Registry(V2)
    assert reg.base is None
    with pytest.raises(R.RegistryError, match="A4"):
        reg.declared_config(reg.runs()[0])


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
def test_real_audit_parses_and_is_open():
    rows = R.audit_rows(MECH / "AUDIT.md")
    assert set(rows.values()) <= set(R.AUDIT_STATUSES)
    assert rows["A02"] == "align"                  # D-004
    assert {f"A{i:02d}" for i in range(1, 22)} | {f"D{i:02d}" for i in range(1, 7)} == set(rows)
    assert R.audit_open_rows(MECH / "AUDIT.md")    # 现在理应没关


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
    R.materialize(R.Registry(d / "registry.yaml"))
    out = _submit(d)
    assert out.returncode == 0, out.stderr
    assert "1 行未关闭" in out.stdout and "本次入队=0" in out.stdout
    assert "sbatch: command not found" not in out.stderr


def test_submit_done_requires_matching_config_sha(mech_copy):
    d, reg, rows = mech_copy
    sha = {r["run_id"]: r["config_sha"] for r in rows}
    _write_metrics(reg, "GA__e2__s42", sha["GA__e2__s42"])           # 完成
    _write_metrics(reg, "GA__e2__s43", "000000000000")               # 配置已变 → 待重跑
    _write_metrics(reg, "GA__e4__s42", sha["GA__e4__s42"], rc=1)     # 跑挂了
    out = _submit(d, "--status")
    lines = dict(ln.split()[::-1] for ln in out.stdout.splitlines()
                 if ln.startswith("  done") or ln.startswith("  todo"))
    assert lines == {"GA__e2__s42": "done", "GA__e2__s43": "todo",
                     "GA__e4__s42": "todo", "GA__e4__s43": "todo"}


def test_submit_run_groups_filter(mech_copy):
    d, _, _ = mech_copy
    out = _submit(d, "--dry-run", RUN_GROUPS="NOPE")
    assert "run 总数=0" in out.stdout
    out = _submit(d, "--dry-run", RUN_GROUPS="GA")
    assert "run 总数=4" in out.stdout
