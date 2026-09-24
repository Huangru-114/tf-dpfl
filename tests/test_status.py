"""
tests/test_status.py  —  声明 vs 实际的对账

真实数据锚点：旧方案 P1 批次（26 个 metrics.json）在 v1 登记表下必须恰好得到
  6 mismatch（def_*：声明防御，实际 none —— F-001）
  1 failed  （rho0：没有 exit_code）
  1 orphan  （stoptest_10edge：登记表里没有）
  18 done
这是「改动前就存在、但没有任何工具报出来」的问题，现在由对账工具报出来。

纯标准库 + pyyaml，本地秒级。
"""

import json
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import registry as R                     # noqa: E402
import status as S                       # noqa: E402

V1 = ROOT / "experiments" / "attack" / "hfl-propagation" / "registry" / "v1.yaml"
V2 = ROOT / "experiments" / "attack" / "hfl-mechanism" / "registry.yaml"

DEF_CELLS = {f"def_{d}_{t}_seed42" for d in ("median", "multi_krum")
             for t in ("flat_baseline", "2edge_distributed", "4edge_distributed")}


# ── 真实数据 ─────────────────────────────────────────────────────────────────
def test_v1_real_data_counts_exactly():
    rep = S.classify(R.Registry(V1))
    assert rep["counts"] == {"todo": 0, "blocked": 0, "failed": 1, "mismatch": 6,
                             "stale": 0, "done": 18, "orphan": 1}


def test_v1_mismatches_are_exactly_the_def_cells_and_name_the_defense():
    rep = S.classify(R.Registry(V1))
    mism = {r["run_id"]: r["detail"] for r in rep["runs"] if r["status"] == "mismatch"}
    assert set(mism) == DEF_CELLS
    for rid, detail in mism.items():
        want = "median" if "median" in rid else "multi_krum"
        assert detail == f"defense: 声明 {want!r} / 实际 'none'"


def test_v1_failed_and_orphan_are_the_known_files():
    rep = S.classify(R.Registry(V1))
    failed = [r for r in rep["runs"] if r["status"] == "failed"]
    assert [(r["run_id"], r["detail"]) for r in failed] == [
        ("rho0_2edge_distributed_seed42", "没有 exit_code")]
    assert rep["orphans"] == [
        "experiments/attack/hfl-propagation/results/stoptest_10edge_seed42.metrics.json"]


def test_v1_old_files_are_done_but_unverified():
    rep = S.classify(R.Registry(V1))
    done = [r for r in rep["runs"] if r["status"] == "done"]
    assert done and all(r["verified"] is False for r in done)


def test_v2_everything_is_blocked_right_now():
    rep = S.classify(R.Registry(V2))
    assert rep["counts"]["blocked"] == 151
    assert all("audit(" in r["detail"] for r in rep["runs"])


def test_main_exit_code_is_nonzero_on_mismatch(capsys):
    assert S.main([str(V1), "--quiet"]) == 1
    assert S.main([str(V2), "--quiet"]) == 0


# ── 合成场景：每个状态各一格 ──────────────────────────────────────────────────
@pytest.fixture
def study(tmp_path):
    d = tmp_path / "study"
    d.mkdir()
    (d / "base.yaml").write_text(yaml.safe_dump({
        "federation": {"n_edges": 2, "edge_rounds": 5, "n_clients": 100,
                       "client_fraction": 0.1},
        "training": {"drift_correction": "hier_fedrep", "local_epochs": 5},
        "backdoor": {"malicious_strategy": "badpfl", "poison_ratio": 0.2,
                     "malicious_per_edge": [5, 5]},
        "defense": {"name": "none"}}))
    (d / "AUDIT.md").write_text("| A01 | x | `done` |\n")
    cells = ["done", "unverified", "failed", "noexit", "mismatch", "stale", "oldproto", "todo"]
    (d / "registry.yaml").write_text(yaml.safe_dump({
        "study": "demo", "protocol": "P2", "layout": "nested", "base": "base.yaml",
        "audit": "AUDIT.md", "available": [],
        "groups": {
            "GA": {"requires": ["audit"], "seeds": [1],
                   "cells": [{"cell": c} for c in cells]},
            "GB": {"requires": ["audit", "S9"], "seeds": [1], "cells": [{"cell": "blk"}]},
        }}, sort_keys=False))
    reg = R.Registry(d / "registry.yaml")
    rows = {r["run_id"]: r for r in R.materialize(reg)}

    def write(cell, *, rc=0, prov=True, protocol="P2", sha=None, **run_over):
        rid = f"GA__{cell}__s1"
        run = {"method": "hier_fedrep", "defense": "none", "attack": "badpfl",
               "n_clients": 100, "n_edges": 2, "edge_rounds": 5, "client_fraction": 0.1,
               "poison_ratio": 0.2, "malicious_per_edge": [5, 5], "local_epochs": 5,
               "seed": 1, **run_over}
        if prov:
            run["provenance"] = {"protocol": protocol,
                                 "config_sha": sha or rows[rid]["config_sha"]}
        m = {"run": run}
        if rc is not None:
            m["exit_code"] = rc
        p = Path(rows[rid]["metrics"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(m))

    write("done")
    write("unverified", prov=False)
    write("failed", rc=1)
    write("noexit", rc=None)
    write("mismatch", n_edges=4)
    write("stale", sha="ffffffffffff")
    write("oldproto", protocol="P1")
    orphan = reg.results_dir / "P2" / "GA" / "GA__ghost__s1.metrics.json"
    orphan.write_text("{}")
    return reg


def test_every_status_is_reached_exactly_where_expected(study):
    rep = S.classify(study)
    got = {r["cell"]: r["status"] for r in rep["runs"]}
    assert got == {"done": "done", "unverified": "done", "failed": "failed",
                   "noexit": "failed", "mismatch": "mismatch", "stale": "stale",
                   "oldproto": "stale", "todo": "todo", "blk": "blocked"}
    assert rep["counts"] == {"todo": 1, "blocked": 1, "failed": 2, "mismatch": 1,
                             "stale": 2, "done": 2, "orphan": 1}
    by = {r["cell"]: r for r in rep["runs"]}
    assert by["done"]["verified"] is True and by["unverified"]["verified"] is False
    assert by["mismatch"]["detail"] == "n_edges: 声明 2 / 实际 4"
    assert "config_sha ffffffffffff" in by["stale"]["detail"]
    assert "口径版本 P1" in by["oldproto"]["detail"]
    assert by["blk"]["detail"] == "缺: S9"


def test_numeric_comparison_is_tolerant_but_not_sloppy():
    assert S._same(0.1, 0.1000000000001) and S._same(5, 5.0) and S._same([5, 5], [5.0, 5])
    assert not S._same(0.1, 0.2) and not S._same([5, 5], [5]) and not S._same(True, 1)
    assert S.run_block_mismatches({"a": None, "b": 1}, {"b": 2}) == [["b", 1, 2]]
