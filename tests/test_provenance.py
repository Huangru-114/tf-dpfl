"""
tests/test_provenance.py  —  `[Provenance]` 行：上游打印、下游解析，两侧分开测

为什么要这一行：FINDINGS F-001 / F-011（experiments/attack/hfl-mechanism/）。
6 个 def_* 格子的 yaml 写着 median / multi_krum，`exp3_cell.sbatch` 的
`--defense none` 静默盖掉了它，metrics.json 里没有任何痕迹。

纯标准库 + pyyaml，本地秒级，不需要 TF。
"""

import ast
import io
import json
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

from utils import provenance as P                  # noqa: E402  (conftest 已把 fedavg/ 放进 sys.path)
from collect_metrics import collect, SCHEMA_VERSION  # noqa: E402

MAIN_PY = ROOT / "fedavg" / "main.py"


# ── config_sha / flat_diff ──────────────────────────────────────────────────
def test_config_sha_ignores_key_order_but_not_values():
    a = {"x": 1, "y": {"b": 2, "a": [1, 2]}}
    b = {"y": {"a": [1, 2], "b": 2}, "x": 1}
    assert P.config_sha(a) == P.config_sha(b)
    assert P.config_sha(a) != P.config_sha({"x": 1, "y": {"b": 3, "a": [1, 2]}})
    assert len(P.config_sha(a)) == 12


def test_flat_diff_reports_exactly_the_changed_leaves():
    before = {"seed": 42, "defense": {"name": "median", "params": {"f": 1}},
              "training": {"lr": 0.1}}
    after = {"seed": 43, "defense": {"name": "none", "params": {"f": 1}},
             "training": {"lr": 0.1, "new": True}}
    assert P.flat_diff(before, after) == [
        ["defense.name", "median", "none"],
        ["seed", 42, 43],
        ["training.new", "<missing>", True],
    ]
    assert P.flat_diff(before, before) == []


def test_def_cell_override_is_caught():
    """反向锚点：正是 F-001 那一格 —— yaml median，CLI 盖成 none。"""
    declared = {"defense": {"name": "median"}}
    actual = {"defense": {"name": "none"}}
    assert P.flat_diff(declared, actual) == [["defense.name", "median", "none"]]


# ── git（只读 .git 文件）─────────────────────────────────────────────────────
def _fake_repo(tmp_path, head, loose=None, packed=None):
    g = tmp_path / ".git"
    (g / "refs" / "heads").mkdir(parents=True)
    (g / "HEAD").write_text(head + "\n")
    for ref, sha in (loose or {}).items():
        f = g / ref
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(sha + "\n")
    if packed:
        (g / "packed-refs").write_text(
            "# pack-refs with: peeled fully-peeled sorted\n"
            + "".join(f"{sha} {ref}\n" for ref, sha in packed.items()))
    return tmp_path


def test_git_commit_from_loose_ref(tmp_path):
    root = _fake_repo(tmp_path, "ref: refs/heads/main", loose={"refs/heads/main": "a" * 40})
    assert P.git_commit(root) == ("a" * 40, "main")


def test_git_commit_from_packed_refs(tmp_path):
    root = _fake_repo(tmp_path, "ref: refs/heads/claude/x",
                      packed={"refs/heads/claude/x": "b" * 40})
    assert P.git_commit(root) == ("b" * 40, "claude/x")


def test_git_commit_detached_head(tmp_path):
    root = _fake_repo(tmp_path, "c" * 40)
    assert P.git_commit(root) == ("c" * 40, None)


def test_git_commit_no_repo_is_none_not_a_guess(tmp_path):
    assert P.git_commit(tmp_path) == (None, None)


@pytest.mark.skipif(shutil.which("git") is None, reason="需要 git 可执行文件做交叉核对")
def test_git_commit_matches_git_rev_parse_on_this_repo():
    sha, _ = P.git_commit(ROOT)
    ref = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    assert sha == ref


# ── 打印 → 解析 round-trip ──────────────────────────────────────────────────
def _fields(**over):
    f = {"protocol": "P1", "git": "abc123def456", "branch": "main", "dirty": "0",
         "config_sha": "0123456789ab", "study": "hfl-mechanism", "group": "G2",
         "run_id": "G2__flat__s42", "host": "node1", "job": "12345",
         "start": "2026-09-24T12:00:00Z", "cli_overrides": []}
    f.update(over)
    return f


def test_roundtrip_all_fields():
    ov = [["defense.name", "median", "none"], ["seed", 42, 43]]
    line = P.format_provenance(_fields(cli_overrides=ov))
    got = P.parse_provenance(line)
    assert got == _fields(cli_overrides=ov)


def test_none_fields_roundtrip_as_none_and_empty_overrides_as_empty_list():
    line = P.format_provenance(_fields(git=None, job=None, run_id=None))
    got = P.parse_provenance(line)
    assert got["git"] is None and got["job"] is None and got["run_id"] is None
    assert got["cli_overrides"] == []          # 「没有覆盖」，不是「不知道」


def test_separator_inside_override_value_does_not_break_fields():
    ov = [["federation.note", "a | b", "c|d"]]
    got = P.parse_provenance(P.format_provenance(_fields(cli_overrides=ov)))
    assert got["cli_overrides"] == ov
    assert got["run_id"] == "G2__flat__s42"


def test_unknown_future_field_does_not_blank_known_ones():
    """与 `[设定]` 的全或无正则不同：多一个字段，已有字段照样读得出来。"""
    line = P.format_provenance(_fields()).replace(" | start=", " | future=x | start=")
    got = P.parse_provenance(line)
    assert got["start"] == "2026-09-24T12:00:00Z" and got["future"] == "x"


def test_non_provenance_line_is_none():
    assert P.parse_provenance("[设定] client_fraction=0.1") is None


def test_print_provenance_prints_overrides_and_the_line(tmp_path):
    cfg = {"meta": {"study": "s", "group": "G0", "run_id": "G0__x__s42"}}
    buf = io.StringIO()
    with redirect_stdout(buf):
        fields = P.print_provenance(cfg, declared_sha="deadbeef0000",
                                    cli_overrides=[["defense.name", "median", "none"]],
                                    repo_root=tmp_path)
    out = buf.getvalue().splitlines()
    assert out[0] == "[Config] CLI 覆盖 yaml: defense.name 'median' → 'none'"
    got = P.parse_provenance(out[-1])
    assert got["run_id"] == "G0__x__s42" and got["group"] == "G0"
    assert got["config_sha"] == "deadbeef0000"
    assert got["protocol"] == P.PROTOCOL_VERSION
    assert got["git"] is None                   # tmp_path 不是仓库
    assert fields["cli_overrides"] == [["defense.name", "median", "none"]]


# ── 下游：collect_metrics ───────────────────────────────────────────────────
def test_collect_puts_provenance_into_run_block():
    line = P.format_provenance(_fields(cli_overrides=[["defense.name", "median", "none"]]))
    m = collect("[Config] loading x.yaml\n" + line + "\n[Round 1]\n")
    assert m["schema_version"] == SCHEMA_VERSION == 2
    prov = m["run"]["provenance"]
    assert prov["run_id"] == "G2__flat__s42" and prov["protocol"] == "P1"
    assert "cli_overrides" not in prov
    assert m["run"]["cli_overrides"] == [["defense.name", "median", "none"]]


def test_old_log_without_provenance_gives_none_not_empty():
    m = collect("[Config] loading x.yaml\n[Round 1]\n")
    assert m["run"]["provenance"] is None
    assert m["run"]["cli_overrides"] is None    # 不知道 ≠ 没有覆盖


# ── 上游：main.load_config 真的接上了（AST，main.py import TF，本地跑不了）──
def _load_config_fn():
    tree = ast.parse(MAIN_PY.read_text())
    return next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "load_config")


def _first_line_calling(fn, name):
    lines = [n.lineno for n in ast.walk(fn)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]
    return min(lines) if lines else None


def test_load_config_hashes_the_yaml_before_any_cli_change():
    fn = _load_config_fn()
    sha_at = _first_line_calling(fn, "config_sha")
    apply_at = _first_line_calling(fn, "apply_experiment_args")
    assert sha_at is not None, "load_config 没有计算 config_sha"
    assert sha_at < apply_at, "config_sha 必须在 CLI 改动之前算（它代表 yaml 原文）"


def test_load_config_diffs_after_all_overrides_and_prints_provenance():
    fn = _load_config_fn()
    diff_at = _first_line_calling(fn, "flat_diff")
    prov_at = _first_line_calling(fn, "print_provenance")
    assert diff_at is not None and prov_at is not None
    # --override 循环在 flat_diff 之前 → 它的改动也被记下
    for_loops = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.For)
                 and "override" in ast.unparse(n.iter)]
    assert for_loops and max(for_loops) < diff_at
