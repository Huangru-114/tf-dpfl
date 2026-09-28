"""
tests/test_pack_submit.py  —  一卡多跑接进提交脚本（D-052）+ stale 不重交（D-053）

核心要求：**不触发 OOM、尽量省机时**。
  · PACK 未设 → 与以前逐字相同（一卡一跑，cell.sbatch）；
  · 同一个包只放同一格子的不同 seed；格子第一次先交 PROBE_K（缺省 3，D-069）的首包，其余 held；
  · 探路回来后按 gpu.json 的真实峰值定 K；OOM 过 → 降一档；
  · pack.sbatch 把被吞掉的 OOM 判成非 0 退出码（run 照样 exit 0 但结果无效）；
  · exit 0 但 config_sha 不符 → stale，默认不重交；
  · 余数跨格子合包（D-060）：只合余数、只合真实峰值相近且无 OOM 的格子、只在作业数变少时合。

纯 bash + 标准库 + pyyaml：合成一个 2 格 × 5 seed 的研究，PATH 上放一个 sbatch 替身记录参数。
"""

import json
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
import registry as R                                # noqa: E402
from collect_metrics import collect                 # noqa: E402

MECH = ROOT / "experiments" / "attack" / "hfl-mechanism"
SEEDS = [42, 43, 44, 45, 46]


# ══════════════════════════════════════════════════════════════════════════
# 合成研究 + sbatch 替身
# ══════════════════════════════════════════════════════════════════════════
def _study(d: Path, audit_status="done", cells=("e2", "e4"), seeds=SEEDS):
    d.mkdir(parents=True, exist_ok=True)
    (d / "base.yaml").write_text(yaml.safe_dump({
        "seed": 1, "federation": {"n_edges": 2, "edge_rounds": 5, "n_clients": 100},
        "defense": {"name": "none"}}))
    (d / "AUDIT.md").write_text(f"| A01 | x | `{audit_status}` |\n")
    (d / "registry.yaml").write_text(yaml.safe_dump({
        "study": "demo", "protocol": "P2", "layout": "nested", "base": "base.yaml",
        "audit": "AUDIT.md", "available": [],
        "groups": {"GA": {"requires": ["audit"], "seeds": list(seeds),
                          "factors": {"e": [{"label": c, "set": {"federation.edge_rounds": i + 1}}
                                            for i, c in enumerate(cells)]}}},
    }, sort_keys=False))
    reg = R.Registry(d / "registry.yaml")
    rows = R.materialize(reg)
    return reg, {r["run_id"]: r for r in rows}


@pytest.fixture
def study(tmp_path):
    d = tmp_path / "study"
    reg, rows = _study(d)
    for f in ("submit.sh", "submit_lib.sh"):
        shutil.copy(MECH / f, d / f)
    return d, reg, rows


@pytest.fixture
def fake_sbatch(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "sbatch_calls.txt"
    (bindir / "sbatch").write_text(
        "#!/bin/bash\n"
        f'printf "%s\\t" "$@" >> "{log}"; echo >> "{log}"\n'
        'echo "Submitted batch job 1"\n')
    (bindir / "sbatch").chmod(0o755)
    return bindir, log


def _run(script: Path, *args, bindir=None, mech_dir=None, **env):
    e = {**os.environ, "TFDPFL_ROOT": str(ROOT),
         "PATH": (f"{bindir}:" if bindir else "") + "/usr/bin:/bin"}
    for k in ("PACK", "PROBE_K", "PACK_MEM_PCT", "PACK_CTX_MIB", "RESUBMIT_STALE", "RUN_GROUPS",
              "PACK_MIX", "PACK_MIX_TOL_PCT"):
        e.pop(k, None)
    if mech_dir is not None:
        e["TFDPFL_MECH_DIR"] = str(mech_dir)
    e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=e)


def _calls(log: Path):
    if not log.exists():
        return []
    return [ln.rstrip("\t").split("\t") for ln in log.read_text().splitlines() if ln.strip()]


def _pack_calls(log: Path):
    """[(n, tag, [run_id…])]：sbatch -c 4n --job-name=… pack.sbatch <tag> (cfg rid met)×n"""
    out = []
    for c in _calls(log):
        if not any(a.endswith("pack.sbatch") for a in c):
            continue
        i = next(j for j, a in enumerate(c) if a.endswith("pack.sbatch"))
        tag, triples = c[i + 1], c[i + 2:]
        assert len(triples) % 3 == 0
        rids = triples[1::3]
        assert c[0] == "-c" and int(c[1]) == 4 * len(rids), c
        assert c[2] == f"--job-name=exp3v2-pack{len(rids)}"
        out.append((len(rids), tag, rids))
    return out


def _metrics_path(reg, run_id):
    return Path(R.read_index(reg)[run_id]["metrics"])


def _done(reg, rows, run_id, rc=0, sha=None):
    p = _metrics_path(reg, run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"exit_code": rc, "run": {"provenance": {
        "config_sha": sha or rows[run_id]["config_sha"]}}}))


def _gpu_json(reg, cell, k, first_seed=42, **fields):
    d = _metrics_path(reg, f"GA__{cell}__s42").parent
    d.mkdir(parents=True, exist_ok=True)
    base = {"k": k, "runs": [], "wall_s": 1, "n_samples": 3, "util_mean": 50.0, "util_max": 90.0,
            "mem_max_mib": None, "mem_total_mib": None, "run_peak_max_mib": None,
            "n_oom": 0, "n_mem_warnings": 0, "run_peak_mib": [], "oom_runs": []}
    base.update(fields)
    (d / f"GA__{cell}__pack-k{k}-s{first_seed}.gpu.json").write_text(json.dumps(base, indent=2))


# ══════════════════════════════════════════════════════════════════════════
# PACK 未设：逐字等于以前
# ══════════════════════════════════════════════════════════════════════════
def test_pack_unset_is_one_cell_sbatch_per_run(study, fake_sbatch):
    d, reg, rows = study
    bindir, log = fake_sbatch
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d)
    assert out.returncode == 0, out.stderr
    calls = _calls(log)
    assert len(calls) == 10
    idx = R.read_index(reg)
    for c in calls:
        assert c[0].endswith("/cell.sbatch") and len(c) == 4, c      # 没有 -c、没有 pack
        rid = c[2]
        assert c[1] == idx[rid]["config"] and c[3] == idx[rid]["metrics"]
    assert "本次入队=10" in out.stdout


def test_pack_one_is_the_same_as_unset(study, fake_sbatch):
    d, _, _ = study
    bindir, log = fake_sbatch
    _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=1)
    assert len(_calls(log)) == 10 and not _pack_calls(log)


# ══════════════════════════════════════════════════════════════════════════
# 探路：每格一个 PROBE_K 的包，其余 held。缺省 PROBE_K=3（D-069：G6 已实测 K=3 放得下）；
# 显存没测过的新配置类型显式写 PROBE_K=2（D-048 / D-052 原规则）。
# ══════════════════════════════════════════════════════════════════════════
def test_first_submission_is_one_k3_pack_per_cell_by_default(study, fake_sbatch):
    d, _, _ = study
    bindir, log = fake_sbatch
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    assert out.returncode == 0, out.stderr
    packs = _pack_calls(log)
    assert packs == [(3, "GA__e2__pack-k3-s42", ["GA__e2__s42", "GA__e2__s43", "GA__e2__s44"]),
                     (3, "GA__e4__pack-k3-s42", ["GA__e4__s42", "GA__e4__s43", "GA__e4__s44"])]
    assert "本次入队=6" in out.stdout and "held=4" in out.stdout
    assert len(_calls(log)) == 2                                   # 没有漏交成单跑


def test_probe_k2_is_still_available_for_untested_config_types(study, fake_sbatch):
    """D-069 之前的缺省；10 edge / R20 这类显存没测过的配置类型第一次交时显式写 PROBE_K=2。"""
    d, _, _ = study
    bindir, log = fake_sbatch
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3, PROBE_K=2)
    assert out.returncode == 0, out.stderr
    packs = _pack_calls(log)
    assert packs == [(2, "GA__e2__pack-k2-s42", ["GA__e2__s42", "GA__e2__s43"]),
                     (2, "GA__e4__pack-k2-s42", ["GA__e4__s42", "GA__e4__s43"])]
    assert "本次入队=4" in out.stdout and "held=6" in out.stdout


def test_three_seed_cell_is_submitted_whole_by_default(study, fake_sbatch):
    """FLR / G3 的形状：每格 3 个 seed → 一个 K=3 包交完，没有 held（探路 K=2 时要交两次）。"""
    d, reg, rows = study
    bindir, log = fake_sbatch
    for cell in ("e2", "e4"):
        for s in (45, 46):
            _done(reg, rows, f"GA__{cell}__s{s}")                   # 每格只剩 42 / 43 / 44
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    assert [p[0] for p in _pack_calls(log)] == [3, 3]
    assert "held=0" in out.stdout


def test_probe_k_is_capped_by_pack_and_by_pending_runs(study, fake_sbatch):
    d, reg, rows = study
    bindir, log = fake_sbatch
    for s in SEEDS[:4]:
        _done(reg, rows, f"GA__e2__s{s}")                          # e2 只剩 1 个
    _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3, PROBE_K=2)
    packs = _pack_calls(log)
    assert (1, "GA__e2__pack-k1-s46", ["GA__e2__s46"]) in packs


# ══════════════════════════════════════════════════════════════════════════
# 探路回来：按实测定 K；不跨格混包；OOM 降档
# ══════════════════════════════════════════════════════════════════════════
def _probe_done(reg, rows, cell):
    for s in (42, 43):
        _done(reg, rows, f"GA__{cell}__s{s}")


def test_measured_peak_allows_k3_for_the_remaining_three(study, fake_sbatch):
    d, reg, rows = study
    bindir, log = fake_sbatch
    _probe_done(reg, rows, "e2")
    # 97871 × 85% / (20000 + 1024) = 3.96 → 3
    _gpu_json(reg, "e2", 2, mem_total_mib=97871, run_peak_max_mib=20000.0, mem_max_mib=60000)
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    packs = _pack_calls(log)
    assert (3, "GA__e2__pack-k3-s44", ["GA__e2__s44", "GA__e2__s45", "GA__e2__s46"]) in packs
    assert (3, "GA__e4__pack-k3-s42", ["GA__e4__s42", "GA__e4__s43", "GA__e4__s44"]) in packs  # e4 仍在探路
    assert "GA__e2：K=3" in out.stdout


def test_measured_peak_too_large_drops_to_k2(study, fake_sbatch):
    d, reg, rows = study
    bindir, log = fake_sbatch
    _probe_done(reg, rows, "e2")
    # 97871 × 85% / (40000 + 1024) = 2.03 → 2
    _gpu_json(reg, "e2", 2, mem_total_mib=97871, run_peak_max_mib=40000.0)
    _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    e2 = [p for p in _pack_calls(log) if "__e2__" in p[1]]
    assert [p[0] for p in e2] == [2, 1]


def test_without_tf_peak_falls_back_to_whole_gpu_reading_per_run(study, fake_sbatch):
    """没有 [GPUMem]（旧日志）→ 用整卡 memory.used / k（偏大 → 偏保守）。
    F-048 的 DET K=3 读数 96076 / 3 = 32025 → 97871 × 85% / 33049 = 2.5 → 2。"""
    d, reg, rows = study
    bindir, log = fake_sbatch
    _probe_done(reg, rows, "e2")
    _gpu_json(reg, "e2", 3, mem_total_mib=97871, mem_max_mib=96076)
    _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    e2 = [p for p in _pack_calls(log) if "__e2__" in p[1]]
    assert [p[0] for p in e2] == [2, 1]


def test_without_any_memory_reading_never_exceeds_the_proven_k(study, fake_sbatch):
    d, reg, rows = study
    bindir, log = fake_sbatch
    _probe_done(reg, rows, "e2")
    _gpu_json(reg, "e2", 2)                                         # 没有 nvidia-smi 的包
    _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    e2 = [p for p in _pack_calls(log) if "__e2__" in p[1]]
    assert [p[0] for p in e2] == [2, 1]


def test_oom_caps_k_below_the_pack_that_failed(study, fake_sbatch):
    d, reg, rows = study
    bindir, log = fake_sbatch
    _probe_done(reg, rows, "e2")
    _gpu_json(reg, "e2", 2, mem_total_mib=97871, run_peak_max_mib=10000.0)
    _gpu_json(reg, "e2", 3, first_seed=44, mem_total_mib=97871, run_peak_max_mib=10000.0,
              n_oom=1, oom_runs=["GA__e2__s45"])
    _done(reg, rows, "GA__e2__s44", rc=86)                          # OOM → 非 0 → todo
    _done(reg, rows, "GA__e2__s45", rc=86)
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    e2 = [p for p in _pack_calls(log) if "__e2__" in p[1]]
    # 显存读数本身允许 K=3（97871×85%/11024 = 7），是 OOM 把它压到 2：剩 3 个 → 2 + 1
    assert e2 == [(2, "GA__e2__pack-k2-s44", ["GA__e2__s44", "GA__e2__s45"]),
                  (1, "GA__e2__pack-k1-s46", ["GA__e2__s46"])]
    assert "出过 OOM" in out.stdout


def test_packs_never_mix_cells(study, fake_sbatch):
    """满包永远同格子；余数 2 + 2 合成 3 + 1 仍是 2 个作业 → 不省就不合（D-060），各交各的。"""
    d, reg, rows = study
    bindir, log = fake_sbatch
    for cell in ("e2", "e4"):
        _gpu_json(reg, cell, 2, mem_total_mib=97871, run_peak_max_mib=5000.0)
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    packs = _pack_calls(log)
    assert sorted(p[0] for p in packs) == [2, 2, 3, 3]              # 每格 5 = 3 + 2
    for _, tag, rids in packs:
        cells = {r.split("__")[1] for r in rids}
        assert len(cells) == 1 and tag.startswith(f"GA__{cells.pop()}__pack-")
    assert "不省" in out.stdout


def test_done_runs_never_enter_a_pack(study, fake_sbatch):
    d, reg, rows = study
    bindir, log = fake_sbatch
    _gpu_json(reg, "e2", 2, mem_total_mib=97871, run_peak_max_mib=5000.0)
    _done(reg, rows, "GA__e2__s44")
    _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3, RUN_GROUPS="GA")
    rids = [r for p in _pack_calls(log) for r in p[2]]
    assert "GA__e2__s44" not in rids and len(rids) == len(set(rids))


# ══════════════════════════════════════════════════════════════════════════
# 余数跨格子合包（D-060）—— G6 的形状：3 个格子 × 3 个 seed，探路包用掉 s42 / s43
# ══════════════════════════════════════════════════════════════════════════
G6_PEAK = 16951.9            # 2026-09-28 G6 探路包回传的真实峰值（三臂 16764–16952 MiB）


@pytest.fixture
def g6like(tmp_path):
    d = tmp_path / "g6"
    reg, rows = _study(d, cells=("a", "b", "c"), seeds=(42, 43, 44))
    for f in ("submit.sh", "submit_lib.sh"):
        shutil.copy(MECH / f, d / f)
    return d, reg, rows


def _mixed_gpu_json(reg, cells, k, first_seed, **fields):
    d = _metrics_path(reg, f"GA__{cells[0]}__s42").parent
    d.mkdir(parents=True, exist_ok=True)
    base = {"k": k, "runs": [], "mem_total_mib": 97871, "run_peak_max_mib": None,
            "mem_max_mib": None, "n_oom": 0, "n_mem_warnings": 0}
    base.update(fields)
    (d / f"GA__mix-{'+'.join(cells)}__pack-k{k}-s{first_seed}.gpu.json").write_text(json.dumps(base))


def _g6_probed(reg, rows, cells=("a", "b", "c"), **fields):
    for c in cells:
        _probe_done(reg, rows, c)
        _gpu_json(reg, c, 2, **({"mem_total_mib": 97871, "run_peak_max_mib": G6_PEAK} | fields))


def test_g6_leftovers_with_equal_real_peaks_share_one_gpu(g6like, fake_sbatch):
    """真实场景：三臂各剩 s44 → 不合是 3 个 K=1 作业，合了是 1 个 K=3 作业。"""
    d, reg, rows = g6like
    bindir, log = fake_sbatch
    _g6_probed(reg, rows)
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    assert _pack_calls(log) == [(3, "GA__mix-a+b+c__pack-k3-s44",
                                 ["GA__a__s44", "GA__b__s44", "GA__c__s44"])]
    assert "3 → 1 个作业" in out.stdout


def test_pack_mix_0_restores_one_cell_per_pack(g6like, fake_sbatch):
    d, reg, rows = g6like
    bindir, log = fake_sbatch
    _g6_probed(reg, rows)
    _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3, PACK_MIX=0)
    assert _pack_calls(log) == [(1, f"GA__{c}__pack-k1-s44", [f"GA__{c}__s44"]) for c in "abc"]


def test_bad_pack_mix_value_is_rejected(g6like, fake_sbatch):
    d, _, _ = g6like
    bindir, log = fake_sbatch
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3, PACK_MIX=2)
    assert out.returncode != 0 and not _calls(log)


def test_peaks_beyond_tolerance_are_not_mixed_at_all(g6like, fake_sbatch):
    """一组里有一个格子峰值差太多 → 整组都不合（不做部分合包）。"""
    d, reg, rows = g6like
    bindir, log = fake_sbatch
    _g6_probed(reg, rows, cells=("a", "b"))
    _g6_probed(reg, rows, cells=("c",), run_peak_max_mib=20000.0)      # (20000 − 16951) / 20000 = 15%
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    assert [p[0] for p in _pack_calls(log)] == [1, 1, 1]
    assert "相差超过 10%" in out.stdout


def test_tolerance_is_configurable(g6like, fake_sbatch):
    d, reg, rows = g6like
    bindir, log = fake_sbatch
    _g6_probed(reg, rows, cells=("a", "b"))
    _g6_probed(reg, rows, cells=("c",), run_peak_max_mib=20000.0)
    _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3, PACK_MIX_TOL_PCT=20)
    assert [p[1] for p in _pack_calls(log)] == ["GA__mix-a+b+c__pack-k3-s44"]


def test_cell_without_a_real_peak_is_not_mixed(g6like, fake_sbatch):
    """只有整卡读数（没有 [GPUMem]）的格子不参与合包；其余两个照合。"""
    d, reg, rows = g6like
    bindir, log = fake_sbatch
    _g6_probed(reg, rows, cells=("a", "b"))
    _probe_done(reg, rows, "c")
    _gpu_json(reg, "c", 2, mem_total_mib=97871, mem_max_mib=67016)
    _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    assert sorted(_pack_calls(log)) == [
        (1, "GA__c__pack-k1-s44", ["GA__c__s44"]),
        (2, "GA__mix-a+b__pack-k2-s44", ["GA__a__s44", "GA__b__s44"])]


@pytest.mark.parametrize("bad", [{"n_oom": 1, "oom_runs": ["GA__c__s43"]}, {"n_mem_warnings": 3}])
def test_cell_with_oom_or_memory_warnings_is_not_mixed(g6like, fake_sbatch, bad):
    d, reg, rows = g6like
    bindir, log = fake_sbatch
    _g6_probed(reg, rows, cells=("a", "b"))
    _g6_probed(reg, rows, cells=("c",), **bad)
    _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    tags = sorted(p[1] for p in _pack_calls(log))
    assert tags == ["GA__c__pack-k1-s44", "GA__mix-a+b__pack-k2-s44"]


def test_oom_in_a_mixed_pack_caps_every_cell_in_it(g6like, fake_sbatch):
    """合包出了 OOM（run 记 86 → todo）→ 包里每个格子 K ≤ k − 1、且不再合包；不在包里的格子不受影响。"""
    d, reg, rows = g6like
    bindir, log = fake_sbatch
    _g6_probed(reg, rows)
    _mixed_gpu_json(reg, ["a", "b"], 3, 44, run_peak_max_mib=G6_PEAK, n_oom=1,
                    oom_runs=["GA__a__s44"])
    for c in "ab":
        _done(reg, rows, f"GA__{c}__s44", rc=86)
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    assert sorted(p[1] for p in _pack_calls(log)) == [
        "GA__a__pack-k1-s44", "GA__b__pack-k1-s44", "GA__c__pack-k1-s44"]
    assert "GA__a：K=2" in out.stdout and "GA__b：K=2" in out.stdout
    assert "GA__c：K=3" in out.stdout                              # c 不在那个合包里


def test_a_mixed_pack_counts_as_memory_data_for_its_cells(g6like, fake_sbatch):
    """格子自己没有包、只参与过合包 → 用合包的读数定 K，不再探路。"""
    d, reg, rows = g6like
    bindir, log = fake_sbatch
    _mixed_gpu_json(reg, ["a", "b", "c"], 3, 42, run_peak_max_mib=G6_PEAK)
    for c in "abc":
        _done(reg, rows, f"GA__{c}__s42")
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    assert "探路" not in out.stdout
    # 每格剩 s43 / s44 两个（< K=3）→ 余数 2+2+2 = 6 → 2 个 K=3 合包（原本 3 个）
    assert [p[0] for p in _pack_calls(log)] == [3, 3]


# ══════════════════════════════════════════════════════════════════════════
# 门槛 / dry-run：PACK 下同样不交
# ══════════════════════════════════════════════════════════════════════════
def test_audit_gate_still_blocks_under_pack(tmp_path, fake_sbatch):
    d = tmp_path / "study"
    _study(d, audit_status="open")
    for f in ("submit.sh", "submit_lib.sh"):
        shutil.copy(MECH / f, d / f)
    bindir, log = fake_sbatch
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK=3)
    assert out.returncode == 0, out.stderr
    assert not _calls(log) and "本次入队=0" in out.stdout
    assert "would sbatch -c 12 pack  GA__e2__pack-k3-s42" in out.stdout


def test_dry_run_under_pack_submits_nothing(study, fake_sbatch):
    d, _, _ = study
    bindir, log = fake_sbatch
    out = _run(d / "submit.sh", "--dry-run", bindir=bindir, mech_dir=d, PACK=3)
    assert not _calls(log) and out.stdout.count("would sbatch -c 12 pack") == 2


def test_bad_pack_value_is_rejected(study, fake_sbatch):
    d, _, _ = study
    bindir, log = fake_sbatch
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, PACK="three")
    assert out.returncode == 2 and not _calls(log)


# ══════════════════════════════════════════════════════════════════════════
# stale（D-053）
# ══════════════════════════════════════════════════════════════════════════
def test_stale_is_reported_and_not_resubmitted_by_default(study, fake_sbatch):
    d, reg, rows = study
    bindir, log = fake_sbatch
    for rid in rows:
        _done(reg, rows, rid)
    _done(reg, rows, "GA__e2__s42", sha="000000000000")             # 配置在结果回来之后改过
    st = _run(d / "submit.sh", "--status", bindir=bindir, mech_dir=d)
    assert "  stale  GA__e2__s42" in st.stdout and "过期(stale)=1" in st.stdout
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d)
    assert not _calls(log) and "RESUBMIT_STALE=1" in out.stdout
    out = _run(d / "submit.sh", bindir=bindir, mech_dir=d, RESUBMIT_STALE=1)
    assert [c[2] for c in _calls(log)] == ["GA__e2__s42"]


# ══════════════════════════════════════════════════════════════════════════
# pilot 的提交脚本同样支持
# ══════════════════════════════════════════════════════════════════════════
def test_pilot_submit_supports_pack_and_stale(tmp_path, fake_sbatch):
    mech = tmp_path / "mech"
    reg, rows = _study(mech / "pilot")
    shutil.copy(MECH / "pilot" / "submit_pilot.sh", mech / "pilot" / "submit_pilot.sh")
    shutil.copy(MECH / "submit_lib.sh", mech / "submit_lib.sh")
    bindir, log = fake_sbatch
    out = _run(mech / "pilot" / "submit_pilot.sh", bindir=bindir, PACK=3)
    assert out.returncode == 0, out.stderr
    assert [p[1] for p in _pack_calls(log)] == ["GA__e2__pack-k3-s42", "GA__e4__pack-k3-s42"]
    assert all(c[3].endswith("/pack.sbatch") for c in _calls(log))
    _done(reg, rows, "GA__e2__s46", sha="000000000000")
    st = _run(mech / "pilot" / "submit_pilot.sh", "--status", bindir=bindir)
    assert "  stale  GA__e2__s46" in st.stdout


# ══════════════════════════════════════════════════════════════════════════
# pack.sbatch：OOM 判定与显存记录（直接执行脚本里的 python 片段）
# ══════════════════════════════════════════════════════════════════════════
def _heredocs(path: Path):
    return re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY\n", path.read_text(encoding="utf-8"), re.S)


def _finalize_snippet():
    snips = [s for s in _heredocs(MECH / "pack.sbatch") if "oom_rc" in s]
    assert len(snips) == 1
    return snips[0]


def _summary_snippet():
    snips = [s for s in _heredocs(MECH / "pack.sbatch") if "run_peak_max_mib" in s]
    assert len(snips) == 1
    return snips[0]


def _finalize(tmp_path, log_text, rc=0, gpu_mem=None):
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"exit_code": None, "gpu_mem": gpu_mem}))
    lg = tmp_path / "run.log"
    lg.write_text(log_text)
    out = subprocess.run([sys.executable, "-", str(m), str(rc), "T", "3", "0", "local",
                          str(lg), "86"], input=_finalize_snippet(), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return int(out.stdout.strip()), json.loads(m.read_text())


def test_swallowed_oom_turns_exit_zero_into_the_oom_code(tmp_path):
    """客户端里的 OOM 被 _collect_updates_* 吞掉 → run exit 0，但结果无效（陷阱 #23 同类）。"""
    log = ("    [ERROR] Client 7: OOM when allocating tensor with shape[32,64,32,32]\n"
           "[Edge 0] 1 个客户端本轮失败并被丢弃: [7]\n")
    rc, d = _finalize(tmp_path, log, rc=0, gpu_mem={"peak_mib": 30000.0, "n_samples": 5})
    assert rc == 86 and d["exit_code"] == 86
    assert d["pack"]["oom"] is True and d["pack"]["peak_mib"] == 30000.0


def test_memory_warning_is_counted_not_judged_oom(tmp_path):
    log = ("W tensorflow/tsl/framework/bfc_allocator.cc:296] Allocator (GPU_0_bfc) ran out of "
           "memory trying to allocate 2.3GiB with freed_by_count=0. The caller indicates that "
           "this is not a failure, but this may mean that there could be performance gains\n")
    rc, d = _finalize(tmp_path, log, rc=0)
    assert rc == 0 and d["exit_code"] == 0
    assert d["pack"]["oom"] is False and d["pack"]["mem_warnings"] == 1


def test_clean_run_keeps_its_exit_code(tmp_path):
    rc, d = _finalize(tmp_path, "[Cloud] Training complete.\n", rc=0)
    assert rc == 0 and d["pack"]["oom"] is False
    rc, d = _finalize(tmp_path, "Traceback (most recent call last):\n", rc=1)
    assert rc == 1                                                   # 非 0 原样保留


def test_gpu_summary_records_capacity_real_peaks_and_oom(tmp_path):
    csv = tmp_path / "g.csv"
    csv.write_text("40, 60000, 97871\n90, 90000, 97871\n")
    ms = []
    for i, (pk, oom) in enumerate([(20000.5, False), (31000.0, True)]):
        p = tmp_path / f"m{i}.json"
        p.write_text(json.dumps({"pack": {"peak_mib": pk, "oom": oom, "mem_warnings": i}}))
        ms.append(str(p))
    out_json = tmp_path / "T.gpu.json"
    out = subprocess.run([sys.executable, "-", str(csv), str(out_json), "2", "100", "r0", "r1", *ms],
                         input=_summary_snippet(), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    g = json.loads(out_json.read_text())
    assert g["mem_total_mib"] == 97871 and g["mem_max_mib"] == 90000
    assert g["run_peak_mib"] == [20000.5, 31000.0] and g["run_peak_max_mib"] == 31000.0
    assert g["n_oom"] == 1 and g["oom_runs"] == ["r1"] and g["n_mem_warnings"] == 1
    assert g["runs"] == ["r0", "r1"]
    # submit_lib.sh 是纯 bash、只 grep 标量：每个标量键各占一行
    text = out_json.read_text()
    for key in ("k", "mem_total_mib", "run_peak_max_mib", "mem_max_mib", "n_oom"):
        assert re.search(rf'^\s*"{key}": [0-9.]+,?$', text, re.M), key


def test_pack_sbatch_samples_total_memory():
    code = "\n".join(ln for ln in (MECH / "pack.sbatch").read_text().splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "--query-gpu=utilization.gpu,memory.used,memory.total" in code
    assert 'RCS[$i]=$RC_EFF' in code                                # OOM 码进 WORST / exit


# ══════════════════════════════════════════════════════════════════════════
# [GPUMem]：服务器打、collect_metrics 解析
# ══════════════════════════════════════════════════════════════════════════
def test_gpu_mem_is_parsed_as_the_max_peak():
    log = ("[GPUMem] Round 1 | peak_mib=1200.5 | current_mib=800.0\n"
           "[GPUMem] Round 2 | peak_mib=3400.0 | current_mib=900.0\n")
    m = collect(log)
    assert m["gpu_mem"] == {"peak_mib": 3400.0, "n_samples": 2}


def test_old_log_without_gpu_mem_is_null_not_zero():
    assert collect("[Cloud] Training complete.\n")["gpu_mem"] is None


def test_gpu_mem_line_is_printed_after_run_round_including_backdoor_eval():
    src = (ROOT / "fedavg" / "server" / "server.py").read_text()
    run_body = src[src.index("    def run(self, logger=None):"):]
    assert run_body.index("self.run_round(r)") < run_body.index("gpu_mem_line(r)")
    fn = src[src.index("def gpu_mem_line"):src.index("class CloudServer")]
    assert 'list_physical_devices("GPU")' in fn and "get_memory_info" in fn


def test_gpu_mem_line_is_silent_without_a_gpu():
    pytest.importorskip("tensorflow")
    sys.path.insert(0, str(ROOT / "fedavg"))
    from server.server import gpu_mem_line         # noqa: E402
    import tensorflow as tf
    if tf.config.list_physical_devices("GPU"):
        pytest.skip("有 GPU 的机器上这一行会打")
    assert gpu_mem_line(1) is None


# ══════════════════════════════════════════════════════════════════════════
# pack.sbatch 端到端（假 main.py，真 collect_metrics）：bash 的接线本身
# ══════════════════════════════════════════════════════════════════════════
FAKE_MAIN = r'''
import sys
cfg = sys.argv[sys.argv.index("--config") + 1]
tag = open(cfg).read().strip()
print("[Checksum] Round 1 | global=abcdef012345")
print("[GPUMem] Round 1 | peak_mib=%s | current_mib=10.0" % ("1500.0" if tag == "a" else "2500.0"))
if tag == "b":                        # 客户端里的 OOM 被吞掉，进程照样 exit 0
    print("    [ERROR] Client 3: OOM when allocating tensor with shape[32,64,32,32]")
if tag == "c":
    sys.exit(3)
print("[Cloud] Training complete.")
'''


@pytest.mark.skipif(shutil.which("taskset") is None, reason="需要 taskset")
def test_pack_sbatch_end_to_end_marks_oom_and_propagates_exit_codes(tmp_path):
    repo = tmp_path / "repo"
    (repo / "fedavg").mkdir(parents=True)
    (repo / "harness").mkdir()
    shutil.copy(ROOT / "cluster_env.sh", repo / "cluster_env.sh")
    shutil.copy(ROOT / "harness" / "collect_metrics.py", repo / "harness" / "collect_metrics.py")
    shutil.copytree(ROOT / "fedavg" / "utils", repo / "fedavg" / "utils")
    (repo / "fedavg" / "main.py").write_text(FAKE_MAIN)
    args = []
    for t in ("a", "b", "c"):
        (repo / f"{t}.yaml").write_text(t)
        args += [f"{t}.yaml", f"R_{t}", f"out/R_{t}.metrics.json"]
    env = {**os.environ, "SLURM_SUBMIT_DIR": str(repo), "TFDPFL_PY": sys.executable,
           "TFDPFL_LOGDIR": str(tmp_path / "logs"), "TFDPFL_CORES_PER_RUN": "1",
           "PATH": "/usr/bin:/bin"}
    out = subprocess.run(["bash", str(MECH / "pack.sbatch"), "T3", *args],
                         capture_output=True, text=True, env=env)
    if "不启动" in out.stdout:
        pytest.skip(out.stdout.strip().splitlines()[-1])
    m = {t: json.loads((repo / "out" / f"R_{t}.metrics.json").read_text()) for t in "abc"}
    assert m["a"]["exit_code"] == 0 and m["a"]["pack"]["oom"] is False
    assert m["a"]["pack"]["peak_mib"] == 1500.0 and m["a"]["gpu_mem"]["peak_mib"] == 1500.0
    assert m["b"]["exit_code"] == 86 and m["b"]["pack"]["oom"] is True      # 被吞掉的 OOM
    assert m["c"]["exit_code"] == 3                                          # 真失败原样
    assert out.returncode != 0                                               # WORST 带出来
    g = json.loads((repo / "out" / "T3.gpu.json").read_text())
    assert g["k"] == 3 and g["runs"] == ["R_a", "R_b", "R_c"]
    assert g["run_peak_mib"] == [1500.0, 2500.0, 2500.0] and g["run_peak_max_mib"] == 2500.0
    assert g["oom_runs"] == ["R_b"] and g["n_oom"] == 1
    assert g["mem_total_mib"] is None                                        # 没有 nvidia-smi
