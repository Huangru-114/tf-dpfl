"""所有作业脚本必须走 `cluster_env.sh` 的 `$PY`，且用 Arrhenius 的 SLURM 头。

**为什么值一条测试**：CLAUDE.md 开头那段说得很清楚 ——「静默地跑在错误的环境里
是最难查的一类问题」。裸 `python3 main.py` 在 Arrhenius 上报的是
numpy/tensorflow 的 ImportError，**看起来像「依赖没装」**，而真实原因是
根本没进容器（`/nobackup/proj/disk/naiss2025-22-1095/personal/ziangg/tensorflow.sif`）。

这件事**已经发生过一次**：`exp3_cell.sbatch` 在 Alvis 时期被改成裸 `python3`
+ Alvis 的 SLURM 头（`--gpus-per-node=A40:1` / `--account=...` 不带 `-gpu`），
换回 Arrhenius 之后没有跟着改回来；新写的 `calib_cell.sbatch` 又照抄了它。
两个脚本都会在 GPU 排到之后才炸。

纯 stdlib，本地秒级。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SIF = "/nobackup/proj/disk/naiss2025-22-1095/personal/ziangg/tensorflow.sif"

# 会提交到计算节点的脚本。cluster_env.sh 自己是那个收口处，不在检查范围内。
JOB_SCRIPTS = sorted(
    p for p in list(ROOT.glob("*.sh")) + list(ROOT.glob("*.sbatch"))
    + list(ROOT.glob("experiments/**/*.sbatch")) + list(ROOT.glob("experiments/**/*.sh"))
    if p.name != "cluster_env.sh"
)

# 真正吃 GPU、必须带完整 SLURM 头的那些（其余是登录节点跑的驱动脚本）
SBATCH_SCRIPTS = [p for p in JOB_SCRIPTS
                  if p.read_text(encoding="utf-8").lstrip().startswith("#!/bin/bash")
                  and "#SBATCH" in p.read_text(encoding="utf-8")]


def _code(path: Path) -> str:
    """去掉注释行 —— 注释里出现 `python3` 不算调用它。"""
    return "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                     if not ln.lstrip().startswith("#"))


def test_there_are_job_scripts_to_check():
    """反向自检：正则/路径写错时不能悄悄变成「零个文件全部通过」。"""
    assert len(JOB_SCRIPTS) >= 6, [p.name for p in JOB_SCRIPTS]
    assert len(SBATCH_SCRIPTS) >= 4, [p.name for p in SBATCH_SCRIPTS]


@pytest.mark.parametrize("path", JOB_SCRIPTS, ids=lambda p: p.name)
def test_no_bare_python_invocation(path):
    """任何 `python3 <脚本>` / `python <脚本>` 都必须是 `$PY ...`。"""
    bad = [ln.strip() for ln in _code(path).splitlines()
           if re.search(r"(?<![\w$/])python3?\s+(-m\s+)?[\w./]+", ln)
           and "$PY" not in ln and "PY=" not in ln]
    assert not bad, (
        f"{path.name} 里有裸 python 调用（必须走 $PY，否则不在容器里）：\n  "
        + "\n  ".join(bad))


@pytest.mark.parametrize("path", JOB_SCRIPTS, ids=lambda p: p.name)
def test_scripts_source_cluster_env(path):
    """要用 `$PY` 就得先 source cluster_env.sh —— 否则 $PY 是空的，
    `$PY main.py` 会退化成 `main.py`（当成可执行文件找不到）。"""
    code = _code(path)
    if "$PY" not in code:
        pytest.skip(f"{path.name} 不调 python")
    assert "cluster_env.sh" in code, f"{path.name} 用了 $PY 却没 source cluster_env.sh"


@pytest.mark.parametrize("path", SBATCH_SCRIPTS, ids=lambda p: p.name)
def test_slurm_header_is_arrhenius(path):
    """Arrhenius 的头：`-A naiss2026-4-650-gpu` + `-p gpu`。

    Alvis 式的 `--account=naiss2026-4-650`（不带 -gpu）+ `--gpus-per-node=A40:1`
    在 Arrhenius 上会被拒或排不到卡。
    """
    head = "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                     if ln.startswith("#SBATCH"))
    assert re.search(r"^#SBATCH\s+-A\s+naiss2026-4-650-gpu\s*$", head, re.M), \
        f"{path.name} 的 account 不是 Arrhenius 的 naiss2026-4-650-gpu：\n{head}"
    assert re.search(r"^#SBATCH\s+-p\s+gpu\s*$", head, re.M), \
        f"{path.name} 少了 -p gpu：\n{head}"
    assert "--gpus-per-node" not in head, \
        f"{path.name} 还带着 Alvis 的 --gpus-per-node：\n{head}"


def test_cluster_env_points_at_the_arrhenius_container():
    """容器路径只应出现在 cluster_env.sh 这一个地方，且是对的那个。"""
    env = (ROOT / "cluster_env.sh").read_text(encoding="utf-8")
    assert SIF in env, f"cluster_env.sh 里的容器路径不是 {SIF}"
    assert "apptainer exec --nv" in env
    assert "--bind" in env, "少了 --bind：容器里看不到仓库的兄弟/上级目录（踩过的坑）"


@pytest.mark.parametrize("path", JOB_SCRIPTS, ids=lambda p: p.name)
def test_container_path_is_not_hardcoded_outside_cluster_env(path):
    """新脚本不要再硬写容器路径 —— CLAUDE.md 明确要求收口到 cluster_env.sh。"""
    assert SIF not in _code(path), \
        f"{path.name} 硬写了容器路径；改成 source cluster_env.sh 用 $PY"
