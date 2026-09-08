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


def _stdlib_only_lines(path: Path) -> set:
    """紧跟在 `# stdlib-only:` 注释块之后的那些行，允许用裸 python3。

    有些脚本（如 read_calibration.py）**刻意**不 import TF，登录节点上裸
    python3 就能跑、不必进容器。没有这个出口的话，守卫会逼着所有东西都套
    容器 —— 那会让「提交作业」这件事绑在「容器此刻可用」上。
    标记必须写明**为什么**是 stdlib-only，不能只写标记。
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    out, armed = set(), False
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if stripped.startswith("#"):
            if "stdlib-only:" in stripped:
                armed = True
            continue
        if armed:
            out.add(i)
            armed = False
    return out


def test_there_are_job_scripts_to_check():
    """反向自检：正则/路径写错时不能悄悄变成「零个文件全部通过」。"""
    assert len(JOB_SCRIPTS) >= 6, [p.name for p in JOB_SCRIPTS]
    assert len(SBATCH_SCRIPTS) >= 4, [p.name for p in SBATCH_SCRIPTS]


@pytest.mark.parametrize("path", JOB_SCRIPTS, ids=lambda p: p.name)
def test_no_bare_python_invocation(path):
    """任何 `python3 <脚本>` / `python <脚本>` 都必须是 `$PY ...`。"""
    allowed = _stdlib_only_lines(path)
    bad = [ln.strip() for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines())
           if not ln.lstrip().startswith("#")
           and re.search(r"(?<![\w$/])python3?\s+(-m\s+)?[\w./]+", ln)
           and "$PY" not in ln and "PY=" not in ln
           and i not in allowed]
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
    assert re.search(r"^#SBATCH\s+--gpus\s+1\s*$", head, re.M), \
        f"{path.name} 少了 --gpus 1：\n{head}"
    assert "--gpus-per-node" not in head, \
        f"{path.name} 还带着 Alvis 的 --gpus-per-node：\n{head}"


def test_cluster_env_points_at_the_arrhenius_container():
    """容器路径只应出现在 cluster_env.sh 这一个地方，且是对的那个。"""
    env = (ROOT / "cluster_env.sh").read_text(encoding="utf-8")
    assert SIF in env, f"cluster_env.sh 里的容器路径不是 {SIF}"
    assert "apptainer exec --nv" in env


@pytest.mark.parametrize("path", JOB_SCRIPTS, ids=lambda p: p.name)
def test_container_path_is_not_hardcoded_outside_cluster_env(path):
    """新脚本不要再硬写容器路径 —— CLAUDE.md 明确要求收口到 cluster_env.sh。"""
    assert SIF not in _code(path), \
        f"{path.name} 硬写了容器路径；改成 source cluster_env.sh 用 $PY"


# ══════════════════════════════════════════════════════════════════════════
# 8 月在 Arrhenius 上跑通的设计：--bind <仓库上一级> 撑起一切
# ══════════════════════════════════════════════════════════════════════════
TRAIN_SCRIPTS = [p for p in JOB_SCRIPTS
                 if re.search(r"\$(PY|RUN) main\.py", _code(p))]


def test_there_are_training_scripts_to_check():
    """反向自检：正则失配时不能退化成「零个文件全部通过」。"""
    assert len(TRAIN_SCRIPTS) >= 4, [p.name for p in TRAIN_SCRIPTS]


def test_bind_defaults_to_the_repo_parent():
    """**`--bind <仓库上一级>` 是这套设计的地基**，不要再拆。

    2026-08 全部 Arrhenius run（含 exp3 的 3C 全批，作者机 arrhenius1）
    都靠它：`cd $ROOT/fedavg` 能读到兄弟目录 `$ROOT/experiments/`、
    上一级的 `tfdpfl-logs/`、以及 `$ROOT/../data/datasets`（keras 缓存），
    全部因为绑定根覆盖了整片。

    2026-09 我们误以为 --bind 无效（真实原因是自检带 `--nv` 在**登录节点**
    起不来容器），连续三轮把 cwd/日志/路径全改掉，每改一轮换一个
    FileNotFoundError。见 CLAUDE.md 陷阱 #17。
    """
    env = (ROOT / "cluster_env.sh").read_text(encoding="utf-8")
    assert re.search(r'TFDPFL_BIND="\$\{TFDPFL_BIND:-\$\(cd "\$_TFDPFL_ROOT/\.\." && pwd\)\}"',
                     env), "TFDPFL_BIND 的默认值不再是仓库上一级"
    # 必须盯 **$PY 那一行**：自检命令里也有 --bind，泛泛地搜整个文件会被它
    # 蒙混过去（第一次写就是这样，反向锚点没红才发现）。
    m = re.search(r'^\s*PY="apptainer exec[^"]*"', env, re.M)
    assert m, "找不到 $PY 的赋值行"
    assert "--bind $TFDPFL_BIND" in m.group(0), f"$PY 里没有 --bind：{m.group(0)}"
    assert "--nv" in m.group(0), f"$PY 里没有 --nv（训练要 GPU）：{m.group(0)}"


def test_self_check_does_not_use_nv():
    """自检只关心**挂载**，不需要 GPU。带 `--nv` 会在登录节点直接失败，
    并把人引向完全错误的方向（报「看不到仓库目录」，实际是没驱动）。"""
    env = (ROOT / "cluster_env.sh").read_text(encoding="utf-8")
    m = re.search(r'_TFDPFL_CHECK="([^"]+)"', env)
    assert m, "找不到自检用的命令"
    assert "--nv" not in m.group(1), f"自检还带着 --nv：{m.group(1)}"
    assert "--bind" in m.group(1), "自检没带 --bind，测不出挂载"


@pytest.mark.parametrize("path", TRAIN_SCRIPTS, ids=lambda p: p.name)
def test_training_runs_from_fedavg(path):
    """cwd = `$ROOT/fedavg` 是 8 月验证过的形式（--bind 覆盖了跨目录访问）。"""
    code = _code(path)
    assert 'cd "$ROOT/fedavg"' in code, f"{path.name} 不再 cd 进 fedavg"
    assert re.search(r"\$(PY|RUN) main\.py", code), \
        f"{path.name} 没有以 `$PY main.py` 的形式调用"


@pytest.mark.parametrize("path", JOB_SCRIPTS, ids=lambda p: p.name)
def test_log_dir_defaults_beside_the_repo(path):
    """日志在仓库**上一级** `tfdpfl-logs/` —— 大日志天然不在 git 里，
    而 `--bind <上一级>` 保证容器看得见。"""
    code = _code(path)
    if "LOGDIR=" not in code:
        pytest.skip(f"{path.name} 不写日志")
    assert "tfdpfl-logs" in code, \
        f"{path.name} 的日志不在上一级的 tfdpfl-logs（8 月设计）"
