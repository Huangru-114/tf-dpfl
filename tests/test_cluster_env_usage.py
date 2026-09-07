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


def test_bind_is_off_by_default():
    """**默认不加 --bind**，与 Arrhenius 上实测可用的写法一致：

        apptainer exec --nv <abs .sif> python -m <module>

    这台机器的 apptainer 已在系统级把 /nobackup 挂进容器；再显式 --bind
    反而会失败，且失败信息长得像「容器里看不到仓库目录」。
    我们一度以为 --bind 是必须的，那条经验在这台机器上不成立。
    """
    env = (ROOT / "cluster_env.sh").read_text(encoding="utf-8")
    assert 'TFDPFL_BIND="${TFDPFL_BIND:-}"' in env, \
        "TFDPFL_BIND 又有默认值了 —— 默认必须为空（不绑）"
    assert 'PY="apptainer exec --nv $TFDPFL_SIF python3"' in env, \
        "缺少不带 --bind 的那条路径"
    # 但仍要保留显式打开的能力（换集群时可能需要）
    assert 'if [ -n "$TFDPFL_BIND" ]; then' in env, \
        "TFDPFL_BIND 设了值时应该仍然生效"


@pytest.mark.parametrize("path", JOB_SCRIPTS, ids=lambda p: p.name)
def test_container_path_is_not_hardcoded_outside_cluster_env(path):
    """新脚本不要再硬写容器路径 —— CLAUDE.md 明确要求收口到 cluster_env.sh。"""
    assert SIF not in _code(path), \
        f"{path.name} 硬写了容器路径；改成 source cluster_env.sh 用 $PY"


# ══════════════════════════════════════════════════════════════════════════
# cwd 必须留在仓库根：apptainer 只自动挂 $PWD
# ══════════════════════════════════════════════════════════════════════════
TRAIN_SCRIPTS = [p for p in JOB_SCRIPTS
                 if re.search(r"\$(PY|RUN) (fedavg/)?main\.py", _code(p))]


def test_there_are_training_scripts_to_check():
    assert len(TRAIN_SCRIPTS) >= 4, [p.name for p in TRAIN_SCRIPTS]


@pytest.mark.parametrize("path", TRAIN_SCRIPTS, ids=lambda p: p.name)
def test_training_runs_from_the_repo_root_not_from_fedavg(path):
    """apptainer **只自动挂 $PWD**（2026-09 实测）。`cd $ROOT/fedavg` 之后，
    兄弟目录 `$ROOT/experiments/` 在 $PWD 之外 -> 容器里看不见 ->
    `FileNotFoundError: .../experiments/calibration/xxx.yaml`，而文件明明在。

    `$PY fedavg/main.py` 的 sys.path[0] 仍是 `fedavg/`，import 一行都不用改。
    """
    code = _code(path)
    assert 'cd "$ROOT/fedavg"' not in code, \
        f"{path.name} 还在 cd 进 fedavg —— 兄弟目录会看不见"
    assert re.search(r"\$(PY|RUN) fedavg/main\.py", code), \
        f"{path.name} 没有以 fedavg/main.py 的形式调用"


@pytest.mark.parametrize("path", JOB_SCRIPTS, ids=lambda p: p.name)
def test_log_dir_defaults_inside_the_repo(path):
    """日志默认目录必须在 $PWD 之内。放在仓库**上一级**（旧的 tfdpfl-logs）
    同样在容器视野之外 —— collect_metrics 会读不到自己刚写的日志。
    `.gitignore` 已忽略 `logs/`，大日志照样不进 git。
    """
    code = _code(path)
    if "LOGDIR=" not in code:
        pytest.skip(f"{path.name} 不写日志")
    assert "../tfdpfl-logs" not in code, \
        f"{path.name} 的日志默认在仓库上一级 —— 容器看不见"
    # 注意默认值里可能有**嵌套** ${...}（experiment_tf.sh 就是
    # `${TFDPFL_LOGDIR:-${SLURM_SUBMIT_DIR:-$ROOT}/logs}`），所以不能用 [^}]*
    # —— 第一版就栽在这儿，是正则错不是脚本错。
    m = re.search(r'LOGDIR="\$\{TFDPFL_LOGDIR:-(.*)\}"', code)
    assert m, f"{path.name} 没有可识别的 LOGDIR 默认值"
    assert m.group(1).rstrip("}").endswith("/logs"), \
        f"{path.name} 的 LOGDIR 默认值不在仓库内：{m.group(1)}"
