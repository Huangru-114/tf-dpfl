# cluster_env.sh  —  解析「用什么 python 跑」的**唯一**地方（被 source，不要直接执行）
#
# ══════════════════════════════════════════════════════════════════════════
# 集群上所有 python 都必须在 apptainer 容器里跑，否则**一个库都找不到**。
# 裸 `python3 xxx.py` 在集群上必然 ImportError —— 而且报的是 numpy/tensorflow
# 找不到这种看起来像「环境没装好」的错，很容易被误判成别的问题。
#
# 用法（所有脚本统一这么写）：
#     source "$ROOT/cluster_env.sh"
#     $PY -m pytest tests/
#     $PY main.py --config ...
#
# 覆盖方式：
#     TFDPFL_PY="python3"        强制用裸 python（本地开发 / 容器外调试）
#     TFDPFL_SIF=/path/to.sif    换容器
#     TFDPFL_BIND=/some/path     换绑定根（默认：仓库的**上一级**目录）
#     TFDPFL_SKIP_ENV_CHECK=1    跳过启动自检
# ══════════════════════════════════════════════════════════════════════════

TFDPFL_SIF="${TFDPFL_SIF:-/nobackup/proj/disk/naiss2025-22-1095/personal/ziangg/tensorflow.sif}"

# 本文件所在目录 = 仓库根
_TFDPFL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# ── 绑定根：默认取仓库的**上一级** ────────────────────────────────────────
# **为什么必须显式 --bind**（踩过的坑）：apptainer 默认只把 `$PWD` 和 `$HOME`
# 挂进容器。而本仓库的脚本会跨目录访问：
#     cd $ROOT/fedavg  →  读 $ROOT/experiments/smoke-base.yaml   （$PWD 的兄弟目录）
#     cd $ROOT         →  读 $ROOT/../tfdpfl-logs/*.log          （$PWD 的上一级）
# 这两处在容器里都**不存在**，报的是 FileNotFoundError —— 看起来像「文件没了」，
# 实际文件好好的，只是容器看不见。绑定仓库上一级可同时覆盖仓库本身与 tfdpfl-logs。
TFDPFL_BIND="${TFDPFL_BIND:-$(cd "$_TFDPFL_ROOT/.." && pwd)}"

if [ -n "${TFDPFL_PY:-}" ]; then
    # 显式覆盖，最高优先级
    PY="$TFDPFL_PY"
    PY_MODE="override"
elif command -v apptainer >/dev/null 2>&1 && [ -f "$TFDPFL_SIF" ]; then
    # 集群：容器存在 → 必须走容器
    # buildenv 提供 CUDA 运行时；module 在非集群环境不存在，故先探测。
    if type module >/dev/null 2>&1; then
        module load GPU/buildenv-nvhpc/25.9-cu13.0 2>/dev/null || \
            echo "[env] 警告：module load GPU/buildenv-nvhpc/25.9-cu13.0 失败，GPU 可能不可用" >&2
    fi
    PY="apptainer exec --nv --bind $TFDPFL_BIND $TFDPFL_SIF python3"
    PY_MODE="apptainer"

    # ── keras 数据缓存：绕开 $HOME 下的软链 ─────────────────────────────
    # `~/.keras/datasets` 常是指向共享盘的符号链接。容器里若看不到链接目标，
    # 它就是**悬空**的，keras 会抛一句与真实原因无关的
    # `FileExistsError: File exists: ~/.keras/datasets`。
    # 绑定根下有 data/datasets 就直接指过去，不碰软链。
    # 判据用 data/ 而不是 data/datasets/：datasets 子目录可能还没建出来
    # （$HOME 下的软链常常指向一个尚未创建的目标），建不出来才是真问题。
    if [ -z "${TFDPFL_KERAS_HOME:-}" ] && [ -d "$TFDPFL_BIND/data" ]; then
        mkdir -p "$TFDPFL_BIND/data/datasets" 2>/dev/null || true
        export TFDPFL_KERAS_HOME="$TFDPFL_BIND/data"
        # apptainer 默认继承宿主环境变量；APPTAINERENV_ 前缀是显式保证，两条都设，
        # 免得哪天容器换成 --cleanenv 就静默失效。
        export APPTAINERENV_TFDPFL_KERAS_HOME="$TFDPFL_KERAS_HOME"
        echo "[env] TFDPFL_KERAS_HOME = $TFDPFL_KERAS_HOME"
    fi
else
    # 本地：无 apptainer / 无容器 → 裸 python。
    # 本地没有 TF，需要 TF 的测试会自动 skip（见 tests/conftest.py）。
    PY="python3"
    PY_MODE="local"
fi

# 明确打印用的是哪条路径 —— 静默地切换执行环境是最难查的一类问题。
echo "[env] python = $PY   (mode=$PY_MODE)"

# ── 启动自检：容器能不能看见仓库 ───────────────────────────────────────────
# 一次容器启动（~1s），换掉一个跑到一半才 FileNotFoundError 的 GPU 作业。
# 曾经就是漏了 --bind，五个 smoke 全废在这上面。
if [ "$PY_MODE" = "apptainer" ] && [ -z "${TFDPFL_SKIP_ENV_CHECK:-}" ]; then
    if ! $PY -c "import sys, os; sys.exit(0 if os.path.isdir(sys.argv[1]) else 1)" \
            "$_TFDPFL_ROOT" 2>/dev/null; then
        echo "[env] ✗ 容器里看不到仓库目录 $_TFDPFL_ROOT" >&2
        echo "[env]   当前绑定根：--bind $TFDPFL_BIND" >&2
        echo "[env]   apptainer 默认只挂 \$PWD 和 \$HOME；用 TFDPFL_BIND=<能覆盖仓库和日志目录的路径> 覆盖。" >&2
        return 1 2>/dev/null || exit 1
    fi
fi
