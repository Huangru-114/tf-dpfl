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
#
# **分三步查，因为这三种病的修法完全不同**（旧版只报「看不到仓库」这一种，
# 而且用 2>/dev/null 把 apptainer 自己的报错吞了 —— 等于自己把诊断信息删掉）：
#   1. 容器根本起不来（--nv 没驱动 / .sif 读不了 / apptainer 配置问题）
#   2. 容器起得来，但 --bind 那个挂载点在镜像里不存在且没有 overlay/underlay
#      → apptainer 直接失败。这是 HPC 上最常见的一种。
#   3. 挂上了，但路径里有软链，容器里解析不到目标
_tfdpfl_try() {           # $* = 在容器里跑的 python 语句；回显 stderr
    apptainer exec --nv ${1:+--bind "$1"} "$TFDPFL_SIF" python3 -c "$2" 2>&1
}

if [ "$PY_MODE" = "apptainer" ] && [ -z "${TFDPFL_SKIP_ENV_CHECK:-}" ]; then
    _tfdpfl_ok=1
    if ! $PY -c "import sys, os; sys.exit(0 if os.path.isdir(sys.argv[1]) else 1)" \
            "$_TFDPFL_ROOT" >/dev/null 2>&1; then
        _tfdpfl_ok=0
    fi

    if [ "$_tfdpfl_ok" -eq 0 ]; then
        echo "[env] ✗ 容器里看不到仓库目录 $_TFDPFL_ROOT" >&2
        echo "[env]   当前绑定根：--bind $TFDPFL_BIND" >&2
        echo "[env]   ── 正在分步定位（三种病因修法完全不同）──" >&2

        # 步骤 1：不带 --bind，容器本身起不起得来
        _out="$(_tfdpfl_try "" "print('container-ok')")"
        if ! printf '%s' "$_out" | grep -q "container-ok"; then
            echo "[env]   [1/3] ✗ 容器**根本起不来**（与 --bind 无关）。apptainer 原话：" >&2
            printf '%s\n' "$_out" | sed 's/^/[env]        /' >&2
            echo "[env]   常见原因：登录节点没有 NVIDIA 驱动 -> 去掉 --nv 试试；" >&2
            echo "[env]             或 .sif 读不到 / apptainer 配置问题。" >&2
            echo "[env]   临时绕过（只做不需要 GPU 的事时）：TFDPFL_SKIP_ENV_CHECK=1" >&2
            return 1 2>/dev/null || exit 1
        fi
        echo "[env]   [1/3] ✓ 容器能起来（不带 --bind）" >&2

        # 步骤 2：带 --bind 还能不能起来
        _out="$(_tfdpfl_try "$TFDPFL_BIND" "print('bind-ok')")"
        if ! printf '%s' "$_out" | grep -q "bind-ok"; then
            echo "[env]   [2/3] ✗ 加上 --bind $TFDPFL_BIND 之后容器起不来。apptainer 原话：" >&2
            printf '%s\n' "$_out" | sed 's/^/[env]        /' >&2
            echo "[env]   最常见原因：这个挂载点在镜像里不存在，而 apptainer 没开" >&2
            echo "[env]   overlay/underlay，创建不出多级目录。修法：" >&2
            echo "[env]     换一个层级更浅、镜像里已存在的绑定根，例如" >&2
            echo "[env]       TFDPFL_BIND=/nobackup  bash <你的脚本>" >&2
            echo "[env]   ⚠️ 不要用 src:dst 改挂载点（如 ...:/mnt）：本仓库的脚本" >&2
            echo "[env]     全程用**宿主机的绝对路径**，改了挂载点之后容器里" >&2
            echo "[env]     那些路径就不存在了，只是把错误换个地方报。" >&2
            return 1 2>/dev/null || exit 1
        fi
        echo "[env]   [2/3] ✓ 带 --bind 也能起来 -> 挂载本身没问题" >&2

        # 步骤 3：挂上了但路径看不见 -> 多半是软链
        _real_root="$(readlink -f "$_TFDPFL_ROOT" 2>/dev/null || echo "$_TFDPFL_ROOT")"
        _real_bind="$(readlink -f "$TFDPFL_BIND" 2>/dev/null || echo "$TFDPFL_BIND")"
        echo "[env]   [3/3] 挂载正常但路径不可见 -> 检查软链：" >&2
        echo "[env]        仓库   $_TFDPFL_ROOT" >&2
        echo "[env]        readlink -f -> $_real_root" >&2
        echo "[env]        绑定根 $TFDPFL_BIND" >&2
        echo "[env]        readlink -f -> $_real_bind" >&2
        if [ "$_real_bind" != "$TFDPFL_BIND" ] || [ "$_real_root" != "$_TFDPFL_ROOT" ]; then
            echo "[env]   ⇒ 路径里有**软链**。容器绑的是字面路径，链接目标不在里面。" >&2
            echo "[env]     修法：绑真实路径 —— TFDPFL_BIND='$_real_bind' bash <你的脚本>" >&2
        else
            echo "[env]   ⇒ 没有软链。请把上面三步的输出发我。" >&2
        fi
        return 1 2>/dev/null || exit 1
    fi
fi
