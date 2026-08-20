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
# ══════════════════════════════════════════════════════════════════════════

TFDPFL_SIF="${TFDPFL_SIF:-/nobackup/proj/disk/naiss2025-22-1095/personal/ziangg/torch_fl.sif}"

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
    PY="apptainer exec --nv $TFDPFL_SIF python3"
    PY_MODE="apptainer"
else
    # 本地：无 apptainer / 无容器 → 裸 python。
    # 本地没有 TF，需要 TF 的测试会自动 skip（见 tests/conftest.py）。
    PY="python3"
    PY_MODE="local"
fi

# 明确打印用的是哪条路径 —— 静默地切换执行环境是最难查的一类问题。
echo "[env] python = $PY   (mode=$PY_MODE)"
