#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 04:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=24G
#SBATCH --job-name=tfdpfl-smoke
#
# run_smoke.sh  —  集群侧 L2：一条命令跑完 + 自动吐出小 metrics.json
#
# **两种跑法**（上面的 #SBATCH 头在用 bash 跑时只是注释，不影响）：
#     sbatch run_smoke.sh <axis> <method> <attack_method> [defense] [exp_id] [framework]
#     bash   run_smoke.sh ...        # 已在计算节点上 / 交互式 salloc 时
#
# **不要**在登录节点用 bash 直接跑完整 smoke —— 它要 GPU 且跑几分钟。
# python 一律经 apptainer 容器（见 cluster_env.sh），裸 python3 在集群上一个库都找不到。
#
# 用法（仓库根目录）：
#     sbatch run_smoke.sh <axis> <method> <attack_method> [defense] [exp_id] [framework]
# 例：
#     sbatch run_smoke.sh attack neurotoxin neurotoxin none  exp001
#     sbatch run_smoke.sh defense flame     badnet     flame exp001
#     # 正交性那一格：攻击轴不变，只换 PFL 方法轴
#     sbatch run_smoke.sh attack orthogonalization badnet none exp001 hier_pfedme
#
# framework 缺省时用 smoke-base.yaml 里写死的 drift_correction（hierfedavg）。
# 合法取值见 main.py:FRAMEWORK_MAP（hier_fedavg / hier_pfedme / hier_ditto / …）。
#
# 产出：
#     experiments/<axis>/<method>/<exp_id>.metrics.json   ← 小，回传 git
#     ${TFDPFL_LOGDIR:-<仓库同级>/tfdpfl-logs}/smoke_<...>.log   ← 大，留集群
#     （不写 /tmp：sbatch 下 /tmp 在计算节点上，作业结束后从登录节点拿不到）
#
# 判据由你在 current-focus.md 里事先写死；本脚本只负责「客观、自动、无需人肉判读」。

set -euo pipefail

AXIS="${1:?用法: sbatch run_smoke.sh <axis> <method> <attack_method> [defense] [exp_id]}"
METHOD="${2:?缺 method}"
ATTACK="${3:?缺 attack_method}"
DEFENSE="${4:-none}"
EXP_ID="${5:-exp001}"
FRAMEWORK="${6:-}"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

OUTDIR="$ROOT/experiments/$AXIS/$METHOD"
# 大日志（留集群，永不进 git）。与 experiment_tf.sh 同一套约定：写到仓库同级的
# tfdpfl-logs/，**不要写 /tmp** —— sbatch 下 /tmp 在计算节点上，作业结束后
# 从登录节点根本拿不到，而 Step 1 的判据要 grep 这个日志。
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
LOG="$LOGDIR/smoke_${AXIS}_${METHOD}_${ATTACK}_${DEFENSE}${FRAMEWORK:+_$FRAMEWORK}.log"

mkdir -p "$OUTDIR" "$LOGDIR"

FW_ARGS=()
if [ -n "$FRAMEWORK" ]; then
    FW_ARGS=(--framework "$FRAMEWORK")
fi

echo "[smoke] axis=$AXIS method=$METHOD attack=$ATTACK defense=$DEFENSE" \
     "framework=${FRAMEWORK:-<yaml 默认>}"
echo "[smoke] 大日志 -> $LOG （留集群，不进 git）"

cd "$ROOT/fedavg"
set +e
$PY main.py \
    --config "$ROOT/experiments/smoke-base.yaml" \
    --attack_method "$ATTACK" \
    --defense "$DEFENSE" \
    "${FW_ARGS[@]}" \
    2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

cd "$ROOT"
$PY harness/collect_metrics.py "$LOG" -o "$OUTDIR/$EXP_ID.metrics.json"

echo "[smoke] run exit code = $RC"
echo "[smoke] 回传这一个文件即可：$OUTDIR/$EXP_ID.metrics.json"
exit $RC
