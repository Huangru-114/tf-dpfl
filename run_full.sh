#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 12:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=24G
#SBATCH --job-name=tfdpfl-full
#
# run_full.sh  —  与 run_smoke.sh 同结构的 L2 **全长** run 驱动。
#
# 与 run_smoke.sh 的唯一区别：
#   1. 第 7 个参数 = 配置文件路径（run_smoke.sh 硬编码 smoke-base.yaml，全长跑不能用）。
#   2. -t 12:00:00（全长 run 可能数小时，smoke 的 4h 不够）。
#
# 用法（仓库根目录）：
#   sbatch run_full.sh <axis> <method> <attack> [defense] [exp_id] [framework] [config_rel_path]
# 例（Bad-PFL × fedrep 全长）：
#   sbatch run_full.sh attack bad-pfl badpfl none exp002 hier_fedavg_fedrep \
#          experiments/attack/bad-pfl/full.yaml
#
# python 一律经 apptainer 容器（cluster_env.sh），裸 python3 在集群上找不到库。
# 产出：experiments/<axis>/<method>/<exp_id>.metrics.json（小，回传 git）
#       大日志留集群（tfdpfl-logs/，永不进 git）。

set -euo pipefail

AXIS="${1:?用法: sbatch run_full.sh <axis> <method> <attack> [defense] [exp_id] [framework] [config]}"
METHOD="${2:?缺 method}"
ATTACK="${3:?缺 attack_method}"
DEFENSE="${4:-none}"
EXP_ID="${5:-exp002}"
FRAMEWORK="${6:-}"
CONFIG_REL="${7:-experiments/smoke-base.yaml}"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

CONFIG_ABS="$ROOT/$CONFIG_REL"
[ -f "$CONFIG_ABS" ] || { echo "[full] 配置不存在: $CONFIG_ABS"; exit 2; }

OUTDIR="$ROOT/experiments/$AXIS/$METHOD"
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
LOG="$LOGDIR/full_${AXIS}_${METHOD}_${ATTACK}_${DEFENSE}${FRAMEWORK:+_$FRAMEWORK}.log"

mkdir -p "$OUTDIR" "$LOGDIR"

FW_ARGS=()
if [ -n "$FRAMEWORK" ]; then
    FW_ARGS=(--framework "$FRAMEWORK")
fi

echo "[full] axis=$AXIS method=$METHOD attack=$ATTACK defense=$DEFENSE" \
     "framework=${FRAMEWORK:-<yaml 默认>} config=$CONFIG_REL"
echo "[full] 大日志 -> $LOG （留集群，不进 git）"

cd "$ROOT/fedavg"
set +e
$PY main.py \
    --config "$CONFIG_ABS" \
    --attack_method "$ATTACK" \
    --defense "$DEFENSE" \
    "${FW_ARGS[@]}" \
    2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

cd "$ROOT"
$PY harness/collect_metrics.py "$LOG" -o "$OUTDIR/$EXP_ID.metrics.json"

echo "[full] run exit code = $RC"
echo "[full] 回传这一个文件即可：$OUTDIR/$EXP_ID.metrics.json"
exit $RC
