#!/bin/bash
#SBATCH --account=naiss2026-4-650
#SBATCH --job-name=sweep_phase3
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase3_%j.log
# ══════════════════════════════════════════════════════════════════════════
# exp3_cell.sbatch  —  跑 Experiment 3 的**一个格子**（一个 config × 一个 seed）。
#   由 run_exp3.sh 提交，一般不手动调用。
#
#   sbatch experiments/attack/hfl-propagation/exp3_cell.sbatch <config_rel> <seed> <exp_id>
#
# 与 run_full.sh 的关键区别：**日志名带 exp_id → 每格独立**。run_full.sh 的日志名只由
# axis/method/attack/defense/framework 决定，本实验 8 格这些参数全相同 → 并发会互相覆盖
# 日志、collect_metrics 读错 → 每格指标全乱。所以本实验必须走这个带 exp_id 的独立日志。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

CONFIG_REL="${1:?用法: sbatch exp3_cell.sbatch <config_rel> <seed> <exp_id>}"
SEED="${2:?缺 seed}"
EXP_ID="${3:?缺 exp_id}"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

CONFIG_ABS="$ROOT/$CONFIG_REL"
[ -f "$CONFIG_ABS" ] || { echo "[exp3] 配置不存在: $CONFIG_ABS"; exit 2; }

OUTDIR="$ROOT/experiments/attack/hfl-propagation/results"
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
LOG="$LOGDIR/exp3_${EXP_ID}.log"          # ← 带 exp_id，每格独立
mkdir -p "$OUTDIR" "$LOGDIR"

echo "[exp3] cell=$EXP_ID config=$CONFIG_REL seed=$SEED"
echo "[exp3] 大日志 -> $LOG （留集群，不进 git）"

cd "$ROOT/fedavg"
set +e
python3 main.py \
    --config "$CONFIG_ABS" \
    --attack_method badpfl \
    --defense none \
    --framework hier_fedavg_fedrep \
    --seed "$SEED" \
    2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

cd "$ROOT"
METRICS="$OUTDIR/$EXP_ID.metrics.json"
$PY harness/collect_metrics.py "$LOG" -o "$METRICS"
# 注入 exit_code（run_full.sh 不写，matrix 的 experiment_tf.sh 才写）→ run_exp3.sh 靠它断点续跑：
# 只有 exit_code==0 的格子才算完成，跑挂的（即使写出了带 errors 的 metrics.json）会被重试。
$PY - "$METRICS" "$RC" <<'PY'
import json, sys
p, rc = sys.argv[1], int(sys.argv[2])
d = json.load(open(p)); d["exit_code"] = rc
json.dump(d, open(p, "w"), indent=2, ensure_ascii=False)
PY

echo "[exp3] run exit code = $RC"
echo "[exp3] 回传这一个文件即可：$OUTDIR/$EXP_ID.metrics.json"
exit $RC
