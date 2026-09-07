#!/bin/bash
#SBATCH --account=naiss2026-4-650
#SBATCH --job-name=calib_stageB
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/calib_%j.log
# ══════════════════════════════════════════════════════════════════════════
# calib_cell.sbatch  —  Stage B 标定的**一个格子**（一个 local_epochs × 一个 seed）。
#   由 run_calibration.sh 提交，一般不手动调用。
#
#   sbatch experiments/calibration/calib_cell.sbatch <config_rel> <seed> <exp_id>
#
# 与 exp3_cell.sbatch 逐字同构（同一套 Alvis header、同样带 exp_id 的独立日志、
# 同样注入 exit_code 供断点续跑），只差 --time 12h（标定预算是 150 有效轮，
# 不是 400）与产物目录。
#
# ⚠️ **训练用裸 `python3` 而不是 cluster_env.sh 的 `$PY`**，与 exp3_cell.sbatch 一致
#    —— 那是 commit 7ae2966（"exp3 alvis part"）刻意改的：Alvis 上跑的是仓库里的
#    virtualenv，不是 apptainer 容器。`$PY` 只留给 collect_metrics.py（纯 stdlib，
#    两种解释器下都能跑）。**改这一行之前先确认你在哪个集群上。**
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

CONFIG_REL="${1:?用法: sbatch calib_cell.sbatch <config_rel> <seed> <exp_id>}"
SEED="${2:?缺 seed}"
EXP_ID="${3:?缺 exp_id}"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

CONFIG_ABS="$ROOT/$CONFIG_REL"
[ -f "$CONFIG_ABS" ] || { echo "[calib] 配置不存在: $CONFIG_ABS"; exit 2; }

OUTDIR="$ROOT/experiments/calibration/results"
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
LOG="$LOGDIR/calib_${EXP_ID}.log"          # ← 带 exp_id，每格独立
mkdir -p "$OUTDIR" "$LOGDIR"

echo "[calib] cell=$EXP_ID config=$CONFIG_REL seed=$SEED"
echo "[calib] 大日志 -> $LOG （留集群，不进 git）"

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
$PY - "$METRICS" "$RC" <<'PYEOF'
import json, sys
p, rc = sys.argv[1], int(sys.argv[2])
d = json.load(open(p)); d["exit_code"] = rc
json.dump(d, open(p, "w"), indent=2, ensure_ascii=False)
PYEOF

echo "[calib] run exit code = $RC"
echo "[calib] 回传这一个文件即可：$OUTDIR/$EXP_ID.metrics.json"
exit $RC
