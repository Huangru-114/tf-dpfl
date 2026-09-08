#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 12:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=24G
#SBATCH --job-name=calib
# ══════════════════════════════════════════════════════════════════════════
# calib_cell.sbatch  —  Stage B 标定的**一个格子**（一个 local_epochs × 一个 seed）。
#   由 run_calibration.sh 提交，一般不手动调用。
#
#   sbatch experiments/calibration/calib_cell.sbatch <config_rel> <seed> <exp_id>
#
# **与 exp3_cell.sbatch 逐字同构**（同一套 SLURM 头、同样 source cluster_env.sh
# 拿 $PY、同样 cd $ROOT/fedavg、同样带 exp_id 的独立日志、同样注入 exit_code
# 供断点续跑），只差 --time 12h 与产物目录。
#
# ⚠️ 不要「优化」cwd 或日志位置：`cd $ROOT/fedavg` + 上一级的 tfdpfl-logs
#    之所以成立，是因为 cluster_env.sh 的 `--bind <仓库上一级>` 把整片都挂进
#    容器了。2026-09 我们一度以为 --bind 无效而把这些全改掉，结果连续三轮
#    FileNotFoundError —— 见 CLAUDE.md 陷阱 #17。
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
$PY main.py \
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

echo "[calib] run exit code = $RC"
echo "[calib] 回传这一个文件即可：$OUTDIR/$EXP_ID.metrics.json"
exit $RC
