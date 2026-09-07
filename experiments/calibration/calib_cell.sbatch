#!/bin/bash
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 12:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --job-name=tfdpfl-calib
# ══════════════════════════════════════════════════════════════════════════
# calib_cell.sbatch  —  Stage B 标定的**一个格子**（一个 local_epochs × 一个 seed）。
#   由 run_calibration.sh 提交，一般不手动调用。
#
#   sbatch experiments/calibration/calib_cell.sbatch <config_rel> <seed> <exp_id>
#
# 与 exp3_cell.sbatch 逐字同构（同一套 Arrhenius SLURM 头、同样带 exp_id 的
# 独立日志、同样注入 exit_code 供断点续跑），只差 --time 12h（标定预算是
# 150 有效轮，不是 400）与产物目录。
#
# ⚠️ **所有 python 一律走 `$PY`**（cluster_env.sh 解析出的 apptainer 容器，
#    路径见那个文件里的 TFDPFL_SIF）。裸 `python3` 在 Arrhenius 上一个库都
#    找不到。仓库里一度有两个脚本用裸 python3（Alvis 时期的遗留），已改回；
#    守卫 tests/test_cluster_env_usage.py 挡住重演。
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
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/logs}"
LOG="$LOGDIR/calib_${EXP_ID}.log"          # ← 带 exp_id，每格独立
mkdir -p "$OUTDIR" "$LOGDIR"

echo "[calib] cell=$EXP_ID config=$CONFIG_REL seed=$SEED"
echo "[calib] 大日志 -> $LOG （留集群，不进 git）"

# ⚠️ **cwd 必须留在仓库根**（2026-09 实测）：apptainer **只自动挂 $PWD**。
#    `cd $ROOT/fedavg` 之后，兄弟目录 $ROOT/experiments/ 与上一级的日志目录
#    都在 $PWD 之外 -> 容器里看不见 -> FileNotFoundError，而文件明明在。
#    `$PY fedavg/main.py` 的 sys.path[0] 仍是 fedavg/，import 一行都不用改。
cd "$ROOT"
set +e
$PY fedavg/main.py \
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
