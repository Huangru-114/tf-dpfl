#!/bin/bash
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 24:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --job-name=tfdpfl-exp3
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
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/logs}"
LOG="$LOGDIR/exp3_${EXP_ID}.log"          # ← 带 exp_id，每格独立
mkdir -p "$OUTDIR" "$LOGDIR"

echo "[exp3] cell=$EXP_ID config=$CONFIG_REL seed=$SEED"
echo "[exp3] 大日志 -> $LOG （留集群，不进 git）"

# ⚠️ **cwd 放在仓库的上一级**（2026-09-08 实测）：apptainer **只自动挂 $PWD**，
#    而本仓库的作业要同时够到三处：
#        $ROOT/experiments/...      配置
#        $ROOT/logs/...             日志
#        $ROOT/../data/datasets/    keras 数据缓存（TFDPFL_KERAS_HOME 指向它）
#    只有把 cwd 放到 $ROOT/.. 才一次覆盖全部。cwd 留在 $ROOT 时，`../data`
#    在容器里 isdir=False，于是 resolve_keras_home 静默跳过 TFDPFL_KERAS_HOME、
#    回退到 ~/.keras 的悬空软链，最后 mkdir 撞上容器只读根：
#        OSError: [Errno 30] Read-only file system: '.../ziangg/data'
#    所有仓库内路径因此一律写成 "$ROOT/..." 的绝对形式。
#    `$PY "$ROOT/fedavg/main.py"` 的 sys.path[0] 仍是 $ROOT/fedavg，import 不变。
cd "$ROOT/.."
set +e
# **必须走 $PY**（cluster_env.sh 解析出的 apptainer 容器）。裸 python3 在
# Arrhenius 上一个库都找不到，而且报的是 numpy/tensorflow 的 ImportError,
# 看起来像「依赖没装」。守卫：tests/test_cluster_env_usage.py。
$PY "$ROOT/fedavg/main.py" \
    --config "$CONFIG_ABS" \
    --attack_method badpfl \
    --defense none \
    --framework hier_fedavg_fedrep \
    --seed "$SEED" \
    2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

cd "$ROOT/.."
METRICS="$OUTDIR/$EXP_ID.metrics.json"
$PY "$ROOT/harness/collect_metrics.py" "$LOG" -o "$METRICS"
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
