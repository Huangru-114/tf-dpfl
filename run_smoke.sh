#!/bin/bash
# run_smoke.sh  —  集群侧 L2：一条命令跑完 + 自动吐出小 metrics.json
#
# 用法（仓库根目录）：
#     bash run_smoke.sh <axis> <method> <attack_method> [defense] [exp_id]
# 例：
#     bash run_smoke.sh attack neurotoxin neurotoxin none      exp001
#     bash run_smoke.sh defense flame     badnet     flame     exp001
#
# 产出：
#     experiments/<axis>/<method>/<exp_id>.metrics.json   ← 小，回传 git
#     /tmp/smoke_<...>.log                                ← 大，留集群
#
# 判据由你在 current-focus.md 里事先写死；本脚本只负责「客观、自动、无需人肉判读」。

set -euo pipefail

AXIS="${1:?用法: bash run_smoke.sh <axis> <method> <attack_method> [defense] [exp_id]}"
METHOD="${2:?缺 method}"
ATTACK="${3:?缺 attack_method}"
DEFENSE="${4:-none}"
EXP_ID="${5:-exp001}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="$ROOT/experiments/$AXIS/$METHOD"
LOG="/tmp/smoke_${AXIS}_${METHOD}_${ATTACK}_${DEFENSE}.log"

mkdir -p "$OUTDIR"

echo "[smoke] axis=$AXIS method=$METHOD attack=$ATTACK defense=$DEFENSE"
echo "[smoke] 大日志 -> $LOG （留集群，不进 git）"

cd "$ROOT/fedavg"
set +e
python main.py \
    --config "$ROOT/experiments/smoke-base.yaml" \
    --attack_method "$ATTACK" \
    --defense "$DEFENSE" \
    2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

cd "$ROOT"
python harness/collect_metrics.py "$LOG" -o "$OUTDIR/$EXP_ID.metrics.json"

echo "[smoke] run exit code = $RC"
echo "[smoke] 回传这一个文件即可：$OUTDIR/$EXP_ID.metrics.json"
exit $RC
