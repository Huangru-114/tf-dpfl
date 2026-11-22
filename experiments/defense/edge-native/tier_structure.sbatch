#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=24G
#SBATCH --job-name=d0b
# ══════════════════════════════════════════════════════════════════════════
# tier_structure.sbatch  —  补充实验 D0b（N-015 / D-135）：TBS k1-snap 三个种子的快照上核对三层个性化的结构
#
#   sbatch experiments/defense/edge-native/tier_structure.sbatch
#
# 前提：TBS__k1-snap__s42–44 已回传（results/P2/TBS/ 里有 metrics.json，快照在 $ROOT/../tfdpfl-dumps）。
# 只读快照、不训练（fedavg/analysis/tier_structure.py；比较是纯 numpy，TF 只用来建模型取索引）。
# 输出：analysis/d0_structure.json（回传这一个文件）；任何一条不过 → exit 3，停下、6.1 需要重新审视（PREREG-v2 §二）。
# 训练从 fedavg/ 跑（cd "$ROOT/fedavg"），容器经 cluster_env.sh（CLAUDE.md：--bind 仓库上一级，不要拆）。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

HERE="$ROOT/experiments/defense/edge-native"
DUMPS="${TFDPFL_DUMPDIR:-$ROOT/../tfdpfl-dumps}"
OUT="$HERE/analysis/d0_structure.json"
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
LOG="$LOGDIR/d0b_tier_structure.${SLURM_JOB_ID:-local}.log"
for s in 42 43 44; do
    [ -f "$HERE/results/P2/TBS/TBS__k1-snap__s$s.metrics.json" ] || { echo "[d0b] TBS__k1-snap__s$s 还没回传"; exit 2; }
done
mkdir -p "$LOGDIR" "$HERE/analysis"

echo "[d0b] out=$OUT；大日志 -> $LOG （留集群，不进 git）"
cd "$ROOT/fedavg"
$PY -m analysis.tier_structure --run TBS__k1-snap__s42 --run TBS__k1-snap__s43 --run TBS__k1-snap__s44 \
    --here "$HERE" --dumps-root "$DUMPS" --out "$OUT" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "[d0b] exit=$RC（3 = 有一条结构核查不过）；回传这一个文件：${OUT#"$ROOT"/}"
exit $RC
