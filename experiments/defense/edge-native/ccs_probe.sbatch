#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=32G
#SBATCH --job-name=ccsprobe
# ══════════════════════════════════════════════════════════════════════════
# ccs_probe.sbatch  —  阶段三 SA-C：CCS 对抗样本「标准化空间 vs 像素空间」的离线对比（fedavg/analysis/ccs_space_probe.py；D-109 Q1）
#
#   sbatch experiments/defense/edge-native/ccs_probe.sbatch                         # 缺省 SNAP__collocated__s42 第 15 轮
#   sbatch experiments/defense/edge-native/ccs_probe.sbatch SNAP__collocated__s42 6
#
# 零训练：SNAP 快照 → 若干良性端的 fresh-PM → 四个变体（input / pixel × 0 / 15 步）的对抗样本统计量 + 每批墙钟。
# 只传参数，不改配置（陷阱 #19）。输出：analysis/ccs_space_probe.<run_id>.r<round>.json（回传这一个文件；大日志留集群）。
# 训练从 fedavg/ 跑（cd "$ROOT/fedavg"），容器经 cluster_env.sh（CLAUDE.md：--bind 仓库上一级，不要拆）。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

RUN_ID="${1:-SNAP__collocated__s42}"
ROUND="${2:-15}"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

HERE="$ROOT/experiments/defense/edge-native"
CONFIG="$HERE/configs/$RUN_ID.yaml"
METRICS="$HERE/results/P2/SNAP/$RUN_ID.metrics.json"
DUMPS="${TFDPFL_DUMPDIR:-$ROOT/../tfdpfl-dumps}"
OUT="$HERE/analysis/ccs_space_probe.${RUN_ID}.r${ROUND}.json"
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
LOG="$LOGDIR/ccsprobe_${RUN_ID}.r${ROUND}.${SLURM_JOB_ID:-local}.log"
[ -f "$CONFIG" ] || { echo "[ccs-probe] 配置不存在: $CONFIG"; exit 2; }
[ -f "$METRICS" ] || { echo "[ccs-probe] metrics.json 不存在: $METRICS"; exit 2; }
mkdir -p "$LOGDIR"

echo "[ccs-probe] run=$RUN_ID round=$ROUND out=$OUT"
echo "[ccs-probe] 大日志 -> $LOG （留集群，不进 git）"
cd "$ROOT/fedavg"
$PY -m analysis.ccs_space_probe --config "$CONFIG" --metrics "$METRICS" --dumps-root "$DUMPS" \
    --round "$ROUND" --out "$OUT" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "[ccs-probe] exit=$RC；回传这一个文件：${OUT#"$ROOT"/}"
exit $RC
