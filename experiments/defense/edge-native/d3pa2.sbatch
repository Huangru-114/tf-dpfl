#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 02:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=32G
#SBATCH --job-name=d3pa2
# ══════════════════════════════════════════════════════════════════════════
# d3pa2.sbatch  —  P-a 追加探针（加大训练量）：一个 SNAP 集中 run 的快照（fedavg/analysis/d3_pa2.py）
#   规则：预注册 N-016（本目录 FINDINGS.md）、决定 D-140；参数清单本目录 d3pa2_configs.yaml。
#
#   sbatch experiments/defense/edge-native/d3pa2.sbatch SNAP__collocated__s42 calibrate     # 第 1 步（在仓库根目录提交）
#   bash experiments/defense/edge-native/d3pa2_submit.sh main                               # 第 2 步：冻结清单提交之后
#
# calibrate：只跑「训全部主干」× 12 档训练量 × 第 15 / 60 轮 → analysis/d3pa2/<run>.calibrate.json
# main：读 analysis/d3pa2_frozen.json 的那一档 × 三臂 × 第 15 / 60 轮 → analysis/d3pa2/<run>.main.json
# 只读快照、不训练联邦；快照 = $ROOT/../tfdpfl-dumps 里 metrics.json 的 manifest 指向的文件（sha 核对）。
# exit 3 = 有快照轮没过重现闸 → 先排查，结果不进判定。回传那一个 JSON；大日志留集群。
# 训练从 fedavg/ 跑（cd "$ROOT/fedavg"），容器经 cluster_env.sh（CLAUDE.md：--bind 仓库上一级，不要拆）。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

RUN_ID="${1:?用法: sbatch d3pa2.sbatch <run_id> <calibrate|main>}"
STAGE="${2:?缺 stage（calibrate / main）}"
case "$STAGE" in calibrate|main) ;; *) echo "[d3pa2] stage 只能是 calibrate / main：$STAGE"; exit 2 ;; esac

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

HERE="$ROOT/experiments/defense/edge-native"
DUMPS="${TFDPFL_DUMPDIR:-$ROOT/../tfdpfl-dumps}"
OUTDIR="$HERE/analysis/d3pa2"
OUT="$OUTDIR/${RUN_ID}.${STAGE}.json"
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
LOG="$LOGDIR/d3pa2_${RUN_ID}.${STAGE}.${SLURM_JOB_ID:-local}.log"
FROZEN_ARGS=()
if [ "$STAGE" = main ]; then
    [ -f "$HERE/analysis/d3pa2_frozen.json" ] || { echo "[d3pa2] 没有冻结清单 analysis/d3pa2_frozen.json（先 select 并提交）"; exit 2; }
    FROZEN_ARGS=(--frozen "$HERE/analysis/d3pa2_frozen.json")
fi
[ -f "$HERE/configs/$RUN_ID.yaml" ] || { echo "[d3pa2] 配置不存在: $HERE/configs/$RUN_ID.yaml"; exit 2; }
mkdir -p "$OUTDIR" "$LOGDIR"

echo "[d3pa2] run=$RUN_ID stage=$STAGE out=$OUT"
echo "[d3pa2] 大日志 -> $LOG （留集群，不进 git）"
cd "$ROOT/fedavg"
$PY -m analysis.d3_pa2 --stage "$STAGE" --run "$RUN_ID" --here "$HERE" --dumps-root "$DUMPS" \
    "${FROZEN_ARGS[@]}" --out "$OUT" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "[d3pa2] exit=$RC（3 = 有快照轮没过重现闸）；回传这一个文件：${OUT#"$ROOT"/}"
exit $RC
