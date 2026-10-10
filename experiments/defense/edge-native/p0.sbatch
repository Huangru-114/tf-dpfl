#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 12:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=32G
#SBATCH --job-name=p0
# ══════════════════════════════════════════════════════════════════════════
# p0.sbatch  —  阶段三 P0：一个 SNAP run 的快照上的离线反事实（fedavg/analysis/p0_snapshot.py；规则 FINDINGS N-008）
#
#   sbatch experiments/defense/edge-native/p0.sbatch SNAP__collocated__s42 screen
#   sbatch experiments/defense/edge-native/p0.sbatch SNAP__collocated__s43 frozen        # 冻结清单 commit 之后
#   sbatch experiments/defense/edge-native/p0.sbatch SNAP__collocated__s42 screen --only pgd-e8-frozen-low --rounds 6   # 探路
#
# 只传参数，不改配置（陷阱 #19）：配置 = 本目录 configs/<run_id>.yaml（SNAP 的 materialize 结果），
# 快照 = $ROOT/../tfdpfl-dumps 里 metrics.json 的 manifest 指向的文件（sha 核对）。
# 输出：analysis/p0/<run_id>.<stage>.json（探路作业 → analysis/p0_probe/，不进判定）；大日志留集群。
# 训练从 fedavg/ 跑（cd "$ROOT/fedavg"），容器经 cluster_env.sh（CLAUDE.md：--bind 仓库上一级，不要拆）。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

RUN_ID="${1:?用法: sbatch p0.sbatch <run_id> <screen|frozen> [--only …] [--rounds …]}"
STAGE="${2:?缺 stage（screen / frozen）}"
shift 2

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

HERE="$ROOT/experiments/defense/edge-native"
CONFIG="$HERE/configs/$RUN_ID.yaml"
METRICS="$HERE/results/P2/SNAP/$RUN_ID.metrics.json"
DUMPS="${TFDPFL_DUMPDIR:-$ROOT/../tfdpfl-dumps}"
OUTDIR="$HERE/analysis/p0"
case " $* " in *" --only "*) OUTDIR="$HERE/analysis/p0_probe" ;; esac
OUT="$OUTDIR/${RUN_ID}.${STAGE}.json"
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
LOG="$LOGDIR/p0_${RUN_ID}.${STAGE}.${SLURM_JOB_ID:-local}.log"
FROZEN_ARGS=()
[ "$STAGE" = frozen ] && FROZEN_ARGS=(--frozen "$HERE/analysis/p0_frozen.json")
[ -f "$CONFIG" ] || { echo "[p0] 配置不存在: $CONFIG"; exit 2; }
[ -f "$METRICS" ] || { echo "[p0] metrics.json 不存在: $METRICS"; exit 2; }
mkdir -p "$OUTDIR" "$LOGDIR"

echo "[p0] run=$RUN_ID stage=$STAGE out=$OUT"
echo "[p0] 大日志 -> $LOG （留集群，不进 git）"
cd "$ROOT/fedavg"
$PY -m analysis.p0_snapshot --config "$CONFIG" --metrics "$METRICS" --dumps-root "$DUMPS" \
    --grid "$HERE/p0_configs.yaml" --stage "$STAGE" "${FROZEN_ARGS[@]}" --out "$OUT" "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "[p0] exit=$RC（3 = V0 不过，见 JSON 的 snapshots.*.v0）；回传这一个文件：${OUT#"$ROOT"/}"
exit $RC
