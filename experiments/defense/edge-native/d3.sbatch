#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 03:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=32G
#SBATCH --job-name=d3
# ══════════════════════════════════════════════════════════════════════════
# d3.sbatch  —  阶段三 D3 离线探针：一个 run 的快照上的 P-a / P-b / P-c（fedavg/analysis/d3_probes.py）
#   规则 PREREG-v2 §五（预注册 N-015 @ 25163a5）；实现取舍 DECISIONS D-138；参数清单本目录 d3_configs.yaml。
#
#   sbatch experiments/defense/edge-native/d3.sbatch SNAP__collocated__s42                 # 在仓库根目录提交
#   sbatch experiments/defense/edge-native/d3.sbatch SNAP__collocated__s42 --rounds 15     # 探路（只跑一轮，结果不进判定）
#   bash experiments/defense/edge-native/d3_submit.sh                                       # 9 个 run 一次交完
#
# 这个 run 跑哪些探针、哪几轮由 d3_configs.yaml 决定：SNAP 集中 = P-a + P-b 参照 + P-c；SNAP 分散 = P-c；TBS k1-snap = P-b + P-c。
# 只读快照、不训练；快照 = $ROOT/../tfdpfl-dumps 里 metrics.json 的 manifest 指向的文件（sha 核对）。
# 输出：analysis/d3/<run_id>.json（探路 → analysis/d3_probe/，不进判定）；大日志留集群。
# exit 3 = 有快照轮没过重现闸（离线重算的第 t 轮主列与 run 记录对不上）→ 先排查，结果不进判定。
# 训练从 fedavg/ 跑（cd "$ROOT/fedavg"），容器经 cluster_env.sh（CLAUDE.md：--bind 仓库上一级，不要拆）。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

RUN_ID="${1:?用法: sbatch d3.sbatch <run_id> [--rounds …]}"
shift 1

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

HERE="$ROOT/experiments/defense/edge-native"
DUMPS="${TFDPFL_DUMPDIR:-$ROOT/../tfdpfl-dumps}"
OUTDIR="$HERE/analysis/d3"
case " $* " in *" --rounds "*) OUTDIR="$HERE/analysis/d3_probe" ;; esac
OUT="$OUTDIR/${RUN_ID}.json"
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
LOG="$LOGDIR/d3_${RUN_ID}.${SLURM_JOB_ID:-local}.log"
[ -f "$HERE/configs/$RUN_ID.yaml" ] || { echo "[d3] 配置不存在: $HERE/configs/$RUN_ID.yaml"; exit 2; }
mkdir -p "$OUTDIR" "$LOGDIR"

echo "[d3] run=$RUN_ID out=$OUT"
echo "[d3] 大日志 -> $LOG （留集群，不进 git）"
cd "$ROOT/fedavg"
$PY -m analysis.d3_probes --run "$RUN_ID" --here "$HERE" --dumps-root "$DUMPS" --out "$OUT" "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "[d3] exit=$RC（3 = 有快照轮没过重现闸，见 JSON 的 rounds.*.reproduce）；回传这一个文件：${OUT#"$ROOT"/}"
exit $RC
