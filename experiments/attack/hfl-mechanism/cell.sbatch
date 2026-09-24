#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 24:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=24G
#SBATCH --job-name=exp3v2
# ══════════════════════════════════════════════════════════════════════════
# cell.sbatch  —  跑改版实验 3 的**一个 run**。由 submit.sh 提交，一般不手动调用。
#
#   sbatch experiments/attack/hfl-mechanism/cell.sbatch <config_rel> <run_id> <metrics_rel>
#
# **只传 --config，不传任何覆盖参数**。旧方案的 exp3_cell.sbatch 固定传
# `--defense none`，静默盖掉了 yaml 里的防御（FINDINGS F-001）。这里的配置由
# harness/registry.py materialize 出来，已经是完整的（含 seed 与 meta 块），
# 跑出来的 config_sha 必须等于 INDEX.tsv 里的值，submit.sh 才算它完成。
# 守卫：tests/test_registry.py::test_cell_sbatch_passes_no_overrides
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

CONFIG_REL="${1:?用法: sbatch cell.sbatch <config_rel> <run_id> <metrics_rel>}"
RUN_ID="${2:?缺 run_id}"
METRICS_REL="${3:?缺 metrics_rel}"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

CONFIG_ABS="$ROOT/$CONFIG_REL"
[ -f "$CONFIG_ABS" ] || { echo "[exp3v2] 配置不存在: $CONFIG_ABS"; exit 2; }

METRICS="$ROOT/$METRICS_REL"
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
LOG="$LOGDIR/exp3v2_${RUN_ID}.log"          # 带 run_id，每个 run 独立
mkdir -p "$(dirname "$METRICS")" "$LOGDIR"

echo "[exp3v2] run=$RUN_ID config=$CONFIG_REL"
echo "[exp3v2] 大日志 -> $LOG （留集群，不进 git）"

cd "$ROOT/fedavg"
set +e
$PY main.py --config "$CONFIG_ABS" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

cd "$ROOT"
$PY harness/collect_metrics.py "$LOG" -o "$METRICS"
# 注入 exit_code：submit.sh 与 harness/status.py 都要求它存在且为 0 才算完成。
$PY - "$METRICS" "$RC" <<'PY'
import json, sys
p, rc = sys.argv[1], int(sys.argv[2])
d = json.load(open(p)); d["exit_code"] = rc
json.dump(d, open(p, "w"), indent=2, ensure_ascii=False)
PY

echo "[exp3v2] run exit code = $RC"
echo "[exp3v2] 回传这一个文件即可：$METRICS_REL"
exit $RC
