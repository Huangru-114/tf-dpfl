#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 24:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --job-name=tfdpfl-matrix
#SBATCH --output=/dev/null      # 每格自己写日志到 $SCRATCH，见下（大日志绝不进 git）
#SBATCH --error=/dev/null
#
# ══════════════════════════════════════════════════════════════════════════
# experiment_tf.sh  —  三维实验矩阵的**单格执行器**（SLURM job array 的一个 task）
#
# 不要直接 sbatch 这个文件；用：
#     bash submit_matrix.sh              # 提交所有未完成的格子
#     bash submit_matrix.sh --dry-run    # 只看要提交什么
#
# 每个 array task 从 manifest 里取第 $SLURM_ARRAY_TASK_ID 行，跑那一格，
# 然后把**全量日志压成一个 KB 级 metrics.json**。回程只带这个 json。
#
# 手动跑单格（调试用）：
#     MANIFEST=experiments/matrix/manifest.tsv LINE=3 bash experiment_tf.sh
# ══════════════════════════════════════════════════════════════════════════

set -uo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"

MANIFEST="${MANIFEST:-experiments/matrix/manifest.tsv}"
LINE="${LINE:-${SLURM_ARRAY_TASK_ID:-1}}"

if [ ! -f "$MANIFEST" ]; then
    echo "[ERROR] 找不到 manifest：$MANIFEST（先跑 bash submit_matrix.sh）" >&2
    exit 2
fi

# manifest 每行： cell_id \t attack \t method \t defense \t seed \t base_config \t overrides
IFS=$'\t' read -r CELL ATTACK METHOD DEFENSE SEED BASE_CONFIG OVERRIDES \
    < <(sed -n "${LINE}p" "$MANIFEST")

if [ -z "${CELL:-}" ]; then
    echo "[ERROR] manifest 第 $LINE 行为空" >&2
    exit 2
fi

# ── 产物落点 ──────────────────────────────────────────────────────────────
# 小结果（进 git）：experiments/matrix/<cell>.metrics.json
# 大日志（留集群）：$LOGDIR/<cell>.log   —— 永远不要放进仓库
OUTDIR="$ROOT/experiments/matrix"
LOGDIR="${TFDPFL_LOGDIR:-${SLURM_SUBMIT_DIR:-$ROOT}/../tfdpfl-logs}"
mkdir -p "$OUTDIR" "$LOGDIR"

METRICS="$OUTDIR/$CELL.metrics.json"
LOG="$LOGDIR/$CELL.log"

# ── 断点续跑：已完成的格子直接跳过 ─────────────────────────────────────────
# 只跳过**成功**完成的格子；跑挂的（exit_code != 0）必须重试，
# 否则失败会被 metrics.json 的存在永久掩盖。
if [ -s "$METRICS" ] && grep -q '"exit_code": *0' "$METRICS" \
   && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] $CELL 已成功完成，跳过（FORCE=1 可强制重跑）"
    exit 0
fi

echo "════════════════════════════════════════════════════════════"
echo " cell     : $CELL"
echo " attack   : $ATTACK"
echo " method   : $METHOD"
echo " defense  : $DEFENSE"
echo " seed     : $SEED"
echo " job      : ${SLURM_JOB_ID:-local}[${SLURM_ARRAY_TASK_ID:-.}]"
echo " node     : $(hostname)"
echo " log      : $LOG   (留集群，不进 git)"
echo " metrics  : $METRICS  (回传 git)"
echo "════════════════════════════════════════════════════════════"

# ── 环境 ─────────────────────────────────────────────────────────────────
# python 必须在 apptainer 容器里跑；容器路径与 module load 统一在 cluster_env.sh。
# （此前这里硬写 $ROOT/tensorflow.sif —— 与实际容器路径不一致。）
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"
RUN="$PY"

# ── 组装 override 参数 ────────────────────────────────────────────────────
OVERRIDE_ARGS=(--override "training.drift_correction=$METHOD")
if [ -n "${OVERRIDES:-}" ]; then
    IFS=';' read -ra EXTRA <<< "$OVERRIDES"
    for kv in "${EXTRA[@]}"; do
        kv="$(echo "$kv" | xargs)"          # 去空白
        [ -n "$kv" ] && OVERRIDE_ARGS+=(--override "$kv")
    done
fi

# ── 跑（cwd=fedavg，与 import 语义一致）───────────────────────────────────
cd "$ROOT/fedavg"
$RUN main.py \
    --config "$ROOT/$BASE_CONFIG" \
    --attack_method "$ATTACK" \
    --defense "$DEFENSE" \
    --seed "$SEED" \
    "${OVERRIDE_ARGS[@]}" \
    > "$LOG" 2>&1
RC=$?

# ── 回收：全量日志 → 一个小 json ─────────────────────────────────────────
cd "$ROOT"
$RUN harness/collect_metrics.py "$LOG" -o "$METRICS"
CRC=$?

# 把这一格的坐标写进 json，让汇总脚本不用解析文件名
$RUN - "$METRICS" "$CELL" "$ATTACK" "$METHOD" "$DEFENSE" "$SEED" "$RC" <<'PY'
import json, sys
path, cell, attack, method, defense, seed, rc = sys.argv[1:8]
try:
    with open(path) as f:
        d = json.load(f)
except Exception:
    d = {}
d["cell"] = {"id": cell, "attack": attack, "method": method,
             "defense": defense, "seed": int(seed), "exit_code": int(rc)}
with open(path, "w") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
print(f"[cell] {cell} exit={rc}")
PY

if [ $RC -ne 0 ]; then
    echo "[FAIL] $CELL 退出码 $RC —— 错误摘要见 $METRICS 的 errors / log_tail 字段" >&2
    echo "----- 日志末尾 20 行 -----" >&2
    tail -20 "$LOG" >&2
fi

exit $RC
