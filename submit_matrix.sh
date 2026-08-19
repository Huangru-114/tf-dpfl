#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# submit_matrix.sh  —  展开三维矩阵 → 生成 manifest → sbatch 一个 job array
#
#   bash submit_matrix.sh                # 提交所有**未完成**的格子
#   bash submit_matrix.sh --dry-run      # 只打印，不提交
#   bash submit_matrix.sh --force        # 连已完成的一起重跑
#   bash submit_matrix.sh --status       # 只看进度，不提交
#
# 矩阵定义在 matrix.conf。断点续跑靠 experiments/matrix/<cell>.metrics.json
# 是否存在——所以中断后重跑这条命令即可，不会重复烧机时。
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DRY_RUN=0; FORCE=0; STATUS_ONLY=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        --force)   FORCE=1 ;;
        --status)  STATUS_ONLY=1 ;;
        *) echo "未知参数: $a" >&2; exit 2 ;;
    esac
done

# shellcheck source=matrix.conf
source "$ROOT/matrix.conf"

OUTDIR="$ROOT/experiments/matrix"
MANIFEST="$OUTDIR/manifest.tsv"
mkdir -p "$OUTDIR"

# ── 不正交的组合（CLAUDE.md 陷阱 #1）：攻击类会替换掉 PFL 方法类 ──────────
NON_ORTHOGONAL_ATTACKS=" neurotoxin cerp badpfl "
FEDAVG_LIKE=" hierfedavg fedavg "

is_non_orthogonal() {   # $1=attack $2=method
    [[ "$NON_ORTHOGONAL_ATTACKS" == *" $1 "* ]] && [[ "$FEDAVG_LIKE" != *" $2 "* ]]
}

# 「已完成」= metrics.json 存在 **且** 那一格是成功退出的。
# 跑挂的格子也会写出 metrics.json（里面带 errors/log_tail），若只看文件存在，
# 失败的格子会被永久跳过、再也不会重试 —— 矩阵里会留下一批静默的空洞。
is_done() {   # $1=metrics 路径
    [ -s "$1" ] || return 1
    grep -q '"exit_code": *0' "$1"
}

# ── 展开矩阵 ─────────────────────────────────────────────────────────────
: > "$MANIFEST"
total=0; skipped_done=0; skipped_ortho=0; queued=0; retried=0
declare -a QUEUE_LINES

for attack in $ATTACKS; do
for method in $METHODS; do
for defense in $DEFENSES; do
for seed in $SEEDS; do
    total=$((total + 1))
    cell="${attack}__${method}__${defense}__seed${seed}"

    if [ "$SKIP_NON_ORTHOGONAL" = "true" ] && is_non_orthogonal "$attack" "$method"; then
        skipped_ortho=$((skipped_ortho + 1))
        continue
    fi

    # manifest 始终写全（experiment_tf.sh 按行号取），完成与否只影响是否入队
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$cell" "$attack" "$method" "$defense" "$seed" \
        "$BASE_CONFIG" "$EXTRA_OVERRIDES" >> "$MANIFEST"
    line=$(wc -l < "$MANIFEST")

    if is_done "$OUTDIR/$cell.metrics.json" && [ "$FORCE" -eq 0 ]; then
        skipped_done=$((skipped_done + 1))
        continue
    fi
    if [ -s "$OUTDIR/$cell.metrics.json" ]; then
        retried=$((retried + 1))            # 上次跑挂了，这次重试
    fi
    QUEUE_LINES+=("$line")
    queued=$((queued + 1))
done; done; done; done

echo "════════════════ 矩阵状态 ════════════════"
echo " 全矩阵格子数        : $total"
echo " 跳过（不正交组合）  : $skipped_ortho"
echo " 跳过（已完成）      : $skipped_done"
echo " 重试（上次失败）    : $retried"
echo " 本次待提交          : $queued"
echo " manifest            : $MANIFEST"
echo "══════════════════════════════════════════"

if [ "$STATUS_ONLY" -eq 1 ]; then exit 0; fi
if [ "$queued" -eq 0 ]; then echo "没有待跑的格子。"; exit 0; fi

# ── 组装 sbatch --array 的行号列表（逗号分隔，SLURM 原生支持）─────────────
ARRAY_SPEC=$(IFS=,; echo "${QUEUE_LINES[*]}")
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"        # 同时最多几个 task，别把队列占满

echo "sbatch --array=${ARRAY_SPEC}%${MAX_CONCURRENT} experiment_tf.sh"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "(--dry-run，未提交)"
    echo "前 10 个待跑格子："
    for l in "${QUEUE_LINES[@]:0:10}"; do
        printf '  %s\n' "$(cut -f1 "$MANIFEST" | sed -n "${l}p")"
    done
    exit 0
fi

FORCE=$FORCE MANIFEST="$MANIFEST" \
    sbatch --array="${ARRAY_SPEC}%${MAX_CONCURRENT}" \
           --export=ALL,MANIFEST="$MANIFEST",FORCE="$FORCE" \
           "$ROOT/experiment_tf.sh"
