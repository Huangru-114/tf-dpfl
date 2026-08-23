#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# run_exp3.sh  —  提交 Experiment 3A/3B 全部格子（8 拓扑 × N seed）。登录节点跑。
#
#   bash experiments/attack/hfl-propagation/run_exp3.sh              # 提交所有未完成格子
#   bash experiments/attack/hfl-propagation/run_exp3.sh --dry-run    # 只打印，不提交
#   bash experiments/attack/hfl-propagation/run_exp3.sh --status     # 只看进度
#   bash experiments/attack/hfl-propagation/run_exp3.sh --force      # 连已完成的一起重跑
#   SEEDS="42" bash .../run_exp3.sh                                  # 先只跑 1 个 seed 探路
#   CONFIGS="4edge_collocated 4edge_distributed" bash .../run_exp3.sh # 只跑指定格
#
# 断点续跑：results/<exp_id>.metrics.json 里 exit_code==0 即算完成、跳过。跑挂的会重试。
# 每格一个独立 SLURM job（exp3_cell.sbatch），日志带 exp_id 互不覆盖。
#
# ⚠️ 默认 8 拓扑 × 3 seed = 24 个 GPU 全长 job（每个数小时）。先 --dry-run 看清单，
#    或 SEEDS="42" 跑一轮探路，再放全量。
# ══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

REL="experiments/attack/hfl-propagation"
JOB="$REL/exp3_cell.sbatch"
OUTDIR="$ROOT/$REL/results"
mkdir -p "$OUTDIR"

# 可被环境变量覆盖；去掉 2edge_distributed 若你信任 exp007 已覆盖它
CONFIGS="${CONFIGS:-2edge_collocated 2edge_distributed \
                    4edge_collocated 4edge_distributed 4edge_mixed \
                    10edge_collocated 10edge_distributed 10edge_mixed}"
SEEDS="${SEEDS:-42 43 44}"

DRY=0; FORCE=0; STATUS=0
for a in "$@"; do case "$a" in
    --dry-run) DRY=1 ;;
    --force)   FORCE=1 ;;
    --status)  STATUS=1 ;;
    *) echo "未知参数: $a" >&2; exit 2 ;;
esac; done

is_done() {   # $1 = metrics 路径
    [ -s "$1" ] && grep -q '"exit_code": *0' "$1"
}

total=0; done_n=0; queued=0
for cell in $CONFIGS; do
    cfg="$REL/$cell.yaml"
    [ -f "$ROOT/$cfg" ] || { echo "跳过：找不到 $cfg" >&2; continue; }
    for seed in $SEEDS; do
        total=$((total + 1))
        exp_id="${cell}_seed${seed}"
        metrics="$OUTDIR/$exp_id.metrics.json"

        if is_done "$metrics" && [ "$FORCE" -eq 0 ]; then
            done_n=$((done_n + 1))
            [ "$STATUS" -eq 1 ] && echo "  done   $exp_id"
            continue
        fi
        if [ "$STATUS" -eq 1 ]; then
            echo "  todo   $exp_id"; continue
        fi
        if [ "$DRY" -eq 1 ]; then
            echo "  would sbatch  $JOB  $cfg  $seed  $exp_id"; queued=$((queued + 1)); continue
        fi
        sbatch "$JOB" "$cfg" "$seed" "$exp_id"
        queued=$((queued + 1))
    done
done

echo "──────────────────────────────────────────────"
echo "格子总数=$total  已完成=$done_n  本次${DRY:+(dry-run)}入队=$queued"
[ "$STATUS" -eq 1 ] && echo "(仅状态；未提交)"
[ "$DRY" -eq 1 ]    && echo "(dry-run；未提交。去掉 --dry-run 即真正 sbatch)"
exit 0
