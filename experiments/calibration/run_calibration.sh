#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# run_calibration.sh  —  提交 Stage B 标定的 6 格（local_epochs {1,3,5} × seed {42,43}）。
#                        登录节点跑。
#
#   bash experiments/calibration/run_calibration.sh --dry-run   # 先看清单
#   bash experiments/calibration/run_calibration.sh             # 提交未完成的格子
#   bash experiments/calibration/run_calibration.sh --status    # 只看进度
#   bash experiments/calibration/run_calibration.sh --force     # 连已完成的一起重跑
#   EPOCHS="1 3" SEEDS="42" bash experiments/calibration/run_calibration.sh
#
# 回答的问题（导师意见 #5）：等算力下哪个 (local_epochs, n_rounds) 组合最快到平台。
# 单变量 = local_epochs；其余逐字继承 3A 的锚点拓扑 2edge_distributed。
#
# 断点续跑：results/<exp_id>.metrics.json 里 exit_code==0 即算完成、跳过。
#
# ⚠️ 与 run_exp3.sh 的同一个坑（陷阱 #15）：重跑前必须把旧的
#    results/*.metrics.json 移走，否则一个作业都不会提交，而输出显示「已完成=6」，
#    看起来一切正常。--force 是另一条路。
# ══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

# 解析出 $PY（Arrhenius 上 = apptainer 容器）。本脚本自己只调 sbatch，
# source 它是为了：(a) 打印那行 `[env] python = ...`，跑之前扫一眼；
# (b) 末尾的读数提示能给出**在这台机器上真的能跑**的命令，而不是裸 python3。
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

REL="experiments/calibration"
JOB="$REL/calib_cell.sbatch"
OUTDIR="$ROOT/$REL/results"
mkdir -p "$OUTDIR"

EPOCHS="${EPOCHS:-1 3 5}"
SEEDS="${SEEDS:-42 43}"

DRY=0; FORCE=0; STATUS=0
for a in "$@"; do case "$a" in
    --dry-run) DRY=1 ;;
    --force)   FORCE=1 ;;
    --status)  STATUS=1 ;;
    *) echo "未知参数: $a" >&2; exit 2 ;;
esac; done

is_done() { [ -s "$1" ] && grep -q '"exit_code": *0' "$1"; }

total=0; done_n=0; queued=0
for ep in $EPOCHS; do
    for seed in $SEEDS; do
        cell="local_epochs${ep}_seed${seed}"
        cfg="$REL/$cell.yaml"
        if [ ! -f "$ROOT/$cfg" ]; then
            echo "跳过：找不到 $cfg" >&2; continue
        fi
        total=$((total + 1))
        metrics="$OUTDIR/$cell.metrics.json"

        if is_done "$metrics" && [ "$FORCE" -eq 0 ]; then
            done_n=$((done_n + 1))
            [ "$STATUS" -eq 1 ] && echo "  done   $cell"
            continue
        fi
        if [ "$STATUS" -eq 1 ]; then echo "  todo   $cell"; continue; fi
        if [ "$DRY" -eq 1 ]; then
            echo "  would sbatch  $JOB  $cfg  $seed  $cell"; queued=$((queued + 1)); continue
        fi
        # seed 同时在 config 里（自描述）和命令行上（main.py 的 --seed 覆盖它）。
        # 两处一致是 run_calibration.sh 的责任，config 文件名里也带着它。
        sbatch "$JOB" "$cfg" "$seed" "$cell"
        queued=$((queued + 1))
    done
done

echo "──────────────────────────────────────────────"
echo "格子总数=$total  已完成=$done_n  本次${DRY:+(dry-run)}入队=$queued"
[ "$STATUS" -eq 1 ] && echo "(仅状态；未提交)"
[ "$DRY" -eq 1 ]    && echo "(dry-run；未提交。去掉 --dry-run 即真正 sbatch)"
echo
echo "跑完后读数：$PY experiments/calibration/read_calibration.py"
echo "  （read_calibration.py 是纯 stdlib，不需要 TF；容器内外都跑得动）"
exit 0
