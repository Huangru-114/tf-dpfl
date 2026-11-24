#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# d3pa2_submit.sh  —  P-a 追加探针（N-016）的作业提交（登录节点上跑，只调 sbatch）
#
#   bash experiments/defense/edge-native/d3pa2_submit.sh calibrate   # 第 1 步：1 个作业（SNAP 集中 s42）
#   bash experiments/defense/edge-native/d3pa2_submit.sh main        # 第 2 步：3 个作业（s42–44），要先有入库的冻结清单
#
# 已有结果（analysis/d3pa2/<run>.<stage>.json）的跳过；要重跑先把那个文件移走。
# 第 1 步回传后本地：python3 harness/d3pa2_verdict.py select experiments/defense/edge-native/analysis/d3pa2/SNAP__collocated__s42.calibrate.json
# 第 2 步回传后本地：python3 harness/d3pa2_verdict.py judge
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

STAGE="${1:?用法: bash d3pa2_submit.sh <calibrate|main>}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$ROOT/experiments/defense/edge-native"
cd "$ROOT"

case "$STAGE" in
    calibrate) RUNS="SNAP__collocated__s42" ;;
    main)
        RUNS="SNAP__collocated__s42 SNAP__collocated__s43 SNAP__collocated__s44"
        [ -f "$HERE/analysis/d3pa2_frozen.json" ] || { echo "[d3pa2] 没有冻结清单 analysis/d3pa2_frozen.json：先 select、提交、git pull"; exit 2; }
        grep -q '"label": "selected"' "$HERE/analysis/d3pa2_frozen.json" || { echo "[d3pa2] 冻结清单没有选出训练量（not_removable / invalid）→ 不交正式作业"; exit 2; }
        ;;
    *) echo "[d3pa2] stage 只能是 calibrate / main：$STAGE"; exit 2 ;;
esac

n=0
for run in $RUNS; do
    if [ -f "$HERE/analysis/d3pa2/$run.$STAGE.json" ]; then
        echo "[d3pa2] 跳过 $run（已有 analysis/d3pa2/$run.$STAGE.json）"
        continue
    fi
    sbatch "$HERE/d3pa2.sbatch" "$run" "$STAGE" && n=$((n + 1))
done
echo "[d3pa2] 交了 $n 个作业；回传 experiments/defense/edge-native/analysis/d3pa2/*.$STAGE.json"
