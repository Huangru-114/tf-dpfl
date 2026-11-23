#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# d3_submit.sh  —  D3 离线探针：d3_configs.yaml 里的 9 个 run 各交一个 d3.sbatch（登录节点上跑，只调 sbatch）
#
#   bash experiments/defense/edge-native/d3_submit.sh          # 在仓库根目录
#   ONLY="TBS__k1-snap__s42" bash experiments/defense/edge-native/d3_submit.sh    # 只交某几个（空格分隔）
#
# 已有结果（analysis/d3/<run_id>.json）的 run 跳过；要重跑先把那个文件移走。
# 回传：analysis/d3/*.json（9 个）→ 本地 `python3 harness/d3_verdict.py` → analysis/d3_verdict.json。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$ROOT/experiments/defense/edge-native"
cd "$ROOT"

RUNS="SNAP__collocated__s42 SNAP__collocated__s43 SNAP__collocated__s44
TBS__k1-snap__s42 TBS__k1-snap__s43 TBS__k1-snap__s44
SNAP__distributed__s42 SNAP__distributed__s43 SNAP__distributed__s44"
[ -n "${ONLY:-}" ] && RUNS="$ONLY"

n=0
for run in $RUNS; do
    if [ -f "$HERE/analysis/d3/$run.json" ]; then
        echo "[d3] 跳过 $run（已有 analysis/d3/$run.json）"
        continue
    fi
    sbatch "$HERE/d3.sbatch" "$run" && n=$((n + 1))
done
echo "[d3] 交了 $n 个作业；回传 experiments/defense/edge-native/analysis/d3/*.json"
