#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# run_evidence.sh  —  数据划分泄漏的确凿证据（**不需要 GPU**，容器内秒~分钟级）
#
#   bash run_evidence.sh [config相对路径]
#
# 默认用 experiments/attack/hfl-propagation/3c_R5.yaml（exp007 锚点的那一格）。
# 调用仓库里**真实的** noniid_partition / split_client_train_test，
# 把 legacy（修复前的合并）与 fixed（当前代码）两条路径各跑一遍，
# 数出官方 test set 有多少张落进客户端的训练集。
#
# 登录节点可以跑：它只做分区，不训练、不碰 GPU。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

CONFIG="${1:-experiments/attack/hfl-propagation/3c_R5.yaml}"

echo "== 数据划分证据 =="
echo "   config = $CONFIG"
echo
$PY harness/evidence_data_split.py --config "$ROOT/$CONFIG"
rc=$?

echo
if [ $rc -eq 0 ]; then
    echo "[run_evidence] ✅ 判据全部成立。把上面那张表直接贴给导师即可。"
else
    echo "[run_evidence] ❌ 有判据不成立 —— 见上面标 ❌ 的行。"
fi
exit $rc
