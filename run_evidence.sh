#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# run_evidence.sh  —  数据划分与 ASR 探针的证据（**不需要 GPU**，容器内秒~分钟级）
#
#   bash run_evidence.sh [config相对路径]
#
# 默认用 experiments/attack/hfl-propagation/3c_R5.yaml（exp007 锚点的那一格）。
# 调用仓库里**真实的** noniid_partition / split_client_train_test，量出四件事：
#   1. 分区是不是划分（每个 index 恰好归一个客户端）
#   2. 训练池 ∩ 留出池 是不是空 —— 这是「合并不是 bug」的依据
#   3. 旧探针（官方 x_test）有多少张落进训练池、其中多少落进**恶意端**分片
#   4. 新探针（留出池）与训练池的交集必须是 0
#
# 登录节点可以跑：它只做分区，不训练、不碰 GPU。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# cwd 放仓库上一级：本脚本要 load_cifar10，需要 $ROOT/../data 可见。
# 详见 CLAUDE.md 的容器一节 / 陷阱 #17。
cd "$ROOT/.."
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

CONFIG="${1:-experiments/attack/hfl-propagation/3c_R5.yaml}"

echo "== 数据划分证据 =="
echo "   config = $CONFIG"
echo
$PY "$ROOT/harness/evidence_data_split.py" --config "$ROOT/$CONFIG"
rc=$?

echo
if [ $rc -eq 0 ]; then
    echo "[run_evidence] ✅ 判据全部成立。把上面那张表直接贴给导师即可。"
else
    echo "[run_evidence] ❌ 有判据不成立 —— 见上面标 ❌ 的行。"
fi
exit $rc
