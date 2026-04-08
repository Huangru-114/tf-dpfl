#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# run_experiments.sh
# HierDP-FL experiment runner
#
# Usage:
#   bash run_experiments.sh [cifar10|cifar100|gtsrb]
#
# Runs four configurations:
#   1. Non-DP baseline
#   2. DP-SGD (uniform, auto-calibrated to ε=10)
#   3. DP-SGD (uniform, ε=3)
#   4. DP-SGD with epoch-wise cosine noise schedule
#   5. DP-SGD with layer-wise adaptive clipping
#   6. DP-SGD with client-wise adaptive noise
# ──────────────────────────────────────────────────────────────────────────
#SBATCH -A naiss2025-22-1095
#SBATCH -p alvis
#SBATCH --gpus-per-node=T4:1
#SBATCH -t 2-00:00:00               # 单个 agent 最长跑 6 小时
#SBATCH -J TF-DPFL
#SBATCH -o logs/tfdpfl_%A_%a.out     # %A=array job id, %a=task id
#SBATCH -e logs/tfdpfl_%A_%a.err

DATASET=${1:-cifar10}
echo "Dataset: $DATASET"

# Common FL topology
CLIENTS=10
EDGES=2
ROUNDS=100
TAU1=5        # local updates per client
TAU2=2        # edge aggregation steps
BATCH=64
LR=0.02
MOMENTUM=0.9
IID=0         # shard non-IID

# Select model based on dataset
if [ "$DATASET" = "gtsrb" ]; then
  MODEL="convnet"
else
  MODEL="cnn_complex"
fi

BASE_CMD="python hierdpfl.py \
  --dataset $DATASET \
  --model $MODEL \
  --num_clients $CLIENTS \
  --num_edges $EDGES \
  --num_communication $ROUNDS \
  --num_local_update $TAU1 \
  --num_edge_aggregation $TAU2 \
  --batch_size $BATCH \
  --lr $LR \
  --momentum $MOMENTUM \
  --iid $IID \
  --seed 42"

# ── 1. Non-DP baseline ─────────────────────────────────────────────────
#echo ""
#echo "=== [1/6] Non-DP baseline ==="
#eval $BASE_CMD

# ── 2. DP-SGD ε=10 (uniform, auto σ) ──────────────────────────────────
#echo ""
#echo "=== [2/6] DP-SGD ε=10 uniform (auto σ) ==="
#eval $BASE_CMD \
#  --dp_enabled \
#  --dp_epsilon 10.0 \
#  --dp_delta 1e-5 \
#  --dp_max_grad_norm 1.0 \
#  --dp_adaptive_mode uniform

# ── 3. DP-SGD ε=3 (tighter privacy) ───────────────────────────────────
echo ""
echo "=== [3/6] DP-SGD ε=3 uniform (tight privacy) ==="
eval $BASE_CMD \
 --dp_enabled \
 --dp_epsilon 3.0 \
 --dp_delta 1e-5 \
 --dp_max_grad_norm 0.5 \
 --dp_adaptive_mode uniform

# # ── 4. Epoch-wise cosine noise schedule ───────────────────────────────
# echo ""
# echo "=== [4/6] DP-SGD ε=10 epoch-wise cosine schedule ==="
# eval $BASE_CMD \
#   --dp_enabled \
#   --dp_epsilon 10.0 \
#   --dp_delta 1e-5 \
#   --dp_max_grad_norm 1.0 \
#   --dp_adaptive_mode epoch_wise

# # ── 5. Layer-wise adaptive clipping ───────────────────────────────────
# echo ""
# echo "=== [5/6] DP-SGD ε=10 layer-wise adaptive clipping ==="
# eval $BASE_CMD \
#   --dp_enabled \
#   --dp_epsilon 10.0 \
#   --dp_delta 1e-5 \
#   --dp_max_grad_norm 1.0 \
#   --dp_adaptive_mode layer_wise

# # ── 6. Client-wise adaptive noise ─────────────────────────────────────
# echo ""
# echo "=== [6/6] DP-SGD ε=10 client-wise adaptive noise ==="
# eval $BASE_CMD \
#   --dp_enabled \
#   --dp_epsilon 10.0 \
#   --dp_delta 1e-5 \
#   --dp_max_grad_norm 1.0 \
#   --dp_adaptive_mode client_wise

# echo ""
# echo "All experiments complete. TensorBoard logs in ./runs/"
# echo "Launch TensorBoard: tensorboard --logdir runs/"
