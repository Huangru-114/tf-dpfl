#!/bin/bash
#SBATCH --job-name=HierFL-sweep
#SBATCH --account=naiss2026-4-650
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --array=0-5            # 同时起 4 个 agent 并行抢 run（按需调整）

set -e
mkdir -p logs
cd ~/tf-dpfl/fedavg
source ~/tf-dpfl/bin/activate
export WANDB_MODE=online

# 把下面 SWEEP_ID 换成第①步输出的那个
wandb agent 3445339138-kth-royal-institute-of-technology/HierFL-cifar10/e5ii9705