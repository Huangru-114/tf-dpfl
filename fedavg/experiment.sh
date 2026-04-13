#!/bin/bash
#SBATCH --job-name=HierFLcifar10
#SBATCH --account=naiss2026-4-650          # C3SE 项目号，格式如 NAISS2024-X-XXX
#SBATCH --output=logs/HierFL_%j.out        # %j 自动替换为 job ID
#SBATCH --error=logs/HierFL_%j.err
#SBATCH --time=24:00:00                    # 预估时间，先给宽裕一点

# ── GPU 资源 ──────────────────────────────────────────────────
#SBATCH --gpus-per-node=V100:1              # A40 适合这种中等规模实验


# ════════════════════════════════════════════════════════════════
# 环境初始化
# ══════════════════════════════════════════════════
set -e  # 任何命令失败立即退出，避免静默错误

echo "========================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Job Name    : $SLURM_JOB_NAME"
echo "Node        : $SLURM_NODELIST"
echo "Start Time  : $(date)"
echo "Working Dir : $(pwd)"
echo "========================================"

# 创建 logs 目录（如果不存在）
mkdir -p logs


echo "[Setup] Python: $(which python)"
echo "[Setup] TF version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "[Setup] GPU available: $(python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))')"

# ════════════════════════════════════════════════════════════════
# 实验配置
# ════════════════════════════════════════════════════════════════

# 项目根目录
PROJECT_DIR=~/tf-dpfl/fedavg
cd $PROJECT_DIR


export WANDB_MODE=online   # 改成 offline 则本地缓存，实验后手动 sync



# ════════════════════════════════════════════════════════════════
# 运行实验
# ════════════════════════════════════════════════════════════════

# 实验组 A：IID vs Non-IID
# 每次修改 run_name 和对应参数，提交不同的 job

RUN_NAME="fedavg-noniid-alpha0.5-$(date +%m%d-%H%M)"

# 用 Python 临时覆盖 config 里的参数，不用手动改 yaml
python -u main.py \
    --config config/config.yaml \
    --override wandb.enabled=true
# python centralized/train.py
# python wideres_sweep.py --sweep
echo "========================================"
echo "End Time : $(date)"
echo "========================================"