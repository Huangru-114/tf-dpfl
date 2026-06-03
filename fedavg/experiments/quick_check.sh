#!/usr/bin/env bash
# experiments/quick_check.sh  –  任务4：快速验证（20 轮，不追求收敛）
#
# 目的：确认所有分层后门指标都能正常记录到 wandb，
#       特别是 local/asr_same_edge 与 local/asr_diff_edge 是否出现且有差异。
#
# 配置：Hier-FedAvg + FT × {D1 pathological-2, D4 dirichlet-0.5} × BadNet
#       n_rounds=20，eval_interval=5（每 5 轮算一次分层 ASR + 遗忘曲线）。
#
# 用法（在有 GPU / 数据 / wandb 出口的环境）：
#   cd fedavg && bash experiments/quick_check.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # 切到 fedavg/

COMMON_OVERRIDES=(
  --override federation.n_rounds=20
  --override backdoor.eval_interval=5
  --override evaluation.eval_interval=5
  --override backdoor.n_malicious=4
  --override backdoor.malicious_placement=spread
)

echo "==================================================================="
echo " Quick check 1/2: Hier-FedAvg+FT | D1_patho_2 | BadNet | 20 rounds"
echo "==================================================================="
python main.py \
  --framework hier_fedavg_ft \
  --distribution_config D1_patho_2 \
  --attack_method badnet \
  --seed 42 \
  "${COMMON_OVERRIDES[@]}"

echo "==================================================================="
echo " Quick check 2/2: Hier-FedAvg+FT | D4_dir_05 | BadNet | 20 rounds"
echo "==================================================================="
python main.py \
  --framework hier_fedavg_ft \
  --distribution_config D4_dir_05 \
  --attack_method badnet \
  --seed 42 \
  "${COMMON_OVERRIDES[@]}"

echo ""
echo "[Quick check] done. 在 wandb 中核对以下 key 是否出现且有值："
echo "  global/asr, global/acc, edge/asr_mean, edge/asr_std,"
echo "  local/asr_benign_mean, local/asr_malicious_mean,"
echo "  local/asr_same_edge, local/asr_diff_edge,  <-- 重点：两者是否有差异"
echo "  feature/separation_score_global, feature/separation_score_local,"
echo "  forgetting/epoch_0..4"
