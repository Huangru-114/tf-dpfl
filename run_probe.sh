#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 08:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=24G
#SBATCH --job-name=tfdpfl-probe
#
# run_probe.sh  —  Exp 0.3：训练（落 checkpoint）+ 逐 checkpoint 跑 paired counterfactual 探针
#
# 跑法（仓库根目录）：
#     sbatch run_probe.sh [config] [outdir]
#     sbatch run_probe.sh experiments/attack/bad-pfl/exp03/probe_smoke.yaml
#
# 两阶段，可分别跳过：
#     PROBE_SKIP_TRAIN=1  只跑探针（checkpoint 已存在时复用，省一次训练）
#     PROBE_SKIP_PROBE=1  只跑训练（先把 checkpoint 攒出来）
#
# 产出：
#     <outdir>/probe_rows_r####.csv      layer 级长表 —— **小，回传 git**
#     <outdir>/probe_summary_r####.json  含三条判据 —— **小，回传 git**
#     <ckptdir>/*.npz                    checkpoint —— **大，留集群**（.gitignore 已挡）
#     <ckptdir>/*.manifest.json          manifest —— 小，放行进 git
#     $LOGDIR/probe_<tag>.log            日志 —— 大，留集群
#
# 判据（写在 experiments/attack/bad-pfl/exp03/current-focus.md，此处只负责产出数字）：
#   1 仪表：良性端 ‖Δθ_BD‖ ≈ 0    不过 → 下面所有数字都不可读，先修仪表
#   2 信号：恶意端 ‖Δθ_BD‖/‖Δθ_stochastic‖ 显著 > 1
#           不过 → 「参数空间不可定位」，**这是结论不是 bug**
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
source "$ROOT/cluster_env.sh"          # 解析出 $PY（容器 or 本地），并打印 [env] 行

CONFIG="${1:-experiments/attack/bad-pfl/exp03/probe_smoke.yaml}"
OUTDIR="${2:-$(dirname "$CONFIG")}"
TAG="$(basename "${CONFIG%.yaml}")"
LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
mkdir -p "$LOGDIR" "$OUTDIR"

# ⚠️ GPU 上 cuDNN 的部分 kernel 默认是非确定性的 —— 那会直接毁掉 paired counterfactual：
# 两条 clean twin 就对不上，Δθ_BD 里混进硬件噪声。判据 1 正是查这件事的。
# 这个开关必须在 import tensorflow **之前**设。
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1

# checkpoint 目录从 config 里读（cwd=fedavg 时 config 里写的是相对 fedavg 的路径）
CKPTDIR="$($PY - "$CONFIG" <<'PYEOF'
import sys, yaml, os
c = yaml.safe_load(open(sys.argv[1]))
d = (c.get("probe") or {}).get("checkpoint_dir", "checkpoints")
print(os.path.normpath(os.path.join("fedavg", d)) if not os.path.isabs(d) else d)
PYEOF
)"
echo "[probe] config=$CONFIG  outdir=$OUTDIR  ckptdir=$CKPTDIR  tag=$TAG"

# ── 阶段 1：训练（在配置的轮次落 checkpoint）────────────────────────────────
if [ "${PROBE_SKIP_TRAIN:-0}" != "1" ]; then
  echo "[probe] === 阶段 1：训练 + 落 checkpoint ==="
  ( cd "$ROOT/fedavg" && $PY main.py --config "$ROOT/$CONFIG" ) \
      2>&1 | tee "$LOGDIR/probe_train_${TAG}.log"
else
  echo "[probe] PROBE_SKIP_TRAIN=1 —— 复用已有 checkpoint"
fi

# ── 阶段 2：逐 checkpoint 跑探针 ────────────────────────────────────────────
if [ "${PROBE_SKIP_PROBE:-0}" = "1" ]; then
  echo "[probe] PROBE_SKIP_PROBE=1 —— 到此为止"; exit 0
fi

shopt -s nullglob
CKPTS=( "$CKPTDIR"/*.npz )
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "[probe] ✗ $CKPTDIR 里没有 checkpoint。"
  echo "        检查 config 的 probe.checkpoint_rounds 是否非空、且 ≤ federation.n_rounds。"
  exit 1
fi
echo "[probe] 找到 ${#CKPTS[@]} 个 checkpoint"

FAILED=0
for ck in "${CKPTS[@]}"; do
  name="$(basename "${ck%.npz}")"
  echo "[probe] === 阶段 2：$name ==="
  # 探针失败一格不该拖垮其余格（每格独立，断点续跑）
  if ! ( cd "$ROOT/fedavg" && $PY -m probe.run_probe \
            --config "$ROOT/$CONFIG" \
            --checkpoint "$ROOT/${ck#./}" \
            --out "$ROOT/$OUTDIR" \
            --benign "${PROBE_N_BENIGN:-3}" \
            --malicious "${PROBE_N_MALICIOUS:-3}" ) \
        2>&1 | tee "$LOGDIR/probe_${name}.log"; then
    echo "[probe] ✗ $name 失败（见 $LOGDIR/probe_${name}.log）"
    FAILED=$((FAILED + 1))
  fi
done

echo
echo "════════════════════════════════════════════════════════════"
echo "[probe] 判据汇总"
for js in "$OUTDIR"/probe_summary_*.json; do
  $PY - "$js" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
v, r = d["verdicts"], d["run"]
print(f"  {sys.argv[1].split('/')[-1]}  round={r['round']}")
print(f"    判据1 仪表 pass={v['instrument_ok']['pass']}  "
      f"良性 ‖Δθ_BD‖max={v['instrument_ok']['benign_bd_norm']['max']}")
print(f"    判据2 信号 pass={v['signal']['pass']}  "
      f"恶意 ‖Δθ_BD‖/‖Δθ_stoch‖ min={v['signal']['malicious_bd_over_stochastic']['min']}")
if d["client_failures"]:
    print(f"    ⚠️ 失败客户端: {d['client_failures']}")
PYEOF
done
echo "════════════════════════════════════════════════════════════"
[ "$FAILED" -eq 0 ] || { echo "[probe] $FAILED 个 checkpoint 失败"; exit 1; }
echo "[probe] 完成。回传 $OUTDIR/probe_rows_*.csv 与 probe_summary_*.json（小），npz 留集群。"
