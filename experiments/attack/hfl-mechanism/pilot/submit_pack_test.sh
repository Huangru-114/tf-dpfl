#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# submit_pack_test.sh  —  一卡多跑的测试（D-047「先测后用」）。登录节点跑，**纯 bash，不调 python**。
#
#   bash experiments/attack/hfl-mechanism/pilot/submit_pack_test.sh --dry-run
#   bash experiments/attack/hfl-mechanism/pilot/submit_pack_test.sh            # 提交 K=2、K=3 两个作业
#   PACK_KS="3" bash …/submit_pack_test.sh                                   # 只交某几个 K
#
# 每个作业 = DET 配置（2edge、5 个云轮、不停轮）× K 份，挤在一张 GPU 上（../pack.sbatch），
# 每份绑 4 个核（sbatch -c 4K）—— 与单跑的 DET 同配置、同核数，只差「是否同卡」。
# 结果写 pilot/results/P1/PACK/：PACK_K<K>__s<i>.metrics.json 与 PACK_K<K>.gpu.json。
# 判定：harness/pack_test.py（checksum 必须等于 DET 第二轮；加速比 ≥ 1.5 才采用）。
# 已有 exit_code 为 0 的结果就跳过该 K（重交前先把旧文件移走）。
# ══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${TFDPFL_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
JOB="$HERE/../pack.sbatch"
CFG="experiments/attack/hfl-mechanism/pilot/configs/DET__rep1__s42.yaml"
OUT="experiments/attack/hfl-mechanism/pilot/results/P1/PACK"

DRY=0
for a in "$@"; do case "$a" in
    --dry-run) DRY=1 ;;
    *) echo "未知参数: $a" >&2; exit 2 ;;
esac; done

[ -f "$ROOT/$CFG" ] || { echo "[pack-test] 找不到 DET 配置: $CFG"; exit 2; }

for K in ${PACK_KS:-2 3}; do
    TAG="PACK_K${K}"
    args=("$TAG")
    all_done=1
    for ((s = 1; s <= K; s++)); do
        m="$OUT/${TAG}__s${s}.metrics.json"
        args+=("$CFG" "${TAG}__s${s}" "$m")
        if ! { [ -s "$ROOT/$m" ] && grep -q '"exit_code": *0' "$ROOT/$m"; }; then all_done=0; fi
    done
    if [ "$all_done" -eq 1 ]; then echo "  done   $TAG（已有结果，跳过）"; continue; fi
    if [ "$DRY" -eq 1 ]; then
        echo "  would sbatch -c $((4 * K)) --mem=$((24 * K))G --job-name=exp3v2-pack$K pack.sbatch ${args[*]}"
        continue
    fi
    (cd "$ROOT" && sbatch -c $((4 * K)) --mem=$((24 * K))G --job-name="exp3v2-pack$K" "$JOB" "${args[@]}")   # 主机内存按 K（D-070）
done
echo "回传后判定：harness/pack_test.py（纯标准库）"
[ "$DRY" -eq 1 ] && echo "(dry-run；未提交)"
exit 0
