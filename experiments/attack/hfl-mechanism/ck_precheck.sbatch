#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 02:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=24G
#SBATCH --job-name=ckpre
# ══════════════════════════════════════════════════════════════════════════
# ck_precheck.sbatch  —  G8 快照上的 c_k 离线预检（S6a，D-085 方案 §二）。只读，不训练。
#
#   sbatch experiments/attack/hfl-mechanism/ck_precheck.sbatch
#
# 对 G8 的 3 个 seed 各载入第 30 / 70 轮快照（manifest 在各 G8 metrics.json 的 dumps.snapshots，
# 文件在 $ROOT/../tfdpfl-dumps/，不进 git），每个 edge 配 NCM head，算 c_0…c_9（定向 PGD 失败率）。
# 回传只有一个小文件：experiments/attack/hfl-mechanism/analysis/ck_precheck.json
# （每 seed × 轮 × edge 的 c_k 与样本数；没有任何张量）。
#
# 单卡预计约 10–20 分钟（**没有实测**）：每个 (seed, 轮, edge) 要 K=10 个类 × 10 步 PGD × 500 张。
# 这是 G8 存盘（D-073）的第一个消费者；分析做完再决定删存盘。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

M="$ROOT/experiments/attack/hfl-mechanism"
DUMPS="${TFDPFL_DUMPDIR:-$ROOT/../tfdpfl-dumps}"
TMP="${TMPDIR:-/tmp}/ckpre.${SLURM_JOB_ID:-local}"
mkdir -p "$TMP" "$M/analysis"

RC=0
cd "$ROOT/fedavg"
for S in 42 43 44; do
    echo "[ckpre] seed=$S"
    $PY -m analysis.ck_snapshot \
        --config "$M/configs/G8__a__s${S}.yaml" \
        --metrics "$M/results/P2/G8/G8__a__s${S}.metrics.json" \
        --dumps-root "$DUMPS" \
        --out "$TMP/ck_s${S}.json" || { echo "[ckpre] seed=$S 失败"; RC=1; }
done

cd "$ROOT"
$PY - "$TMP" "$M/analysis/ck_precheck.json" <<'PY'
import glob, json, sys
tmp, out = sys.argv[1], sys.argv[2]
runs = {}
for p in sorted(glob.glob(f"{tmp}/ck_s*.json")):
    d = json.load(open(p))
    runs[str(d["seed"])] = d
json.dump({"purpose": "S6a c_k 预检（D-085）：G8 快照第 30 / 70 轮，NCM head + 定向 PGD 失败率",
           "runs": runs}, open(out, "w"), indent=1)
print(f"[ckpre] {out}  ({len(runs)} seed)")
PY
exit $RC
