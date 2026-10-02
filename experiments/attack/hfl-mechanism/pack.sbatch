#!/bin/bash
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gpus 1
#SBATCH -t 24:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=48G
#SBATCH --job-name=exp3v2-pack
# ══════════════════════════════════════════════════════════════════════════
# pack.sbatch  —  一张 GPU 上同时跑 K 个 run（D-047「一卡多跑，先测后用」）。
#
#   sbatch -c $((4*K)) experiments/attack/hfl-mechanism/pack.sbatch <tag> \
#          <config_rel_1> <run_id_1> <metrics_rel_1>  [<config_rel_2> <run_id_2> <metrics_rel_2> …]
#
# 为什么：pilot 的 run 是 ResNet-10、batch 32、客户端串行 —— GPU 大概率大部分时间闲着。
# 计费按 GPU 小时，K 个 run 挤一张卡，若墙钟涨得比 K 慢，每个 run 的机时就降了。
# **能快多少、会不会影响确定性，没有证据** —— 由 pilot/submit_pack_test.sh + harness/pack_test.py 回答。
#
# 与 cell.sbatch 的区别只有三处，其余（只传 --config、cd fedavg、日志在上一级、collect + exit_code）照抄：
#   1. TF_FORCE_GPU_ALLOW_GROWTH=true —— 代码没开显存按需增长，第一个进程会占满显存，第二个起不来。
#   2. 每个 run 用 taskset 绑到**恰好 4 个**自己的核上（TFDPFL_CORES_PER_RUN，缺省 4 = cell.sbatch 的 -c 4）。
#      **核数会改变结果**：CPU 上同一配置 2 核与 4 核的 checksum 不同，而同为 2 核时「并发」与「单跑」逐位相同
#      （FINDINGS F-047）。所以每个 run 看到的核数必须与单跑一样，否则与一卡一跑的结果不可比。
#      可用核不够 K×4、或没有 taskset → 拒绝启动（不静默降级成「核数随 K 变」）。
#   3. 后台采样 nvidia-smi（每 15 s），结束时写一个小 JSON（利用率均值 / 峰值、显存峰值、墙钟）
#      到第一个 run 的 metrics 旁边：<tag>.gpu.json。没有 nvidia-smi 就跳过（本地 CPU 验证）。
#      D-052 起另记：整卡容量 mem_total_mib；每个 run 的**真实**显存峰值 run_peak_mib（来自日志的
#      [GPUMem]，TF 分配器实际占用 —— 整卡 memory.used 含按需增长预留的块，偏大）；OOM 的 run。
#      提交脚本（submit_lib.sh）按这几个数定下一包的 K。
#   4. OOM 判定（D-052）：日志里出现 ResourceExhaustedError / OOM when allocating /
#      CUDA_ERROR_OUT_OF_MEMORY 就算 OOM —— **包括被 _collect_updates_* 吞掉的客户端异常**
#      （run 照样 exit 0，恶意端被踢出聚合，结果无效，陷阱 #23 同类）。这种 run 的 exit_code
#      记成 OOM_RC（非 0）→「完成 = exit 0 + sha 一致」的判据不用改就会把它当未完成重交。
#      TF 的「ran out of memory trying to allocate … not a failure」是警告不是失败：只计数，不判 OOM。
#   5. **主机 OOM**（D-070）：run 的退出码 137（SIGKILL）= 作业 cgroup 超了 --mem，OOM killer 杀了它
#      （2026-09-28 G6 s44 的 K=3 包在 48G 下就是这样死的，sacct 3083773 = OUT_OF_MEMORY，F-060）。
#      exit_code 保留 137（→ todo），pack.host_oom = true，并计入 gpu.json 的 n_oom / oom_runs（另记 n_host_oom）
#      → 提交脚本的降档规则把该格子 K 压到 k − 1。gpu.json 另记本作业申请的内存 mem_req_mb。
#   6. 日志与 nvidia-smi 采样文件名带 SLURM 作业号：同一个 run 被重复提交时，两个作业不再互相截断日志（F-060）。
# -c / --mem 可在 sbatch 命令行覆盖：submit_lib.sh 按 K 传 -c 4K --mem=<每 run GB × K>G；
# 下面的 48G 只是直接 sbatch 本脚本时的缺省（= K=2 × 24G）。
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

TAG="${1:?用法: sbatch pack.sbatch <tag> <config_rel> <run_id> <metrics_rel> [...]}"
shift
if [ $# -lt 3 ] || [ $(( $# % 3 )) -ne 0 ]; then
    echo "[pack] 参数必须是 (config_rel run_id metrics_rel) 的三元组，收到 $# 个"; exit 2
fi

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
source "$ROOT/cluster_env.sh"
export TF_FORCE_GPU_ALLOW_GROWTH=true

LOGDIR="${TFDPFL_LOGDIR:-$ROOT/../tfdpfl-logs}"
mkdir -p "$LOGDIR"

# ── 解析三元组；先全部校验，再一起启动（不留半启动的作业）─────────────────
CONFIGS=(); RUN_IDS=(); METRICS=(); LOGS=()
while [ $# -gt 0 ]; do
    CONFIGS+=("$ROOT/$1"); RUN_IDS+=("$2"); METRICS+=("$ROOT/$3")
    LOGS+=("$LOGDIR/exp3v2_$2.${SLURM_JOB_ID:-local}.log")    # 带作业号：重复提交不互相截断（D-070）
    shift 3
done
K=${#RUN_IDS[@]}
for c in "${CONFIGS[@]}"; do
    [ -f "$c" ] || { echo "[pack] 配置不存在: $c"; exit 2; }
done

# ── 分核：把本作业可用的核平均分成 K 组 ────────────────────────────────────
expand_cpus() {                      # "0-3,8,10-11" → "0 1 2 3 8 10 11"
    local out=() part a b c
    IFS=',' read -ra parts <<< "$1"
    for part in "${parts[@]}"; do
        if [[ $part == *-* ]]; then
            a=${part%-*}; b=${part#*-}
            for ((c = a; c <= b; c++)); do out+=("$c"); done
        else
            out+=("$part")
        fi
    done
    echo "${out[@]}"
}
ALLOWED=$(awk '/^Cpus_allowed_list/ {print $2}' /proc/self/status 2>/dev/null)
read -ra CPUS <<< "$(expand_cpus "${ALLOWED:-}")"
PER="${TFDPFL_CORES_PER_RUN:-4}"
if ! command -v taskset >/dev/null 2>&1; then
    echo "[pack] 没有 taskset，无法保证每个 run 的核数与单跑相同（F-047）→ 不启动"; exit 2
fi
if [ "${#CPUS[@]}" -lt $(( K * PER )) ]; then
    echo "[pack] 可用核 ${#CPUS[@]} < K×每 run 核数 = ${K}×${PER} → 不启动（用 sbatch -c $(( K * PER )) 提交）"; exit 2
fi
echo "[pack] tag=$TAG  K=$K  可用核=${#CPUS[@]}  每个 run 绑 ${PER} 核"

# ── GPU 采样 ──────────────────────────────────────────────────────────────
GPU_CSV="$LOGDIR/exp3v2_${TAG}.${SLURM_JOB_ID:-local}.gpu.csv"
SMI_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
        -l 15 > "$GPU_CSV" 2>/dev/null &
    SMI_PID=$!
fi

# ── 并行启动 K 个 run ─────────────────────────────────────────────────────
T_START=$(date +%s)
cd "$ROOT/fedavg"
PIDS=()
for ((i = 0; i < K; i++)); do
    CORES=$(IFS=,; echo "${CPUS[*]:$((i * PER)):$PER}")
    echo "[pack] 启动 ${RUN_IDS[$i]}  taskset -c $CORES  日志 -> ${LOGS[$i]}"
    taskset -c "$CORES" $PY main.py --config "${CONFIGS[$i]}" > "${LOGS[$i]}" 2>&1 &
    PIDS+=($!)
done

RCS=()
for ((i = 0; i < K; i++)); do
    wait "${PIDS[$i]}"
    RCS+=($?)
    echo "[pack] ${RUN_IDS[$i]} 结束，exit=${RCS[$i]}"
done
T_END=$(date +%s)
[ -n "$SMI_PID" ] && kill "$SMI_PID" 2>/dev/null

# ── 逐个收尾：collect_metrics + exit_code + 打包信息（照抄 cell.sbatch）──────
cd "$ROOT"
OOM_RC=86                            # OOM 专用退出码（D-052）：非 0 → 提交脚本当未完成
WORST=0
for ((i = 0; i < K; i++)); do
    mkdir -p "$(dirname "${METRICS[$i]}")"
    $PY harness/collect_metrics.py "${LOGS[$i]}" -o "${METRICS[$i]}"
    # 只取最后一行、且必须是整数：容器包装层往 stdout 多打一行也不会把退出码弄坏
    RC_EFF=$($PY - "${METRICS[$i]}" "${RCS[$i]}" "$TAG" "$K" "$i" "${SLURM_JOB_ID:-local}" \
             "${LOGS[$i]}" "$OOM_RC" <<'PY' | tail -n 1
import json, os, re, sys
p, rc, tag, k, slot, job = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
log, oom_rc = sys.argv[7], int(sys.argv[8])
text = open(log, errors="replace").read() if os.path.isfile(log) else ""
oom = bool(re.search(r"ResourceExhaustedError|OOM when allocating|CUDA_ERROR_OUT_OF_MEMORY", text))
warn = len(re.findall(r"ran out of memory trying to allocate", text))
d = json.load(open(p))
d["exit_code"] = rc
if oom and rc == 0:
    d["exit_code"] = oom_rc          # 被吞掉的 OOM：run 退出 0 但结果无效
host_oom = rc == 137                 # SIGKILL：作业 cgroup 超 --mem，被 OOM killer 杀（D-070）
d["pack"] = {"tag": tag, "k": k, "slot": slot, "job": job, "oom": oom, "host_oom": host_oom,
             "mem_warnings": warn, "peak_mib": (d.get("gpu_mem") or {}).get("peak_mib")}
json.dump(d, open(p, "w"), indent=2, ensure_ascii=False)
print(d["exit_code"])
PY
)
    [[ "$RC_EFF" =~ ^[0-9]+$ ]] && RCS[$i]=$RC_EFF
    [ "${RCS[$i]}" -ne 0 ] && WORST=${RCS[$i]}
done

# ── GPU 汇总（小 JSON，可回传）────────────────────────────────────────────
GPU_JSON="$(dirname "${METRICS[0]}")/${TAG}.gpu.json"
$PY - "$GPU_CSV" "$GPU_JSON" "$K" "$((T_END - T_START))" "${RUN_IDS[@]}" "${METRICS[@]}" <<'PY'
import json, os, sys
csv_path, out, k, wall = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
runs, metrics = sys.argv[5:5 + k], sys.argv[5 + k:5 + 2 * k]
util, mem, total = [], [], []
if os.path.isfile(csv_path):
    for ln in open(csv_path):
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) >= 2 and parts[0].replace(".", "").isdigit():
            util.append(float(parts[0])); mem.append(float(parts[1]))
            if len(parts) >= 3 and parts[2].replace(".", "").isdigit():
                total.append(float(parts[2]))
peaks, oom, warn, host = [], [], 0, 0
for rid, mp in zip(runs, metrics):
    try:
        pk = json.load(open(mp)).get("pack") or {}
    except (OSError, ValueError):
        pk = {}
    peaks.append(pk.get("peak_mib"))
    if pk.get("oom") or pk.get("host_oom"):     # 主机 OOM 与 GPU OOM 一样让下一包降档（D-070）
        oom.append(rid)
    host += 1 if pk.get("host_oom") else 0
    warn += int(pk.get("mem_warnings") or 0)
_req = os.environ.get("SLURM_MEM_PER_NODE", "")
known = [p for p in peaks if p is not None]
# 标量键各占一行：submit_lib.sh 是纯 bash，只 grep 标量
json.dump({"k": k, "runs": runs, "wall_s": wall, "n_samples": len(util),
           "util_mean": round(sum(util) / len(util), 1) if util else None,
           "util_max": max(util) if util else None,
           "mem_max_mib": max(mem) if mem else None,
           "mem_total_mib": max(total) if total else None,
           "run_peak_max_mib": max(known) if known else None,
           "n_oom": len(oom),
           "n_host_oom": host,
           "mem_req_mb": int(_req) if _req.isdigit() else None,
           "n_mem_warnings": warn,
           "run_peak_mib": peaks,
           "oom_runs": oom},
          open(out, "w"), indent=2)
print(f"[pack] GPU 汇总 -> {out}")
PY

echo "[pack] 墙钟 $((T_END - T_START)) s；回传 ${K} 个 metrics.json + ${TAG}.gpu.json"
exit "$WORST"