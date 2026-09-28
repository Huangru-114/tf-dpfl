#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# submit.sh  —  提交改版实验 3 的 run。登录节点跑，**纯 bash，不调 python**。
#
#   bash experiments/attack/hfl-mechanism/submit.sh --status     # 看进度（不提交）
#   bash experiments/attack/hfl-mechanism/submit.sh --dry-run    # 列出会提交什么
#   bash experiments/attack/hfl-mechanism/submit.sh              # 提交所有未完成的 run
#   RUN_GROUPS="G0 G2" bash ./submit.sh                         # 只处理这些组（不能叫 GROUPS：那是 bash 的内置变量）
#   PACK=3 RUN_GROUPS=G6 bash ./submit.sh                       # 一卡多跑（D-048 / D-052，见 submit_lib.sh）
#
# 读 configs/INDEX.tsv（harness/registry.py --materialize 生成）。一个 run 算「完成」要同时满足：
#   1. metrics.json 存在；2. exit_code 为 0；3. 里面的 config_sha 等于 INDEX.tsv 的值。
# 第 3 条是旧方案 run_exp3.sh 没有的：改了配置而结果还是旧的，旧脚本会照样算「已完成」
# （陷阱 #15）。配置一变 sha 就变，这样的旧结果标 `stale`（D-053）：不算完成，但**默认不重交**
# —— 已判完的组不会因为 base.yaml 改了一个评估开关就整组重跑；要重跑设 RESUBMIT_STALE=1。
#
# 一卡多跑：PACK=K（缺省 1 = 一卡一跑，与以前逐字相同）。同一个包只放同一格子的不同 seed；
# 每个格子先交一个 PROBE_K（缺省 3，D-069）的首包，显存数据回来后再按实测峰值定 K；OOM 自动降档。
# 显存没测过的新配置类型（如 10 edge / R20）交时写 PROBE_K=2。
# 规则与参数见 submit_lib.sh 的页首。**交完等回传再跑下一次**：脚本不查 SLURM 队列。
#
# **门槛**：AUDIT.md 里只要还有 `open` / `align` 的行，就拒绝提交（DECISIONS D-006）。
# --status 与 --dry-run 照样能看，只是不提交。没有绕过门槛的开关，这是刻意的。
# ══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="${TFDPFL_MECH_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ROOT="${TFDPFL_ROOT:-$(cd "$HERE/../../.." && pwd)}"
INDEX="$HERE/configs/INDEX.tsv"
AUDIT="$HERE/AUDIT.md"
JOB="$HERE/cell.sbatch"
PACK_JOB="$HERE/pack.sbatch"
ONLY_GROUPS="${RUN_GROUPS:-}"
# shellcheck source=submit_lib.sh
source "$HERE/submit_lib.sh"

DRY=0; STATUS=0
for a in "$@"; do case "$a" in
    --dry-run) DRY=1 ;;
    --status)  STATUS=1 ;;
    *) echo "未知参数: $a" >&2; exit 2 ;;
esac; done

# ── 审计门槛 ────────────────────────────────────────────────────────────────
n_open=$(grep -E '^\|[[:space:]]*[AD][0-9]{2}[[:space:]]*\|' "$AUDIT" \
         | grep -cE '\|[[:space:]]*`(open|align)`[[:space:]]*\|' || true)
gate_closed=1
if [ "$n_open" -gt 0 ]; then
    gate_closed=0
    echo "[submit] AUDIT.md 还有 $n_open 行未关闭（open/align）→ 不会提交任何 run（D-006）"
fi

if [ ! -f "$INDEX" ]; then
    echo "[submit] 没有 $INDEX"
    echo "[submit] 先 materialize：harness/registry.py $HERE/registry.yaml --materialize（纯标准库 + PyYAML）"
    echo "[submit] （base 要等 A4 之后才有；在那之前这里本来就是空的）"
    exit 0
fi

SUBMIT_OK=0
[ "$DRY" -eq 0 ] && [ "$STATUS" -eq 0 ] && [ "$gate_closed" -eq 1 ] && SUBMIT_OK=1
total=0; done_n=0; todo_n=0; stale_n=0; queued=0
while IFS=$'\t' read -r run_id group cell seed sha config metrics; do
    [ "$run_id" = "run_id" ] && continue                 # 表头
    if [ -n "$ONLY_GROUPS" ] && [[ " $ONLY_GROUPS " != *" $group "* ]]; then continue; fi
    total=$((total + 1))
    case "$metrics" in /*) m="$metrics" ;; *) m="$ROOT/$metrics" ;; esac
    state=$(run_state "$m" "$sha")
    if [ "$state" = done ]; then
        done_n=$((done_n + 1))
        [ "$STATUS" -eq 1 ] && echo "  done   $run_id"
        continue
    fi
    if [ "$state" = stale ]; then
        stale_n=$((stale_n + 1))
        if [ "$STATUS" -eq 1 ]; then echo "  stale  $run_id"; continue; fi
        [ "$RESUBMIT_STALE" = 1 ] || continue
    else
        todo_n=$((todo_n + 1))
        if [ "$STATUS" -eq 1 ]; then echo "  todo   $run_id"; continue; fi
    fi
    if [ "$PACK" -gt 1 ]; then
        pack_add "$group" "$cell" "$seed" "$config" "$run_id" "$metrics" "$m"; continue
    fi
    if [ "$SUBMIT_OK" -eq 0 ]; then
        echo "  would sbatch  $run_id  ($config)"; continue
    fi
    (cd "$ROOT" && sbatch "$JOB" "$config" "$run_id" "$metrics")
    queued=$((queued + 1))
done < "$INDEX"
[ "$STATUS" -eq 0 ] && [ "$PACK" -gt 1 ] && pack_flush

echo "──────────────────────────────────────────────"
echo "run 总数=$total  已完成=$done_n  未完成=$todo_n  过期(stale)=$stale_n  本次入队=$queued  held=$held"
[ "$stale_n" -gt 0 ] && [ "$RESUBMIT_STALE" != 1 ] && echo "(stale 默认不重交；要重跑设 RESUBMIT_STALE=1)"
[ "$STATUS" -eq 1 ] && echo "(仅状态；未提交)"
[ "$DRY" -eq 1 ] && echo "(dry-run；未提交)"
[ "$gate_closed" -eq 0 ] && [ "$STATUS" -eq 0 ] && echo "(审计门槛未过；未提交)"
exit 0
