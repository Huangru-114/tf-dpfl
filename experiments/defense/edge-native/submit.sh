#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# submit.sh  —  提交阶段三（edge 原生防御）的 run。登录节点跑，**纯 bash，不调 python**。
#
#   bash experiments/defense/edge-native/submit.sh --status
#   bash experiments/defense/edge-native/submit.sh --dry-run
#   PACK=3 RUN_GROUPS=SNAP bash experiments/defense/edge-native/submit.sh   # 一卡三跑（规则见 hfl-mechanism/submit_lib.sh）
#
# 结构照抄 hfl-mechanism/pilot/submit_pilot.sh，只读**本目录**的 configs/INDEX.tsv
# （harness/registry.py experiments/defense/edge-native/registry.yaml --materialize 生成，**不带 --group**）。
# 与 pilot 不同：本表是 P2 口径 → 保留 hfl-mechanism/AUDIT.md 的门槛（D-006；2026-09-27 起已全部关闭）。
# 「完成」= metrics.json 存在、exit_code 为 0、config_sha 等于 INDEX；exit 0 但 sha 不等 → stale，默认不重交（D-053）。
# 作业脚本复用 hfl-mechanism 的 cell.sbatch / pack.sbatch（只传 --config，不传任何覆盖；陷阱 #19）。
# 守卫：tests/test_edge_native_registry.py
# ══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${TFDPFL_ROOT:-$(cd "$HERE/../../.." && pwd)}"
MECH="$ROOT/experiments/attack/hfl-mechanism"
INDEX="$HERE/configs/INDEX.tsv"
AUDIT="$MECH/AUDIT.md"
JOB="$MECH/cell.sbatch"
PACK_JOB="$MECH/pack.sbatch"
ONLY_GROUPS="${RUN_GROUPS:-}"
# shellcheck source=../../attack/hfl-mechanism/submit_lib.sh
source "$MECH/submit_lib.sh"

DRY=0; STATUS=0
for a in "$@"; do case "$a" in
    --dry-run) DRY=1 ;;
    --status)  STATUS=1 ;;
    *) echo "未知参数: $a" >&2; exit 2 ;;
esac; done

# ── 审计门槛（同 hfl-mechanism/submit.sh）────────────────────────────────────
n_open=$(grep -E '^\|[[:space:]]*[AD][0-9]{2}[[:space:]]*\|' "$AUDIT" \
         | grep -cE '\|[[:space:]]*`(open|align)`[[:space:]]*\|' || true)
gate_closed=1
if [ "$n_open" -gt 0 ]; then
    gate_closed=0
    echo "[edge-native] AUDIT.md 还有 $n_open 行未关闭（open/align）→ 不会提交任何 run（D-006）"
fi

if [ ! -f "$INDEX" ]; then
    echo "[edge-native] 没有 $INDEX —— 先 materialize（不带 --group）："
    echo "[edge-native]   harness/registry.py $HERE/registry.yaml --materialize（纯标准库 + PyYAML）"
    exit 0
fi

SUBMIT_OK=0
[ "$DRY" -eq 0 ] && [ "$STATUS" -eq 0 ] && [ "$gate_closed" -eq 1 ] && SUBMIT_OK=1
total=0; done_n=0; todo_n=0; stale_n=0; queued=0; inq_n=0
queue_scan                                             # 防重交（D-070）
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
    if run_in_queue "$run_id"; then                     # 已交、还在排队 / 在跑 → 不重交（D-070）
        inq_n=$((inq_n + 1))
        echo "  queued $run_id（已在 SLURM 队列里，不重交）"
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
    if [ "$SUBMIT_OK" -eq 0 ]; then echo "  would sbatch  $run_id  ($config)"; continue; fi
    (cd "$ROOT" && sbatch --comment="exp3v2:$run_id" "$JOB" "$config" "$run_id" "$metrics")
    queued=$((queued + 1))
done < "$INDEX"
[ "$STATUS" -eq 0 ] && [ "$PACK" -gt 1 ] && pack_flush

echo "──────────────────────────────────────────────"
echo "edge-native run 总数=$total  已完成=$done_n  未完成=$todo_n  过期(stale)=$stale_n  已在队列=$inq_n  本次入队=$queued  held=$held"
[ "$stale_n" -gt 0 ] && [ "$RESUBMIT_STALE" != 1 ] && echo "(stale 默认不重交；要重跑设 RESUBMIT_STALE=1)"
[ "$STATUS" -eq 1 ] && echo "(仅状态；未提交)"
[ "$DRY" -eq 1 ] && echo "(dry-run；未提交)"
[ "$gate_closed" -eq 0 ] && [ "$STATUS" -eq 0 ] && echo "(审计门槛未过；未提交)"
echo "回传后对账：harness/status.py $HERE/registry.yaml（纯标准库 + PyYAML）；SNAP 的有效性闸见 registry.yaml / FINDINGS N-008"
exit 0
