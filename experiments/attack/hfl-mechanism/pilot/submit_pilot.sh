#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# submit_pilot.sh  —  提交 A4 的可行性 pilot（D-029 / D-031 / A15 的确定性对）。
# 登录节点跑，**纯 bash，不调 python**。
#
#   bash experiments/attack/hfl-mechanism/pilot/submit_pilot.sh --status
#   bash experiments/attack/hfl-mechanism/pilot/submit_pilot.sh --dry-run
#   bash experiments/attack/hfl-mechanism/pilot/submit_pilot.sh
#   RUN_GROUPS="DET" bash .../pilot/submit_pilot.sh        # 只提交某几组
#   PACK=3 bash .../pilot/submit_pilot.sh                  # 一卡多跑（与 submit.sh 同一套规则，见 ../submit_lib.sh）
#
# 为什么不复用上一级的 submit.sh：那个脚本在 AUDIT.md 还有 open/align 行时**拒绝提交**
# （D-006），而这些 pilot 恰恰是关闭 A08 / A26 / A15 / A25 的前提。D-029 明确写着
# pilot「登记在单独的 pilot 登记表（P1 口径、不进 P2），不违反 D-006」——
# 所以这里只读 **pilot/configs/INDEX.tsv**，绝不碰上一级的 configs/（守卫
# tests/test_pilot_a4.py::test_submit_pilot_only_reads_the_pilot_index）。
#
# 「完成」的判据与 submit.sh 相同：metrics.json 存在、exit_code 为 0、config_sha 等于 INDEX；
# exit 0 但 sha 不等 → stale，默认不重交（RESUBMIT_STALE=1 才交，D-053）。
# 作业脚本复用上一级的 cell.sbatch / pack.sbatch（只传 --config，不传任何覆盖）。
# ══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${TFDPFL_ROOT:-$(cd "$HERE/../../../.." && pwd)}"
INDEX="$HERE/configs/INDEX.tsv"
JOB="$HERE/../cell.sbatch"
PACK_JOB="$HERE/../pack.sbatch"
ONLY_GROUPS="${RUN_GROUPS:-}"
# shellcheck source=../submit_lib.sh
source "$HERE/../submit_lib.sh"

DRY=0; STATUS=0
for a in "$@"; do case "$a" in
    --dry-run) DRY=1 ;;
    --status)  STATUS=1 ;;
    *) echo "未知参数: $a" >&2; exit 2 ;;
esac; done

if [ ! -f "$INDEX" ]; then
    echo "[pilot] 没有 $INDEX —— 先 materialize："
    echo "[pilot]   harness/registry.py $HERE/registry.yaml --materialize（纯标准库 + PyYAML）"
    exit 0
fi

SUBMIT_OK=0
[ "$DRY" -eq 0 ] && [ "$STATUS" -eq 0 ] && SUBMIT_OK=1
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
    if [ "$DRY" -eq 1 ]; then echo "  would sbatch  $run_id  ($config)"; continue; fi
    (cd "$ROOT" && sbatch --comment="exp3v2:$run_id" "$JOB" "$config" "$run_id" "$metrics")
    queued=$((queued + 1))
done < "$INDEX"
[ "$STATUS" -eq 0 ] && [ "$PACK" -gt 1 ] && pack_flush

echo "──────────────────────────────────────────────"
echo "pilot run 总数=$total  已完成=$done_n  未完成=$todo_n  过期(stale)=$stale_n  已在队列=$inq_n  本次入队=$queued  held=$held"
[ "$stale_n" -gt 0 ] && [ "$RESUBMIT_STALE" != 1 ] && echo "(stale 默认不重交；要重跑设 RESUBMIT_STALE=1)"
[ "$STATUS" -eq 1 ] && echo "(仅状态；未提交)"
[ "$DRY" -eq 1 ] && echo "(dry-run；未提交)"
echo "回传后判定：harness/pilot_a4.py $HERE/registry.yaml（纯标准库 + PyYAML）"
exit 0
