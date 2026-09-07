#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# test_exp3.sh  —  Experiment 3A/3B 的 GPU 前 preflight（登录节点/容器内秒~分钟级，无 GPU）
#
#   bash experiments/attack/hfl-propagation/test_exp3.sh
#
# 四步，任一步失败即非零退出 —— 绿了再去 sbatch 烧 GPU：
#   1. L1：本工作相关的测试模块（布点 / 强制参与 / 回程解析 / 播种完整性 /
#      测试集隔离 / 参与配额 / 无定义指标 / exp3 配置不变量）
#   2. 全部 16 个 config 过 config_validate（长度/容量早筛）
#   3. 布点 dry-check：对每个 config 真跑 block_assignment + resolve_malicious_ids，
#      断言每个 edge 的恶意端数 == malicious_per_edge（不训练、不用 GPU）
#   4. **旧结果不得遮挡重跑**：run_exp3.sh 按 exit_code:0 跳过已完成格子，
#      results/ 里残留的旧 metrics.json 会让整批实验一个作业都不提交
#      而输出显示「已完成=N」——这类失败最难发现，所以在这里挡住。
#
# 只跑与本工作相关的测试，**不**跑全量 tests/ —— 避免被既有的红灯干扰判读。
# （既有红灯是陷阱 #4 的 test_neurotoxin_mask ×2；CLAUDE.md 里写的「6 条」已过期，
#  test_badpfl_trigger 那 4 条已于 2026-08-21 修好。开工前请自己实测一次记基线。）
# ══════════════════════════════════════════════════════════════════════════
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
# shellcheck source=cluster_env.sh
source "$ROOT/cluster_env.sh"

# 默认扫描本目录所有 *.yaml（3A/3B 8 个 + 3C 7 个 + flat 1 个 = 16）；CONFIGS= 可缩小范围
CONFIGS="${CONFIGS:-$(cd "$HERE" && ls *.yaml 2>/dev/null | sed 's/\.yaml$//' | tr '\n' ' ')}"

fail() { echo -e "\n[test_exp3] ❌ $1"; exit 1; }

# ── 1. L1 ─────────────────────────────────────────────────────────────────
echo "== [1/4] L1 =="
$PY -m pytest -q \
    tests/test_malicious_placement.py \
    tests/test_forced_participation.py \
    tests/test_collect_metrics.py \
    tests/test_seeding_completeness.py \
    tests/test_no_test_leakage.py \
    tests/test_participation_quota.py \
    tests/test_undefined_metrics_are_null.py \
    tests/test_exp3_config_invariants.py \
    tests/test_eval_timing.py \
    tests/test_plot_exp3_style.py || fail "L1 未通过"

# ── 2. config_validate ────────────────────────────────────────────────────
echo -e "\n== [2/4] config_validate 全部 config =="
( cd "$ROOT/fedavg" && $PY - "$ROOT" $CONFIGS <<'PY'
import sys, yaml, io, contextlib
sys.path.insert(0, ".")
import config_validate as cv
root = sys.argv[1]
ok = True
for name in sys.argv[2:]:
    p = f"{root}/experiments/attack/hfl-propagation/{name}.yaml"
    cfg = yaml.safe_load(open(p))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cv.validate_config(cfg)
        ne = cfg["federation"]["n_edges"]; pe = cfg["backdoor"]["malicious_per_edge"]
        assert len(pe) == ne and sum(pe) == 10, (name, pe, ne)
        print(f"  ok  {name:<20} n_edges={ne} per_edge={pe}")
    except Exception as e:
        print(f"  FAIL {name}: {str(e).splitlines()[0][:70]}"); ok = False
sys.exit(0 if ok else 1)
PY
) || fail "config_validate 未通过"

# ── 3. 布点 dry-check（真跑 block + resolve，不训练）──────────────────────
echo -e "\n== [3/4] 布点 dry-check：每 edge 恶意端数是否 == malicious_per_edge =="
( cd "$ROOT/fedavg" && $PY - "$ROOT" $CONFIGS <<'PY'
import sys, yaml
sys.path.insert(0, ".")
from data.clustering import block_assignment
from attack.backdoor import resolve_malicious_ids
root = sys.argv[1]
ok = True
for name in sys.argv[2:]:
    cfg = yaml.safe_load(open(f"{root}/experiments/attack/hfl-propagation/{name}.yaml"))
    n_clients = int(cfg["federation"]["n_clients"])
    n_edges   = int(cfg["federation"]["n_edges"])
    per_edge  = cfg["backdoor"]["malicious_per_edge"]
    seed      = int(cfg.get("seed", 42))
    # 复刻 main.py 的接线：block 提前算 → 喂 resolve_malicious_ids
    assign = block_assignment(list(range(n_clients)), n_edges)
    mal = resolve_malicious_ids(cfg["backdoor"], n_clients, assignments=assign, seed=seed)
    got = [0] * n_edges
    for cid in mal:
        got[int(assign[cid])] += 1
    if got != list(per_edge):
        print(f"  FAIL {name}: 每 edge 恶意数 {got} != 期望 {per_edge}"); ok = False
    else:
        # 顺带打印 E0 命中的 id，肉眼确认落在连续段里
        e0 = sorted(c for c in mal if assign[c] == 0)
        print(f"  ok  {name:<20} per_edge={got}  E0_ids={e0}")
sys.exit(0 if ok else 1)
PY
) || fail "布点 dry-check 未通过"

# ── 4. 旧结果不得遮挡重跑 ─────────────────────────────────────────────────
echo -e "\n== [4/4] results/ 里没有会让 run_exp3.sh 跳过格子的旧 metrics.json =="
STALE=$(find "$HERE/results" -maxdepth 1 -name '*.metrics.json' 2>/dev/null \
        -exec grep -l '"exit_code": *0' {} + 2>/dev/null | wc -l)
if [ "$STALE" -gt 0 ]; then
    echo "  发现 $STALE 个带 exit_code:0 的旧结果，run_exp3.sh 会跳过对应格子："
    find "$HERE/results" -maxdepth 1 -name '*.metrics.json' -printf '    %f\n' 2>/dev/null | head -5
    echo "  若是**有意续跑**，忽略本条即可；若是要**重跑**，先把它们移进"
    echo "  results/archive-<原因>/ 再来（旧的那批已在 results/archive-pre-fix/）。"
    fail "旧结果会遮挡重跑（这是提示，不是代码错误——确认意图后再继续）"
fi
echo "  ok  results/ 干净，16 个格子都会真的被提交"

echo -e "\n[test_exp3] ✅ 全部通过 —— 可以 sbatch 跑实验了。"
