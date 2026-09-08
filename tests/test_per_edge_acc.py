"""
tests/test_per_edge_acc.py  —  逐 edge 精度面板 `[Acc]` 的两侧

**这个面板回答什么**：后门的**干净精度代价是逐 edge 的** —— 被污染的 edge 掉多少、
干净 edge 掉多少，是两个不同的数。而 `server.py` 此前把 `em_accs[]` / `pm_accs[]`
**算出来之后立刻塌成一个按样本加权的均值**（`run_round` 里的 `avg_em_acc` /
`avg_pm_acc`），逐 edge 的值当场丢弃 → metrics.json 里根本问不了这个问题。

**两侧分开测**（陷阱 #10）：
  - 上游侧：AST + 源码断言 `run_round` 真的打印了 `[Acc]` 行，**且聚合表达式没被动过**；
  - 下游侧：`collect_metrics` 真的解析成 `per_edge_acc_rounds` / `per_edge_acc_final`。

⚠️ **本文件不能证明的事**：逐 edge 的数与全局均值在**真实运行**中数值自洽
（逐 edge 按 n_samples 加权 == avg_em_acc）—— 那要跑起来才知道，本地没有 TF。
这里只在**合成日志**上锁住这条恒等式的形式，真正的确认在集群 smoke（L2）。

纯 Python（ast + 正则），本地秒级。
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

from collect_metrics import collect      # noqa: E402

SERVER = ROOT / "fedavg" / "server" / "server.py"
SRC = SERVER.read_text(encoding="utf-8")


def _run_round_src() -> str:
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_round":
            return ast.get_source_segment(SRC, node)
    pytest.fail("server.py 里找不到 run_round —— 本测试的定位假设失效了")


# ── 上游侧 ────────────────────────────────────────────────────────────────
def test_run_round_prints_the_per_edge_acc_line():
    src = _run_round_src()
    assert "[Acc] Round " in src, "run_round 没有打印 [Acc] 逐 edge 行"
    for field in ("em_acc=", "pm_acc=", "n_clients=", "n_samples=", "edge{"):
        assert field in src, f"[Acc] 行缺字段 {field}"


def test_per_edge_pm_is_grouped_by_edge():
    src = _run_round_src()
    assert "per_edge_pm" in src, "没有按 edge 分组 PM 精度"
    assert "per_edge_pm[edge.edge_id]" in src, \
        "per_edge_pm 的键必须是 edge.edge_id —— 用列表下标会在 edge 顺序变化时静默错位"


def test_aggregate_expressions_are_untouched():
    """**这条是「纯新增」的守卫。**

    加逐 edge 面板不能改变任何已发表口径。两个加权均值的表达式必须逐字不变；
    改了它们 = 全部历史 acc 数字与新数字不可比，而日志上看不出来。
    """
    for expr in (
        "avg_em_acc  = float(sum(a * n / total_em_n for a, n in zip(em_accs, em_ns)))",
        "avg_pm_acc  = float(sum(a * n / total_pm_n for a, n in zip(pm_accs, pm_ns)))",
    ):
        assert expr in SRC, f"聚合表达式被改动了：\n  期望 {expr}"


def test_unevaluated_pm_prints_na_not_zero():
    """未到 PM 评估轮打 `n/a`。打 0 会被读成「精度归零」（陷阱 #13 的同一个坑）。"""
    src = _run_round_src()
    assert '"n/a" if v is None' in src, "[Acc] 行没有 None → n/a 的格式化"
    assert 'pm_acc={_f4(' in src, "pm_acc 没走可空格式化，None 会炸或被写成 0"


# ── 下游侧 ────────────────────────────────────────────────────────────────
def _log(rounds=(1, 2), pm_at=(2,), n_edges=2):
    """合成日志：em_acc 每轮都有，pm_acc 只在 pm_at 的轮有。"""
    L = ["[Config] loading cfg.yaml", "[Config] run_name = r_seed42"]
    for r in rounds:
        L.append(f"[Round   {r}] Broadcasting to {n_edges} edges...")
        L.append(f"  [Cloud] GM=0.10 | EM=0.50 | loss=2.0 | time=1.0s | "
                 f"comm=1.0MB (total=1MB)")
        for e in range(n_edges):
            pm = f"{0.60 + 0.01 * e:.4f}" if r in pm_at else "n/a"
            L.append(f"[Acc] Round {r} | edge{e} | em_acc={0.70 + 0.01 * e:.4f} | "
                     f"pm_acc={pm} | n_clients={25 + e} | n_samples={1000 * (e + 1)}")
    return "\n".join(L)


def test_collect_parses_the_per_edge_acc_panel():
    out = collect(_log())
    pea = out["per_edge_acc_rounds"]
    assert set(pea) == {1, 2}
    assert [e["edge_id"] for e in pea[1]] == [0, 1]
    assert pea[1][0]["em_acc"] == pytest.approx(0.70)
    assert pea[1][1]["n_samples"] == 2000
    assert pea[1][1]["n_clients"] == 26


def test_pm_na_becomes_none_not_zero():
    pea = collect(_log())["per_edge_acc_rounds"]
    assert pea[1][0]["pm_acc"] is None       # 第 1 轮不是 PM 评估轮
    assert pea[1][0]["pm_acc"] != 0.0        # 铁律 #5
    assert pea[2][0]["pm_acc"] == pytest.approx(0.60)


def test_final_snapshot_picks_the_last_round_that_actually_has_pm():
    """末轮快照必须落在**有 pm_acc** 的轮上。

    em_acc 每轮都有、pm_acc 只在评估轮有 —— 直接取 max(round) 往往落在一个
    pm_acc 全是 None 的轮上，于是「最终个性化精度」在 metrics.json 里凭空消失。
    """
    out = collect(_log(rounds=(1, 2, 3), pm_at=(2,)))
    final = out["per_edge_acc_final"]
    assert [e["pm_acc"] for e in final] == [pytest.approx(0.60), pytest.approx(0.61)]


def test_weighted_mean_over_edges_is_the_invariant_downstream_relies_on():
    """逐 edge 值按 n_samples 加权 == 全局均值。

    这里只在合成数据上锁住**形式**（下游分析会这么用）；真实运行中的一致性
    要在集群 smoke 里核对 —— 本地没有 TF，跑不了 server.run_round。
    """
    pea = collect(_log())["per_edge_acc_rounds"][2]
    tot = sum(e["n_samples"] for e in pea)
    em = sum(e["em_acc"] * e["n_samples"] / tot for e in pea)
    pm = sum(e["pm_acc"] * e["n_samples"] / tot for e in pea)
    assert em == pytest.approx((0.70 * 1000 + 0.71 * 2000) / 3000)
    assert pm == pytest.approx((0.60 * 1000 + 0.61 * 2000) / 3000)


def test_log_without_acc_lines_yields_empty_not_bogus():
    """老日志没有 [Acc] 行 → 空 dict / 空 list，不是伪造的零值。"""
    out = collect("[Config] loading cfg.yaml\n[Round   1] Broadcasting to 2 edges...")
    assert out["per_edge_acc_rounds"] == {}
    assert out["per_edge_acc_final"] == []
