"""
tests/test_undefined_metrics_are_null.py  —  无定义的分组必须是 null，不能是 0.0

**这条测试存在的理由**（已经污染了 Experiment 3 全部产出的 bug）：
`attack/backdoor_eval.py` 的聚合助手曾经是

    def _mean(xs):
        return float(np.mean(xs)) if len(xs) else 0.0    # ← 空组返回 0.0

于是「这个指标在本配置下**无定义**」和「后门完全没有传过去」在数值上完全一样。
后者是个强结论，前者什么都不是。已提交的 metrics.json 里的实际后果：

  · 所有 *distributed* 布点（每个 edge 都有恶意端）→ 不存在「无恶意 edge 的良性端」
    → `diff_edge_asr = 0.000`（2edge/4edge_distributed、全部 3C 格、flat 基线）
  · `10edge_collocated`（E0 里 10/10 都是恶意端，没有良性端）
    → `same_edge_asr = 0.000`，且 `per_edge[0].client_benign = 0.000`
    —— 后者还被画进了逐 edge 传播图，把那条线拉到底。

约定与防御判决一致（坐标类防御记 `admitted=None` 而非 0）：
**无定义就留空，绝不用 0 填充后当数值参与统计。**

日志侧记号是 `n/a`，回程解析成 JSON 的 `null`。
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEDAVG = ROOT / "fedavg"
sys.path.insert(0, str(ROOT / "harness"))

import collect_metrics as CM       # 纯标准库，本地可跑


# ─────────────────────────── 解析器：n/a → None ───────────────────────────

_LOG_NA = """\
[Round 5]
[Backdoor] Round 5 | GM_ASR=0.931 | EM_ASR=0.922 | local_benign=0.709 \
(same_edge=n/a, diff_edge=0.709) | local_malicious=0.963
[Backdoor] Round 5 | edge0 | edge_asr=0.922 | client_benign=n/a | \
client_malicious=0.963 | n_benign=0 | n_malicious=10 | has_malicious=True
[Backdoor] Round 5 | edge1 | edge_asr=0.901 | client_benign=0.709 | \
client_malicious=n/a | n_benign=10 | n_malicious=0 | has_malicious=False
"""


def test_na_parses_to_none_not_zero():
    """10edge_collocated 的真实形状：E0 全恶意 → same_edge / client_benign 无定义。"""
    out = CM.collect(_LOG_NA)
    rnd = out["rounds"][0]
    assert rnd["same_edge_asr"] is None, (
        f"same_edge_asr 解析成了 {rnd['same_edge_asr']!r}；无定义必须是 None，"
        " 0.0 会被读成「后门一点都没传到同 edge 的良性端」")
    assert rnd["diff_edge_asr"] == 0.709, "有定义的字段不能被误伤"
    assert rnd["global_asr"] == 0.931


def test_na_in_per_edge_panel_parses_to_none():
    out = CM.collect(_LOG_NA)
    e0, e1 = out["per_edge_final"]
    assert e0["client_benign"] is None, "E0 没有良性端 → client_benign 无定义"
    assert e0["n_benign"] == 0, "计数是真实的 0，不是无定义"
    assert e0["client_malicious"] == 0.963
    assert e1["client_malicious"] is None, "E1 没有恶意端 → client_malicious 无定义"
    assert e1["client_benign"] == 0.709


def test_old_logs_with_zeros_still_parse():
    """
    向后兼容：历史日志里那些 0.000 仍然要解析得出来（值为 0.0）。
    它们语义上是错的，但重新生成旧 metrics.json 的能力不能丢 ——
    区分真假 0 靠的是同一行的 n_benign / n_malicious 计数。
    """
    old = ("[Backdoor] Round 5 | GM_ASR=0.931 | EM_ASR=0.922 | local_benign=0.709 "
           "(same_edge=0.000, diff_edge=0.000) | local_malicious=0.963\n")
    rnd = CM.collect(old)["rounds"][0]
    assert rnd["same_edge_asr"] == 0.0 and rnd["diff_edge_asr"] == 0.0


def test_opt_helper_never_returns_zero_for_na():
    assert CM._opt("n/a") is None
    assert CM._opt("0.000") == 0.0


# ─────────────────── 产出侧：空组返回 None、打印用 None-safe 格式 ───────────────────

def _fn_named(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} 里找不到 {name}")


def test_empty_group_helpers_return_none():
    """
    `_mean` / `_std` 是 evaluate_hierarchical_asr 里的闭包，用 AST 检查它们的
    空分支——不需要 numpy/TF，本地可跑。
    """
    outer = _fn_named(FEDAVG / "attack" / "backdoor_eval.py",
                      "evaluate_hierarchical_asr")
    seen = set()
    for node in ast.walk(outer):
        if not (isinstance(node, ast.FunctionDef) and node.name in ("_mean", "_std")):
            continue
        seen.add(node.name)
        src = ast.unparse(node)
        assert "else None" in src, (
            f"{node.name} 的空分支不是 None：\n{src}\n"
            " 0.0 会让「无定义」与「ASR 为零」不可区分。")
    assert seen == {"_mean", "_std"}, f"只找到 {seen}，结构变了请更新本测试"


def test_printer_is_none_safe_for_group_metrics():
    """
    `f"{None:.3f}"` 会直接抛 TypeError —— 所以打印必须走 None-safe 的格式化。
    断言那几个分组字段不再是裸的 `:.3f`。
    """
    src = (FEDAVG / "server" / "backdoor_server.py").read_text(encoding="utf-8")
    for field in ("local_asr_same_edge", "local_asr_diff_edge",
                  "local_asr_benign_mean", "local_asr_malicious_mean"):
        assert f"metrics['{field}']:.3f" not in src, (
            f"{field} 仍在用裸的 :.3f 格式化；空组时它是 None，会抛 TypeError。"
            " 请走 _f3()（None → \"n/a\"）。")
    assert '"n/a" if v is None' in src, "找不到 None → n/a 的格式化助手"
