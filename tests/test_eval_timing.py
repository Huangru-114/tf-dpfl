"""
tests/test_eval_timing.py  —  评估耗时仪表的两侧

**为什么要两侧分开测**（陷阱 #10 的教训）：
解析器认得某个格式 ≠ 代码真的会打印它。这里
  - 上游侧：AST 断言 `_backdoor_eval` 真的发出了 `[Timing]` 行、且分段计时
    确实包住了那几个阶段；
  - 下游侧：`collect_metrics` 真的把它解析成 `timing_rounds` / `timing_summary`。

**这个仪表回答什么**：`[Cloud] … time=Xs` 测的是 `CloudServer.run_round` 的
t0→elapsed，而 `_backdoor_eval` 是在 `super().run_round()` 返回**之后**才调用的，
所以那个数**不含**后门评估。标定 (local_epochs, n_rounds, eval_interval) 要读的
「一轮花多少、其中多少在评估上」此前在 metrics.json 里完全看不到。

纯 Python（ast + 正则），本地秒级，不需要 TF。
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

from collect_metrics import collect          # noqa: E402

BD_SERVER = ROOT / "fedavg" / "server" / "backdoor_server.py"


def _log(timing=True, n_rounds=3, feature="n/a", drift="n/a"):
    L = []
    for r in range(1, n_rounds + 1):
        L.append(f"[Round   {r}] Broadcasting to 2 edges...")
        L.append(f"  [Cloud] GM=0.1{r}00 | EM=0.5{r}00 | loss=2.0 | "
                 f"time={r * 10}.0s | comm=1.0MB (total=1MB)")
        L.append(f"[Backdoor] Round {r} | GM_ASR=0.0{r}0 | EM_ASR=0.1{r}0 | "
                 f"local_benign=0.2{r}0 (same_edge=0.3{r}0, diff_edge=0.4{r}0) | "
                 f"local_malicious=0.9{r}0")
        if timing:
            L.append(f"[Timing] Round {r} | asr={r}.0s | feature={feature}s | "
                     f"forgetting=2.0s | drift={drift}s | total={r + 2}.0s")
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════
# 1. 上游侧：代码真的会打印这一行
# ══════════════════════════════════════════════════════════════════════════
def _backdoor_eval_src() -> str:
    tree = ast.parse(BD_SERVER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_backdoor_eval":
            return ast.get_source_segment(BD_SERVER.read_text(encoding="utf-8"), node)
    pytest.fail("backdoor_server.py 里找不到 _backdoor_eval")


def test_backdoor_eval_emits_timing_line():
    src = _backdoor_eval_src()
    assert "[Timing] Round" in src, (
        "_backdoor_eval 必须发出可解析的 [Timing] 行 —— 否则 metrics.json 里"
        "永远看不到评估花了多少时间")
    for phase in ("asr=", "feature=", "forgetting=", "drift=", "total="):
        assert phase in src, f"[Timing] 行缺少 {phase} 分段"


def test_timing_uses_perf_counter_not_wall_clock():
    """`time.time()` 会被 NTP 校时拽走；计时一律用单调钟。"""
    src = _backdoor_eval_src()
    assert "perf_counter" in src
    assert "time.time()" not in src


def test_every_optional_phase_is_timed():
    """四个阶段各自被计时包住，而不是只测一个总数。

    只有总数的话，「ASR 慢」与「遗忘曲线慢」分不开 —— 而这两件的优化手段
    完全不同（前者调 asr_max_samples，后者调 forgetting_epochs）。
    """
    src = _backdoor_eval_src()
    for var in ("t_asr", "t_feature", "t_forget", "t_drift"):
        assert f"{var} = time.perf_counter() - " in src, f"{var} 没有被计时"


def test_unused_phases_report_na_not_zero():
    """未启用的阶段打 n/a。0.0 会被读成「跑了但是很快」，与「根本没跑」混同。"""
    src = _backdoor_eval_src()
    assert "t_feature = t_forget = t_drift = None" in src
    assert '"n/a" if v is None' in src


# ══════════════════════════════════════════════════════════════════════════
# 2. 下游侧：解析器读得出来
# ══════════════════════════════════════════════════════════════════════════
def test_timing_rounds_are_collected():
    m = collect(_log())
    assert [t["round"] for t in m["timing_rounds"]] == [1, 2, 3]
    assert m["timing_rounds"][0]["asr_s"] == pytest.approx(1.0)
    assert m["timing_rounds"][2]["total_s"] == pytest.approx(5.0)
    assert m["timing_rounds"][0]["forgetting_s"] == pytest.approx(2.0)


def test_disabled_phase_parses_to_none_not_zero():
    m = collect(_log(feature="n/a", drift="n/a"))
    assert all(t["feature_s"] is None for t in m["timing_rounds"])
    assert all(t["drift_s"] is None for t in m["timing_rounds"])
    # 而启用的阶段是真数字
    assert all(t["asr_s"] is not None for t in m["timing_rounds"])


def test_round_time_is_captured_from_cloud_line():
    m = collect(_log())
    assert [a["round_time"] for a in m["acc_rounds"]] == [10.0, 20.0, 30.0]


def test_round_time_is_none_in_legacy_log_without_time_field():
    log = "\n".join([
        "[Round   1] Broadcasting to 2 edges...",
        "  [Cloud] GM=0.1100 | EM=0.5100 | loss=2.0 | comm=1.0MB (total=1MB)",
    ])
    m = collect(log)
    assert m["acc_rounds"][0]["round_time"] is None
    assert m["acc_rounds"][0]["gm_acc"] == pytest.approx(0.11), (
        "加 round_time 不能破坏原有字段的解析")


# ══════════════════════════════════════════════════════════════════════════
# 3. 汇总：训练 vs 评估的墙钟拆分
# ══════════════════════════════════════════════════════════════════════════
def test_timing_summary_splits_train_and_eval():
    m = collect(_log())
    s = m["timing_summary"]
    assert s["round_time_total_s"] == pytest.approx(60.0)   # 10+20+30
    assert s["bd_eval_total_s"] == pytest.approx(12.0)      # 3+4+5
    assert s["wall_total_s"] == pytest.approx(72.0)
    assert s["bd_eval_fraction"] == pytest.approx(12.0 / 72.0, abs=1e-4)
    assert s["n_rounds_timed"] == 3 and s["n_bd_evals"] == 3


def test_timing_summary_phase_means_skip_undefined():
    m = collect(_log(feature="n/a"))
    ph = m["timing_summary"]["bd_phase_mean_s"]
    assert ph["asr"] == pytest.approx(2.0)          # (1+2+3)/3
    assert ph["forgetting"] == pytest.approx(2.0)
    assert ph["feature"] is None, "未启用的阶段不能被算成 0 拉低均值"
    assert ph["drift"] is None


def test_timing_summary_is_none_not_zero_without_timing_lines():
    """老日志（没有 [Timing] 行）不能得到 0.0 —— 那会被读成「评估不花时间」。"""
    m = collect(_log(timing=False))
    s = m["timing_summary"]
    assert m["timing_rounds"] == []
    assert s["bd_eval_total_s"] is None
    assert s["bd_eval_fraction"] is None
    assert s["wall_total_s"] is None
    # 但训练侧仍然读得出来
    assert s["round_time_total_s"] == pytest.approx(60.0)


def test_empty_log_does_not_crash():
    m = collect("")
    assert m["timing_rounds"] == []
    assert m["timing_summary"]["round_time_total_s"] is None
    assert m["timing_summary"]["bd_eval_fraction"] is None
