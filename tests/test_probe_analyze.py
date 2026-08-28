"""
tests/test_probe_analyze.py  —  Δ → 指标行的装配不变量（纯 numpy，本地秒级）。

守两件容易静默出错的事：
  1. **scope 隔离**（陷阱 #9）：backbone 与 head 绝不进同一个展平向量。
  2. 判据函数（writer.verdicts）在「仪表坏了」和「无信号」两种情形下给出**不同**结论
     —— 这两者的原始外观是一样的，判据不区分开就等于没有判据。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))

from probe.analyze import analyze_probe, ROW_COLUMNS          # noqa: E402
from probe.writer import (hardware_floor, write_rows,          # noqa: E402
                          write_summary, verdicts)


# 玩具模型：张量 0/1 = conv 层（backbone），张量 2/3 = head 的 kernel+bias
REF = [np.zeros((2, 2)), np.zeros(2), np.zeros((2, 3)), np.zeros(3)]
GROUPS = {"conv": [0, 1], "head": [2, 3]}
BASE_IDX, HEAD_IDX = [0, 1], [2, 3]


def _w(vals):
    """按 REF 的形状把一串标量摊成权重列表（前 4+2=6 个给 backbone，后 6+3 给 head）。"""
    a = np.asarray(vals, dtype=float)
    return [a[:4].reshape(2, 2), a[4:6], a[6:12].reshape(2, 3), a[12:15]]


def _result(clean, poison, clean_b, *, malicious=True, cid=0, rnd=5):
    d = {}
    for space in ("upload", "personal"):
        c, p, b = _w(clean), _w(poison), _w(clean_b)
        d[space] = {
            "clean": c, "poison": p,
            "bd": [x - y for x, y in zip(p, c)],
            "stochastic": [x - y for x, y in zip(c, b)],
        }
    return {"round": rnd, "client_id": cid, "was_malicious": malicious,
            "seed_a": 1, "seed_b": 2, "loss": {}, "deltas": d}


def test_rows_cover_every_layer_plus_an_all_row_per_scope():
    out = analyze_probe(_result(np.ones(15), np.ones(15) * 2, np.ones(15)),
                        REF, GROUPS, BASE_IDX, HEAD_IDX, checkpoint="T3")
    got = {(r["weight_space"], r["scope"], r["layer"]) for r in out["rows"]}
    assert ("upload", "backbone", "conv") in got
    assert ("upload", "backbone", "__all__") in got
    assert ("upload", "head", "head") in got
    assert ("personal", "head", "__all__") in got


def test_scope_isolation_head_tensors_never_enter_the_backbone_scope():
    """陷阱 #9：head 层不该出现在 backbone scope 的任何一行里。"""
    out = analyze_probe(_result(np.ones(15), np.ones(15) * 2, np.ones(15)),
                        REF, GROUPS, BASE_IDX, HEAD_IDX)
    backbone_layers = {r["layer"] for r in out["rows"] if r["scope"] == "backbone"}
    assert backbone_layers == {"conv", "__all__"}


def test_backbone_metrics_are_immune_to_a_wildly_diverging_head():
    """
    head 位移放大 1e6 倍，backbone 的指标必须逐位不变 —— 这正是分 scope 的理由。
    """
    clean = np.ones(15)
    poison = np.concatenate([np.ones(6) * 2, np.ones(9) * 2])
    poison_huge = np.concatenate([np.ones(6) * 2, np.ones(9) * 2e6])
    a = analyze_probe(_result(clean, poison, clean), REF, GROUPS, BASE_IDX, HEAD_IDX)
    b = analyze_probe(_result(clean, poison_huge, clean), REF, GROUPS, BASE_IDX, HEAD_IDX)
    key = lambda o: [(r["layer"], r["bd_update_norm"], r["cosine_bd_clean"])
                     for r in o["rows"] if r["scope"] == "backbone"
                     and r["weight_space"] == "upload"]
    assert key(a) == key(b)


def test_bd_delta_is_poison_minus_clean():
    """Δθ_BD 的定义本身：clean=1、poison=3 → bd 恒为 2，backbone 6 个坐标 → ‖·‖=2√6。"""
    out = analyze_probe(_result(np.ones(15), np.ones(15) * 3, np.ones(15)),
                        REF, GROUPS, BASE_IDX, HEAD_IDX)
    row = next(r for r in out["rows"] if r["scope"] == "backbone"
               and r["layer"] == "__all__" and r["weight_space"] == "upload")
    assert row["bd_update_norm"] == pytest.approx(2.0 * np.sqrt(6))
    assert row["n_params"] == 6


def test_identical_clean_twins_give_zero_stochastic():
    out = analyze_probe(_result(np.ones(15), np.ones(15) * 2, np.ones(15)),
                        REF, GROUPS, BASE_IDX, HEAD_IDX)
    assert all(r["stochastic_norm"] == pytest.approx(0.0) for r in out["rows"])


def test_summary_carries_the_stage1_ratio():
    out = analyze_probe(_result(np.ones(15), np.ones(15) * 3, np.ones(15) * 0.5),
                        REF, GROUPS, BASE_IDX, HEAD_IDX)
    s = next(s for s in out["summaries"]
             if s["scope"] == "backbone" and s["weight_space"] == "upload")
    # bd = 2 (每坐标), stochastic = 0.5 → 比值 4
    assert s["bd_over_stochastic"] == pytest.approx(4.0)


def test_summary_ratio_is_nan_when_stochastic_is_zero():
    """随机性对照恰为 0 时比值无定义。返回 inf/0 会让判据静默通过。"""
    out = analyze_probe(_result(np.ones(15), np.ones(15) * 3, np.ones(15)),
                        REF, GROUPS, BASE_IDX, HEAD_IDX)
    s = next(s for s in out["summaries"] if s["scope"] == "backbone")
    assert np.isnan(s["bd_over_stochastic"])


def test_row_columns_are_all_present():
    out = analyze_probe(_result(np.ones(15), np.ones(15) * 2, np.ones(15)),
                        REF, GROUPS, BASE_IDX, HEAD_IDX)
    for r in out["rows"]:
        assert set(ROW_COLUMNS) <= set(r)


# ── 判据 ────────────────────────────────────────────────────────────────────
def _summ(role, bd, stoch):
    return {"role": role, "bd_norm": bd, "stochastic_norm": stoch,
            "bd_over_stochastic": (bd / stoch) if stoch else float("nan")}


def test_verdict_broken_instrument_and_no_signal_are_distinguishable():
    """
    这是本文件最重要的一条：两种情形的**原始外观相同**（恶意端比值都上不去），
    但含义完全不同 —— 一个是 bug，一个是科学结论。判据必须分开报。
    """
    broken = verdicts([_summ("benign", 5.0, 5.0), _summ("malicious", 6.0, 5.0)])
    assert broken["instrument_ok"]["pass"] is False

    no_signal = verdicts([_summ("benign", 0.0, 5.0), _summ("malicious", 6.0, 5.0)])
    assert no_signal["instrument_ok"]["pass"] is True      # 仪表好的
    assert no_signal["signal"]["pass"] is False            # 但没信号 → 是结论


def test_verdict_passes_when_signal_clears_threshold():
    v = verdicts([_summ("benign", 0.0, 1.0), _summ("malicious", 10.0, 1.0)],
                 ratio_threshold=3.0)
    assert v["instrument_ok"]["pass"] and v["signal"]["pass"]


def test_verdict_signal_uses_the_worst_malicious_client_not_the_mean():
    """一个恶意端不过就不算过 —— 用均值会被另一个高比值的端掩盖过去。"""
    v = verdicts([_summ("benign", 0.0, 1.0),
                  _summ("malicious", 100.0, 1.0), _summ("malicious", 1.0, 1.0)],
                 ratio_threshold=3.0)
    assert v["signal"]["pass"] is False


# ── 判据在两种 determinism 模式下的语义（陷阱 #11）─────────────────────────
def test_hardware_floor_is_the_benign_bd_norm():
    """
    良性端没有攻击 mixin，翻 is_malicious 是无操作 → 两条 twin 代码路径逐字节相同。
    所以它们的 ‖Δθ_BD‖ 里不含任何算法内容，只有"同一段计算跑两遍是否一致"。
    """
    assert hardware_floor([_summ("benign", 0.25, 9.0),
                           _summ("malicious", 99.0, 9.0)])["max"] == 0.25


def test_cpu_mode_asserts_a_zero_hardware_floor():
    v = verdicts([_summ("benign", 1e-3, 1.0), _summ("malicious", 10.0, 1.0)],
                 determinism="cpu")
    assert v["instrument_ok"]["asserted"] is True
    assert v["instrument_ok"]["pass"] is False


def test_gpu_measure_mode_does_not_assert_the_floor_it_only_reports_it():
    """
    GPU 上 clean twin 残差本来就非零（cuDNN 非确定性 kernel）。把它判成"仪表坏了"
    是把硬件性质误读成 bug —— 那个模式下地板不断言，只并进判据 2 的分母。
    """
    v = verdicts([_summ("benign", 1e-3, 1.0), _summ("malicious", 10.0, 1.0)],
                 determinism="gpu-measure")
    assert v["instrument_ok"]["asserted"] is False
    assert v["instrument_ok"]["pass"] is True
    assert v["instrument_ok"]["hardware_floor"]["max"] == 1e-3
    assert v["determinism"] == "gpu-measure"


def test_signal_is_divided_by_the_larger_of_the_two_floors():
    """
    硬件地板 5.0 > SGD 地板 1.0 时，只跟 SGD 噪声比会把信号高估 5 倍。
    signal=10 → 必须是 10/5=2（不过阈值 3），而不是 10/1=10。
    """
    v = verdicts([_summ("benign", 5.0, 1.0), _summ("malicious", 10.0, 1.0)],
                 ratio_threshold=3.0, determinism="gpu-measure")
    assert v["signal"]["signal_over_floor"]["min"] == pytest.approx(2.0)
    assert v["signal"]["pass"] is False


def test_sgd_floor_still_wins_when_it_is_the_larger_one():
    v = verdicts([_summ("benign", 0.0, 1.0), _summ("malicious", 10.0, 2.0)],
                 ratio_threshold=3.0)
    assert v["signal"]["signal_over_floor"]["min"] == pytest.approx(5.0)
    assert v["signal"]["pass"] is True


def test_cpu_mode_reduces_to_the_plain_stochastic_ratio():
    """hardware_floor=0 时新公式必须退化成旧行为（除以 SGD 地板）。"""
    v = verdicts([_summ("benign", 0.0, 4.0), _summ("malicious", 12.0, 4.0)])
    assert v["signal"]["signal_over_floor"]["min"] == pytest.approx(3.0)


# ── 落盘 ────────────────────────────────────────────────────────────────────
def test_write_rows_and_summary_roundtrip(tmp_path):
    out = analyze_probe(_result(np.ones(15), np.ones(15) * 2, np.ones(15) * 0.5),
                        REF, GROUPS, BASE_IDX, HEAD_IDX, checkpoint="T1")
    csv_path = write_rows(tmp_path / "rows.csv", out["rows"])
    text = csv_path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(ROW_COLUMNS)
    assert len(text.strip().splitlines()) == len(out["rows"]) + 1

    js = write_summary(tmp_path / "s.json", {"summaries": out["summaries"]})
    import json
    loaded = json.loads(js.read_text(encoding="utf-8"))
    assert len(loaded["summaries"]) == len(out["summaries"])


def test_write_summary_turns_nan_into_null_not_a_crash(tmp_path):
    """json.dumps 默认把 nan 写成裸 NaN —— 那不是合法 JSON，pandas/jq 都读不了。"""
    import json
    js = write_summary(tmp_path / "s.json", {"x": float("nan"), "y": [1.0, float("inf")]})
    loaded = json.loads(js.read_text(encoding="utf-8"))
    assert loaded["x"] is None and loaded["y"] == [1.0, None]
