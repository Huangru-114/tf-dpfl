"""
tests/test_eval_detail.py  —  S9 / D-072：评估细节的解析值测试（纯 numpy，本地秒级）

手工构造 softmax 概率与标签，正确值可以直接写出来：
  · margin = log p_t − max_{k≠t} log p_k（与 logit 差相等）；
  · 客户端 ASR 用整数 argmax 计数，与 asr_counts 的过滤口径同式（逐位相同）；
  · y_t 偏置、按类 ASR、非目标翻转率、分位数、空组 → None。
另测 kvline 的列表编码（None 写 na，不写含 "/" 的 n/a）。
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from attack import eval_detail as ED
from attack.asr_counting import asr_counts, rates_from_counts
from utils.kvline import fmt_list, format_kv, parse_kv, parse_list

T, K = 0, 4          # 目标类 0，四个类
FEDAVG = Path(__file__).resolve().parent.parent / "fedavg"


def _softmax_rows(logits):
    z = np.asarray(logits, np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


# ── margin ──────────────────────────────────────────────────────────────────
def test_margin_equals_logit_difference():
    logits = [[3.0, 1.0, 0.0, -1.0],     # t 最大：3 − 1 = 2
              [0.0, 2.5, 1.0, 0.0],      # 最大的非 t 是 2.5：0 − 2.5 = −2.5
              [1.0, 1.0, 0.0, 0.0]]      # 平局：0
    m = ED.margins(_softmax_rows(logits), T)
    np.testing.assert_allclose(m, [2.0, -2.5, 0.0], atol=1e-6)


def test_margin_clips_underflow_instead_of_inf():
    p = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)          # log 0 会是 −inf
    m = ED.margins(p, T)
    assert np.isfinite(m).all()
    assert m[0] == pytest.approx(-np.log(ED.TINY))            # ≈ 87.34


# ── 客户端记录 ───────────────────────────────────────────────────────────────
def _probe():
    #   y     pred  说明
    #   1     0     命中
    #   2     0     命中
    #   3     2     翻到非目标、非真实类
    #   1     1     没翻
    #   0     0     目标类：不计入过滤口径
    y = np.array([1, 2, 3, 1, 0])
    pred = np.array([0, 0, 2, 1, 0])
    logits = np.full((5, K), -1.0)
    logits[np.arange(5), pred] = 2.0
    return _softmax_rows(logits), pred, y


def test_client_asr_is_the_integer_count_and_matches_asr_counts_bitwise():
    probs, pred, y = _probe()
    r = ED.client_record(client_id=7, edge_id=1, malicious=False, target=T, n_classes=K,
                         trig_probs=probs, trig_pred=pred, y_probe=y)
    filt, _ = rates_from_counts(asr_counts(pred, y, T))
    assert r["asr"] == filt == 2 / 4
    assert r["n"] == 4
    assert r["flip_n"] == 1
    assert list(r["cls_hits"]) == [0, 1, 1, 0]
    assert list(r["cls_n"]) == [0, 2, 1, 1]
    np.testing.assert_allclose(sorted(r["margins"]), [-3.0, -3.0, 3.0, 3.0], atol=1e-5)
    assert r["margin_med"] == pytest.approx(0.0, abs=1e-5)


def test_clean_accuracy_and_target_bias():
    yc = np.array([0, 1, 1, 2, 3, 3])
    pc = np.array([0, 1, 0, 2, 0, 3])          # 非目标 5 张里 2 张被判成 0
    r = ED.client_record(client_id=1, edge_id=0, malicious=True, target=T, n_classes=K,
                         clean_pred=pc, y_clean=yc)
    assert r["acc"] == 4 / 6
    assert r["yt_clean"] == 2 / 5
    assert r["asr"] is None and r["margin_med"] is None          # 没有触发探针 → 无定义


def test_probe_with_only_target_class_is_undefined_not_zero():
    y = np.array([0, 0]); pred = np.array([0, 1])
    r = ED.client_record(client_id=1, edge_id=0, malicious=False, target=T, n_classes=K,
                         trig_probs=_softmax_rows(np.eye(K)[pred]), trig_pred=pred, y_probe=y)
    assert r["asr"] is None and r["n"] == 0


# ── 汇总 ────────────────────────────────────────────────────────────────────
def _rec(cid, mal, asr_hits, n, acc=None, yt=None, edge=0):
    y = np.array([1] * n + [0])                     # 末尾一张目标类（不计入）
    pred = np.array([0] * asr_hits + [1] * (n - asr_hits) + [0])
    logits = np.full((n + 1, K), -1.0)
    logits[np.arange(n + 1), pred] = 1.0
    r = ED.client_record(client_id=cid, edge_id=edge, malicious=mal, target=T, n_classes=K,
                         trig_probs=_softmax_rows(logits), trig_pred=pred, y_probe=y)
    r["acc"], r["yt_clean"] = acc, yt
    return r


def test_summary_quantiles_pooling_and_groups():
    recs = [_rec(0, False, 1, 4, yt=0.1), _rec(1, False, 3, 4, yt=0.3),
            _rec(2, False, 4, 4, yt=None), _rec(3, True, 4, 4, acc=0.2),
            _rec(4, True, 0, 4, acc=0.4)]
    s = ED.summarize(recs, T, K)
    ben = [0.25, 0.75, 1.0]
    assert s["benign_asr_p50"] == pytest.approx(np.percentile(ben, 50))
    assert s["benign_asr_p10"] == pytest.approx(np.percentile(ben, 10))
    assert s["benign_asr_p90"] == pytest.approx(np.percentile(ben, 90))
    assert s["benign_asr_gt50"] == pytest.approx(2 / 3)
    assert s["malicious_clean_acc"] == pytest.approx(0.3)
    assert s["yt_clean_benign"] == pytest.approx(0.2)            # None 不参与
    # 池化的 margin：良性端 12 张合格样本里 8 张命中（+2）、4 张没命中（−2）
    assert s["margin_p50"] == pytest.approx(2.0, abs=1e-5)
    assert s["cls_asr"] == [None, pytest.approx(8 / 12), None, None]
    assert s["flip_other"] == 0.0
    assert (s["n_benign"], s["n_malicious"]) == (3, 2)


def test_empty_groups_are_none_not_zero():
    s = ED.summarize([_rec(3, True, 4, 4, acc=0.9)], T, K)       # 只有恶意端
    for k in ("benign_asr_p10", "benign_asr_p50", "benign_asr_p90", "benign_asr_gt50",
              "yt_clean_benign", "margin_p50", "flip_other"):
        assert s[k] is None, k
    assert s["malicious_clean_acc"] == pytest.approx(0.9)
    s = ED.summarize([_rec(0, False, 1, 4)], T, K)
    assert s["malicious_clean_acc"] is None


def test_summary_values_are_python_floats_for_kvline():
    s = ED.summarize([_rec(0, False, 1, 4, yt=0.1), _rec(1, True, 2, 4, acc=0.5)], T, K)
    for k, v in s.items():
        if k == "cls_asr":
            assert all(x is None or type(x) is float for x in v)
        elif v is not None:
            assert type(v) in (float, int), (k, type(v))


# ── logits 打包 ─────────────────────────────────────────────────────────────
def test_pack_logits_keeps_per_sample_identity_and_argmax():
    probs, pred, y = _probe()
    yc = np.array([1, 0, 2]); pc = np.array([1, 0, 0])
    r = ED.client_record(client_id=9, edge_id=2, malicious=False, target=T, n_classes=K,
                         trig_probs=probs, trig_pred=pred, y_probe=y,
                         clean_pred=pc, y_clean=yc, clean_probs=_softmax_rows(np.eye(K)[pc]),
                         keep_probs=True)
    z = ED.pack_logits([r])
    assert z["trig_logp"].dtype == np.float16 and z["trig_logp"].shape == (5, K)
    assert list(z["trig_client"]) == [9] * 5 and list(z["clean_client"]) == [9] * 3
    assert list(z["trig_pred"]) == list(pred) and list(z["clean_pred"]) == list(pc)
    np.testing.assert_array_equal(np.argmax(z["trig_logp"], 1), pred)
    assert list(z["client_edge"]) == [2]


def test_records_do_not_keep_arrays_unless_asked():
    probs, pred, y = _probe()
    r = ED.client_record(client_id=1, edge_id=0, malicious=False, target=T, n_classes=K,
                         trig_probs=probs, trig_pred=pred, y_probe=y)
    assert "trig_logp" not in r and "clean_logp" not in r


# ── kvline 列表 ─────────────────────────────────────────────────────────────
def test_list_encoding_round_trips_none_as_na():
    vals = [0.5, None, 3, 0.125]
    s = fmt_list(vals)
    assert s == "0.5000/na/3/0.1250"
    line = format_kv("[ClientEval]", {"asr": s, "ids": fmt_list([4, 7])}, round_idx=3, edge_id=1)
    d = parse_kv(line, "[ClientEval]")
    assert parse_list(d["asr"]) == [0.5, None, 3, 0.125]
    assert parse_list(d["ids"]) == [4, 7]


def test_single_element_and_empty_lists():
    d = parse_kv(format_kv("[X]", {"a": fmt_list([5]), "b": fmt_list([None]),
                                   "c": fmt_list([])}), "[X]")
    assert parse_list(d["a"]) == [5]
    assert parse_list(d["b"]) == [None]
    assert parse_list(d["c"]) == []
    assert parse_list(None) is None


# ── 接线守卫（AST，本地可跑）──────────────────────────────────────────────────
def _func(path, name):
    tree = ast.parse((FEDAVG / path).read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(fn, name):
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


@pytest.mark.parametrize("name", ["compute_asr_four_way", "compute_asr_on_dataset"])
def test_each_probe_calls_the_trigger_exactly_once(name):
    """第二次调用触发器会让同一列后面所有探针的 ξ 随机数错位（并改动生成器的 BN 统计）。"""
    fn = _func("attack/backdoor_eval.py", name)
    assert len(_calls(fn, "trigger_fn")) == 1


@pytest.mark.parametrize("name", ["compute_asr_four_way", "compute_asr_on_dataset",
                                  "_acc_on_dataset"])
def test_detail_is_keyword_only_and_defaults_to_none(name):
    fn = _func("attack/backdoor_eval.py", name)
    kw = [a.arg for a in fn.args.kwonlyargs]
    assert kw == ["detail"]
    assert isinstance(fn.args.kw_defaults[0], ast.Constant) and fn.args.kw_defaults[0].value is None


def test_detail_lines_are_emitted_after_the_asr_timer_stops():
    """[Timing] asr / [TimingASR] main 要与改动前可比：细节与存盘不计入。"""
    src = (FEDAVG / "server/backdoor_server.py").read_text(encoding="utf-8")
    body = src[src.index("def _backdoor_eval"):]
    assert body.index("t_asr = time.perf_counter() - _t0") < body.index("self._emit_detail(")


# ── 纯度守卫 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("rel", ["attack/eval_detail.py", "utils/dumps.py"])
def test_instrumentation_modules_touch_no_rng_and_no_tf(rel):
    """仪表只读已有结果：不 import TF、不碰任何随机数（F-045 的确定性前提）。"""
    src = (FEDAVG / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m.split(".")[0] in ("tensorflow", "random") for m in mods), mods
    for bad in ("np.random", "default_rng", "random."):
        assert bad not in src, f"{rel} 里出现了 {bad}"
