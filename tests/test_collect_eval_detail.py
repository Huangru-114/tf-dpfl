"""
tests/test_collect_eval_detail.py  —  S9：collect_metrics schema 7（评估细节 + dumps manifest）

日志行一律用**生产代码自己的打印函数**造（format_kv / fmt_list / eval_detail / dumps.dump_line），
所以这里测的是「打印 ↔ 解析」同源；另测：
  · 老日志（没有这些行）→ 新字段全为 null / 空，不是 0；
  · sha 长得像数字（如 123456e78901）时照样原样取回（同 [Checksum] 的坑）；
  · collect_metrics（纯标准库，登录节点也能跑）里抄的列名与 eval_detail 的一致。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import collect_metrics as CM                                    # noqa: E402
from attack import eval_detail as ED                            # noqa: E402
from utils import dumps as D                                    # noqa: E402
from utils.kvline import fmt_list, format_kv                    # noqa: E402

T, K = 0, 4


def _rec(cid, edge, mal, hits, n):
    y = np.array([1] * n)
    pred = np.array([0] * hits + [1] * (n - hits))
    probs = np.full((n, K), 0.1, np.float32)
    probs[np.arange(n), pred] = 0.7
    return ED.client_record(client_id=cid, edge_id=edge, malicious=mal, target=T, n_classes=K,
                            trig_probs=probs, trig_pred=pred, y_probe=y,
                            clean_pred=np.array([1, 0]), y_clean=np.array([1, 2]))


def _bd_line(r):
    return (f"[Backdoor] Round {r} | GM_ASR=0.500 | EM_ASR=0.400 | local_benign=0.300 "
            f"(same_edge=0.200, diff_edge=n/a) | local_malicious=1.000")


def _log(rounds=(1, 2), sha="123456e78901"):
    recs = [_rec(0, 0, True, 4, 4), _rec(1, 0, False, 1, 4), _rec(2, 1, False, 3, 4)]
    lines = []
    for r in rounds:
        lines.append(_bd_line(r))
        pooled = ED.summarize(recs, T, K)
        fields = {k: v for k, v in pooled.items() if k not in ("n_benign", "n_malicious")}
        fields["cls_asr"] = fmt_list(pooled["cls_asr"])
        lines.append(format_kv("[EvalDetail]", fields, round_idx=r))
        for eid in (0, 1):
            er = [x for x in recs if x["edge_id"] == eid]
            s = ED.summarize(er, T, K)
            lines.append(format_kv("[EvalDetailEdge]", {k: s[k] for k in ED.EDGE_FIELDS},
                                   round_idx=r, edge_id=eid))
            lines.append(format_kv("[ClientEval]", {k: fmt_list(v) for k, v
                                                    in ED.client_columns(er).items()},
                                   round_idx=r, edge_id=eid))
        info = {"path": f"G8__a__s42.1/logits_r{r:03d}.npz", "bytes": 1000, "sha": sha,
                "write_s": 0.1}
        lines.append(D.dump_line(r, "logits", info))
    snap = {"path": "G8__a__s42.1/snapshot_r002.npz", "bytes": 5000, "sha": "00ab12cd34ef",
            "write_s": 1.0}
    lines.append(D.dump_line(2, "snapshot", snap))
    lines.append(format_kv("[Dump]", {"kind": "logits", "path": "x.npz", "error": "OSError: full"},
                           round_idx=2))
    return "\n".join(lines) + "\n", recs


def test_rounds_get_the_pooled_distribution_columns():
    text, recs = _log()
    m = CM.collect(text)
    assert m["schema_version"] == 7
    want = ED.summarize(recs, T, K)
    row = m["rounds"][0]
    assert row["round"] == 1
    assert row["benign_asr_p50"] == round(want["benign_asr_p50"], 4)
    assert row["malicious_clean_acc"] == round(want["malicious_clean_acc"], 4)
    assert row["cls_asr"] == [None] + [round(v, 4) if v is not None else None
                                       for v in want["cls_asr"][1:]]
    assert m["final"]["margin_p50"] == row["margin_p50"]


def test_per_edge_detail_is_compact_rows():
    m = CM.collect(_log()[0])
    assert m["per_edge_detail_columns"] == ["edge_id", "margin_p50", "benign_asr_p90",
                                            "yt_clean_benign"]
    rows = m["per_edge_detail_rounds"][2]
    assert [r[0] for r in rows] == [0, 1]
    assert rows[0][2] == round(ED.summarize([_rec(1, 0, False, 1, 4)], T, K)["benign_asr_p90"], 4)


def test_client_final_is_columnar_and_from_the_last_round():
    m = CM.collect(_log()[0])
    cf = m["client_final"]
    assert cf["round"] == 2
    assert cf["client_id"] == [0, 1, 2] and cf["edge_id"] == [0, 0, 1]
    assert cf["malicious"] == [True, False, False]
    assert cf["asr"] == [1.0, 0.25, 0.75]
    assert cf["n"] == [4, 4, 4]


def test_dumps_manifest_keeps_hex_sha_and_errors():
    m = CM.collect(_log()[0])
    d = m["dumps"]
    assert d["logits"] == {"count": 2, "bytes": 2000, "dir": "G8__a__s42.1"}
    assert d["snapshots"] == [{"round": 2, "path": "G8__a__s42.1/snapshot_r002.npz",
                               "bytes": 5000, "sha": "00ab12cd34ef"}]
    assert d["errors"] == [{"round": 2, "kind": "logits", "error": "OSError: full"}]
    # 长得像数字的 sha 也原样取回
    line = next(ln for ln in _log()[0].splitlines() if "logits_r001" in ln)
    assert CM.RE_DUMP_SHA.search(line).group(1) == "123456e78901"


def test_old_logs_give_null_not_zero():
    m = CM.collect(_bd_line(1) + "\n")
    row = m["rounds"][0]
    for k in ("benign_asr_p50", "margin_p50", "yt_clean_benign", "malicious_clean_acc",
              "flip_other", "cls_asr"):
        assert row[k] is None, k
    assert m["per_edge_detail_rounds"] == {}
    assert m["client_final"] is None
    assert m["dumps"] == {"logits": {"count": 0, "bytes": 0, "dir": None},
                          "snapshots": [], "errors": []}


def test_copied_column_names_match_the_producer():
    """collect_metrics 要在登录节点用纯标准库跑，不能 import eval_detail（numpy）→ 抄了一份。"""
    assert CM.EDGE_DETAIL_COLUMNS[1:] == ED.EDGE_FIELDS
    assert CM.CLIENT_FINAL_COLUMNS == ED.CLIENT_COLUMNS


def test_metrics_growth_stays_inside_the_budget():
    """70 个评估点 × 4 edge × 25 端的规模：新字段合计 ≤ 70 KB（.git 已 360 MB，S9 设计复核）。"""
    import json
    recs = [_rec(i, i // 25, i < 10, 3, 110) for i in range(100)]
    lines = []
    for r in range(1, 71):
        lines.append(_bd_line(r))
        pooled = ED.summarize(recs, T, K)
        f = {k: v for k, v in pooled.items() if k not in ("n_benign", "n_malicious")}
        f["cls_asr"] = fmt_list(pooled["cls_asr"])
        lines.append(format_kv("[EvalDetail]", f, round_idx=r))
        for e in range(4):
            er = [x for x in recs if x["edge_id"] == e]
            s = ED.summarize(er, T, K)
            lines.append(format_kv("[EvalDetailEdge]", {k: s[k] for k in ED.EDGE_FIELDS},
                                   round_idx=r, edge_id=e))
            lines.append(format_kv("[ClientEval]", {k: fmt_list(v) for k, v in
                                                    ED.client_columns(er).items()},
                                   round_idx=r, edge_id=e))
        lines.append(D.dump_line(r, "logits", {"path": f"G8__a__s42.1/logits_r{r:03d}.npz",
                                               "bytes": 600000, "sha": "0" * 12, "write_s": 0.1}))
    with_new = len(json.dumps(CM.collect("\n".join(lines)), indent=2, ensure_ascii=False))
    old = "\n".join(_bd_line(r) for r in range(1, 71))
    without = len(json.dumps(CM.collect(old), indent=2, ensure_ascii=False))
    assert with_new - without < 70_000, with_new - without
