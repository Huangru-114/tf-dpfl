"""
tests/test_eval_detail_tf.py  —  S9 / D-072 / D-073：评估细节与两个存盘开关的接线（需要 TF）

用 test_eval_integration 的小 HFL（2 edge × 3 FedRep 客户端，2 个 Bad-PFL 恶意端）跑一轮：
  · 细节用的是**主列那一次前向**：良性端客户端 ASR 的均值 == local_asr_benign_mean（逐位）；
  · 两个开关开 / 关各跑一轮，[Checksum] 与全部已有的评估行逐字相同（开关不改任何数）；
  · 快照 == 在线的 edge / 全局权重，且 compose_pm(快照) 的哈希 == 评估所用模型；
  · logits 文件的 argmax == 同一行的 trig_pred（fp16 并列除外，以 uint8 预测为准），恶意端标记与样本数对得上。
"""

import hashlib
import json
import re

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import test_eval_integration as TEI                          # noqa: E402
from server.backdoor_server import BackdoorCloudServer       # noqa: E402
from utils.kvline import collect_kv, parse_list              # noqa: E402
from utils.pm import compose_pm                              # noqa: E402

OLD_TAGS = ("[Checksum]", "[Backdoor]", "[ASR4]", "[Acc]", "[Stale]", "[StaleASR]", "[Cloud] GM=")


def _old_lines(out):
    keep = []
    for ln in out.splitlines():
        if ln.startswith(OLD_TAGS) or ln.strip().startswith(OLD_TAGS):
            keep.append(re.sub(r"time=[\d.]+s", "time=?", ln))
    return keep


def _hash(ws):
    return hashlib.sha256(b"".join(np.ascontiguousarray(w).tobytes() for w in ws)).hexdigest()


def test_detail_reuses_the_main_forward_pass(monkeypatch, capsys):
    cfg, cloud, clients, edges = TEI._setup(True)
    seen = {}
    orig = BackdoorCloudServer._emit_detail

    def spy(self, round_idx, metrics, keep):
        seen["recs"] = list(metrics.get("client_detail") or [])
        seen["benign_mean"] = metrics["local_asr_benign_mean"]
        seen["probe_order"] = metrics.get("probe_order")
        return orig(self, round_idx, metrics, keep)
    monkeypatch.setattr(BackdoorCloudServer, "_emit_detail", spy)
    cloud.run_round(1)
    out = capsys.readouterr().out

    ben = [r["asr"] for r in seen["recs"] if not r["malicious"] and r["asr"] is not None]
    assert ben and float(np.mean(ben)) == seen["benign_mean"]            # 逐位
    assert {r["client_id"] for r in seen["recs"]} == {c.client_id for c in clients}
    kinds = [k for k, _, _ in seen["probe_order"]]
    assert kinds[0] == "global" and kinds[1:3] == ["edge", "edge"] and kinds[3:] == ["client"] * 6

    pooled = [d for d in collect_kv(out.splitlines(), "[EvalDetail]") if "edge_id" not in d]
    assert len(pooled) == 1 and pooled[0]["round"] == 1
    assert len(parse_list(pooled[0]["cls_asr"])) == TEI.NCLS
    edge_rows = collect_kv(out.splitlines(), "[EvalDetailEdge]")
    assert sorted(d["edge_id"] for d in edge_rows) == [0, 1]
    ce = collect_kv(out.splitlines(), "[ClientEval]")
    ids = sorted(i for d in ce for i in parse_list(d["ids"]))
    assert ids == sorted(c.client_id for c in clients)
    assert cloud._last_bd_metrics is None or "client_detail" not in cloud._last_bd_metrics


def test_switches_change_no_existing_number_and_dump_what_they_claim(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("TFDPFL_DUMPDIR", str(tmp_path))
    _, cloud_a, _, _ = TEI._setup(True)
    cloud_a.run_round(1)
    out_a = capsys.readouterr().out

    _, cloud_b, clients, edges = TEI._setup(
        True, evaluation={"dump_logits_every": 1, "snapshot_rounds": "1"})
    captured = {}
    orig_pm = cloud_b.main_pm

    def rec_pm(c, slot="victim"):
        m = orig_pm(c, slot)
        if slot == "victim":
            captured[int(c.client_id)] = _hash(m.get_weights())
        return m
    monkeypatch.setattr(cloud_b, "main_pm", rec_pm)
    cloud_b.run_round(1)
    out_b = capsys.readouterr().out

    assert _old_lines(out_a) == _old_lines(out_b)                      # 开关不改任何已有的数
    dumps = collect_kv(out_b.splitlines(), "[Dump]")
    assert sorted(d["kind"] for d in dumps) == ["logits", "snapshot"]
    assert all("error" not in d for d in dumps)
    assert not collect_kv(out_a.splitlines(), "[Dump]")

    root = tmp_path
    snap = np.load(root / next(d["path"] for d in dumps if d["kind"] == "snapshot"))
    meta = json.loads(str(snap["meta_json"]))
    assert meta["round"] == 1 and meta["evaluated"] and meta["edge_matches_eval"] is True
    assert meta["resumable"] is False and meta["has_generator"]
    g = cloud_b.global_model.get_weights()
    for i, w in enumerate(g):
        np.testing.assert_array_equal(snap[f"global_{i:03d}"], w)
    for e in edges:
        ew = [snap[f"edge{e.edge_id}_{i:03d}"] for i in range(len(e.model.get_weights()))]
        for a, b in zip(ew, e.model.get_weights()):
            np.testing.assert_array_equal(a, b)
    for c in clients:                                                   # 快照复原 fresh-PM
        idx = [int(i) for i in snap[f"client{c.client_id}_idx"]]
        e = cloud_b.edge_of(c)
        ew = [snap[f"edge{e.edge_id}_{i:03d}"] for i in range(len(e.model.get_weights()))]
        val = [snap[f"client{c.client_id}_{j:03d}"] for j in range(len(idx))]
        pm = compose_pm(ew, val, idx) if idx else ew
        assert _hash(pm) == captured[int(c.client_id)], c.client_id

    lg = np.load(root / next(d["path"] for d in dumps if d["kind"] == "logits"))
    # fp16 对数概率可能出现并列（CPU 替身里 893 张中有 1 张：最大的两类都是 −2.09375）→
    # 以同文件的 uint8 预测为准；argmax 只要求与它一致或恰好并列
    lp = lg["trig_logp"].astype(np.float32)
    tie = lp[np.arange(len(lp)), lg["trig_pred"]] == lp.max(axis=1)
    assert np.all((np.argmax(lp, 1) == lg["trig_pred"]) | tie)
    assert set(lg["client_ids"]) == {c.client_id for c in clients}
    mal = dict(zip(lg["client_ids"].tolist(), lg["client_malicious"].tolist()))
    assert {k for k, v in mal.items() if v} == TEI.MAL
    assert len(lg["clean_y"]) == 12 * len(clients)                      # 每端留出 12 张
