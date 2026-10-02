"""
tests/test_s6_tf.py  —  S6a：G1 的便宜记录的 TF 侧（真跑 cloud / edge / client；D-085）

小 HFL（2 edge × 3 FedRep 客户端，2 个 Bad-PFL 恶意端，official BN、固定攻击者、fresh-PM，
R > 1 + 交错调度 + 统一网格），夹具同 test_eval_grid_tf。

硬要求（同 S5 / D-055）：**记录不得改变训练**
  · 三个记录开关（update_geometry / post_agg_eval / frozen_trigger）全开 vs 全关：逐轮 [Checksum] 相同，
    全量点上已有的评估行逐字相同；
  · 一次云聚合后评估点 / 一次冻结触发器评估前后，训练可见的状态清单逐位相同；
  · 反向锚点：去掉 random 围栏，开关就改变训练（冻结槽第一次创建消耗 Python random，F-078）。

记录的内容：
  · [UpdateGeo] 的范数 == 同一批上传独立算出的 ‖Δ‖；恶意标记对得上；每个 (edge, edge 轮) 一行；
  · 草图落盘（每云轮一个 npz，条数 = 该轮的上传数）；
  · 云聚合后评估点只在第 2 个云轮起、有效轮号 = (g−1)·R、不进 history / _eval_seq / 停止判据；
  · 冻结列：生成器没动时，冻结的 ξ + δ 与主列的触发器逐位相同；生成器被训动之后冻结值不变、
    主列的值变了（这是它存在的理由）；退出后真生成器逐位复原。
"""

import random
import re

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import test_eval_integration as TEI                          # noqa: E402
from test_eval_grid_tf import _prod_seeding, _state, _after_one_edge_round, OLD_TAGS  # noqa: E402
from server import eval_grid as EG                           # noqa: E402
from server import update_geometry as UG                     # noqa: E402
from utils.kvline import collect_kv, parse_list              # noqa: E402

S6 = {"update_geometry": True, "post_agg_eval": True, "frozen_trigger": True,
      "update_sketch_dim": 64}
FED = {"edge_schedule": "interleaved"}
R, G, N = 4, 2, 3


def _run(capsys, monkeypatch, s6, tmp_path=None):
    ev = {"eval_interval": EG.full_interval(G, R), "eval_grid": G, **(s6 or {})}
    random.seed(42)
    if tmp_path is not None:
        monkeypatch.setenv("TFDPFL_DUMPDIR", str(tmp_path))
    cfg, cloud, clients, edges = TEI._setup(True, evaluation=ev,
                                            federation=dict(FED, edge_rounds=R, n_rounds=N))
    _prod_seeding(monkeypatch)
    cfg["backdoor"]["eval_interval"] = ev["eval_interval"]
    cloud.bd_eval_interval = ev["eval_interval"]
    capsys.readouterr()
    cloud.run()
    return capsys.readouterr().out, cloud, clients, edges


def _checksums(out):
    return [ln for ln in out.splitlines() if ln.startswith("[Checksum]")]


def _old_by_round(out):
    chunks, cur = {}, None
    for ln in out.splitlines():
        m = re.match(r"\s*\[Round\s+(\d+)\]", ln)
        if m:
            cur = int(m.group(1))
            chunks[cur] = []
            continue
        if cur is not None and (ln.startswith(OLD_TAGS) or ln.strip().startswith(OLD_TAGS)):
            chunks[cur].append(re.sub(r"time=[\d.]+s", "time=?", ln))
    return chunks


# ══════════════════════════════════════════════════════════════════════════
# 记录不改变训练
# ══════════════════════════════════════════════════════════════════════════
def test_switches_on_off_same_training_and_same_numbers(capsys, monkeypatch, tmp_path):
    off, cloud_off, *_ = _run(capsys, monkeypatch, None)
    on, cloud_on, *_ = _run(capsys, monkeypatch, S6, tmp_path)
    assert _checksums(off) and _checksums(off) == _checksums(on)            # 训练逐轮逐位相同
    assert _old_by_round(off) == _old_by_round(on)                          # 已有评估行逐字相同
    # 轻评估点上已有的数也相同（只多了列）
    lo = {(d["round"], d["effective_round"]): d for d in collect_kv(off.splitlines(), "[Light]")}
    ln = {(d["round"], d["effective_round"]): d for d in collect_kv(on.splitlines(), "[Light]")}
    assert lo and set(lo) == set(ln)
    for k in lo:
        for f in ("pm_acc", "em_acc", "edge_asr", "local_benign_asr", "same_edge_asr",
                  "diff_edge_asr", "local_malicious_asr"):
            assert lo[k][f] == ln[k][f], (k, f)
    # 计数器 / 历史也不动（post-agg 与冻结列不是评估点）
    assert cloud_off._eval_seq == cloud_on._eval_seq
    skip = ("round_time", "avg_client_time")                                # 墙钟
    assert ({k: v for k, v in cloud_off.history.items() if k not in skip}
            == {k: v for k, v in cloud_on.history.items() if k not in skip})


def test_without_the_fence_the_switches_change_training(capsys, monkeypatch, tmp_path):
    """反向锚点：去掉 random 复原 → 冻结槽第一次创建推进 Python random，legacy 管线的洗牌随之变。"""
    off, *_ = _run(capsys, monkeypatch, None)
    monkeypatch.setattr(random, "setstate", lambda st: None)
    on, *_ = _run(capsys, monkeypatch, S6, tmp_path)
    assert _checksums(off) != _checksums(on)


def test_post_agg_and_frozen_leave_every_training_state_untouched(monkeypatch):
    cfg, cloud, clients, edges = TEI._setup(
        True, evaluation={"eval_grid": 1, **S6},
        federation=dict(FED, edge_rounds=2, n_rounds=2))
    _prod_seeding(monkeypatch)
    cloud._on_round_broadcast(1)                       # 冻结（第 1 云轮不做 post-agg）
    for e in edges:
        e.run_edge_round(1, 1)
    before = _state(cloud, clients, edges)
    cloud._post_agg_eval(2)
    cloud._frozen_eval(1, 1, 1, "light")
    # 生成器的 moving 统计量在 eval_delta(training=True) 下会变，_state 不含它们；
    # 其余全部（含生成器权重与 Adam）逐位不变
    assert _state(cloud, clients, edges) == before


# ══════════════════════════════════════════════════════════════════════════
# 记录的内容
# ══════════════════════════════════════════════════════════════════════════
def test_update_geometry_matches_an_independent_recompute(capsys, monkeypatch, tmp_path):
    seen = []
    orig = UG.UpdateGeometry.observe

    def spy(self, edge_id, edge_round, client_updates, edge_weights, base_idx):
        seen.append((int(edge_id), int(edge_round),
                     [(int(u.client_id), np.linalg.norm(
                         UG.body_delta(u[0], edge_weights, base_idx).astype(np.float64)))
                      for u in client_updates]))
        return orig(self, edge_id, edge_round, client_updates, edge_weights, base_idx)
    monkeypatch.setattr(UG.UpdateGeometry, "observe", spy)
    out, cloud, clients, edges = _run(capsys, monkeypatch, S6, tmp_path)
    rows = collect_kv(out.splitlines(), "[UpdateGeo]")
    assert len(rows) == len(seen) == N * R * 2                              # 每 (云轮, edge 轮, edge) 一行
    by_key = {(d["round"], d["edge_id"], d["edge_round"]): d for d in rows}
    assert len(by_key) == len(rows)
    for d in rows:
        cids = parse_list(d["cid"])
        mal = parse_list(d["mal"])
        norm = parse_list(d["norm"])
        assert len(cids) == len(mal) == len(norm) == len(parse_list(d["cos_edge"]))
        assert [bool(m) for m in mal] == [c in TEI.MAL for c in cids]
        for v in parse_list(d["cos_global"]) + parse_list(d["cos_edge"]):
            assert v is None or -1.0 - 1e-6 <= v <= 1.0 + 1e-6
    # 打印的范数 == 独立重算（逐 edge 轮顺序一致：seen 按调用顺序，rows 按 flush 顺序）
    flat_seen = {}
    for eid, er, ups in seen:
        flat_seen.setdefault((eid, er), []).append([n for _, n in ups])
    got = {}
    for d in rows:
        got.setdefault((d["edge_id"], d["edge_round"]), []).append(parse_list(d["norm"]))
    assert set(flat_seen) == set(got)
    for k in flat_seen:
        for a, b in zip(flat_seen[k], got[k]):
            np.testing.assert_allclose(a, b, atol=5e-5)


def test_sketches_are_dumped_once_per_cloud_round(capsys, monkeypatch, tmp_path):
    out, cloud, *_ = _run(capsys, monkeypatch, S6, tmp_path)
    dumps = [d for d in collect_kv(out.splitlines(), "[Dump]") if d.get("kind") == "sketch"]
    assert [d["round"] for d in dumps] == list(range(1, N + 1))
    rows = collect_kv(out.splitlines(), "[UpdateGeo]")
    for d in dumps:
        z = np.load(tmp_path / d["path"])
        n_rows = sum(len(parse_list(r["cid"])) for r in rows if r["round"] == d["round"])
        assert z["sketch"].shape == (n_rows, 64) and z["sketch"].dtype == np.float16
        assert len(z["client_id"]) == len(z["edge_id"]) == len(z["edge_round"]) == n_rows
        assert {int(c) for c in z["client_id"][z["malicious"]]} <= TEI.MAL


def test_post_agg_points_only_from_round_two_and_stay_out_of_history(capsys, monkeypatch, tmp_path):
    out, cloud, *_ = _run(capsys, monkeypatch, S6, tmp_path)
    pa = collect_kv(out.splitlines(), "[PostAgg]")
    assert [(d["round"], d["edge_round"], d["effective_round"]) for d in pa] == [
        (g, 0, (g - 1) * R) for g in range(2, N + 1)]
    assert all(d["pm_acc"] is not None and d["local_benign_asr"] is not None for d in pa)
    assert len(collect_kv(out.splitlines(), "[PostAggEdge]")) == 2 * len(pa)
    assert len(cloud.history["round"]) == N                                 # 一轮一个，post-agg 不在里面


def test_frozen_columns_cover_post_light_and_full_points(capsys, monkeypatch, tmp_path):
    out, *_ = _run(capsys, monkeypatch, S6, tmp_path)
    fz = collect_kv(out.splitlines(), "[FrozenASR]")
    phases = [d["phase"] for d in fz]
    assert set(phases) == {"post", "light", "full"}
    assert phases.count("post") == N - 1
    assert phases.count("full") == sum(EG.is_full_round(G, R, g) for g in range(1, N + 1))
    assert phases.count("light") == sum(len(EG.light_edge_rounds(G, R, g)) for g in range(1, N + 1))
    assert all(d["local_benign"] is not None for d in fz)
    assert len(collect_kv(out.splitlines(), "[FrozenASREdge]")) == 2 * len(fz)


def test_frozen_trigger_equals_main_trigger_until_the_generator_moves(monkeypatch):
    cfg, cloud, clients, edges = _after_one_edge_round(monkeypatch)
    cloud.config["evaluation"]["frozen_trigger"] = True
    cloud._frozen_on = True
    att = cloud.eval_attacker
    gen = att._atk_generator
    x = np.random.default_rng(1).normal(size=(8, TEI.IMG, TEI.IMG, 3)).astype(np.float32)
    y = np.arange(8) % TEI.NCLS

    # 冻结 = 此刻的状态；生成器没动 → 冻结 ξ + δ 与主列的触发器逐位相同
    # （主列用 fresh-PM 的 attacker，冻结用同一时刻拷下的权重）
    cloud.begin_pm_eval()
    cloud._freeze_trigger(1)
    main = cloud._attacker_trigger(1, 0, "fresh", rng=np.random.default_rng(5))(None, x, y)
    with cloud._frozen_state() as xm:
        fro = (x + att.eval_xi(xm, x, y, rng=np.random.default_rng(5)) + att.eval_delta(x))
    np.testing.assert_array_equal(main, fro)

    # 生成器被训动（用最小的扰动模拟一次生成器更新）：冻结值不变，主列变
    w0 = [np.array(w, copy=True) for w in gen.get_weights()]
    gen.set_weights([w + 0.05 * (np.random.default_rng(7).normal(size=w.shape).astype(w.dtype)
                                 if w.dtype.kind == "f" else 0) for w in w0])
    main2 = cloud._attacker_trigger(1, 0, "fresh", rng=np.random.default_rng(5))(None, x, y)
    moved = [np.array(w, copy=True) for w in gen.get_weights()]
    with cloud._frozen_state() as xm:
        fro2 = (x + att.eval_xi(xm, x, y, rng=np.random.default_rng(5)) + att.eval_delta(x))
    np.testing.assert_array_equal(fro2, fro)
    assert not np.array_equal(main2, main)
    for a, b in zip(moved, gen.get_weights()):                              # 退出后真生成器逐位复原
        np.testing.assert_array_equal(a, b)
