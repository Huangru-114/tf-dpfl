"""
阶段三 SA0：`fedavg/analysis/p0_snapshot.py`（P0 的离线反事实驱动）端到端。需要 TF。

在一个很小的真 HFL（tests/test_eval_integration._setup：2 edge × 3 个 FedRep 客户端，2 个 Bad-PFL 恶意端）上：
  1. 跑第 1、2 云轮，第 1 轮末存快照、第 2 轮开头做云聚合后评估点，记下 run 自己算出的数；
  2. 另建一个**全新**的世界（同配置、没训练过），用驱动从快照恢复；
  3. V0 的三项**逐位**相同：FedAvg(快照 edge) = 快照 global；离线云聚合后点 = run 第 2 轮的 [PostAgg]；
     离线第 1 轮全量点主列 = run 第 1 轮的主列；
  4. 加固 → FedAvg → 评估跑得通、可复现；附加读数跑得通。
这就是 N-008 的 V0 在 CPU 上的预演：GPU 上不逐位 = 驱动没复原 run 的状态（而不是加固有效）。
"""

import copy
import random

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import test_eval_integration as TEI                         # noqa: E402
import server.backdoor_server as BS                         # noqa: E402
from analysis import p0_snapshot as P0                       # noqa: E402
from hardening import spec                                   # noqa: E402
from hardening.core import EdgeHardener                      # noqa: E402

EV = {"post_agg_eval": True, "snapshot_rounds": "1"}
FED = {"edge_schedule": "interleaved", "edge_rounds": 2, "n_rounds": 2}


def _per_edge(asr, key="client_benign"):
    return {int(d["edge_id"]): d.get(key) for d in asr.get("per_edge") or []}


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """跑第 1、2 云轮的「原 run」：记下第 1 轮全量点主列、第 2 轮云聚合后点，返回快照路径。"""
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("dumps")
    mp.setenv("TFDPFL_DUMPDIR", str(tmp))
    random.seed(42)
    cfg, cloud, clients, edges = TEI._setup(True, evaluation=dict(EV), federation=dict(FED))
    rec = {"post": [], "full": []}
    orig_emit = cloud._emit_light

    def emit(round_idx, er, eff, acc, asr, eval_s, tag="[Light]"):
        if tag == "[PostAgg]":
            rec["post"].append({"round": round_idx, "acc": copy.deepcopy(acc),
                                "client_benign": _per_edge(asr), "pooled": asr.get("local_asr_benign_mean"),
                                "margin": {int(k): v.get("margin_p50") for k, v in (asr.get("detail_edge") or {}).items()}})
        return orig_emit(round_idx, er, eff, acc, asr, eval_s, tag=tag)
    mp.setattr(cloud, "_emit_light", emit)
    orig_eha = BS.evaluate_hierarchical_asr

    def eha(*a, **k):
        m = orig_eha(*a, **k)
        if not k.get("light"):
            rec["full"].append(_per_edge(m))
        return m
    mp.setattr(BS, "evaluate_hierarchical_asr", eha)
    cloud.run_round(1)
    n_full_r1 = len(rec["full"])
    cloud.run_round(2)
    mp.undo()
    snaps = sorted(tmp.rglob("snapshot_r001.npz"))
    assert len(snaps) == 1 and n_full_r1 >= 1 and rec["post"] and rec["post"][0]["round"] == 2
    return {"snap": snaps[0], "rec": rec, "cfg": cfg}


def _world(cfg_override=None):
    random.seed(42)
    cfg, cloud, clients, edges = TEI._setup(True, evaluation=dict(EV), federation=dict(FED))
    w = P0.World()
    w.config, w.cloud, w.global_model = cfg, cloud, cloud.global_model
    w.edges, w.clients = list(cloud.edge_servers), clients
    w.malicious = set(TEI.MAL)
    w.R, w.seed = FED["edge_rounds"], 42
    w.victims, w.placement = [], "distributed"                  # MAL = {0, 3} → 两个 edge 都有攻击者
    r = np.random.default_rng(9)
    w.clean = {int(e.edge_id): (r.normal(size=(16, TEI.IMG, TEI.IMG, 3)).astype(np.float32),
                                (np.arange(16) % 4).astype(np.int64)) for e in w.edges}
    return w


@pytest.fixture(scope="module")
def restored(source):
    w = _world()
    snap = P0.load_snapshot(source["snap"], w)
    P0.restore(w, snap)
    return w, snap


# ── V0 的三项逐位相同 ─────────────────────────────────────────────────────────
def test_v0_fedavg_of_the_snapshot_edges_is_the_snapshot_global_bitwise(restored):
    w, snap = restored
    assert spec.max_abs_diff(P0.fedavg(w, snap["edges"]), snap["global"]) == 0.0


def test_v0_offline_post_agg_point_equals_the_runs_bitwise(source, restored):
    w, snap = restored
    ev = P0.eval_post(w, {int(e.edge_id): snap["global"] for e in w.edges}, t=1)
    run = source["rec"]["post"][0]
    assert {e: v["client_benign"] for e, v in ev["per_edge"].items()} == run["client_benign"]
    assert {e: v["pm_acc"] for e, v in ev["per_edge"].items()} == \
        {int(e): v["pm_acc"] for e, v in run["acc"]["per_edge"].items()}
    assert ev["pm_acc"] == run["acc"]["pm_acc"] and ev["pooled_benign"] == run["pooled"]
    assert {e: v["margin_p50"] for e, v in ev["per_edge"].items()} == run["margin"]


def test_v0_offline_full_point_equals_the_runs_main_column_bitwise(source, restored):
    w, snap = restored
    assert P0.eval_full_main(w, snap, 1) == source["rec"]["full"][0]


def test_eval_is_repeatable(restored):
    w, snap = restored
    a = P0.eval_post(w, {int(e.edge_id): snap["global"] for e in w.edges}, t=1)
    b = P0.eval_post(w, {int(e.edge_id): snap["global"] for e in w.edges}, t=1)
    assert a == b


def test_snapshot_from_another_world_is_refused(source):
    w = _world()
    w.malicious = {0, 4}
    with pytest.raises(RuntimeError, match="恶意端不一致"):
        P0.load_snapshot(source["snap"], w)


# ── 加固 → FedAvg → 评估 ─────────────────────────────────────────────────────
GRID = spec.expand_grid({"bn": ["frozen"], "budgets": {"low": {"epochs": 1, "pgd_steps": 2}},
                         "objectives": {"pgd": {"eps_px": ["8/255"]}, "cft": {}}})


def test_eval_config_runs_and_is_reproducible(restored):
    w, snap = restored
    h = EdgeHardener(w.global_model, w.config, probe={"steps": 10, "lr": 0.05}, batch=8)
    by = {c["id"]: c for c in GRID}
    a = P0.eval_config(w, h, snap, by["pgd-e8-frozen-low"], 0, 1, 0.05)
    b = P0.eval_config(w, h, snap, by["pgd-e8-frozen-low"], 0, 1, 0.05)
    assert a["eval"] == b["eval"]
    assert set(a["harden"]) == {"0", "1"} and all(v["steps"] == 2 for v in a["harden"].values())
    base = P0.eval_post(w, {int(e.edge_id): snap["global"] for e in w.edges}, t=1)
    assert a["eval"] != base                                    # 加固真的改了模型
    # 只加固攻击者所在的 edge（「只洗自己」）：其余 edge 原样
    new, infos = P0.harden_all(w, h, snap, by["cft-frozen-low"], 1, 1, 0.05, only_edges={0})
    assert set(infos) == {0}
    for x, y in zip(new[1], snap["edges"][1]):
        np.testing.assert_array_equal(x, y)


def test_per_snapshot_aux_readouts_run(restored):
    pytest.importorskip("sklearn")
    w, snap = restored
    P0.set_edges(w, {int(e.edge_id): snap["global"] for e in w.edges})
    tn = P0.trigger_norms(w, 1, 8)
    assert tn["n"] == 8 and tn["delta"]["pixel_max"] <= 4 / 255 + 1e-6       # δ = ε·tanh(·)，不裁剪
    # ξ 的界（σ）只对合法像素成立：这个小世界的探针是 N(0,1) 噪声、超出 valid_range，ξ 的裁剪会把它拉回来 →
    # 这里只查字段齐全；真实数据上的界由 SA0 的 GPU 探路作业读出（附加读数 trigger_norms）。
    assert {"input_max", "pixel_max", "pixel_p50"} <= set(tn["xi"]) and tn["delta_plus_xi"]["pixel_max"] > 0
    cc = P0.ccs_clu(w, snap)
    assert set(cc) == {0, 1} and all("status" in v for v in cc.values())


# ── run 记录的数 ─────────────────────────────────────────────────────────────
def test_recorded_and_victim_jump_from_metrics():
    m = {"per_edge_post_agg_rounds": {"7": [[0, 0.9, 0.95, 1.0, 0.8, 0.8, 5.0, 1.0],
                                            [1, 0.5, 0.40, None, 0.8, 0.8, 1.0, 0.6],
                                            [2, 0.5, 0.30, None, 0.8, 0.8, 1.0, 0.6]]},
         "per_edge_rounds": {"6": [{"edge_id": 0, "client_benign": 0.9}, {"edge_id": 1, "client_benign": 0.2},
                                   {"edge_id": 2, "client_benign": 0.1}]}}
    rec = P0.recorded(m, 6)
    assert rec["post"] == {0: 0.95, 1: 0.40, 2: 0.30} and rec["full"] == {0: 0.9, 1: 0.2, 2: 0.1}
    assert P0.victim_jump(rec, [1, 2]) == pytest.approx(0.2)
    assert P0.victim_jump(rec, []) is None
