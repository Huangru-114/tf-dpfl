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
            mg = P0.edge_margins(m, a[5], k.get("n_classes"))           # 同一次前向的逐端记录（run 的 [EvalDetailEdge]）
            rec["full"].append({e: {"client_benign": v, "margin_p50": mg.get(e)} for e, v in _per_edge(m).items()})
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
    """小世界里 ASR 常为 0（0 == 0 分辨不出复原错没错）→ 连 margin 中位数一起逐位比；真实非零 ASR 的 V0 由 GPU 探路作业验证。"""
    w, snap = restored
    got = P0.eval_full_main(w, snap, 1, detail=True)
    assert got == source["rec"]["full"][0]
    assert any(v["margin_p50"] not in (None, 0.0) for v in got.values())
    assert P0.eval_full_main(w, snap, 1) == {e: v["client_benign"] for e, v in got.items()}


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
    cc = P0.bn_private_clu(w, snap)
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


def test_frozen_aux_runs_and_reuses_the_main_hardened_weights(restored):
    """冻结阶段的附加读数在小世界上跑通、每个键都在；own / ck 复用主评估的加固权重 = 用 salt 0 重算的结果（逐位配对）。"""
    pytest.importorskip("sklearn")
    w, snap = restored
    h = EdgeHardener(w.global_model, w.config, probe={"steps": 10, "lr": 0.05},
                     kstar={"n": 8, "eps_px": "4/255", "steps": 1}, batch=8)
    grid = GRID
    by = {c["id"]: c for c in grid}
    index_of = {c["id"]: i for i, c in enumerate(grid)}
    frozen = {"frozen": ["pgd-e8-frozen-low"]}
    aux = {"cloud_side": "top3", "n_clean": {"n": [8]}, "head_diag": "top3", "e0_own": "top3",
           "ck_before_after": {"n": 8}}
    kept = {}
    P0.eval_config(w, h, snap, by["pgd-e8-frozen-low"], index_of["pgd-e8-frozen-low"], 1, 0.05, keep=kept)
    a = P0.frozen_aux(w, h, snap, frozen, by, index_of, 1, 0.05, aux, kept)
    b = P0.frozen_aux(w, h, snap, frozen, by, index_of, 1, 0.05, aux)          # 不给 → salt 0 重算
    assert set(a) == {"cloud_side", "n_clean", "head_diag", "own", "ck"}
    assert a["own"] == b["own"] and a["ck"] == b["ck"]


# ── 第三轮审查：GPU 入口 main() 端到端、集中布点的受害 edge 路径、真实 SNAP 的 J_t ────────
import json                                                  # noqa: E402
from pathlib import Path                                      # noqa: E402

import yaml                                                   # noqa: E402

E2E_GRID = {"version": 1, "batch": 8, "n_clean": 16,
            "probe": {"steps": 5, "lr": 0.05, "l2": 1e-4}, "kstar": {"n": 8, "eps_px": "4/255", "steps": 1},
            "grid": {"bn": ["frozen"], "budgets": {"low": {"epochs": 1, "pgd_steps": 1}},
                     "objectives": {"pgd": {"eps_px": ["8/255"]}, "cft": {}}},
            "select": {"top": 1, "seed": 42, "placement": "collocated", "snapshots": [1]},
            "aux": {"damping_lambda": [0.5], "trigger_norms": {"n": 8}, "bn_private_clu": True,
                    "cloud_side": "top3", "n_clean": {"configs": "top1", "n": [8]}, "e0_own": "top3",
                    "ck_before_after": {"configs": "top3", "n": 8}, "head_diag": "top3"}}


def _metrics(source, snap_path, bump=0.0):
    rec = source["rec"]
    cols = list(P0.POST_EDGE_COLUMNS)
    rows = []
    for e, v in rec["post"][0]["client_benign"].items():
        r = [None] * len(cols)
        r[cols.index("edge_id")], r[cols.index("client_benign")] = e, (None if v is None else v + bump)
        rows.append(r)
    full = [{"edge_id": e, "client_benign": v["client_benign"]} for e, v in rec["full"][0].items()]
    return {"dumps": {"snapshots": [{"round": 1, "path": snap_path.name, "sha": spec.sha12(snap_path)}]},
            "per_edge_post_agg_rounds": {"2": rows}, "per_edge_rounds": {"1": full}}


@pytest.fixture()
def e2e(source, restored, tmp_path, monkeypatch):
    w, _ = restored
    monkeypatch.setattr(P0, "build_world", lambda path: w)
    g = tmp_path / "grid.yaml"
    g.write_text(yaml.safe_dump(E2E_GRID))
    snap = Path(source["snap"])

    def call(stage="screen", extra=(), bump=0.0, sha_ok=True):
        m = _metrics(source, snap, bump)
        if not sha_ok:
            m["dumps"]["snapshots"][0]["sha"] = "000000000000"
        (tmp_path / "m.json").write_text(json.dumps(m))
        out = tmp_path / f"{stage}.json"
        argv = ["--config", "unused.yaml", "--metrics", str(tmp_path / "m.json"), "--dumps-root", str(snap.parent),
                "--grid", str(g), "--stage", stage, "--rounds", "1", "--out", str(out), *extra]
        rc = P0.main(argv)
        return rc, json.loads(out.read_text())
    return call, tmp_path


def test_main_screen_writes_a_complete_result(e2e):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
    import p0_verdict as V
    call, _ = e2e
    rc, res = call()
    assert rc == 0 and res["schema"] == P0.SCHEMA and res["stage"] == "screen"
    assert sorted(res["configs"]) == ["cft-frozen-low", "pgd-e8-frozen-low"]           # screen = 网格全部
    assert V.incomplete(res, res["configs"], (1,)) == []
    S = res["snapshots"]["1"]
    assert S["v0"]["pass"] and {"trigger_norms", "bn_private_clu", "damping_0.5"} <= set(S["aux"])


def test_main_frozen_stage_adds_the_matched_cft_and_aux_and_only_skips_them(e2e):
    call, tmp = e2e
    fz = tmp / "frozen.json"
    fz.write_text(json.dumps({"frozen": ["pgd-e8-frozen-low"], "status": "ok"}))
    rc, res = call("frozen", ["--frozen", str(fz)])
    assert rc == 0 and res["configs"] == ["pgd-e8-frozen-low", "cft-frozen-low"]
    assert {"cloud_side", "n_clean", "head_diag", "own", "ck"} <= set(res["snapshots"]["1"]["aux"])
    assert res["frozen_sha"] and len(res["frozen_sha"]) == 12
    rc, res = call("frozen", ["--frozen", str(fz), "--only", "pgd-e8-frozen-low"])
    assert res["configs"] == ["pgd-e8-frozen-low"] and "ck" not in res["snapshots"]["1"]["aux"]


def test_main_refuses_a_snapshot_whose_sha_differs(e2e):
    call, _ = e2e
    with pytest.raises(RuntimeError, match="sha"):
        call(sha_ok=False)


def test_main_v0_failure_stops_and_exits_3(e2e):
    call, _ = e2e
    rc, res = call(bump=0.5)
    assert rc == 3 and not res["snapshots"]["1"]["v0"]["pass"] and res["snapshots"]["1"]["configs"] == {}


def test_collocated_path_reads_the_victim_edges(restored):
    w, snap = restored
    saved = (w.victims, w.placement)
    try:
        w.victims, w.placement = [1], "collocated"
        ev = P0.eval_post(w, {int(e.edge_id): snap["global"] for e in w.edges}, t=1)
        assert ev["victim_asr"] == ev["per_edge"][1]["client_benign"]
        assert ev["margin_v"] == ev["per_edge"][1]["margin_p50"]
    finally:
        w.victims, w.placement = saved


SNAP_RESULTS = Path(__file__).resolve().parent.parent / "experiments/defense/edge-native/results/P2/SNAP"
F094_J = {42: {6: 0.2309, 15: 0.4226}, 43: {6: 0.2497, 15: 0.2244}, 44: {6: 0.336, 15: 0.146}}


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_victim_jump_on_the_real_snap_metrics_matches_f094(seed):
    files = sorted(SNAP_RESULTS.glob(f"SNAP__collocated__s{seed}*.metrics.json"))
    if not files:
        pytest.skip("SNAP 的 metrics.json 不在本地")
    m = json.loads(files[-1].read_text())
    for t, want in F094_J[seed].items():
        assert round(P0.victim_jump(P0.recorded(m, t), [1, 2, 3]), 4) == pytest.approx(want, abs=1e-4)


def test_diverged_config_is_not_evaluated(restored):
    w, snap = restored
    h = EdgeHardener(w.global_model, w.config, probe={"steps": 10, "lr": 0.05}, batch=8)
    by = {c["id"]: c for c in GRID}
    r = P0.eval_config(w, h, snap, by["cft-frozen-low"], 1, 1, 1e30)
    assert r["eval"] is None and set(r["diverged"]) == {"0", "1"}


def test_main_harden_only_records_traces_and_reads_no_asr(e2e):
    call, _ = e2e
    rc, res = call(extra=["--harden-only"])
    S = res["snapshots"]["1"]
    assert rc == 0 and S["aux"] == {}
    for c in S["configs"].values():
        assert c["harden_only"] and "eval" not in c
        assert all(len(h["loss_trace"]) == h["steps"] for h in c["harden"].values())
