"""
tests/test_eval_grid_tf.py  —  S5：统一评估网格的 TF 侧（真跑 cloud / edge / client；D-055 / D-084）

用 test_eval_integration 的小 HFL（2 edge × 3 FedRep 客户端，2 个 Bad-PFL 恶意端，official BN、
固定攻击者、fresh-PM），改成 R > 1 + 交错调度：

  · **评估不改变训练**（D-055 的硬要求）：同一配置开 / 关网格，逐轮 [Checksum] 相同，
    全量点上已有的评估行（[Backdoor] [ASR4] [Acc] [Stale] [StaleASR] [Cloud]）也逐字相同；
  · 一次轻评估前后的**状态清单**逐位相同（客户端 / edge / 全局权重、各 rng、Python random、
    生成器权重与 Adam、服务器计数器）；生成器 BN 的 moving 统计量在 official 模式下**确实变了**
    （训练不读它们 —— 由上一条 checksum 判定，不是假设）；
  · Python random 的复原是**承重的**：去掉它，状态清单那条就红（clone_model 会消耗 random，F-078）；
  · 轻评估的位置 = server/eval_grid.py 的纯 python 规则；网格下 GM / EM 只在全量点算，[Acc] 照打 n/a；
  · `evaluate_hierarchical_asr(light=True)` 的分组值 = 全量调用的（确定性触发器）。

夹具用 legacy 取数管线（训练会读 Python random.shuffle）—— 比 P2 的 per_epoch 更敏感。
另把 Keras 的种子发生器复位成 main.py 的播种方式（`_prod_seeding`）：夹具的 _model() 调了
`tf.keras.utils.set_random_seed`，那之后 clone_model **不**消耗 Python random；main.py 只调
`tf.random.set_seed`，clone_model 会消耗它（F-078 实测）。不复位，这些测试就测不到这条通道。
"""

import hashlib
import random
import re

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import test_eval_integration as TEI                          # noqa: E402
import attack.backdoor_eval as BE                            # noqa: E402
from server import eval_grid as EG                           # noqa: E402
from utils.kvline import collect_kv                          # noqa: E402

def _prod_seeding(monkeypatch):
    """main.py 的播种方式：Keras 没有自己的种子发生器 → 未播种初始化器从 Python random 取种子。"""
    from keras.src import backend as kb
    monkeypatch.setattr(kb._SEED_GENERATOR, "generator", None)


OLD_TAGS = ("[Checksum]", "[Backdoor]", "[ASR4]", "[Acc]", "[Stale]", "[StaleASR]", "[Cloud] GM=")
FED = {"edge_schedule": "interleaved"}


def _run(capsys, monkeypatch, R, n_rounds, grid, interval):
    fed = dict(FED, edge_rounds=R, n_rounds=n_rounds)
    ev = {"eval_interval": interval}
    if grid is not None:
        ev["eval_grid"] = grid
    random.seed(42)                         # 两次 run 从同一个 Python random 状态出发
    cfg, cloud, clients, edges = TEI._setup(True, evaluation=ev, federation=fed)
    _prod_seeding(monkeypatch)
    cfg["backdoor"]["eval_interval"] = interval
    cloud.bd_eval_interval = interval
    capsys.readouterr()
    cloud.run()
    return capsys.readouterr().out, cloud


def _rounds(out):
    """按 `[Round N]` 头切块 → {N: [已有评估行（time= 抹掉）]}。"""
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


def _checksums(out):
    return [ln for ln in out.splitlines() if ln.startswith("[Checksum]")]


# (R, G, n_rounds)：A = 每个云轮都有轻评估、全量每轮；B = 云轮中间的轻评估 + GM / EM 隔轮
CASES = [(4, 2, 2), (2, 3, 3)]


@pytest.mark.parametrize("R,G,n", CASES)
def test_grid_on_off_same_training_and_same_full_point_numbers(capsys, monkeypatch, R, G, n):
    interval = EG.full_interval(G, R)
    off, _ = _run(capsys, monkeypatch, R, n, None, interval)
    on, cloud = _run(capsys, monkeypatch, R, n, G, interval)
    assert _checksums(off) and _checksums(off) == _checksums(on)          # 训练逐轮逐位相同
    a, b = _rounds(off), _rounds(on)
    full = [g for g in range(1, n + 1) if EG.is_full_round(G, R, g)]
    assert full
    for g in full:                                                      # 全量点的评估数值逐字相同
        assert a[g] == b[g], g
    light = collect_kv(on.splitlines(), "[Light]")
    want = [(g, er, eff) for g, er, eff, k in EG.eval_points(G, R, n) if k == "light"]
    assert [(d["round"], d["edge_round"], d["effective_round"]) for d in light] == want
    assert len(collect_kv(on.splitlines(), "[LightEdge]")) == 2 * len(want)
    assert all(d["local_benign_asr"] is not None and d["pm_acc"] is not None for d in light)
    assert not collect_kv(off.splitlines(), "[Light]")


def test_without_the_fence_grid_on_changes_training(capsys, monkeypatch):
    """反向锚点：去掉轻评估前后的 random 复原 → 开网格就改变了训练（第一个轻评估点建草稿模型时
    推进了 Python random，legacy 管线下一个 edge 轮的洗牌随之变）。证明上一条测试测得到这条通道。"""
    R, G, n = 4, 2, 2
    interval = EG.full_interval(G, R)
    off, _ = _run(capsys, monkeypatch, R, n, None, interval)
    monkeypatch.setattr(random, "setstate", lambda st: None)
    on, _ = _run(capsys, monkeypatch, R, n, G, interval)
    assert _checksums(off) != _checksums(on)


def test_gm_em_only_at_full_points_and_acc_lines_still_printed(capsys, monkeypatch):
    R, G, n = 2, 3, 3                                                   # 全量只在第 3 云轮
    on, cloud = _run(capsys, monkeypatch, R, n, G, EG.full_interval(G, R))
    ch = _rounds(on)
    for g in (1, 2):
        cl = [ln for ln in ch[g] if "[Cloud]" in ln]
        assert cl and "GM=n/a | EM=n/a" in cl[0] and "loss=n/a" in cl[0]
        acc = [ln for ln in ch[g] if ln.startswith("[Acc]")]
        assert len(acc) == 2 and all("em_acc=n/a" in ln for ln in acc)   # 不会整条消失
    assert "GM=n/a" not in " ".join(ch[3])
    assert cloud.history["global_acc"][:2] == [None, None]
    assert cloud.history["global_acc"][2] is not None
    t = collect_kv(on.splitlines(), "[TimingAcc]")
    assert t[0]["gm"] is None and t[2]["gm"] is not None


def _state(cloud, clients, edges):
    """一次评估可能碰到、训练会读到的全部状态（复核的状态清单）。"""
    def h(ws):
        return hashlib.sha256(b"".join(np.ascontiguousarray(np.asarray(w)).tobytes()
                                       for w in ws)).hexdigest()
    st = {"global": h(cloud.global_model.get_weights()),
          "py_random": random.getstate(), "np_global": repr(np.random.get_state()[1][:4])}
    for c in clients:
        st[f"c{c.client_id}"] = (h(c.model.get_weights()),
                                 h(getattr(c, "_head_weights", None) or []),
                                 h(getattr(c, "_stats", None) or []),
                                 repr(c.rng.bit_generator.state),
                                 repr(c.data_rng.bit_generator.state))
    for e in edges:
        st[f"e{e.edge_id}"] = (h(e.model.get_weights()), repr(e.rng.bit_generator.state))
    att = cloud.eval_attacker
    gen = att._atk_generator
    st["gen"] = h([v.numpy() for v in gen.variables if "moving" not in v.name])
    opt = att._atk_gen_opt.variables
    st["gen_opt"] = h([v.numpy() for v in (opt() if callable(opt) else opt)])
    st["counters"] = (cloud._eval_seq, cloud._bd_eval_count, cloud._last_bd_metrics,
                      {k: len(v) for k, v in cloud.history.items()})
    return st


def _after_one_edge_round(monkeypatch):
    cfg, cloud, clients, edges = TEI._setup(True, evaluation={"eval_grid": 1},
                                            federation=dict(FED, edge_rounds=2, n_rounds=1))
    _prod_seeding(monkeypatch)
    for e in edges:                                                     # 第 1 个 edge 轮（训出 head）
        e.run_edge_round(1, 1)
    return cfg, cloud, clients, edges


def test_one_light_eval_leaves_every_training_state_untouched(monkeypatch):
    cfg, cloud, clients, edges = _after_one_edge_round(monkeypatch)
    gen = cloud.eval_attacker._atk_generator
    mov0 = [v.numpy().copy() for v in gen.variables if "moving" in v.name]
    before = _state(cloud, clients, edges)
    cloud._light_eval(1, 1, 1)
    assert _state(cloud, clients, edges) == before
    assert cloud._edge_w_cache is None
    # 全量评估的草稿槽没被轻评估建出来（它们的创建时刻在开 / 关网格时必须相同）
    assert "victim" not in cloud._pm_scratch and "attacker" not in cloud._pm_scratch
    assert "light_victim" in cloud._pm_scratch
    # official 模式下 eval_delta 以 training=True 调生成器 → BN moving 统计量确实变了
    mov1 = [v.numpy() for v in gen.variables if "moving" in v.name]
    assert mov0 and any(not np.array_equal(a, b) for a, b in zip(mov0, mov1))


def test_the_python_random_fence_is_load_bearing(monkeypatch):
    """反向锚点：去掉 random 的复原，轻评估（第一次建草稿模型）就会推进 Python random —— 而
    legacy 管线的训练用它洗牌（F-078 的数值证据：clone_model 的未播种初始化器取 random 种子）。"""
    cfg, cloud, clients, edges = _after_one_edge_round(monkeypatch)
    st0 = random.getstate()
    monkeypatch.setattr(random, "setstate", lambda s: None)
    cloud._light_eval(1, 1, 1)
    assert random.getstate() != st0


def test_light_points_land_where_the_pure_rule_says(monkeypatch):
    import test_schedule_alignment as TSA
    for R, G, n in [(4, 2, 2), (20, 5, 2), (2, 5, 5), (10, 5, 2), (5, 5, 3)]:
        cloud, _ = TSA._tiny_cloud("interleaved", R, "effective")
        cloud.eval_grid = G
        got = []
        monkeypatch.setattr(cloud, "_light_eval", lambda g, er, eff: got.append((g, er, eff)))
        for g in range(1, n + 1):
            cloud.collect_and_aggregate(g, cloud.global_model.get_weights())
        want = [(g, er, eff) for g, er, eff, k in EG.eval_points(G, R, n) if k == "light"]
        assert got == want, (R, G)


def test_grid_with_sequential_schedule_is_refused():
    with pytest.raises(ValueError, match="interleaved"):
        TEI._setup(True, evaluation={"eval_grid": 1},
                   federation={"edge_rounds": 2, "edge_schedule": "sequential"})


def test_light_grouping_equals_the_full_call(monkeypatch):
    """light=True 只跳过 global / 干净前向；分组（逐 edge、same / diff、恶意端）与默认路径同一段代码。"""
    cfg, cloud, clients, edges = _after_one_edge_round(monkeypatch)
    trig = lambda model, x, y=None: np.clip(np.asarray(x, np.float32) + 0.5, -3, 3)  # noqa: E731
    kw = dict(fallback_test_ds=cloud.test_dataset, local_model_fn=lambda c: cloud.main_pm(c),
              asr_max_samples=0, asr_columns="four_way")
    cloud.begin_pm_eval()
    full = BE.evaluate_hierarchical_asr(cloud.global_model, edges, clients, cloud.test_dataset,
                                        trig, 0, TEI.MAL, **kw)
    lite = BE.evaluate_hierarchical_asr(cloud.global_model, edges, clients, cloud.test_dataset,
                                        trig, 0, TEI.MAL, light=True, **kw)
    for k in ("edge_asr_mean", "edge_asr_per_node", "local_asr_benign_mean",
              "local_asr_malicious_mean", "local_asr_same_edge", "local_asr_diff_edge",
              "local_asr_mean", "local_asr_benign_unfiltered_mean"):
        assert lite[k] == full[k], k
    assert [(d["edge_id"], d["edge_asr"], d["client_benign"], d["client_malicious"])
            for d in lite["per_edge"]] == [(d["edge_id"], d["edge_asr"], d["client_benign"],
                                            d["client_malicious"]) for d in full["per_edge"]]
    assert lite["global_asr"] is None and lite["global_acc"] is None
    assert lite["local_acc_mean"] is None and lite["edge_acc_mean"] is None
    # S6a ④：轻评估点也留逐客户端记录（只读已算出的概率），但没有干净前向 → acc / yt_clean 为 None
    assert len(lite["client_detail"]) == len(full["client_detail"]) > 0
    assert all(r["acc"] is None and r["yt_clean"] is None for r in lite["client_detail"])
    assert [r["asr"] for r in lite["client_detail"]] == [r["asr"] for r in full["client_detail"]]
    assert [k for k, *_ in lite["probe_order"]].count("global") == 0
