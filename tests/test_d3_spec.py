"""
阶段三 D3 离线探针：`fedavg/analysis/d3_spec.py` 的纯算术（不 import TF，本地秒级）。

规则 PREREG-v2 §五（N-015 @ 25163a5）；实现取舍 DECISIONS D-138。TF 部分见 tests/test_d3_probes_tf.py。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))
from analysis import d3_spec as S                            # noqa: E402

HERE = ROOT / "experiments/defense/edge-native"
CFG = HERE / "d3_configs.yaml"

# 参数清单的内容 sha（D-138，写于任何 D3 数据之前）。改了 d3_configs.yaml 的任何取值 → 这里红：
# 先在 DECISIONS 留记录（为什么改、改在数据之前还是之后），再更新这个数。
FROZEN_CONTENT_SHA = "89160c6203b99416"


@pytest.fixture(scope="module")
def cfg():
    return S.load_config(CFG)


# ── 参数清单 ─────────────────────────────────────────────────────────────────
def test_config_content_sha_is_frozen(cfg):
    assert S.content_sha(cfg) == FROZEN_CONTENT_SHA


def test_config_matches_the_prereg_objects(cfg):
    """PREREG-v2 §五写定的对象：P-a = SNAP 集中 s42–44 第 15 / 60 轮；P-b = TBS k1-snap 第 60 轮（SNAP 集中作参照）；
    P-c = SNAP 集中 + 分散 + TBS k1-snap × 第 6 / 15 / 60 轮。"""
    seeds = (42, 43, 44)
    assert cfg["pa"]["runs"] == [f"SNAP__collocated__s{s}" for s in seeds]
    assert cfg["pa"]["rounds"] == [15, 60]
    assert cfg["pb"]["runs"] == [f"TBS__k1-snap__s{s}" for s in seeds]
    assert cfg["pb"]["reference_runs"] == [f"SNAP__collocated__s{s}" for s in seeds]
    assert cfg["pb"]["rounds"] == [60]
    assert cfg["pc"]["runs"] == ([f"SNAP__collocated__s{s}" for s in seeds] + [f"SNAP__distributed__s{s}" for s in seeds]
                                 + [f"TBS__k1-snap__s{s}" for s in seeds])
    assert cfg["pc"]["rounds"] == [6, 15, 60]
    assert len(cfg["pc"]["runs"]) * len(cfg["pc"]["rounds"]) * 4 == 108       # 108 个边缘模型
    assert cfg["pa"]["acc_gate"] == 0.05 and cfg["reproduce_tol"] == 0.01


def test_pa_uses_p0_cft_train_high_unchanged(cfg):
    """P-a 的训练设置 = P0 的 cft-train-high（批 32、75 步、SGD、lr × 0.1、BN 训练模式），只改训练哪些层。"""
    sys.path.insert(0, str(ROOT / "fedavg"))
    from hardening import spec
    g = yaml.safe_load((HERE / cfg["pa"]["grid"]).read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in spec.expand_grid(g["grid"])}
    c = by_id[cfg["pa"]["base_config"]]
    assert (c["objective"], c["bn"], c["optimizer"], c["epochs"]) == ("cft", "train", "sgd", 5)
    assert c["lr_scale"] == 0.1 and int(g.get("batch", 32)) == 32
    assert spec.augments(c)
    assert (500 // 32) * c["epochs"] == 75                    # 每 epoch 15 批（drop_last）× 5 = 75 步
    assert spec.harden_lr(c, 0.0547490) == pytest.approx(0.00547490)   # P0 s42 屏筛 JSON 的 t=15 lr


def test_arms_and_prefixes(cfg):
    a = S.arm_prefixes(cfg)
    assert list(a) == ["stage4", "stem_stage3", "body"]
    assert a["stage4"] == ("stage4_",) and a["body"] is None
    assert a["stem_stage3"] == ("stem_", "stage1_", "stage2_", "stage3_")
    names = ["stem_conv", "stem_bn", "stage1_conv1", "stage3_bn2", "stage4_sc_conv", "stage4_bn2", "head", None]
    assert S.select_by_prefix(names, a["stage4"]) == [4, 5]
    assert S.select_by_prefix(names, a["stem_stage3"]) == [0, 1, 2, 3]
    assert S.select_by_prefix(names, None) == list(range(len(names)))
    # 两个子集互不相交、并起来恰好是全部主干（ResNet-10 的主干 = stem + stage1–4）
    body = [n for n in names if n not in ("head", None)]
    s4 = set(S.select_by_prefix(body, a["stage4"]))
    s03 = set(S.select_by_prefix(body, a["stem_stage3"]))
    assert not (s4 & s03) and s4 | s03 == set(range(len(body)))


def test_plan_for_each_run(cfg):
    assert S.plan_for(cfg, "SNAP__collocated__s42") == {"pa": [15, 60], "pb": [60], "pc": [6, 15, 60]}
    assert S.plan_for(cfg, "TBS__k1-snap__s43") == {"pb": [60], "pc": [6, 15, 60]}
    assert S.plan_for(cfg, "SNAP__distributed__s44") == {"pc": [6, 15, 60]}
    assert S.plan_for(cfg, "TBS__k2__s42") == {}
    assert S.rounds_needed(S.plan_for(cfg, "SNAP__collocated__s42")) == [6, 15, 60]
    assert len(S.all_runs(cfg)) == 9
    assert S.group_of("TBS__k1-snap__s42") == "TBS" and S.seed_of("TBS__k1-snap__s42") == 42
    with pytest.raises(ValueError):
        S.seed_of("TBS__k1-snap")


@pytest.mark.parametrize("bad", [
    {"pa": {"label_arm": "nope"}},
    {"pa": {"arms": {"none": None, "stage4": ["stage4_"]}}},
    {"pa": {"reestimate": {"method": "ema", "batch": 32, "augment": False}}},
    {"pa": {"reestimate": {"method": "precise", "batch": 32, "augment": True}}},
    {"pc": {"kl_direction": "clean_given_triggered"}},
    {"pc": {"permutation": {"kind": "stratified", "n": 10000, "seed": 1, "sided": "greater"}}},
    {"pc": {"permutation": {"kind": "free", "n": 100, "seed": 1, "sided": "greater"}}},
    {"schema": 2},
])
def test_validate_rejects_unimplemented_choices(cfg, bad):
    c = json.loads(json.dumps(cfg))
    for k, v in bad.items():
        if isinstance(v, dict):
            c[k].update(v)
        else:
            c[k] = v
    with pytest.raises(ValueError):
        S.validate_config(c)


# ── 精确重估的动量日程 ───────────────────────────────────────────────────────
def test_precise_momenta_give_the_sample_weighted_mean():
    sizes = [32] * 11 + [23]                                   # 375 张训练图、批 32
    m = S.precise_momenta(sizes)
    assert m[0] == 0.0 and len(m) == 12
    assert m[1] == pytest.approx(32 / 64) and m[-1] == pytest.approx(352 / 375)
    r = np.random.default_rng(0)
    stats = [r.normal(size=5) for _ in sizes]
    want = np.sum([n * s for n, s in zip(sizes, stats)], axis=0) / sum(sizes)
    got = S.apply_momenta(stats, m, init=r.normal(size=5) * 100)   # 旧统计量完全不留
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_precise_momenta_rejects_empty_batches():
    with pytest.raises(ValueError):
        S.precise_momenta([32, 0])


def test_seq_batches_cover_everything_in_order():
    b = S.seq_batches(375, 32)
    assert b[0] == (0, 32) and b[-1] == (352, 375) and len(b) == 12
    assert sum(e - s for s, e in b) == 375
    assert S.seq_batches(0, 32) == []


# ── run 记录 + 重现闸 ─────────────────────────────────────────────────────────
def test_recorded_edges_on_real_snap_metrics():
    """真实 metrics 的锚点（Explore 读数，与 metrics.json 逐字相同）：SNAP 集中 s42 第 60 / 15 轮。"""
    m = json.loads((HERE / "results/P2/SNAP/SNAP__collocated__s42.metrics.json").read_text(encoding="utf-8"))
    r60 = S.recorded_edges(m, 60)
    assert r60[0] == {"client_benign": 1.0, "margin_p50": 10.0426}
    assert r60[1]["margin_p50"] == 7.9576 and r60[3]["client_benign"] == 0.999
    r15 = S.recorded_edges(m, 15)
    assert r15[0]["margin_p50"] == 6.0398 and r15[1]["client_benign"] == 0.356


def test_recorded_edges_exist_for_every_planned_unit(cfg):
    """108 个后门强度读数都在已入库的 metrics.json 里（P-c 的 S），P-a / P-b 用到的轮次也都有。"""
    for run in S.all_runs(cfg):
        m = json.loads((HERE / f"results/P2/{S.group_of(run)}/{run}.metrics.json").read_text(encoding="utf-8"))
        assert {int(s["round"]) for s in m["dumps"]["snapshots"]} >= {6, 15, 60}
        for t in S.rounds_needed(S.plan_for(cfg, run)):
            rec = S.recorded_edges(m, t)
            assert sorted(rec) == [0, 1, 2, 3]
            assert all(v["margin_p50"] is not None and v["client_benign"] is not None for v in rec.values())


def test_reproduce_check():
    rec = {0: {"client_benign": 1.0, "margin_p50": 10.0}, 1: {"client_benign": 0.3, "margin_p50": -1.0}}
    ok = S.reproduce_check({0: {"client_benign": 1.0, "margin_p50": 10.004},
                            1: {"client_benign": 0.309, "margin_p50": -1.0}}, rec, 0.01)
    assert ok["pass"] and ok["max_abs_diff"] == pytest.approx(0.009)
    bad = S.reproduce_check({0: {"client_benign": 1.0, "margin_p50": 10.02},
                             1: {"client_benign": 0.3, "margin_p50": -1.0}}, rec, 0.01)
    assert not bad["pass"] and "edge 0 的 margin_p50" in bad["reasons"][0]
    miss = S.reproduce_check({0: {"client_benign": 1.0, "margin_p50": None}}, rec, 0.01)
    assert not miss["pass"]                                     # 缺值不能当成比过了
    assert not S.reproduce_check({}, {}, 0.01)["pass"]


# ── 逐端记录 → 逐 edge ───────────────────────────────────────────────────────
def test_edge_summary_matches_run_conventions():
    recs = [
        {"client_id": 0, "edge_id": 0, "malicious": True, "asr": 1.0, "margins": np.array([9.0]), "acc": 0.5},
        {"client_id": 1, "edge_id": 0, "malicious": False, "asr": 0.5, "margins": np.array([1.0, 3.0]), "acc": 0.8},
        {"client_id": 2, "edge_id": 0, "malicious": False, "asr": 1.0, "margins": np.array([5.0]), "acc": 0.6},
        {"client_id": 3, "edge_id": 1, "malicious": False, "asr": None, "margins": np.zeros(0), "acc": 0.9},
    ]
    s = S.edge_summary(recs)
    assert s["0"] == {"asr": 0.75, "margin_p50": 3.0, "acc": pytest.approx(0.7), "n": 2}  # 攻击者不计；margin 池化
    assert s["1"]["asr"] is None and s["1"]["margin_p50"] is None and s["1"]["acc"] == 0.9
    assert S.edge_summary(recs, malicious=True)["0"]["asr"] == 1.0
    assert S.group_mean(s, [0]) == 0.75 and S.group_mean(s, [0, 1]) is None


def test_jsonable_and_trim_info():
    assert S.jsonable({1: np.float32(0.5), "a": np.arange(2), "b": float("nan")}) == {"1": 0.5, "a": [0, 1], "b": None}
    t = S.trim_info({"id": "x", "steps": 75, "loss_trace": [1, 2], "gnorm_body": [1], "lr": 0.1})
    assert t == {"id": "x", "steps": 75, "lr": 0.1}


def test_module_does_not_import_tensorflow():
    src = (ROOT / "fedavg/analysis/d3_spec.py").read_text(encoding="utf-8")
    assert "import tensorflow" not in src and "from tensorflow" not in src


# ── 作业脚本 ─────────────────────────────────────────────────────────────────
def _code(path):
    return "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#"))


def test_job_script_passes_arguments_only():
    code = _code(HERE / "d3.sbatch")
    assert 'source "$ROOT/cluster_env.sh"' in code and 'cd "$ROOT/fedavg"' in code
    assert "$PY -m analysis.d3_probes" in code
    assert "--override" not in code and "--defense" not in code and "--seed" not in code


def test_submit_script_lists_exactly_the_configured_runs(cfg):
    import re
    code = _code(HERE / "d3_submit.sh")
    m = re.search(r'RUNS="([^"]+)"', code)
    assert m and m.group(1).split() == S.all_runs(cfg)
    assert 'sbatch "$HERE/d3.sbatch"' in code and "python" not in code
