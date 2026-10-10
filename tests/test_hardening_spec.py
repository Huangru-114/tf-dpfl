"""
阶段三 SA0：`fedavg/hardening/spec.py` 的纯算术（不 import TF，本地秒级）。

- 配置网格：p0_configs.yaml 展开成 32 个确定的 id（N-008：打开 s42 之前冻结 → 这里钉住）；每个目标的步长约定 = 官方实现；
- 分层子集：干净集按类排序（F-095），子集必须按类比例、不能是前 n 张；
- 探针拟合：确定性、可分数据上能学会；
- 记账、lr、R_H、精度过滤、V0 的算术：手算值。
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

import sys                                                  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))
from hardening import spec                                  # noqa: E402

GRID_YAML = ROOT / "experiments/defense/edge-native/p0_configs.yaml"
SNAP_CFG = ROOT / "experiments/defense/edge-native/configs/SNAP__collocated__s42.yaml"

FROZEN_IDS = [
    "pgd-e4-train-low", "pgd-e4-train-high", "pgd-e4-frozen-low", "pgd-e4-frozen-high",
    "pgd-e8-train-low", "pgd-e8-train-high", "pgd-e8-frozen-low", "pgd-e8-frozen-high",
    "trades-e8-b1-train-low", "trades-e8-b1-train-high", "trades-e8-b1-frozen-low", "trades-e8-b1-frozen-high",
    "trades-e8-b6-train-low", "trades-e8-b6-train-high", "trades-e8-b6-frozen-low", "trades-e8-b6-frozen-high",
    "sau-e8-train-low", "sau-e8-train-high", "sau-e8-frozen-low", "sau-e8-frozen-high",
    "sau-e51-train-low", "sau-e51-train-high", "sau-e51-frozen-low", "sau-e51-frozen-high",
    "tgt-e4-train-low", "tgt-e4-train-high", "tgt-e4-frozen-low", "tgt-e4-frozen-high",
    "cft-train-low", "cft-train-high", "cft-frozen-low", "cft-frozen-high",
]


@pytest.fixture(scope="module")
def grid():
    g = yaml.safe_load(GRID_YAML.read_text(encoding="utf-8"))
    return g, spec.expand_grid(g["grid"])


# ── 网格 ─────────────────────────────────────────────────────────────────────
def test_grid_ids_are_frozen(grid):
    """N-008：配置清单在打开 s42 之前冻结；改了 p0_configs.yaml 或展开规则，这里红。"""
    _, cfgs = grid
    assert [c["id"] for c in cfgs] == FROZEN_IDS


def test_sau_has_both_budgets_d103(grid):
    _, cfgs = grid
    eps = sorted({c["eps_px"] for c in cfgs if c["objective"] == "sau"})
    assert eps == [pytest.approx(8 / 255), pytest.approx(0.2)]
    assert 0.2 * 255 == pytest.approx(51)                       # sau-e51 就是官方的 0.2
    assert all(c["pgd_steps"] == 5 and c["optimizer"] == "adam" and c["lr"] == 1e-4
               for c in cfgs if c["objective"] == "sau")


def test_no_at_ccs_in_p0_d104(grid):
    _, cfgs = grid
    assert not [c for c in cfgs if "ccs" in c["id"]]


def test_step_sizes_follow_the_official_implementations(grid):
    _, cfgs = grid
    by = {c["id"]: c for c in cfgs}
    # Madry：ε = 8/255、10 步 → 步长 2/255
    assert spec.step_px(by["pgd-e8-train-high"]) == pytest.approx(2 / 255)
    # TRADES：ε = 0.031、10 步 → 0.007（这里 ε = 8/255 ≈ 0.03137，同一比例）
    assert spec.step_px(by["trades-e8-b6-train-high"]) == pytest.approx(0.007 / 0.031 * 8 / 255)
    # SAU：adv_lr = trigger_norm（每步 = ε）
    assert spec.step_px(by["sau-e51-frozen-low"]) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        spec.step_px(by["cft-train-low"])


def test_matched_cft_has_the_same_bn_and_budget(grid):
    _, cfgs = grid
    for c in cfgs:
        m = spec.matched_cft(c, cfgs)
        assert m == f"cft-{c['bn']}-{c['budget']}"


def test_parse_px_and_labels():
    assert spec.parse_px("8/255") == pytest.approx(8 / 255)
    assert spec.parse_px(0.2) == 0.2
    assert spec.px_label(4 / 255) == "e4" and spec.px_label(0.2) == "e51" and spec.px_label(0.013) == "e0.013"
    with pytest.raises(ValueError):
        spec.parse_px(True)


def test_unknown_objective_or_bn_is_refused():
    with pytest.raises(ValueError):
        spec.expand_grid({"budgets": {"low": {"epochs": 1, "pgd_steps": 1}}, "objectives": {"nad": {}}})
    with pytest.raises(ValueError):
        spec.expand_grid({"bn": ["eval"], "budgets": {"low": {"epochs": 1, "pgd_steps": 1}},
                          "objectives": {"cft": {}}})


# ── 单位换算（与 Bad-PFL 的 ξ 同一套）──────────────────────────────────────
def test_pixel_budget_maps_to_input_space_per_channel():
    cfg = yaml.safe_load(SNAP_CFG.read_text(encoding="utf-8"))
    e = spec.to_input_space(8 / 255, cfg)
    np.testing.assert_allclose(e * np.array([0.2470, 0.2435, 0.2616]), 8 / 255, rtol=1e-6)
    lo, hi = spec.valid_range(cfg)
    assert np.all(lo < 0) and np.all(hi > 0)


# ── 干净集 ───────────────────────────────────────────────────────────────────
def test_stratified_subset_is_proportional_not_the_first_n():
    y = np.repeat(np.arange(10), [50, 125, 50, 13, 13, 12, 50, 12, 50, 125])   # C1 E1 的干净集，按类排序
    rng = np.random.default_rng(0)
    sub = spec.stratified_subset(y, 100, rng)
    assert len(sub) == 100 and len(set(sub.tolist())) == 100
    counts = np.bincount(y[sub], minlength=10)
    np.testing.assert_array_equal(counts, [10, 25, 10, 3, 3, 2, 10, 2, 10, 25])   # 最大余数法的手算值
    assert set(y[:100].tolist()) == {0, 1}                     # 反向锚点：前 100 张只有两个类
    np.testing.assert_array_equal(spec.stratified_subset(y, 100, np.random.default_rng(0)), sub)
    assert len(spec.stratified_subset(y, 10_000, rng)) == len(y)


def test_epoch_batches_shuffle_and_drop_last():
    rng = np.random.default_rng(1)
    b = spec.epoch_batches(500, 32, rng)
    assert len(b) == 15 and all(len(x) == 32 for x in b)
    flat = np.concatenate(b)
    assert len(set(flat.tolist())) == 480
    assert not np.array_equal(flat, np.arange(480))            # 打乱了


# ── 探针 ─────────────────────────────────────────────────────────────────────
def test_fit_probe_learns_separable_classes_and_is_deterministic():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 8)) * 5
    y = np.repeat(np.arange(4), 30)
    F = centers[y] + rng.normal(size=(120, 8))
    a = spec.fit_probe(F, y, 6, steps=200, lr=0.05)
    b = spec.fit_probe(F, y, 6, steps=200, lr=0.05)
    assert a["train_acc"] == 1.0
    assert a["W"].shape == (8, 6) and a["W"].dtype == np.float32
    np.testing.assert_array_equal(a["W"], b["W"])
    np.testing.assert_array_equal(a["b"], b["b"])


# ── 记账 / lr ───────────────────────────────────────────────────────────────
def test_fb_accounting_hand_values(grid):
    _, cfgs = grid
    by = {c["id"]: c for c in cfgs}
    assert spec.fb_harden(by["cft-train-low"], 500, 32) == 1 * 1 * 15
    assert spec.fb_harden(by["pgd-e8-train-high"], 500, 32) == (10 + 1) * 5 * 15
    assert spec.fb_harden(by["sau-e8-frozen-low"], 500, 32) == (2 * 5 + 3.5) * 1 * 15
    cfg = yaml.safe_load(SNAP_CFG.read_text(encoding="utf-8"))
    # 10 端 / edge 轮 × (5 + 1) epoch × 11 批 × R5
    assert spec.fb_clients_per_cloud_round(cfg) == 10 * 6 * 11 * 5


def test_lr_at_is_the_last_edge_round_of_the_cloud_round():
    cfg = yaml.safe_load(SNAP_CFG.read_text(encoding="utf-8"))
    assert spec.lr_at(cfg, 6) == pytest.approx(0.1 * 0.992 ** 30)
    assert spec.lr_at(cfg, 15) == pytest.approx(0.1 * 0.992 ** 75)


# ── N-008 的算术 ────────────────────────────────────────────────────────────
def test_r_h_hand_values_and_exclusions():
    assert spec.r_h(0.60, 0.45, 0.30) == pytest.approx(0.5)
    assert spec.r_h(0.60, 0.70, 0.30) == pytest.approx(-1 / 3)
    assert spec.r_h(0.60, 0.45, 0.049) is None                 # J_t < 0.05 → 不计
    assert spec.r_h(None, 0.45, 0.3) is None


def _ev(pool, edges):
    return {"pm_acc": pool, "per_edge": {e: {"pm_acc": v} for e, v in edges.items()}}


def test_accuracy_filter_pool_and_edge_thresholds():
    base = _ev(0.86, {0: 0.85, 1: 0.86, 2: 0.87, 3: 0.86})
    assert spec.accuracy_ok(base, _ev(0.845, {0: 0.84, 1: 0.84, 2: 0.86, 3: 0.85}))["ok"]
    r = spec.accuracy_ok(base, _ev(0.83, {0: 0.84, 1: 0.84, 2: 0.86, 3: 0.85}))
    assert not r["ok"] and r["d_pool"] == pytest.approx(0.03)
    r = spec.accuracy_ok(base, _ev(0.85, {0: 0.80, 1: 0.86, 2: 0.87, 3: 0.86}))
    assert not r["ok"] and r["d_edge_max"] == pytest.approx(0.05)
    assert not spec.accuracy_ok(base, _ev(None, {}))["ok"]


def test_v0_check():
    ok = spec.v0_check(0.0, {0: 0.5, 1: None}, {0: 0.505, 1: None}, {0: 0.3}, {0: 0.3})
    assert ok["pass"] and ok["post_max_abs"] == pytest.approx(0.005)
    bad = spec.v0_check(2e-5, {0: 0.5}, {0: 0.52}, {0: 0.3}, {0: None})
    assert not bad["pass"] and len(bad["reasons"]) == 3


def test_max_abs_diff_and_victims():
    assert spec.max_abs_diff([np.zeros(3), np.ones(2)], [np.zeros(3), np.ones(2) * 1.5]) == 0.5
    with pytest.raises(ValueError):
        spec.max_abs_diff([np.zeros(3)], [np.zeros(4)])
    assert spec.victims([10, 0, 0, 0]) == [1, 2, 3] and spec.victims([3, 3, 2, 2]) == []


def test_spec_does_not_import_tensorflow():
    import ast
    src = (ROOT / "fedavg/hardening/spec.py").read_text(encoding="utf-8")
    names = {a.name.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
    assert "tensorflow" not in names
    assert "random" not in names                               # 不碰 Python random
    assert "np.random.seed" not in src and "np.random.rand" not in src
