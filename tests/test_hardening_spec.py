"""
阶段三 SA0：`fedavg/hardening/spec.py` 的纯算术（不 import TF，本地秒级）。

- 配置网格：p0_configs.yaml 展开成 36 个确定的 id（32 个 + SAU 的 4 个匹配 C-ft，D-105；N-008：打开 s42 之前冻结 → 这里钉住，
  连同每个配置的全部字段与 probe / kstar / batch / n_clean）；步长约定（按比例缩放，D-105）；
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
    "cft-sau-train-low", "cft-sau-train-high", "cft-sau-frozen-low", "cft-sau-frozen-high",
]
# 展开后的网格（全部字段）+ 影响加固的全局设置的 sha。改了 p0_configs.yaml 的任何数值或展开规则 → 这里红；
# 那是在改预注册的配置清单（N-008），要先在 DECISIONS 留记录、再更新这个值。
FROZEN_CONTENT_SHA = "76eda6008dd872ed"


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
    # 比例约定（D-105，不是官方的绝对步长）：SAU ε = 8/255 → 步长 8/255（官方绝对 0.2）；TRADES K = 5 → 2·官方的 0.007 比例
    assert spec.step_px(by["sau-e8-frozen-low"]) == pytest.approx(8 / 255)
    assert spec.step_px(by["trades-e8-b6-train-low"]) == pytest.approx(2 * 0.007 / 0.031 * 8 / 255)
    with pytest.raises(ValueError):
        spec.step_px(by["cft-train-low"])


def test_matched_cft_has_the_same_bn_budget_optimizer_and_augmentation_d105(grid):
    _, cfgs = grid
    by = {c["id"]: c for c in cfgs}
    for c in cfgs:
        m = spec.matched_cft(c, cfgs)
        want = f"cft-sau-{c['bn']}-{c['budget']}" if (c["objective"] == "sau" or c["id"].startswith("cft-sau"))\
            else f"cft-{c['bn']}-{c['budget']}"
        assert m == want, c["id"]
        mc = by[m]
        assert (mc["optimizer"], mc["epochs"], spec.augments(mc)) == (c["optimizer"], c["epochs"], spec.augments(c))
        if c["optimizer"] == "adam":
            assert mc["lr"] == c["lr"] and mc["sau"]["amsgrad"] == c["sau"]["amsgrad"]
    # 反向锚点：只按 bn × 预算配（旧规则）会给 SAU 配上 SGD + 增强的 C-ft
    old = next(x for x in cfgs if x["id"] == "cft-frozen-low")
    assert (old["optimizer"], spec.augments(old)) != (by["sau-e8-frozen-low"]["optimizer"],
                                                       spec.augments(by["sau-e8-frozen-low"]))


def test_frozen_grid_content_sha(grid):
    """配置清单的**内容**（不只 id）：每个配置的全部字段 + batch / n_clean / probe / kstar。"""
    import hashlib
    g, cfgs = grid
    blob = json.dumps({"grid": cfgs, "batch": g["batch"], "n_clean": g["n_clean"], "probe": g["probe"],
                       "kstar": g["kstar"], "select": g["select"]}, sort_keys=True, default=str)
    assert hashlib.sha256(blob.encode()).hexdigest()[:16] == FROZEN_CONTENT_SHA


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
    bad = spec.v0_check(2e-5, {0: 0.5}, {0: 0.52}, {0: 0.3}, {0: None}, edges=[0])
    assert not bad["pass"] and len(bad["reasons"]) == 4         # 权重、A 差、一边 None、A 无可比 edge


def test_v0_judges_the_victim_mean_as_n008_says_d105():
    """N-008：|A 离线 − A 记录| ≤ 0.01，A = 受害三 edge 均值。逐 edge 差只记录（攻击者 edge 0 不进 A）。"""
    off = {0: 0.90, 1: 0.50, 2: 0.40, 3: 0.30}
    rec = {0: 0.95, 1: 0.515, 2: 0.39, 3: 0.30}               # 受害 edge 的差 +0.015 / −0.01 / 0 → 均值差 0.00167
    r = spec.v0_check(0.0, off, rec, off, off, edges=[1, 2, 3])
    assert r["pass"] and r["post_abs"] == pytest.approx(0.005 / 3) and r["post_max_abs"] == pytest.approx(0.015)
    # 反向锚点：不限定受害 edge（把攻击者 edge 0 也算进来）会判不过
    assert not spec.v0_check(0.0, off, rec, off, off)["pass"]
    # 均值本身超差 → 不过
    rec2 = {**rec, 1: 0.54, 2: 0.42}
    assert not spec.v0_check(0.0, off, rec2, off, off, edges=[1, 2, 3])["pass"]
    # 受害 edge 一边 None → 不过
    assert not spec.v0_check(0.0, off, {**rec, 2: None}, off, off, edges=[1, 2, 3])["pass"]


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


# ── 新模块不碰全局随机数（第二轮审查：旧守卫只扫 spec.py、按子串）───────────────────
RNG_MODULES = ["fedavg/hardening/spec.py", "fedavg/hardening/core.py", "fedavg/analysis/p0_snapshot.py",
               "fedavg/analysis/bn_cluster.py", "harness/p0_verdict.py"]
PY_RANDOM_DRAWS = {"random", "uniform", "randint", "randrange", "choice", "choices", "shuffle", "sample", "gauss",
                   "normalvariate", "seed", "betavariate", "expovariate", "triangular"}


def _global_rng_calls(src: str) -> list:
    import ast
    bad = []
    for n in ast.walk(ast.parse(src)):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute):
            continue
        f = n.func
        chain = []
        cur = f
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            chain.append(cur.id)
        chain = chain[::-1]
        if chain[:2] in (["np", "random"], ["numpy", "random"]) and len(chain) == 3 \
                and chain[2] not in ("default_rng", "Generator", "PCG64", "SeedSequence"):
            bad.append((".".join(chain), n.lineno))
        elif chain[:2] == ["tf", "random"]:
            bad.append((".".join(chain), n.lineno))
        elif chain[0] == "random" and len(chain) == 2 and chain[1] in PY_RANDOM_DRAWS:
            bad.append((".".join(chain), n.lineno))
    return bad


@pytest.mark.parametrize("path", RNG_MODULES)
def test_new_modules_do_not_draw_from_global_rngs(path):
    assert _global_rng_calls((ROOT / path).read_text(encoding="utf-8")) == []


def test_global_rng_scan_catches_planted_calls():
    planted = "import numpy as np\nimport random\nimport tensorflow as tf\n" \
              "a = np.random.normal(size=3)\nb = random.shuffle(x)\nc = tf.random.uniform((2,))\n" \
              "ok = np.random.default_rng(0)\nst = random.getstate()\n"
    assert [c for c, _ in _global_rng_calls(planted)] == ["np.random.normal", "random.shuffle", "tf.random.uniform"]


# ── D-106：加固 lr 倍数的选择规则（写于 lr 扫描数据之前）───────────────────────────
def _row(div=False, pm=None):
    return {"diverged": div, "pm_acc": pm}


def test_pick_lr_scale_takes_the_largest_stable_scale_with_small_clean_drop():
    base = 0.66
    scan = {"cft-train-low": {"1": _row(pm=0.60), "0.3": _row(pm=0.645), "0.1": _row(pm=0.655),
                              "0.03": _row(pm=0.66), "0.01": _row(pm=0.66)},
            "cft-train-high": {"1": _row(pm=0.55), "0.3": _row(pm=0.639), "0.1": _row(pm=0.65),
                               "0.03": _row(pm=0.66), "0.01": _row(pm=0.66)},
            "pgd-e8-train-low": {k: _row() for k in ("1", "0.3", "0.1", "0.03", "0.01")}}
    r = spec.pick_lr_scale(scan, base, ["cft-train-low", "cft-train-high"])
    assert r["scale"] == 0.1                                         # 0.3：high 降 0.021 > 0.02
    assert any("cft-train-high" in w for w in r["rejected"]["0.3"])
    scan["pgd-e8-train-low"]["0.1"] = _row(div=True)                 # 对抗配置在 0.1 发散 → 退到 0.03
    assert spec.pick_lr_scale(scan, base, ["cft-train-low", "cft-train-high"])["scale"] == 0.03


def test_pick_lr_scale_none_when_nothing_qualifies_and_missing_data_is_not_ok():
    scan = {"cft-frozen-low": {k: _row(div=True) for k in ("1", "0.3", "0.1", "0.03", "0.01")}}
    assert spec.pick_lr_scale(scan, 0.66, ["cft-frozen-low"])["scale"] is None
    scan = {"cft-frozen-low": {"0.01": _row(pm=0.66)}}                 # 只有最小倍数有数据 → 只能选它
    assert spec.pick_lr_scale(scan, 0.66, ["cft-frozen-low"])["scale"] == 0.01
    scan = {"cft-frozen-low": {"0.01": _row(pm=None)}}
    assert spec.pick_lr_scale(scan, 0.66, ["cft-frozen-low"])["scale"] is None


def test_lr_scan_configs_exist_in_the_frozen_grid(grid):
    _, cfgs = grid
    ids = {c["id"] for c in cfgs}
    for bn, cs in spec.LR_SCAN_CONFIGS.items():
        assert set(cs) <= ids and f"sau-e8-{bn}-low" in ids
    assert spec.LR_SCALES == (1.0, 0.3, 0.1, 0.03, 0.01)
