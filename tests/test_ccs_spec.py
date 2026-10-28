"""
阶段三 SA-C（D-109）：CCS 的纯算术部分（`fedavg/defense/ccs_spec.py`）+ 服务端（`fedavg/defense/ccs.py`）+ 配置校验 + 回程解析。不需要 TF。

手算值：
  · DPR：圆心 1、半径内 ≈ 1、半径外 ≈ 0、面积 ≈ 36 边形面积 ½·36·r²·sin 10°；
  · MMD：x = y → 0；一维 x = [[0]]、y = [[1]] → 8 − 2·Σ exp(−0.5 / bw)；
  · max-margin / KL：小 logits 表手算；
  · 对抗步：u + step·sign → 投影 → clamp 的顺序（AT.py）；
  · 聚类：8 个相近 + 2 个离群 → 剔除离群；cluster 关 / 全部接纳 → 与 off 臂的 FedAvg **逐位**相同；edge_cache 池；退化。
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))
sys.path.insert(0, str(ROOT / "harness"))

from aggregation.client_update import ClientUpdate        # noqa: E402
from aggregation.fedavg import aggregate as fedavg_aggregate   # noqa: E402
from defense import ccs_spec as S                          # noqa: E402
from defense.ccs import CCSDefense                         # noqa: E402

pytest.importorskip("sklearn")


# ── DPR 掩码 ──────────────────────────────────────────────────────────────
def test_dpr_mask_is_the_official_36gon():
    m = S.dpr_mask(16, 16, 10, 36, 32, 32)
    assert m.shape == (32, 32) and m[16, 16] == 1.0
    assert m[16, 21] > 0.999 and m[21, 16] > 0.999          # 半径 5：里面
    assert m[16, 29] < 1e-6 and m[3, 16] < 1e-6             # 半径 13：外面
    area = 0.5 * 36 * 10 ** 2 * math.sin(math.radians(10))
    assert abs(float((m > 0.5).sum()) - area) < 15          # 像素格点数 ≈ 多边形面积
    assert np.isfinite(m).all() and m.min() >= 0 and m.max() <= 1


def test_dpr_mask_centre_offsets_follow_rows_then_cols():
    m = S.dpr_mask(10, 20, 6, 36, 32, 32)                   # cx = 行、cy = 列（官方 coordinate[..., 0] = 行号）
    assert m[10, 20] == 1.0 and m[10, 24] > 0.999 and m[14, 20] > 0.999
    assert m[20, 20] < 1e-6 and m[10, 29] < 1e-6


def test_draw_masks_shape_and_centre_range_and_rng_only():
    st = np.random.get_state()[1].copy()
    r = np.random.default_rng(0)
    M = S.draw_masks(r, 5, 32, 32, 10, 36)
    assert M.shape == (5, 32, 32, 1) and M.dtype == np.float32
    assert np.array_equal(np.random.get_state()[1], st)      # 不碰全局 np.random
    r2 = np.random.default_rng(0)
    assert np.array_equal(M, S.draw_masks(r2, 5, 32, 32, 10, 36))


# ── 对抗步 / 损失的 numpy 参照 ─────────────────────────────────────────────
def test_adv_step_order_is_step_project_clamp():
    u0 = np.array([0.5, 0.95, 0.05])
    u = np.array([0.5, 0.95, 0.05])
    out = S.adv_step(u, u0, np.array([1.0, 1.0, -1.0]), step=0.1, eps=0.5)
    assert np.allclose(out, [0.6, 1.0, 0.0])
    out = S.adv_step(np.array([0.9]), np.array([0.5]), np.array([1.0]), step=0.2, eps=0.3)
    assert np.allclose(out, [0.8])                           # 投影先于 clamp


def test_adv_init_mixes_patch_by_mask():
    u = np.full((2, 2), 0.2)
    m = np.array([[1.0, 0.0], [0.5, 0.0]])
    assert np.allclose(S.adv_init(u, m, np.full((2, 2), 0.8)), [[0.8, 0.2], [0.5, 0.2]])


def test_space_affine_input_is_identity_and_pixel_is_mean_std():
    a, b = S.space_affine("input", [0.4, 0.5, 0.6], [0.2, 0.25, 0.3])
    assert np.array_equal(a, np.ones(3, np.float32)) and np.array_equal(b, np.zeros(3, np.float32))
    a, b = S.space_affine("pixel", [0.4, 0.5, 0.6], [0.2, 0.25, 0.3])
    assert np.allclose(a, [0.2, 0.25, 0.3]) and np.allclose(b, [0.4, 0.5, 0.6])


def test_max_margin_hand_values():
    z = np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 5.0]])
    # 样本 0：y=0 → second=1 → 1−3 = −2；样本 1：y=1 → second=5 → 5−2 = 3 → 均值 0.5
    assert S.max_margin(z, [0, 1]) == pytest.approx(0.5)


def test_kl_is_batchmean_kl_clean_to_adv():
    lc = np.log(np.array([[0.5, 0.5], [0.9, 0.1]]))
    la = np.log(np.array([[0.25, 0.75], [0.9, 0.1]]))
    want = (0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)) / 2
    assert S.kl_clean_adv(lc, la) == pytest.approx(want)
    assert S.kl_clean_adv(la, la) == pytest.approx(0.0, abs=1e-12)


def test_mmd_rbf_hand_values():
    x = np.array([[0.0]])
    y = np.array([[1.0]])
    want = 8.0 - 2.0 * sum(math.exp(-0.5 / bw) for bw in S.MMD_BANDWIDTHS)
    assert S.mmd_rbf(x, y) == pytest.approx(want)
    r = np.random.default_rng(1).normal(size=(6, 4))
    assert S.mmd_rbf(r, r) == pytest.approx(0.0, abs=1e-12)


# ── 配置 ──────────────────────────────────────────────────────────────────
def _cfg(**params):
    return {"defense": {"name": "ccs", "layers": ["client", "edge"], "ccs": params}}


def test_defaults_are_the_paper_coefficients_and_the_code_attack():
    p = S.ccs_params(_cfg())
    assert (p["kl"], p["mmd"]) == (0.01, 0.01)                         # 原文（D-109 Q2）
    assert (p["steps"], p["step"], p["eps"], p["radius"], p["n_regions"]) == (15, 0.1, 0.5, 10, 36)
    assert p["space"] == "input" and p["pool"] == "round" and p["clip_norm"] == 10.0


@pytest.mark.parametrize("bad", [{"spaec": "pixel"}, {"space": "rgb"}, {"pool": "cloud"},
                                 {"at": False, "cluster": False}, {"eps": -1}, {"n_regions": 2}])
def test_bad_params_are_rejected(bad):
    with pytest.raises(ValueError):
        S.ccs_params(_cfg(**bad))


def test_settings_row_off_and_on():
    assert S.settings_row({"defense": {"name": "none"}}) == {"ccs": False}
    row = S.settings_row(_cfg(at=False, pool="edge_cache"))
    assert row["ccs"] is True and row["at"] is False and row["pool"] == "edge_cache"
    assert list(row)[1:] == list(S.SETTINGS_KEYS)


# ── 服务端：聚类剔除 ───────────────────────────────────────────────────────
STAT_IDX = [1, 2]


def _ups(vecs, ns=None, seed=0):
    r = np.random.default_rng(seed)
    out = []
    for j, v in enumerate(vecs):
        w = [r.normal(size=(3, 2)).astype(np.float32), np.asarray(v[:2], np.float32),
             np.asarray(v[2:], np.float32), r.normal(size=(2,)).astype(np.float32)]
        out.append(ClientUpdate(w, (ns or [10] * len(vecs))[j], 0.1, 0.0, client_id=100 + j))
    return out


def _defense(**params):
    d = CCSDefense(dict(params))
    d.set_stat_indices(STAT_IDX, label=0)
    return d


def _benign_and_outliers(n_b=8, n_o=2, seed=0):
    r = np.random.default_rng(seed)
    base = np.array([1.0, 2.0, 1.0, 1.0])
    vec = [base + 0.01 * r.normal(size=4) for _ in range(n_b)]
    vec += [np.array([-1.0, 0.5, 3.0, 0.2]) + 0.01 * r.normal(size=4) for _ in range(n_o)]
    return vec


def test_outliers_are_rejected_and_rest_is_fedavg_bitwise(capsys):
    ups = _ups(_benign_and_outliers(), ns=[10, 20, 30, 10, 10, 10, 10, 10, 50, 50])
    d = _defense()
    agg = d.aggregate(ups, ups[0][0])
    d.record_decision(ups)
    assert {108, 109} <= set(d.last_rejected_ids) and d.last_status == "ok"
    want = fedavg_aggregate([ups[j] for j in d.last_admitted])
    assert all(np.array_equal(a, b) for a, b in zip(agg, want))
    line = [ln for ln in capsys.readouterr().out.splitlines() if "[CCS]" in ln][0]
    assert "edge0" in line and "n_pool=10" in line
    assert "rejected=" + "/".join(str(i) for i in d.last_rejected_ids) in line


def test_homogeneous_benign_uploads_still_lose_some_to_noise():
    """官方参数（min_samples=1、min_cluster_size=n//2+2）下，彼此很近的良性端也会有一部分被判成噪声剔除
    —— 这是 CCS 聚类本身的行为（与官方 `hdbscan` 包逐例相同，见下一条），不是这里的 bug。"""
    ups = _ups(_benign_and_outliers(n_b=10, n_o=0))
    d = _defense()
    d.aggregate(ups, ups[0][0])
    d.record_decision(ups)
    assert d.last_status == "ok" and 0 < len(d.last_rejected_ids) < 10


def test_labels_match_the_official_hdbscan_package():
    """官方用 `hdbscan` 包，这里用 sklearn ≥ 1.3 的 HDBSCAN：同一组距离矩阵上标签逐例相同（装了 hdbscan 才跑）。"""
    hdbscan = pytest.importorskip("hdbscan")
    from analysis.bn_cluster import detect
    from sklearn.metrics.pairwise import cosine_similarity
    for seed in range(60):
        r = np.random.default_rng(seed)
        nb, no = [(8, 2), (10, 0), (9, 1), (7, 3), (20, 5)][seed % 5]
        base = r.normal(size=40)
        sc = 10 ** r.uniform(-3, 0)
        X = np.array([base + sc * r.normal(size=40) for _ in range(nb)]
                     + [r.normal(size=40) + base * r.uniform(0, 1) for _ in range(no)])
        D = 1 - cosine_similarity(X)                       # hdbscan_detect.py 原样
        np.fill_diagonal(D, 0)
        D[D < 0] = 0
        lab = hdbscan.HDBSCAN(min_cluster_size=len(X) // 2 + 2, metric="precomputed",
                              allow_single_cluster=True, min_samples=1).fit(D.astype(np.float64)).labels_
        res = detect({i: X[i] for i in range(len(X))})
        if (lab >= 0).any():
            from collections import Counter
            keep = Counter(int(l) for l in lab if l >= 0).most_common(1)[0][0]
            want = [i for i in range(len(X)) if lab[i] != keep]
        else:
            want = []
        assert res["rejected"] == want, seed


def test_cluster_off_is_the_off_arm_fedavg_bitwise():
    ups = _ups(_benign_and_outliers())
    d = _defense(cluster=False)
    agg = d.aggregate(ups, ups[0][0])
    d.record_decision(ups)
    assert d.last_rejected_ids == [] and len(d.last_admitted_ids) == 10 and d.last_status == "off"
    assert all(np.array_equal(a, b) for a, b in zip(agg, fedavg_aggregate(ups)))


def test_degenerate_admits_all_and_is_fedavg_bitwise():
    ups = _ups(_benign_and_outliers(n_b=1, n_o=1))              # n = 2 → min_cluster_size 3 > 2
    d = _defense()
    agg = d.aggregate(ups, ups[0][0])
    d.record_decision(ups)
    assert d.last_status == "degenerate" and d.last_rejected_ids == []
    assert all(np.array_equal(a, b) for a, b in zip(agg, fedavg_aggregate(ups)))


def test_edge_cache_pools_earlier_uploads_and_judges_only_this_rounds():
    vec = _benign_and_outliers(n_b=8, n_o=2)
    d = _defense(pool="edge_cache")
    first = _ups(vec[:3])                       # 100, 101, 102：池 3 < 5 → 退化
    d.aggregate(first, first[0][0])
    assert d.last_status == "degenerate"
    for k in range(3, 8, 3):                    # 再喂几批良性端进缓存（id 不同）
        b = _ups(vec[k:k + 3])
        for j, u in enumerate(b):
            b[j] = ClientUpdate(u[0], u[1], u[2], u[3], client_id=200 + k + j)
        d.aggregate(b, b[0][0])
    out = _ups(vec[8:])                         # 本轮 = 两个离群端
    for j, u in enumerate(out):
        out[j] = ClientUpdate(u[0], u[1], u[2], u[3], client_id=900 + j)
    ref = [np.full_like(w, 7.0) for w in out[0][0]]
    agg = d.aggregate(out, ref)
    d.record_decision(out)
    assert d.last_status == "ok" and d.last_rejected_ids == [900, 901]
    assert all(np.array_equal(a, b) for a, b in zip(agg, ref))      # 本轮全剔 → 本层模型不变


def test_missing_stat_indices_is_an_error():
    d = CCSDefense({})
    ups = _ups(_benign_and_outliers())
    with pytest.raises(RuntimeError, match="统计量索引"):
        d.aggregate(ups, ups[0][0])


def test_defense_registry_and_lazy_mixin_ref():
    from defense import defense_class
    cls = defense_class("ccs")
    assert cls is CCSDefense and cls.layers == {"client", "edge"}
    assert cls.client_mixin_ref == "client.ccs_mixin.CCSClientMixin"
