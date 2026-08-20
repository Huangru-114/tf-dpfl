"""
defense/test_defenses_offline.py  –  纯 numpy 离线单测（无 TF/GPU 依赖）

运行：cd fedavg && python -m defense.test_defenses_offline

每个防御用构造性用例验证其核心行为：
  - trimmed_mean / median：含离群值时去极值/取中位的结果正确
  - multi_krum：剔除方向/距离离群的 client
  - flame：离群更新被聚类过滤 + 范数裁剪，聚合远离离群
  - dnc：奇异方向离群点被标记剔除
"""

import numpy as np

from .factory import create_defense


def _upd(vec, n=10):
    """构造一条 client 更新：weights=[1D array]，样本数 n。"""
    return ([np.asarray(vec, dtype=np.float32)], n, 0.0, 0.0)


def _cfg(name, **params):
    return {"defense": {"name": name, "params": params},
            "federation": {"n_edges": 2}, "backdoor": {"n_malicious": 4}}


def test_trimmed_mean():
    # 4 个良性 ~1.0，1 个离群 100；trim_ratio=0.25 → 每端去 floor(0.25*5)=1 个。
    ups = [_upd([1.0, 1.0]), _upd([1.1, 0.9]), _upd([0.9, 1.1]),
           _upd([1.0, 1.0]), _upd([100.0, 100.0])]
    d = create_defense(_cfg("trimmed_mean", trim_ratio=0.25))
    out = d.aggregate(ups, [np.zeros(2, np.float32)])[0]
    assert np.all(out < 2.0), f"trimmed_mean 未去除离群: {out}"
    print(f"  [trimmed_mean] out={out}  ✓ (离群 100 被去除)")


def test_median():
    ups = [_upd([1.0]), _upd([2.0]), _upd([3.0]), _upd([4.0]), _upd([500.0])]
    d = create_defense(_cfg("median"))
    out = d.aggregate(ups, [np.zeros(1, np.float32)])[0]
    assert abs(out[0] - 3.0) < 1e-5, f"median 应为 3.0，得 {out}"
    print(f"  [median] out={out}  ✓ (中位数=3.0，不受 500 影响)")


def test_multi_krum():
    # 3 个良性聚集在 0 附近，1 个离群在 100；f=1 应排除离群。
    rng = np.random.default_rng(0)
    benign = [_upd(rng.normal(0, 0.1, 5)) for _ in range(5)]
    outlier = _upd(np.full(5, 100.0))
    ups = benign + [outlier]
    d = create_defense(_cfg("multi_krum", num_malicious=1))
    out = d.aggregate(ups, [np.zeros(5, np.float32)])[0]
    assert np.all(np.abs(out) < 5.0), f"multi_krum 未排除离群: {out}"
    print(f"  [multi_krum] ‖out‖={np.linalg.norm(out):.3f}  ✓ (离群被排除)")


def test_flame():
    # 5 个方向一致的良性小更新 + 1 个反向大更新（离群）。ref=0。
    rng = np.random.default_rng(1)
    benign = [_upd(rng.normal(1.0, 0.05, 8)) for _ in range(5)]
    outlier = _upd(np.full(8, -50.0))
    ups = benign + [outlier]
    d = create_defense(_cfg("flame", noise_scale=0.0))  # 关噪声便于断言
    out = d.aggregate(ups, [np.zeros(8, np.float32)])[0]
    # 良性方向为正；离群被过滤后聚合应为正、且远小于 50。
    assert np.mean(out) > 0 and np.all(np.abs(out) < 5.0), f"flame 结果异常: {out}"
    print(f"  [flame] mean={np.mean(out):.3f}  ✓ (反向离群被过滤+裁剪)")


def test_dnc():
    # d=100：10 个高斯点 + 1 个前 10 维=100 的离群点。应被标记剔除。
    rng = np.random.default_rng(2)
    benign = [_upd(rng.normal(0, 1.0, 100)) for _ in range(10)]
    spike = np.zeros(100, np.float32); spike[:10] = 100.0
    outlier = _upd(spike)
    ups = benign + [outlier]
    d = create_defense(_cfg("dnc", num_malicious=1, num_projections=5, subspace_dim=100))
    out = d.aggregate(ups, [np.zeros(100, np.float32)])[0]
    assert np.max(np.abs(out)) < 10.0, f"dnc 未剔除尖峰离群: max={np.max(np.abs(out))}"
    print(f"  [dnc] max|out|={np.max(np.abs(out)):.3f}  ✓ (尖峰离群被剔除)")


def test_none_fallback():
    # name=none → create_defense 返回 None（robust_mean 会回退普通聚合）。
    assert create_defense(_cfg("none")) is None
    print("  [none] create_defense → None  ✓ (回退普通加权平均)")


if __name__ == "__main__":
    print("=== 防御离线单测（纯 numpy）===")
    test_none_fallback()
    test_trimmed_mean()
    test_median()
    test_multi_krum()
    test_flame()
    test_dnc()
    print("=== ALL DEFENSE TESTS PASSED ===")
