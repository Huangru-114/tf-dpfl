"""
tests/test_trigger_norms.py  —  fedavg/analysis/trigger_norms.py（检查 1）。
纯 numpy 部分本地跑；TF 部分（importorskip）用真模型 + 真生成器造一份假快照、把 CIFAR 换成随机图，端到端跑一遍。
"""

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))

from analysis.trigger_norms import perturbation_stats             # noqa: E402

CFG = ROOT / "experiments/attack/hfl-mechanism/configs/G8__a__s42.yaml"


def test_stats_in_pixel_space_by_hand():
    std = np.array([0.5, 0.25, 1.0])
    mean = np.array([0.5, 0.5, 0.5])
    x = np.zeros((2, 2, 2, 3))                                      # 像素 0.5
    xi = np.zeros_like(x)
    xi[0, 0, 0, :] = [0.02, 0.04, 0.01]                             # 像素空间 0.01 / 0.01 / 0.01
    de = np.zeros_like(x)
    de[1] = 0.04 / std                                              # 每个像素 +0.04
    s = perturbation_stats(x, xi, de, mean, std)
    assert s["xi"]["linf"]["max"] == pytest.approx(0.01) and s["xi"]["linf"]["min"] == 0
    assert s["xi"]["l2"]["max"] == pytest.approx((3 * 0.01 ** 2) ** 0.5)
    assert s["delta"]["linf"]["max"] == pytest.approx(0.04)
    assert s["delta"]["l2"]["max"] == pytest.approx((12 * 0.04 ** 2) ** 0.5)
    assert s["total"]["linf"]["p50"] == pytest.approx(0.025)
    assert s["out_of_range"]["frac_pixels"] == 0 and s["n"] == 2 and s["dims"] == 12


def test_out_of_range_counts_overshoot():
    x = np.full((1, 1, 2, 1), 0.99)
    s = perturbation_stats(x, np.zeros_like(x), np.full_like(x, 0.02), [0.0], [1.0])
    assert s["out_of_range"]["frac_pixels"] == 1.0 and s["out_of_range"]["max_overshoot"] == pytest.approx(0.01)


def test_end_to_end_on_a_fake_snapshot(tmp_path, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    monkeypatch.chdir(ROOT / "fedavg")
    import main as M
    from analysis import trigger_norms as T
    from models.autoencoder import build_generator
    from models.cnn import build_model

    config = M.load_config(str(CFG))
    model = build_model(input_shape=(32, 32, 3), num_classes=10, arch=config["model"]["arch"],
                        rep_dim=int(config["model"].get("rep_dim", 64)))
    gen = build_generator(config, channels=3)
    arrays = {f"edge0_{i:03d}": w for i, w in enumerate(model.get_weights())}
    arrays["client16_idx"] = np.array([len(model.get_weights()) - 1], np.int32)
    arrays["client16_000"] = model.get_weights()[-1]
    arrays.update({f"gen_{i:03d}": w for i, w in enumerate(gen.get_weights())})
    arrays["meta_json"] = np.array(json.dumps({"eval_attacker": 16, "client_ids": [16], "client_edge": [0],
                                               "has_generator": True}))
    (tmp_path / "d").mkdir()
    np.savez(tmp_path / "d" / "s.npz", **arrays)
    (tmp_path / "m.json").write_text(json.dumps({"dumps": {"snapshots": [{"round": 30, "path": "d/s.npz"}]}}))
    from data.pixel_space import pixel_stats
    mean, std = pixel_stats(config)
    rng = np.random.default_rng(0)
    xs = ((rng.random((40, 32, 32, 3)) - mean) / std).astype(np.float32)   # 合法像素 [0,1] 标准化后
    ys = rng.integers(0, 10, 40)
    monkeypatch.setattr(M, "load_cifar10", lambda cfg: (None, None, None, None, xs, ys))
    res = T.run(str(CFG), tmp_path / "m.json", tmp_path, 30, n=20)
    sig = float(config["backdoor"]["badpfl_sigma"])
    eps = float(config["backdoor"]["badpfl_epsilon"])
    assert res["n"] == 20 and res["eval_attacker"] == 16
    assert res["xi"]["linf"]["max"] <= sig + 1e-6
    assert res["delta"]["linf"]["max"] <= eps + 1e-6
    assert res["total"]["linf"]["max"] <= sig + eps + 1e-6
    assert res["out_of_range"]["x_plus_xi_out"] == 0.0          # ξ 之后裁剪过；δ 之后才不裁
