"""
analysis/trigger_norms.py  —  检查 1：Bad-PFL 触发器 x_trig − x 的实际范数（G8 快照；只读、不训练）

代码层面（`client/client_badpfl.py`，P2 开关 badpfl_xi=pgd / badpfl_generator=official）：
  ξ  单步 PGD：x₀ = clip(x + u, lo, hi)，u ~ U(−σ, σ)；x₁ = x₀ + σ·sign(∇CE(F(x₀), y))；
     adv = clip(x + clip(x₁ − x, −σ, σ), lo, hi)；ξ = adv − x  →  ‖ξ‖∞ ≤ σ（像素空间），且 x + ξ ∈ [0, 1]
  δ  = tanh(G(x_[0,1])) · ε                                     →  ‖δ‖∞ ≤ ε（像素空间）
  x_trig = x + ξ + δ，**之后不再裁剪**（对齐官方 fba.py:55）  →  ‖ξ + δ‖∞ ≤ σ + ε，x_trig 可越出 [0, 1]
  P2 / G8：σ = ε = 0.01569 ≈ 4/255（base.yaml）。模型输入是逐通道标准化图，σ_in = σ / std_c（data/pixel_space.py）。

本脚本把这些上界换成**实测分布**：载入 G8 第 R 云轮的快照（缺省 30 = 攻击最后一轮），
  · 攻击者 fresh-PM = 快照里攻击者所在 edge 的模型 + 该攻击者的 private_state（与 CloudServer.pm_model 同一个组合）；
  · 生成器 = 快照里评估攻击者的生成器权重；
  · ξ / δ 用 `BadPFLMixin.eval_xi` / `eval_delta` **原样**算（同一份代码，评估口径：F 推理模式、G 按 32 张分块的 batch 统计）；
对 N 张官方 test 图（缺省 500，只取非目标类，标签 = 真值）在像素空间统计每张图的 L∞ / L2（3072 维）与越界像素比例。

用法（cwd = fedavg，经 cluster_env.sh 的容器；**CPU 即可**，登录节点：apptainer exec 不带 --nv）：
    python3 -m analysis.trigger_norms --config ../experiments/attack/hfl-mechanism/configs/G8__a__s42.yaml \\
        --metrics ../experiments/attack/hfl-mechanism/results/P2/G8/G8__a__s42.metrics.json \\
        --dumps-root ../../tfdpfl-dumps --round 30 --out /tmp/trig_s42.json
回传只有 JSON（分位数），不回传张量。纯算术 `perturbation_stats` 不 import TF，本地测试直接测它。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

QUANTS = (50, 99)


def _summ(v: np.ndarray) -> dict:
    v = np.asarray(v, np.float64)
    return {"mean": float(v.mean()), **{f"p{q}": float(np.percentile(v, q)) for q in QUANTS},
            "max": float(v.max()), "min": float(v.min())}


def perturbation_stats(x_in, xi_in, delta_in, mean, std) -> dict:
    """输入空间的 x、ξ、δ（N,H,W,C）→ 像素空间每张图的 L∞ / L2 分布 + x_trig 越出 [0,1] 的像素比例。"""
    std = np.asarray(std, np.float64).reshape(1, 1, 1, -1)
    mean = np.asarray(mean, np.float64).reshape(1, 1, 1, -1)
    xi = np.asarray(xi_in, np.float64) * std
    de = np.asarray(delta_in, np.float64) * std
    x = np.asarray(x_in, np.float64) * std + mean
    out = {}
    for name, d in (("xi", xi), ("delta", de), ("total", xi + de)):
        flat = d.reshape(len(d), -1)
        out[name] = {"linf": _summ(np.abs(flat).max(axis=1)), "l2": _summ(np.sqrt((flat ** 2).sum(axis=1)))}
    trig = x + xi + de
    out["out_of_range"] = {"frac_pixels": float(((trig < 0) | (trig > 1)).mean()),
                           "max_overshoot": float(max(0.0, (-trig).max(), (trig - 1).max())),
                           "x_plus_xi_out": float((((x + xi) < -1e-6) | ((x + xi) > 1 + 1e-6)).mean())}
    out["n"] = int(len(x))
    out["dims"] = int(np.prod(x.shape[1:]))
    return out


def run(config_path, metrics_path, dumps_root, round_idx=30, n=500):
    import tensorflow as tf
    import main as M
    from client.client_badpfl import BadPFLMixin
    from data.pixel_space import pixel_stats
    from models.autoencoder import build_generator
    from models.cnn import build_model
    from utils.pm import compose_pm

    config = M.load_config(config_path)
    M.set_seed(config.get("seed", 42))
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    snap = next((s for s in m["dumps"]["snapshots"] if int(s["round"]) == int(round_idx)), None)
    if snap is None:
        raise SystemExit(f"[trig] metrics.json 里没有第 {round_idx} 轮的快照")
    z = np.load(Path(dumps_root) / snap["path"])
    meta = json.loads(str(z["meta_json"]))
    if not meta.get("has_generator"):
        raise SystemExit("[trig] 快照里没有生成器（has_generator=false）")
    att = int(meta["eval_attacker"])
    edge = dict(zip(meta["client_ids"], meta["client_edge"]))[att]

    model = build_model(input_shape=(config["data"]["img_size"],) * 2 + (3,),
                        num_classes=config["data"]["num_classes"], arch=config["model"]["arch"],
                        rep_dim=int(config["model"].get("rep_dim", 64)))
    n_w = len(model.get_weights())
    edge_w = [z[f"edge{edge}_{i:03d}"] for i in range(n_w)]
    idx = [int(i) for i in z[f"client{att}_idx"]]
    priv = [z[f"client{att}_{i:03d}"] for i in range(len(idx))]
    model.set_weights(compose_pm(edge_w, priv, idx) if idx else edge_w)

    class _Stub:                                       # BadPFLMixin 只用到 config / rng / loss_fn
        def __init__(self, client_id, dataset, model, config, n_samples=None):
            self.client_id, self.model, self.config = client_id, model, config
            self.rng = np.random.default_rng([int(config.get("seed", 42)), int(client_id)])
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    class _Probe(BadPFLMixin, _Stub):
        pass

    probe = _Probe(att, None, model, config)
    gen = build_generator(config, channels=3)
    gen_w = [z[k] for k in sorted(k for k in z.files if k.startswith("gen_"))]
    gen.set_weights(gen_w)
    probe._atk_generator = gen

    _, _, _, _, x_test, y_test = M.load_cifar10(config)
    y_test = np.asarray(y_test).reshape(-1)
    target = int(config["backdoor"].get("target_label", 0))
    keep = np.flatnonzero(y_test != target)[:n]
    x, y = x_test[keep].astype(np.float32), y_test[keep]
    # 与 BackdoorCloudServer._eval_rng(轮, 列 0) 同一种键控（主列）；不碰任何全局 RNG
    rng = np.random.default_rng([int(config.get("seed", 42)), 0xE7A1, int(round_idx), 0])
    xi = probe.eval_xi(model, x, y, rng=rng)
    delta = probe.eval_delta(x)
    mean, std = pixel_stats(config)
    pred = np.argmax(model(x + xi + delta, training=False).numpy(), axis=1)
    return {"config": str(config_path), "seed": int(config.get("seed", 42)), "round": int(round_idx),
            "eval_attacker": att, "attacker_edge": int(edge),
            "eps_pixel": float(config["backdoor"].get("badpfl_epsilon")),
            "sigma_pixel": float(config["backdoor"].get("badpfl_sigma")),
            "asr_on_attacker_pm": float(np.mean(pred == target)),
            **perturbation_stats(x, xi, delta, mean, std)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--dumps-root", required=True)
    ap.add_argument("--round", type=int, default=30)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a.config, a.metrics, a.dumps_root, a.round, a.n)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[trig] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
