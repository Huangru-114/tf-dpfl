"""
analysis/client_profile.py  —  逐客户端的训练集标签直方图（检查 4 的协变量；只建客户端、不训练、不读存盘）

为什么要它：G8 / G8F 用旧 `noniid` 划分（Dirichlet，用全局 np.random），客户端训练集里目标类的占比
只能按该 run 的配置**原样重建划分**才拿得到；metrics.json 里没有（`client_final.yt_clean` 是「干净留出样本被判成 y_t 的比例」，不是训练集占比）。

用法（cwd = fedavg，经 cluster_env.sh 的容器；**CPU 即可**，登录节点可跑：apptainer exec 不带 --nv）：
    python3 -m analysis.client_profile \\
        --config ../experiments/attack/hfl-mechanism/configs/G8__a__s42.yaml \\
        --out ../experiments/attack/hfl-mechanism/analysis/client_profile/G8__a__s42.json
回传只有 JSON（每客户端：edge、训练集大小、10 个类的计数、是否恶意），约 10 KB，不含任何张量。
G8F 与 G8 同 seed 的划分相同（两边 client_final.n 逐个相同）；仍建议各跑一次，由 `harness/tail_clients.py` 核对。

`label_profile` 是纯 numpy（不 import TF），本地测试直接测它；`run` 才 import main（TF）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def label_profile(clients, labels, n_classes: int) -> dict:
    """{client_id: {edge, n_train, counts[K], malicious}}。clients 要有 client_id / assigned_edge / is_malicious /
    _train_src = (images, labels, indices)；`labels` 缺省时用 _train_src 里的全局标签数组。"""
    out = {}
    for c in sorted(clients, key=lambda c: int(c.client_id)):
        _, lab, idx = c._train_src
        lab = np.asarray(labels if labels is not None else lab).reshape(-1)
        cnt = np.bincount(lab[np.asarray(idx, dtype=np.int64)], minlength=n_classes)
        out[str(int(c.client_id))] = {"edge": int(getattr(c, "assigned_edge", -1)), "n_train": int(len(idx)),
                                      "counts": [int(v) for v in cnt[:n_classes]],
                                      "malicious": bool(getattr(c, "is_malicious", False))}
    return out


def run(config_path: str) -> dict:
    import main as M
    config = M.load_config(config_path)
    M.set_seed(config.get("seed", 42))
    name = config["data"].get("dataset", "cifar10").lower()
    if name != "cifar10":
        raise ValueError("只做了 cifar10")
    _, _, x_train, y_train, x_test, y_test = M.load_cifar10(config)
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    from models.cnn import build_model
    model = build_model(input_shape=(config["data"]["img_size"],) * 2 + (3,),
                        num_classes=config["data"]["num_classes"], arch=config["model"]["arch"],
                        rep_dim=int(config["model"].get("rep_dim", 64)))
    clients, baked, _ = M.build_clients(x_all, y_all, model, config, s3_out={})
    M.build_edge_servers(clients, model, config, precomputed_assignments=baked)   # 写 assigned_edge
    K = int(config["data"]["num_classes"])
    return {"config": str(config_path), "seed": int(config.get("seed", 42)),
            "target": int(config["backdoor"].get("target_label", 0)), "n_classes": K,
            "clients": label_profile(clients, y_all, K)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a.config)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[client_profile] {a.out}  ({len(res['clients'])} clients)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
