"""
hardening/online.py  —  阶段三 SA1：在线的「上传前加固」（PLAN §3.3 的 A-pre 臂，D-108）

每个云轮，交错调度的全部 edge 轮跑完、各 edge 把模型交给 cloud 之前（`CloudServer.collect_and_aggregate`），
每个 edge 用自己的干净集（S3 的 `clean_indices`，main.py 挂成 `edge.clean_x / clean_y`）按 `defense.edge_harden`
指定的 P0 配置（`p0_configs.yaml` 展开后的 id）调用 **P0 用过的同一个** `hardening.core.EdgeHardener` 洗一遍，
cloud 聚合的是洗过的权重。edge 自己的模型不变（随后被 cloud 的广播覆盖，与 P0 的反事实一致）。

  · lr：`spec.harden_lr(cfg, spec.lr_at(config, 本云轮))` —— 与 P0 离线对快照 t 的取法相同（D-107 的倍数）；SAU 用官方 Adam。
  · 随机数：`default_rng([seed, 0x5A1, 云轮, edge_id])`，不碰全局 np.random / tf.random；EdgeHardener 的构造与每次加固
    都包在 random.getstate()/setstate() 里（F-078）。
  · 发散（NaN / Inf）：这一次不加固、上传原权重，并在 `[Harden]` 里记 `diverged_at`（在线不能把 NaN 交给 cloud；D-108）。
  · `[Harden]` 每次加固一行；另外每个云轮一行 `[HardenAgg] … | unhardened=<checksum>` = 用**没洗过的**上传物做同一个
    FedAvg 的全局权重短哈希 —— 第 1 云轮它必须等于对照（SNAP）的 `[Checksum] Round 1`（P1 的闸，D-108）。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from hardening import spec

TAG = 0x5A1
GRID_PATH = Path(__file__).resolve().parents[2] / "experiments/defense/edge-native/p0_configs.yaml"


def load_grid(path=GRID_PATH) -> tuple:
    import yaml
    g = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return g, {c["id"]: c for c in spec.expand_grid(g["grid"])}


class OnlineHardener:
    def __init__(self, config: dict, template_model, *, grid_path=GRID_PATH):
        from hardening.core import EdgeHardener          # TF
        self.config = config
        self.cfg_id = str(config["defense"]["edge_harden"])
        g, by_id = load_grid(grid_path)
        if self.cfg_id not in by_id:
            raise ValueError(f"defense.edge_harden={self.cfg_id!r} 不在 {grid_path} 展开的配置里")
        self.cfg = by_id[self.cfg_id]
        self.seed = int(config.get("seed", 42))
        # 草稿模型在**第一次加固时**才建：建模型可能推进 TF / Keras 的随机状态，放在第 1 云轮的训练之后，
        # 才能保证「第一次加固之前的训练与对照逐位相同」（P1 的闸，D-108）。
        self._make = lambda: EdgeHardener(template_model, config, probe=g.get("probe"), kstar=g.get("kstar"),
                                          batch=int(g.get("batch", 32)))
        self.hardener = None

    def harden_upload(self, edge, weights: list, round_idx: int) -> tuple:
        """返回 (上传给 cloud 的权重, info)。发散 → 原权重 + info["diverged_at"]。"""
        x, y = getattr(edge, "clean_x", None), getattr(edge, "clean_y", None)
        if x is None or y is None or len(y) == 0:
            raise ValueError(f"defense.edge_harden：edge {edge.edge_id} 没有干净集（clean_x / clean_y）")
        rng = np.random.default_rng([self.seed, TAG, int(round_idx), int(edge.edge_id)])
        lr = spec.harden_lr(self.cfg, spec.lr_at(self.config, int(round_idx)))
        t0 = time.perf_counter()
        if self.hardener is None:
            self.hardener = self._make()
        new, info = self.hardener.harden(weights, x, y, self.cfg, rng=rng, lr=lr)
        info["seconds_total"] = round(time.perf_counter() - t0, 3)
        if info.get("diverged_at") is not None:
            return weights, info
        return new, info
