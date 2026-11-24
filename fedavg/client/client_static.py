"""
client/client_static.py  –  静态触发器逐批投毒的攻击 mixin（`malicious_strategy: static`，D-141）

BadNets（右下角 3×3 白块）/ Blended（hello_kitty 以 α 混合）/ DBA（每个攻击者一块局部像素条，
评估用 4 块合起来的全局触发器）在 P2 的每轮重洗取数路径（data.batch_pipeline = per_epoch）下的实现。
旧的 `vanilla` 策略（开跑前一次性替换数据集）保持原样、只能配旧取数路径；两者在 config_validate 里互斥。

  - **是 mixin，不是 Client 子类**（CLAUDE.md 陷阱 #1）：与任意 PFL 方法类组合，恶意端仍跑该方法的
    local_train；投毒挂在 `on_batch` 钩子上（FedRep 在 P2 下只有 body 阶段调它，与 Bad-PFL 对照相同）。
  - 用户 11-24 拍板的三条语义：
      · 每批按 `backdoor.poison_sampling`（P2 = 每张独立以 ρ 的概率）抽，读 Bad-PFL 同一个开关；
      · 钩子在 epoch_batches 做完增强之后才被调 → 先增强、再加触发器（恶意端的干净样本也增强）；
      · 收集客户端更新强制串行（main.py，与 Bad-PFL 同列）。
  - 闸门是基类每轮刷新的 `_attack_active`（攻击时间窗 attack_start/stop_round 因此自动可用）；
    闸门关时直接返回，**不消耗**投毒随机数 `self.rng`。
  - 触发器由 main.build_clients 注入（`set_static_trigger`）：DBA = 本端的局部触发器，其余 = build_trigger 的
    统一触发器 —— 与旧路径 `dba_triggers.get(i, bd_trigger)` 同一个表达式。没注入就投毒 → 直接抛错
    （不许恶意端静默变成良性端）。
  - 评估侧：静态触发器不依赖模型，没有 `eval_xi` → main.select_eval_attacker 返回 None → 服务器退回静态
    触发器（backdoor_server._light_asr / _backdoor_eval 的 `fixed` 判断）。
  - 自描述：每次本地训练结束时（on_upload，闸门开着才打）一行 `[Poison]`：本次被投毒张数 / 经过钩子的张数 / 批数。
    这是「恶意端真的投了毒」的直接证据（collect_metrics → poison_counts）。
"""

import numpy as np
import tensorflow as tf

from alignment import get_switch
from attack.poison_mask import poison_mask
from attack.static_poison import apply_static_poison
from utils.kvline import format_kv


class StaticPoisonMixin:

    def __init__(self, client_id, dataset, model, config, n_samples=None):
        super().__init__(client_id, dataset, model, config, n_samples)
        bd = (config or {}).get("backdoor", {}) or {}
        self._atk_target_label = int(bd.get("target_label", 9))
        self._atk_poison_ratio = float(bd.get("poison_ratio", 0.5))
        self._atk_sampling = get_switch(config, "backdoor.poison_sampling")
        self._static_trigger = None
        self._static_n_poisoned = 0
        self._static_n_seen = 0
        self._static_n_batches = 0

    def set_static_trigger(self, trigger_fn):
        """投毒侧触发器：numpy (k,H,W,C) → (k,H,W,C)，作用于模型输入空间（attack/triggers.py）。"""
        self._static_trigger = trigger_fn

    # ── 钩子 ──────────────────────────────────────────────────────────────
    def on_round_start(self, round_idx: int):
        super().on_round_start(round_idx)
        self._static_round = int(round_idx)
        self._static_n_poisoned = 0
        self._static_n_seen = 0
        self._static_n_batches = 0
        if self._attack_active and self._static_trigger is None:
            raise RuntimeError(
                f"Client {self.client_id}: malicious_strategy=static 但没有注入触发器"
                f"（main.build_clients 应调 set_static_trigger）—— 不投毒就等于静默变成良性端。")

    def on_batch(self, x, y):
        """按 poison_sampling 逐批抽样，被选中的样本换成 trigger(x) + 目标标签（eager，在增强之后）。"""
        x, y = super().on_batch(x, y)
        if not self._attack_active:
            return x, y
        n = int(x.shape[0])
        self._static_n_seen += n
        self._static_n_batches += 1
        # 投毒样本的选取走客户端自己的 seeded RNG（self.rng，见 client_base）—— 与 Bad-PFL 同一条流、同一个函数。
        mask = poison_mask(self.rng, n, self._atk_poison_ratio, self._atk_sampling)
        if not mask.any():
            return x, y                       # 一个都没选中：原样返回（同 Bad-PFL）
        y_np = np.asarray(y)
        xp, yp, k = apply_static_poison(np.asarray(x), y_np, mask,
                                        self._static_trigger, self._atk_target_label)
        self._static_n_poisoned += k
        return tf.convert_to_tensor(xp), tf.convert_to_tensor(yp, dtype=tf.as_dtype(y_np.dtype))

    def on_upload(self, upload: list, round_idx: int):
        """纯函数：上传物原样返回。闸门开着时打一行 [Poison]（本次本地训练的投毒计数）。"""
        upload = super().on_upload(upload, round_idx)
        if self._attack_active:
            print(format_kv("[Poison]", {
                "client": int(self.client_id),
                "n_poisoned": int(self._static_n_poisoned),
                "n_seen": int(self._static_n_seen),
                "n_batches": int(self._static_n_batches),
            }, round_idx=round_idx))
        return upload
