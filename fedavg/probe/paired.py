"""
probe/paired.py  —  paired counterfactual runner。

对同一个 checkpoint θ_t，从**完全相同的起点**跑三条轨迹：

    clean_A   order_seed = s_a,  attack = False
    poison    order_seed = s_a,  attack = True     ← 与 clean_A 严格配对
    clean_B   order_seed = s_b,  attack = False    ← clean-vs-clean 随机性对照

    Δθ_BD          = Δθ_poison − Δθ_clean_A     （投毒**额外**诱导出的位移）
    Δθ_stochastic  = Δθ_clean_A − Δθ_clean_B    （纯 SGD 随机性的量级）

**Δθ_BD 只有和 Δθ_stochastic 比较之后才有意义。** 单看 ‖Δθ_BD‖ 大，可能只是
local SGD 的噪声；这正是用户 §四 要求加 clean-vs-clean control 的理由。

投毒开关 = `client.is_malicious`
────────────────────────────────
Bad-PFL 的全部投毒在两个钩子里，两处都以 `if not self.is_malicious: return` 开头
（client/client_badpfl.py:123 on_round_start / :132 on_batch）。Neurotoxin（:100/:109）
与 CerP（:170/:178/:202）同样。所以这个开关**天然攻击无关**，IBA 移植后零改动接入。

**不要为 clean twin 新建一个良性 client 对象** —— 那会重新引入陷阱 #1 那一类的
「恶意端跑的是另一个类」污染。同一个对象、同一条 MRO，只翻一个 bool。

已知不覆盖的一类攻击：vanilla 策略（badnet/blended/dba）把投毒**烘焙进了数据集**
（main.py 的 build_poisoned_dataset），不经任何钩子，`is_malicious` 对它们不是开关。
这类攻击要做 paired 分析需要另配一份干净数据集，本模块显式不支持（见 assert）。
"""
import numpy as np

from .determinism import ClientSnapshot, FrozenOrder


def _copy(ws):
    return [np.array(w, copy=True) for w in ws]


def single_trajectory(client, theta_t, round_idx: int, *, order_seed: int,
                      attack: bool, global_weights=None) -> dict:
    """
    从 θ_t 跑一次 local_train，返回轨迹的三个权重快照。全程被 ClientSnapshot 隔离：
    退出时客户端与共享攻击者状态（含 Bad-PFL 的共享 generator）逐项还原。

    Args:
        theta_t:        本轮广播给客户端的 edge 权重（= checkpoint 里的 edge 模型）
        global_weights: 同时转发的 cloud 权重；None 时用 theta_t（对齐
                        EdgeServerBase.broadcast_to_clients 的 `gw` 缺省行为）
    Returns:
        {"start": [...], "upload": [...], "personal": [...], "loss": float}
        start 是 set_weights 之后**模型的真实起点** —— FedRep 下它是
        [edge backbone, 本客户端私有 head]，不等于 theta_t 本身。
    """
    gw = global_weights if global_weights is not None else theta_t
    with ClientSnapshot(client):
        with FrozenOrder(client, order_seed):
            client.is_malicious = bool(attack)
            client.set_weights(global_weights=gw, edge_weights=theta_t)
            if hasattr(client, "apply_round_lr"):
                client.apply_round_lr(round_idx)
            start = _copy(client.model.get_weights())
            upload, _n, loss, _t = client.local_train(round_idx)
            return {"start": start,
                    "upload": _copy(upload),
                    "personal": _copy(client.model.get_weights()),
                    "loss": float(loss)}


def probe_client(client, theta_t, round_idx: int, *, seed_a: int, seed_b: int,
                 global_weights=None) -> dict:
    """
    三条轨迹 + 四个 Δ（upload / personal 两个权重空间各一套）。

    为什么两个权重空间都要：FedRep 下 `upload`（backbone，交聚合、决定后门怎么
    **传播**）与 `self.model` 终态（个性化模型，local ASR 与 PM acc 在它上面测）
    不是同一个东西（client/hier_fedrep.py:151 vs :163）。只测一个会答错 Q1。

    Returns:
        {
          "round": int, "client_id": int, "was_malicious": bool,
          "seed_a": int, "seed_b": int,
          "loss": {"clean_a":…, "poison":…, "clean_b":…},
          "deltas": {
             "upload":   {"clean": …, "poison": …, "bd": …, "stochastic": …},
             "personal": {…同上…},
          },
        }
        每个 Δ 是与 `get_weights()` 同形的 ndarray 列表（尚未展平）。
    """
    was_mal = bool(getattr(client, "is_malicious", False))

    a = single_trajectory(client, theta_t, round_idx, order_seed=seed_a,
                          attack=False, global_weights=global_weights)
    p = single_trajectory(client, theta_t, round_idx, order_seed=seed_a,
                          attack=True, global_weights=global_weights)
    b = single_trajectory(client, theta_t, round_idx, order_seed=seed_b,
                          attack=False, global_weights=global_weights)

    deltas = {}
    for space in ("upload", "personal"):
        d_clean = [x - s for x, s in zip(a[space], a["start"])]
        d_pois = [x - s for x, s in zip(p[space], p["start"])]
        d_cb = [x - s for x, s in zip(b[space], b["start"])]
        deltas[space] = {
            "clean": d_clean,
            "poison": d_pois,
            "bd": [q - c for q, c in zip(d_pois, d_clean)],
            "stochastic": [c - c2 for c, c2 in zip(d_clean, d_cb)],
        }

    return {
        "round": int(round_idx),
        "client_id": int(client.client_id),
        "was_malicious": was_mal,
        "seed_a": int(seed_a), "seed_b": int(seed_b),
        "loss": {"clean_a": a["loss"], "poison": p["loss"], "clean_b": b["loss"]},
        "deltas": deltas,
    }


def assert_attack_is_hook_gated(config: dict):
    """
    拒绝在「投毒烘焙进数据集」的攻击上跑本模块。

    vanilla 策略（badnet / blended / dba）的投毒样本由 build_poisoned_dataset 直接写进
    client.dataset，不经 on_batch。翻 `is_malicious` 对它们**毫无作用** → 两个 twin
    逐字节相同 → Δθ_BD 恒为 0。那个 0 与"后门无位移"看起来一模一样，是最坏的一类
    假阴性，所以在入口处直接拦掉。
    """
    strategy = str((config or {}).get("backdoor", {})
                   .get("malicious_strategy", "vanilla")).lower()
    hook_gated = {"badpfl", "neurotoxin", "cerp"}
    if strategy not in hook_gated:
        raise ValueError(
            f"malicious_strategy={strategy!r} 的投毒烘焙在数据集里，不经 on_batch 钩子，"
            f"`is_malicious` 对它不是开关 → Δθ_BD 会恒为 0（假阴性）。"
            f"本探针只支持钩子门控的攻击: {sorted(hook_gated)}。")


def probe_edge_round(*_args, **_kwargs):
    """
    **未实现**（HFL 传播会话的入口，本会话只留接口）。

    设计要点，别把它写成平凡指标：无防御时 edge 聚合**就是** Σ wᵢ Δθᵢ 本身，
    所以 `A_edge = cos(Δθ_edge^BD, Σ wᵢ Δθᵢ^BD)` 按字面定义恒等于 1、
    `M_edge` 恒等于 1，与「HFL 是否削弱攻击」无关。

    要有信息量，Δθ_edge^BD 必须**也用 counterfactual 定义**：整个 edge round 序列
    （edge_rounds 轮，含每轮的客户端重选与聚合）跑两遍，攻击开 / 关。此时偏离 1 的
    来源才是真机制 —— `edge_rounds > 1` 时，第 2 轮起良性客户端是从**已被污染的**
    edge 模型出发训练的，他们的更新里也带 BD 分量（间接传播），加上防御的过滤。
    """
    raise NotImplementedError(
        "edge 层嵌套 counterfactual 属于 HFL 传播会话，见本函数 docstring 的设计约定。")
