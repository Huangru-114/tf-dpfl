"""
aggregation/client_update.py  –  带身份的客户端上传记录

**为什么需要**：原本客户端上传是裸的 4-元组 `(weights, n_samples, loss, train_time)`，
不含 client_id。后果是防御（FLAME/Krum/DnC）报出「接纳了第 0,1,4 号更新」时，
这些索引**映射不回真实客户端**，于是算不出防御的检出率 / 误伤率（TPR/FPR）——
而这正是防御轴唯一有意义的指标：只看 ASR 分不清「防御奏效」和「攻击本来就没生效」。

**为什么是 tuple 子类而不是 dataclass**：全仓库有大量
`for weights, n, loss, t in client_updates` / `upd[0]` / `sum(n for _, n, *_ in ...)`
的解包写法。继承 tuple 让这些**一行都不用改**，同时多出 `.client_id`。
新代码请用具名属性（`upd.weights`），旧解包继续有效。
"""


class ClientUpdate(tuple):
    """
    行为完全等同 `(weights, n_samples, loss, train_time)` 的 4-元组，
    额外带 `.client_id`（不参与元组比较 / 解包）。
    """

    # 注：tuple 是变长类型，不支持非空 __slots__，所以实例带一个 __dict__。
    # 每轮每客户端一个 dict，开销可忽略。

    def __new__(cls, weights, n_samples, loss, train_time, client_id=None):
        obj = super().__new__(cls, (weights, n_samples, loss, train_time))
        obj.client_id = None if client_id is None else int(client_id)
        return obj

    # ── 具名访问（新代码优先用这些）────────────────────────────────────────
    @property
    def weights(self):
        return self[0]

    @property
    def n_samples(self):
        return self[1]

    @property
    def loss(self):
        return self[2]

    @property
    def train_time(self):
        return self[3]

    def __repr__(self):
        return (f"ClientUpdate(client_id={self.client_id}, "
                f"n_samples={self[1]}, loss={self[2]:.4f})")


def as_client_update(result, client_id):
    """
    把 client.local_train 的返回值规范成 ClientUpdate。
    已经是 ClientUpdate 时补齐 client_id（不覆盖已有值）。
    """
    if isinstance(result, ClientUpdate):
        if result.client_id is None:
            result.client_id = int(client_id)
        return result
    weights, n_samples, loss, train_time = result
    return ClientUpdate(weights, n_samples, loss, train_time, client_id)


def client_ids(client_updates: list) -> list:
    """取一批上传的 client_id 列表；缺失的位置为 None。"""
    return [getattr(u, "client_id", None) for u in client_updates]
