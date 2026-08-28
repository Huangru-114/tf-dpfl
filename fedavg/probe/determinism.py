"""
probe/determinism.py  —  paired counterfactual 的地基：把「同一个客户端跑两遍」
变成一件**逐比特可复现**的事。

为什么这是整个 Exp 0.3 里最重要的一块
────────────────────────────────────
核心量 Δθ_BD = Δθ_poison − Δθ_clean 是两次训练之差。如果两次训练除了「投毒开关」
之外还有别的差异，那个差异会**整个进到 Δθ_BD 里**。本仓库现状有三处这样的差异：

  1. `hier_fedrep._shuffled_batches()`（同族的 ditto/ditto_rep/pfedme_rep 一样）
     用的是 Python **全局** `random.shuffle` —— 两个 twin 顺序跑，批序必然不同。
  2. `client_fedavg` 直接迭代 `self.dataset`，而 dataset 建时是
     `.shuffle(buf, reshuffle_each_iteration=True)`（data/dataset.py:175）
     —— 同一个 dataset 对象两次迭代顺序也不同。
  3. `BadPFLMixin.on_batch` 会消耗 `self.rng`（投毒 mask 的 shuffle），
     clean twin 不消耗 → 之后两条 RNG 流错位。

不修这三条，Δθ_BD 里绝大部分是批序噪声。而 clean-vs-clean control 会把这件事
呈现成 ‖Δθ_BD‖ ≈ ‖Δθ_stochastic‖ —— **「后门无法定位」和「仪表坏了」外观完全一样**。

本模块给两个 context manager：
  FrozenOrder    钉死批序（由 order_seed 决定，与调用次数无关）
  ClientSnapshot 进入时快照、退出时还原客户端**与共享攻击者状态**
"""
import random

import numpy as np
import tensorflow as tf


# ════════════════════════════════════════════════════════════════════════════
# 批序冻结
# ════════════════════════════════════════════════════════════════════════════

class _FrozenBatches:
    """
    行为像 `self.dataset` 的可迭代对象，但每次迭代给出的是**预先算好**的一个置换。

    为什么预先算好而不是"每次 iter 时现摇一个 rng"：现摇的话，poison twin 若比
    clean twin 多消耗一次迭代（例如 Bad-PFL 的生成器训练也遍历 dataset），
    之后所有 epoch 的顺序就整体错位。预生成 + 按 epoch 序号取，顺序只由
    (order_seed, epoch 序号) 决定，与谁消耗了几次无关。
    """

    def __init__(self, batches, perms):
        self._batches = batches
        self._perms = perms
        self._epoch = 0

    def __iter__(self):
        p = self._perms[self._epoch % len(self._perms)]
        self._epoch += 1
        return iter([self._batches[i] for i in p])

    def __len__(self):
        return len(self._batches)

    def reset(self):
        self._epoch = 0


def _materialize(client):
    """把客户端的数据物化成一个普通 list（只做一次，之后复用）。"""
    cached = getattr(client, "_probe_batches", None)
    if cached is None:
        src = getattr(client, "_batch_list", None)
        cached = list(src) if src is not None else list(client.dataset)
        client._probe_batches = cached
    return cached


class FrozenOrder:
    """
    钉死一个客户端的批序。

    覆盖三条路径（不同 PFL 方法走不同的那一条）：
      * `_shuffled_batches()`        —— fedrep / ditto / ditto_rep / pfedme_rep
      * 直接迭代 `self.dataset`      —— fedavg，以及 Bad-PFL 生成器训练
                                        （client_badpfl.py:100）与 CerP
      * `_batch_list` + 内联 shuffle —— pfedme（它不走 `_shuffled_batches`）
    最后一条靠 `random.seed(order_seed)` 兜底：把进程级 Python RNG 也钉死，
    于是内联的 `random.shuffle` 在两个 twin 里消耗同一条流。

    同时 `tf.random.set_seed(order_seed)`——若模型里有 Dropout 之类的随机层，
    不钉死它就是第四条泄漏路径。
    """

    _EPOCH_POOL = 256          # 预生成的置换个数，够所有方法的 epoch 总数

    def __init__(self, client, order_seed: int):
        self.client = client
        self.order_seed = int(order_seed)
        self._saved = {}

    def __enter__(self):
        c = self.client
        batches = _materialize(c)

        # drop_last：复刻 `_shuffled_batches` 里的 `x.shape[0] == self._batch_size` 过滤，
        # 先过滤再置换，于是置换长度稳定、与 batch 内容无关。
        bs = getattr(c, "_batch_size", None)
        if bs is not None:
            batches = [b for b in batches if int(b[0].shape[0]) == int(bs)]

        rng = np.random.default_rng(self.order_seed)
        n = len(batches)
        perms = [rng.permutation(n) for _ in range(self._EPOCH_POOL)]
        frozen = _FrozenBatches(batches, perms)

        self._saved["dataset"] = c.dataset
        self._saved["_batch_list"] = getattr(c, "_batch_list", None)
        self._saved["_shuffled_batches"] = c.__dict__.get("_shuffled_batches", _MISSING)
        self._saved["random_state"] = random.getstate()

        c.dataset = frozen
        if hasattr(c, "_batch_list"):
            c._batch_list = list(batches)
        if hasattr(c, "_shuffled_batches"):
            # 实例属性遮蔽类方法；退出时删掉这个属性即还原
            c._shuffled_batches = lambda: list(iter(frozen))

        random.seed(self.order_seed)
        tf.random.set_seed(self.order_seed)
        frozen.reset()
        self._frozen = frozen
        return self

    def __exit__(self, *exc):
        c = self.client
        c.dataset = self._saved["dataset"]
        if self._saved["_batch_list"] is not None:
            c._batch_list = self._saved["_batch_list"]
        sb = self._saved["_shuffled_batches"]
        if sb is _MISSING:
            c.__dict__.pop("_shuffled_batches", None)
        else:
            c._shuffled_batches = sb
        random.setstate(self._saved["random_state"])
        return False


class _Missing:
    pass


_MISSING = _Missing()


# ════════════════════════════════════════════════════════════════════════════
# 状态快照 / 还原
# ════════════════════════════════════════════════════════════════════════════

def _is_keras_model(o) -> bool:
    return hasattr(o, "get_weights") and hasattr(o, "set_weights") and hasattr(o, "layers")


def _is_optimizer(o) -> bool:
    return hasattr(o, "apply_gradients") and hasattr(o, "variables")


def _opt_vars(opt):
    """Keras 2 的 `optimizer.variables()` 是方法，Keras 3 是属性 —— 两边都吃。"""
    v = opt.variables
    return list(v() if callable(v) else v)


def _snapshot_optimizer(opt) -> dict:
    """
    按**变量名**存优化器状态，而不是按位置。

    理由：探针启动时客户端还从没训练过，优化器的 slot 变量尚未创建
    （`_opt_vars` 返回空或只有 iterations）；第一个 twin 跑完才建出来。
    按位置还原就会在第二个 twin 上对不上号。按名字还原时，
    **快照里没有的变量一律置零** —— 那正是"刚建好、从未训练"的状态
    （所有 slot 初值为 0，iterations 为 0）。
    """
    return {v.name: np.array(v.numpy(), copy=True) for v in _opt_vars(opt)}


def _restore_optimizer(opt, snap: dict):
    for v in _opt_vars(opt):
        want = snap.get(v.name)
        if want is None:
            v.assign(tf.zeros_like(v))          # 快照时不存在 → 回到"从未训练"
        else:
            v.assign(tf.cast(tf.convert_to_tensor(want), v.dtype))


class ClientSnapshot:
    """
    进入时快照、退出时还原一个客户端的**全部可变状态**。

    漏掉任何一项都会静默污染下一个 twin，而污染的表现是 Δθ 里多出一块无法归因的
    位移 —— 与"后门位移"外观相同。所以这里用**通用扫描**而不是白名单：
    逐个检查 `client.__dict__`，按类型分派。白名单每加一个攻击 mixin 就要改一次，
    改漏了不会报错、只会让数字变脏。

    覆盖：
      * keras Model    —— `client.model`、pFedMe 的 `_local_model`、CerP 的
                          `_atk_ref_model`、**Bad-PFL 的 `_atk_generator`**
                          （main.py:468 起全体恶意端共享**同一个对象**，poison twin
                          会在它上面训 30 步，不还原就污染之后所有探测）
      * keras 优化器   —— `optimizer` / `_head_opt` / `_base_opt` / `_atk_gen_opt`
      * tf.Variable    —— `lr_schedule` / `_head_lr_var` / CerP 的触发器与门控
      * np.random.Generator —— `client.rng`（Bad-PFL 的投毒 mask 消耗它）
      * ndarray / ndarray 列表 —— `_head_weights` / `edge_weights` / `global_weights`
      * 标量           —— `is_malicious` / `_atk_n_poisoned_batches`
      * 进程级         —— `random` 与 `np.random` 的全局状态
    """

    _SCALARS = (bool, int, float, str, type(None))

    def __init__(self, client):
        self.client = client
        self._models = {}
        self._opts = {}
        self._vars = {}
        self._rngs = {}
        self._arrays = {}
        self._scalars = {}

    def __enter__(self):
        c = self.client
        for name, obj in list(c.__dict__.items()):
            if _is_keras_model(obj):
                self._models[name] = [np.array(w, copy=True) for w in obj.get_weights()]
            elif _is_optimizer(obj):
                self._opts[name] = _snapshot_optimizer(obj)
            elif isinstance(obj, tf.Variable):
                self._vars[name] = np.array(obj.numpy(), copy=True)
            elif isinstance(obj, np.random.Generator):
                self._rngs[name] = obj.bit_generator.state
            elif isinstance(obj, np.ndarray):
                self._arrays[name] = np.array(obj, copy=True)
            elif isinstance(obj, list) and obj and all(isinstance(w, np.ndarray) for w in obj):
                self._arrays[name] = [np.array(w, copy=True) for w in obj]
            elif isinstance(obj, self._SCALARS):
                self._scalars[name] = obj
        self._py_random = random.getstate()
        self._np_random = np.random.get_state()
        return self

    def __exit__(self, *exc):
        c = self.client
        for name, w in self._models.items():
            getattr(c, name).set_weights(w)
        for name, snap in self._opts.items():
            _restore_optimizer(getattr(c, name), snap)
        for name, v in self._vars.items():
            getattr(c, name).assign(v)
        for name, state in self._rngs.items():
            getattr(c, name).bit_generator.state = state
        for name, a in self._arrays.items():
            setattr(c, name, [np.array(w, copy=True) for w in a]
                    if isinstance(a, list) else np.array(a, copy=True))
        for name, s in self._scalars.items():
            setattr(c, name, s)
        random.setstate(self._py_random)
        np.random.set_state(self._np_random)
        return False
