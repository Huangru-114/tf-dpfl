"""
tests/test_client_compose.py  —  客户端行为组合机制的不变量

CLAUDE.md 陷阱 #1 的修法是把攻击/防御从「替换方法类」改成「与方法类组合」。
组合本身的性质（MRO 顺序、缓存、无 mixin 时的恒等）是**纯 Python** 的，
不需要 TF，所以本地就能跑 —— 用假的方法类/mixin 手工构造，断言精确。

真实客户端上的钩子行为（on_upload 纯函数性等）在 tests/test_client_hooks.py，
那条需要 TF，在集群跑。
"""

import pytest

from client.compose import compose_client_class, composed_cache_size


# ── 假的「方法类」与两个假 mixin（都不碰 TF）────────────────────────────────
class FakeMethod:
    def __init__(self, client_id, dataset, model, config, n_samples=None):
        self.client_id = client_id
        self.dataset = dataset
        self.model = model
        self.config = config
        self.n_samples = n_samples
        self.trace = ["method_init"]

    def on_batch(self, x, y):
        return x, y

    def on_upload(self, upload, round_idx):
        return upload

    def who(self):
        return "method"


class FakeDefenseMixin:
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trace.append("defense_init")

    def on_batch(self, x, y):
        x, y = super().on_batch(x, y)
        return x + ["defense"], y

    def on_upload(self, upload, round_idx):
        upload = super().on_upload(upload, round_idx)
        return upload + ["defense"]


class FakeAttackMixin:
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trace.append("attack_init")

    def on_batch(self, x, y):
        x, y = super().on_batch(x, y)
        return x + ["attack"], y

    def on_upload(self, upload, round_idx):
        upload = super().on_upload(upload, round_idx)
        return upload + ["attack"]


def _make(cls):
    return cls(client_id=0, dataset=None, model=None, config={}, n_samples=1)


# ══════════════════════════════════════════════════════════════════════════
# 1. 无 mixin 时必须原样返回方法类（不是子类）
# ══════════════════════════════════════════════════════════════════════════
def test_no_mixin_returns_method_class_itself():
    """
    `defense=none` + `strategy=vanilla` 时，客户端类必须与改动前**逐字节一致**。
    返回一个空壳子类也会改变 @tf.function 的 trace 缓存归属，且让 L2「没跑坏」的
    论证失效 —— 所以这里断言的是 `is`，不是 issubclass。
    """
    assert compose_client_class(FakeMethod) is FakeMethod
    assert compose_client_class(FakeMethod, None) is FakeMethod
    assert compose_client_class(FakeMethod, None, None) is FakeMethod


# ══════════════════════════════════════════════════════════════════════════
# 2. 组合类仍然是方法类的子类（陷阱 #1 的核心断言）
# ══════════════════════════════════════════════════════════════════════════
def test_composed_class_is_subclass_of_method():
    cls = compose_client_class(FakeMethod, None, FakeAttackMixin)
    assert issubclass(cls, FakeMethod)
    assert issubclass(cls, FakeAttackMixin)
    # 方法类的行为仍然可达（没有被替换掉）
    assert _make(cls).who() == "method"


# ══════════════════════════════════════════════════════════════════════════
# 3. MRO 顺序：攻击在外层，防御在内层，方法类最内
# ══════════════════════════════════════════════════════════════════════════
def test_mro_order_attack_outermost():
    """
    传入约定是 (MethodCls, DefenseMixin, AttackMixin)，
    组合后 MRO 必须是 Attack → Defense → Method。

    语义：恶意客户端有权把防御要求的客户端侧动作做假或跳过，所以攻击在最外层。
    """
    cls = compose_client_class(FakeMethod, FakeDefenseMixin, FakeAttackMixin)
    mro = [c.__name__ for c in cls.__mro__]
    assert mro.index("FakeAttackMixin") < mro.index("FakeDefenseMixin")
    assert mro.index("FakeDefenseMixin") < mro.index("FakeMethod")


def test_hooks_chain_in_mro_order():
    """
    每个 mixin 都先调 super() 再叠自己的效果 →
    结果顺序是「最内层先追加、最外层后追加」= [method..., defense, attack]。
    这条精确断言就是「攻击有最后发言权」的可执行形式。
    """
    cls = compose_client_class(FakeMethod, FakeDefenseMixin, FakeAttackMixin)
    obj = _make(cls)

    x, _ = obj.on_batch([], None)
    assert x == ["defense", "attack"]

    assert obj.on_upload([], 1) == ["defense", "attack"]


def test_only_defense_mixin_chains():
    """良性客户端（只有防御 mixin）也必须走通链条。"""
    cls = compose_client_class(FakeMethod, FakeDefenseMixin)
    obj = _make(cls)
    assert obj.on_upload([], 1) == ["defense"]


# ══════════════════════════════════════════════════════════════════════════
# 4. __init__ 协作：super() 链必须一路走到方法类
# ══════════════════════════════════════════════════════════════════════════
def test_init_cooperates_through_whole_mro():
    """
    mixin 的 __init__ 必须先 super()、再做自己的事，否则方法类的
    self.model / self.rng / self.lr_schedule 都不存在。
    trace 的顺序精确反映这一点：方法类最先跑完。
    """
    cls = compose_client_class(FakeMethod, FakeDefenseMixin, FakeAttackMixin)
    obj = _make(cls)
    assert obj.trace == ["method_init", "defense_init", "attack_init"]
    # 方法类的构造参数确实传到了
    assert obj.client_id == 0 and obj.n_samples == 1


# ══════════════════════════════════════════════════════════════════════════
# 5. 缓存：同一组合只能产生一个类对象
# ══════════════════════════════════════════════════════════════════════════
def test_composition_is_cached():
    """
    不缓存的话，N 个恶意客户端会生成 N 个**不同的类对象** →
    @tf.function 按类持有 trace 缓存，重复 retrace；isinstance 也不可预测。
    """
    a = compose_client_class(FakeMethod, FakeDefenseMixin, FakeAttackMixin)
    b = compose_client_class(FakeMethod, FakeDefenseMixin, FakeAttackMixin)
    assert a is b

    before = composed_cache_size()
    for _ in range(50):
        compose_client_class(FakeMethod, FakeDefenseMixin, FakeAttackMixin)
    assert composed_cache_size() == before, "重复组合不应新增缓存条目"


def test_different_combinations_are_different_classes():
    only_attack  = compose_client_class(FakeMethod, None, FakeAttackMixin)
    both         = compose_client_class(FakeMethod, FakeDefenseMixin, FakeAttackMixin)
    only_defense = compose_client_class(FakeMethod, FakeDefenseMixin)
    assert len({only_attack, both, only_defense}) == 3


def test_composed_class_has_readable_name():
    """日志里要能一眼看出这个客户端是什么组合。"""
    cls = compose_client_class(FakeMethod, FakeDefenseMixin, FakeAttackMixin)
    assert "FakeAttack" in cls.__name__
    assert "FakeDefense" in cls.__name__
    assert "FakeMethod" in cls.__name__
