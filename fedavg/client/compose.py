"""
client/compose.py  –  客户端行为组合（攻击 mixin / 防御 mixin × PFL 方法类）

**存在理由**（CLAUDE.md 陷阱 #1，架构级）：
原来 main.py 是 `use_cls = MalCls or ClientCls` —— 恶意客户端类**替换掉** PFL 方法类。
而 NeurotoxinClient / CerPClient / BadPFLClient 都继承 FedAvgClient，于是
`drift_correction=hierpfedme` 时良性客户端跑 pFedMe、恶意客户端跑朴素 FedAvg，
上传语义都不同 —— 这足以单独解释 ASR≈0，且与攻击算法本身对不对无关。

正确形态是**组合**而非替换：攻击/防御都是 mixin，与方法类一起 type() 出一个新类，
恶意客户端仍然跑该 PFL 方法的 local_train，只是在其钩子上叠加行为。

MRO 约定
────────
    compose_client_class(MethodCls, DefenseMixin, AttackMixin)
    → type(名字, (AttackMixin, DefenseMixin, MethodCls), {})
    → MRO: AttackMixin → DefenseMixin → MethodCls → FLClientBase

  * 防御 mixin 挂给**所有**客户端 —— 防御方分不出谁是恶意的，这才是主动防御的
    真实语义。只挂良性客户端等于假设防御方已知恶意名单，跑出的 TPR/FPR 不可发表。
  * 攻击 mixin 在**外层** —— 恶意客户端有权把防御要求的客户端侧动作做假或跳过
    （「调不调 super()」= 「是否假装守协议」，是研究变量，不写死）。

缓存
────
按 (类元组) 缓存。不缓存的话 N 个恶意客户端会生成 N 个**不同的类对象** →
@tf.function 按类持有 trace 缓存，重复 retrace；且 isinstance 行为不可预测。
"""

_COMPOSED_CACHE: dict[tuple, type] = {}


def compose_client_class(method_cls: type, *mixins: type) -> type:
    """
    把若干行为 mixin 组合到 PFL 方法类上，返回组合类（带缓存）。

    Args:
        method_cls: PFL 方法的客户端类（FedAvgClient / PFedMeClient / …）
        *mixins:    行为 mixin，**按 MRO 优先级从高到低**传入。
                    约定顺序：compose_client_class(MethodCls, DefenseMixin, AttackMixin)
                    —— 传入顺序 (defense, attack)，组合时反转成 (attack, defense, method)，
                    使攻击在最外层。

    Returns:
        没有 mixin 时**原样返回 method_cls**（不是它的子类）——
        这保证 `defense=none` + `strategy=vanilla` 时类对象与改动前逐字节一致，
        L2「没跑坏」的论证依赖这一条。
    """
    real = tuple(m for m in mixins if m is not None)
    if not real:
        return method_cls

    key = (method_cls,) + real
    cached = _COMPOSED_CACHE.get(key)
    if cached is not None:
        return cached

    # 传入是 (defense, attack, ...)，越靠后越外层 → 反转后放在 bases 前面
    bases = tuple(reversed(real)) + (method_cls,)
    name = "".join(m.__name__.replace("Mixin", "") for m in reversed(real)) \
           + method_cls.__name__
    cls = type(name, bases, {"__doc__":
                             f"组合类：{' → '.join(b.__name__ for b in bases)}"})
    cls.__qualname__ = name
    _COMPOSED_CACHE[key] = cls
    return cls


def composed_cache_size() -> int:
    """当前缓存的组合类数量（测试用）。"""
    return len(_COMPOSED_CACHE)
