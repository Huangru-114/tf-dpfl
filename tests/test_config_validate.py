"""
tests/test_config_validate.py  —  配置兼容性校验

这个仓库里最贵的错误是**静默跑错**：config 被接受、日志正常、跑完才发现那一格
没有意义。校验器的职责就是把这些组合在 run 开始前变成硬错误。

纯 python（config_validate 不 import TF），本地秒级可跑。
"""

import copy

import pytest

from config_validate import (ConfigError, validate_config,
                             METHOD_SUPPORTS_DEFENSE, AGGREGATION_DEFENSES)


BASE = {
    "seed": 42,
    "data": {"num_classes": 10},
    "federation": {"n_clients": 20, "n_edges": 2, "client_fraction": 1.0},
    "training": {"drift_correction": "hierfedavg"},
    "defense": {"name": "none", "params": {}},
    "backdoor": {"enabled": False},
}


def cfg(**patch):
    c = copy.deepcopy(BASE)
    for path, value in patch.items():
        keys = path.split(".")
        d = c
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return c


# ══════════════════════════════════════════════════════════════════════════
# 1. 未知 / 已移除 / 未实现的方法必须拒绝，不能静默退化成 pFedMe
# ══════════════════════════════════════════════════════════════════════════
def test_unknown_method_is_rejected():
    """
    `_select_method_classes` 用 .get(method, PFedMeClient)：拼错的方法名会
    **静默变成 pFedMe**。你以为在跑 A，实际跑的是 B，且没有任何征兆。
    """
    with pytest.raises(ConfigError, match="静默变成 pFedMe"):
        validate_config(cfg(**{"training.drift_correction": "hier_fedavgg"}))


@pytest.mark.parametrize("method", ["fedprox", "feddyn", "scaffold"])
def test_unimplemented_methods_are_rejected(method):
    """config 注释里列过、但当前类分发不实现的取值 —— 同样会静默变成 pFedMe。"""
    with pytest.raises(ConfigError):
        validate_config(cfg(**{"training.drift_correction": method}))


def test_removed_method_gives_migration_message():
    """已移除的方法要给出「为什么移除」，而不是笼统的『未知取值』。"""
    with pytest.raises(ConfigError, match="元梯度"):
        validate_config(cfg(**{"training.drift_correction": "hier_perfedavg"}))


@pytest.mark.parametrize("method", sorted(METHOD_SUPPORTS_DEFENSE))
def test_all_registered_methods_pass(method):
    validate_config(cfg(**{"training.drift_correction": method}))


# ══════════════════════════════════════════════════════════════════════════
# 2. 防御轴：不兼容组合必须拒绝，而不是打印 [Defense] enabled 然后不设防
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("method", sorted(METHOD_SUPPORTS_DEFENSE))
@pytest.mark.parametrize("defense", sorted(AGGREGATION_DEFENSES))
def test_every_registered_method_supports_every_aggregation_defense(method, defense):
    """
    这是三维矩阵可跑的前提：登记在册的每个 PFL 方法都必须能承接每个聚合层防御。
    某个方法的 edge 聚合绕开 robust_mean 时，必须在 METHOD_SUPPORTS_DEFENSE 里
    标 False —— 那样这条断言会失败，提醒你那一整片格子是空的。
    """
    validate_config(cfg(**{"training.drift_correction": method,
                           "defense.name": defense}))


def test_unknown_defense_is_rejected():
    with pytest.raises(ConfigError, match="defense.name"):
        validate_config(cfg(**{"defense.name": "flamee"}))


def test_defense_incompatible_with_method_is_rejected(monkeypatch):
    """构造一个「不支持防御」的方法，确认校验器真的拦得住（而不是只在表里好看）。"""
    monkeypatch.setitem(METHOD_SUPPORTS_DEFENSE, "fake_method", False)
    with pytest.raises(ConfigError, match="静默失效"):
        validate_config(cfg(**{"training.drift_correction": "fake_method",
                               "defense.name": "flame"}))
    # 同一个方法配 defense=none 则应放行
    validate_config(cfg(**{"training.drift_correction": "fake_method",
                           "defense.name": "none"}))


# ══════════════════════════════════════════════════════════════════════════
# 3. 攻击轴 sanity + 正交性告警
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("method", ["hierpfedme", "hier_ditto", "hier_fedrep",
                                    "hier_ditto_rep", "hier_pfedme_rep",
                                    "hierfedavg"])
@pytest.mark.parametrize("strategy", ["neurotoxin", "cerp", "badpfl"])
def test_no_orthogonality_warning_after_mixin_refactor(method, strategy):
    """
    陷阱 #1 已修复：攻击策略现在是 **mixin**，与 PFL 方法类组合而非替换
    （main.py:resolve_client_classes + client/compose.py），所以任何
    「方法 × 策略」组合都不再是不可解释的格子。

    这条测试原本断言的是**反面**（「默认告警、strict 模式报错」），
    即它当初编码的是 bug 的存在。修复后必须反过来断言，否则会把回归当成通过。

    真正的结构守卫在 tests/test_attack_method_orthogonality.py（断言恶意客户端类
    仍是方法类的子类）；这里只保证校验器不再吐已经不成立的警告。
    """
    c = cfg(**{"training.drift_correction": method,
               "backdoor.enabled": True,
               "backdoor.malicious_strategy": strategy,
               "backdoor.trigger": "badnet",
               "backdoor.target_label": 9,
               "backdoor.poison_ratio": 0.5,
               "backdoor.n_malicious": 2})

    warnings = validate_config(c, strict_orthogonality=False)
    assert not any("尚未正交" in w for w in warnings)

    # strict 模式也不该再拦 —— 该参数已失效，只会提示「可以删掉这条配置」
    strict_warnings = validate_config(c, strict_orthogonality=True)
    assert not any("尚未正交" in w for w in strict_warnings)
    assert any("已失效" in w for w in strict_warnings)


@pytest.mark.parametrize("patch,msg", [
    ({"backdoor.target_label": 99}, "target_label"),
    ({"backdoor.poison_ratio": 1.5}, "poison_ratio"),
    ({"backdoor.n_malicious": 999}, "n_malicious"),
    ({"backdoor.malicious_strategy": "neurotoxine"}, "malicious_strategy"),
    ({"backdoor.trigger": "badnett"}, "trigger"),
])
def test_backdoor_sanity_checks(patch, msg):
    base = {"backdoor.enabled": True, "backdoor.malicious_strategy": "vanilla",
            "backdoor.trigger": "badnet", "backdoor.target_label": 9,
            "backdoor.poison_ratio": 0.5, "backdoor.n_malicious": 2}
    base.update(patch)
    with pytest.raises(ConfigError, match=msg):
        validate_config(cfg(**base))


def test_dba_requires_patterns():
    with pytest.raises(ConfigError, match="dba_patterns"):
        validate_config(cfg(**{"backdoor.enabled": True,
                               "backdoor.malicious_strategy": "vanilla",
                               "backdoor.trigger": "dba",
                               "backdoor.target_label": 9,
                               "backdoor.poison_ratio": 0.5,
                               "backdoor.n_malicious": 2}))


# ══════════════════════════════════════════════════════════════════════════
# 4. 拓扑与可复现性
# ══════════════════════════════════════════════════════════════════════════
def test_topology_sanity():
    with pytest.raises(ConfigError, match="n_edges"):
        validate_config(cfg(**{"federation.n_edges": 50}))
    with pytest.raises(ConfigError, match="client_fraction"):
        validate_config(cfg(**{"federation.client_fraction": 0.0}))


def test_missing_seed_warns():
    c = cfg()
    del c["seed"]
    assert any("seed" in w for w in validate_config(c))
