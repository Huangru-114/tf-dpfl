"""
tests/test_attack_method_orthogonality.py  —  陷阱 #1 的数值证据（架构级）

config 把「攻击策略」和「PFL 方法」设计成两个**正交**轴：

    training.drift_correction   ∈ {hierfedavg, hierpfedme, hier_ditto, hier_fedrep, ...}
    backdoor.malicious_strategy ∈ {vanilla, neurotoxin, cerp, badpfl}

但 `main.py:395` 是 `use_cls = MalCls or ClientCls` —— 恶意客户端类**替换掉**了
PFL 方法类。而 NeurotoxinClient / BadPFLClient / CerPClient 都继承 FedAvgClient。

后果：`drift_correction=hierpfedme` 时，99 个良性客户端跑 pFedMe、恶意客户端跑朴素
FedAvg。它上传的东西语义与所有人都不同，既进不了正确的聚合，又天然像个离群点。
**这足以单独解释 neurotoxin / bad-pfl 的 ASR≈0**，且与算法本身对不对无关。

正确形态：攻击策略应是 **mixin**，与方法类组合 —— `type("X", (Strategy, MethodCls), {})`
或等价手段 —— 使恶意客户端仍然跑该 PFL 方法的 local_train，只是在其上叠加投毒/掩码。

需要 TF，本地自动 skip，在集群上跑。
"""

import pytest

pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from main import _select_method_classes, select_malicious_client_class   # noqa: E402


PFL_METHODS = ["hierfedavg", "hierpfedme", "hier_ditto", "hier_ditto_rep",
               "hier_pfedme_rep", "hier_perfedavg", "hier_fedrep"]
ATTACK_STRATEGIES = ["neurotoxin", "cerp", "badpfl"]


def _resolve(method: str, strategy: str):
    """复刻 main.py:386-395 的类选择逻辑。"""
    config = {"training": {"drift_correction": method}}
    method_cls, _ = _select_method_classes(config)
    mal_cls = select_malicious_client_class(strategy)
    return method_cls, (mal_cls or method_cls)


@pytest.mark.parametrize("method", PFL_METHODS)
@pytest.mark.parametrize("strategy", ATTACK_STRATEGIES)
def test_malicious_client_preserves_pfl_method(method, strategy):
    """
    恶意客户端必须仍然是该 PFL 方法的客户端 —— 攻击是叠加在方法上的，不是替换方法。
    """
    method_cls, used_cls = _resolve(method, strategy)

    assert issubclass(used_cls, method_cls), (
        f"drift_correction={method} + malicious_strategy={strategy}：\n"
        f"  良性客户端用 {method_cls.__name__}，恶意客户端用 {used_cls.__name__}，"
        f"后者不是前者的子类。\n"
        f"  → 恶意客户端跑的是另一套训练算法，上传语义与聚合端不匹配。")


@pytest.mark.parametrize("method", PFL_METHODS)
@pytest.mark.parametrize("strategy", ATTACK_STRATEGIES)
def test_malicious_client_does_not_silently_downgrade_to_fedavg(method, strategy):
    """
    更具体的形式：非 fedavg 方法下，恶意客户端的 local_train 不能解析到
    FedAvgClient.local_train —— 那意味着个性化/近端项全部丢失。
    """
    if method in ("hierfedavg", "fedavg"):
        pytest.skip("baseline 本身就是 FedAvg")

    from client.client_fedavg import FedAvgClient

    method_cls, used_cls = _resolve(method, strategy)
    resolved = used_cls.local_train

    assert resolved is not FedAvgClient.local_train, (
        f"drift_correction={method} + malicious_strategy={strategy}：恶意客户端的 "
        f"local_train 解析到 FedAvgClient.local_train —— {method_cls.__name__} 的"
        f"个性化/近端逻辑被整个绕过。")
