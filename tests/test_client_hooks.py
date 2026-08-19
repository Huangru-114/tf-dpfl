"""
tests/test_client_hooks.py  —  客户端钩子协议的不变量

组合机制本身的性质在 tests/test_client_compose.py（纯 Python，本地跑）。
本文件验的是**挂到真实客户端上之后**的性质，需要 TF，在集群跑。

守两件事：

  1. **on_upload 是纯函数** —— 绝不写回 self.model。
     六个 PFL 方法里有五个的 upload ≠ self.model（pFedMe 传锚点 Θ、Ditto 传阶段 A 的
     w_k、Rep 家族传 backbone+私有 head）。旧的 Neurotoxin 实现做
     `self.model.set_weights(w_proj)`，在非 FedAvg 方法下把**掩码后的上传物**盖到
     **个性化模型**上 → PM 评估 / 遗忘曲线 / local ASR 三处指标同时被污染。

  2. **掩码基准是 edge_weights** —— 而不是 self.model 训练前后之差。
     `upload − edge_weights` 才是聚合端真正看到的 delta，且对六个方法都有定义。

  3. **基类钩子是恒等/零** —— 良性、无防御路径数值零变化，这是 L2「没跑坏」的依据。
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from client.client_fedavg import FedAvgClient          # noqa: E402
from client.client_pfedme import PFedMeClient          # noqa: E402
from client.client_neurotoxin import NeurotoxinMixin   # noqa: E402
from client.compose import compose_client_class        # noqa: E402


N_FEAT = 40


def _config():
    return {
        "seed": 42,
        "training": {"learning_rate": 0.1, "lr_decay": 1.0, "local_epochs": 1,
                     "personal_lr_pfedme": 0.01, "moreau_lr_pfedme": 0.005,
                     "lambda2_hier": 15.0, "mu_pfedme": 0.001,
                     "inner_steps_hier": 2},
        "data": {"batch_size": 4, "img_size": 4, "num_classes": 2},
        "backdoor": {"neurotoxin_mask_ratio": 0.5, "neurotoxin_mask_batches": 0},
    }


def _model():
    """
    末层必须带 softmax：`FLClientBase.loss_fn` 是
    `SparseCategoricalCrossentropy(from_logits=False)`，与 models/cnn.py 里
    所有模型末层的 `activation="softmax"` 配套。
    用裸线性层（更别说零初始化）会让梯度恒为 0，训练什么都不做，
    「个性化模型 ≠ 上传物」这类断言就变成在比较两个全零数组 —— 假绿。
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_FEAT,)),
        tf.keras.layers.Dense(2, use_bias=False, activation="softmax",
                              kernel_initializer=tf.keras.initializers.RandomNormal(
                                  stddev=0.1, seed=7)),
    ])


def _dataset(rng, n=8):
    x = rng.normal(size=(n, N_FEAT)).astype(np.float32)
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(4)


def _client(rng, method_cls, with_attack=True):
    cls = compose_client_class(method_cls, None,
                               NeurotoxinMixin if with_attack else None)
    ds = _dataset(rng)
    c = cls(client_id=0, dataset=ds, model=_model(), config=_config(), n_samples=8)
    c.is_malicious = with_attack
    if with_attack:
        c.set_clean_dataset(ds)
    w0 = [w.copy() for w in c.model.get_weights()]
    c.set_weights(global_weights=w0, edge_weights=w0)
    return c


# ══════════════════════════════════════════════════════════════════════════
# 1. 基类钩子必须是恒等 / 零
# ══════════════════════════════════════════════════════════════════════════
def test_base_hooks_are_noops(rng):
    """良性、无防御客户端的钩子必须什么都不做 —— 否则 L2「没跑坏」无从谈起。"""
    c = _client(rng, FedAvgClient, with_attack=False)

    x = tf.constant(np.ones((2, N_FEAT), np.float32))
    y = tf.constant(np.array([0, 1], np.int64))
    x2, y2 = c.on_batch(x, y)
    np.testing.assert_array_equal(x2.numpy(), x.numpy())
    np.testing.assert_array_equal(y2.numpy(), y.numpy())

    assert float(c.on_extra_loss().numpy()) == 0.0

    up = [w.copy() for w in c.model.get_weights()]
    out = c.on_upload(up, 1)
    for a, b in zip(out, up):
        np.testing.assert_array_equal(a, b)

    assert c.get_aux() == {}
    c.set_control({"clip": 1.0})
    assert c._control == {"clip": 1.0}


# ══════════════════════════════════════════════════════════════════════════
# 2. on_upload 必须是纯函数（本会话修的核心 bug）
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("method_cls", [FedAvgClient, PFedMeClient])
def test_on_upload_is_pure(rng, method_cls):
    """
    调用 on_upload 前后，self.model 的权重必须**逐元素相同**。

    旧实现 `self.model.set_weights(w_proj)` 会让这条断言失败 —— 而在 pFedMe /
    Ditto / FedRep 下 self.model 是个性化模型，那正是评估用的东西。
    """
    c = _client(rng, method_cls, with_attack=True)
    # 手工给一个已知掩码，避免依赖梯度
    c._atk_mask = [np.array([[1.0 if j == 0 else 0.0 for j in range(2)]
                             for _ in range(N_FEAT)], np.float32)]

    before = [w.copy() for w in c.model.get_weights()]
    upload = [w + 1.0 for w in before]

    out = c.on_upload(upload, round_idx=1)

    after = c.model.get_weights()
    assert len(before) == len(after)
    for b, a in zip(before, after):
        np.testing.assert_array_equal(
            a, b, err_msg="on_upload 写回了 self.model —— 个性化模型被污染")
    # 返回值确实变了（否则这条测试是空的）
    assert any(not np.array_equal(o, u) for o, u in zip(out, upload))


# ══════════════════════════════════════════════════════════════════════════
# 3. 掩码基准必须是 edge_weights，且投影公式精确
# ══════════════════════════════════════════════════════════════════════════
def test_mask_projects_upload_delta_against_edge_weights(rng):
    """
    on_upload 的结果必须精确等于 edge + (upload − edge)·mask。
    特别地：mask=0 的坐标上，结果必须**逐位等于 edge_weights**（硬投影）。
    """
    c = _client(rng, PFedMeClient, with_attack=True)

    edge = [w.copy() for w in c.edge_weights]
    m = np.zeros((N_FEAT, 2), np.float32)
    m[:, 0] = 1.0                      # 只放行第 0 列
    c._atk_mask = [m]

    upload = [e + rng.normal(size=e.shape).astype(np.float32) for e in edge]
    out = c.on_upload([u.copy() for u in upload], round_idx=1)

    for e, u, mm, o in zip(edge, upload, c._atk_mask, out):
        expected = e + (u - e) * mm
        np.testing.assert_allclose(o, expected, rtol=0, atol=0)
        blocked = mm < 0.5
        np.testing.assert_array_equal(
            o[blocked], e[blocked],
            err_msg="被屏蔽坐标没有回到 edge_weights —— 投影不是硬投影")


def test_mask_reference_is_not_the_personalized_model(rng):
    """
    这条是陷阱 #1 的数值形式：在 pFedMe 下，self.model 是个性化模型 θ̃，
    与上传的锚点 Θ 不是一回事。掩码若以 self.model 为基准（旧实现），
    被屏蔽坐标会回到 θ̃ 而不是 edge_weights —— 上传物语义直接错乱。
    """
    c = _client(rng, PFedMeClient, with_attack=True)
    edge = [w.copy() for w in c.edge_weights]

    # 让 self.model 明显偏离 edge_weights（模拟个性化训练之后的状态）
    c.model.set_weights([w + 7.0 for w in edge])

    m = np.zeros((N_FEAT, 2), np.float32)      # 全屏蔽
    c._atk_mask = [m]
    upload = [e + 3.0 for e in edge]

    out = c.on_upload(upload, round_idx=1)
    for o, e in zip(out, edge):
        np.testing.assert_array_equal(
            o, e, err_msg="全屏蔽时上传物没有回到 edge_weights —— 基准取错了")


# ══════════════════════════════════════════════════════════════════════════
# 4. 端到端：恶意客户端仍然跑该 PFL 方法，个性化模型不被上传物污染
# ══════════════════════════════════════════════════════════════════════════
def test_malicious_pfedme_keeps_personalized_model_separate(rng):
    """
    Neurotoxin × pFedMe 跑完一轮 local_train 后：
      - 返回的 upload 是掩码后的锚点 Θ
      - self.model 仍停在个性化模型 θ̃，且**不等于** upload
    旧实现两者会被强制相等（因为 set_weights(w_proj) 写回了 self.model）。
    """
    c = _client(rng, PFedMeClient, with_attack=True)
    upload, n, loss, _ = c.local_train(round_idx=1)

    assert n == 8
    personalized = c.model.get_weights()
    assert len(personalized) == len(upload)
    assert any(not np.array_equal(p, u) for p, u in zip(personalized, upload)), (
        "个性化模型与上传物逐元素相同 —— on_upload 很可能又写回了 self.model")
