"""
tests/test_probe_checkpoint.py  —  checkpoint 存/载往返（需 TF）。

checkpoint 少存一样东西不会报错，只会让从它复现出来的轨迹是**另一条**轨迹。
最典型的两个：FedRep 的私有 head（个性化状态）和 Bad-PFL 的共享 generator
（跨轮累积的攻击者状态）。这里逐个钉死。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

from client.client_badpfl import BadPFLMixin              # noqa: E402
from client.client_fedavg import FedAvgClient             # noqa: E402
from client.compose import compose_client_class           # noqa: E402
from client.hier_fedrep import HierFedRepClient           # noqa: E402
from models.autoencoder import build_autoencoder          # noqa: E402
from probe import checkpoint as ck                        # noqa: E402

IMG, NC, BS = 16, 3, 8


def _config():
    return {"seed": 42,
            "data": {"dataset": "cifar10", "img_size": IMG, "batch_size": BS,
                     "num_classes": NC},
            "training": {"learning_rate": 0.05, "lr_decay": 0.99, "local_epochs": 1,
                         "plocal_epochs": 1, "head_lr_rep": 0.01},
            "backdoor": {"target_label": 2, "poison_ratio": 0.5, "badpfl_gen_steps": 2}}


def _model():
    L = tf.keras.layers
    inp = L.Input((IMG, IMG, 3))
    x = L.Conv2D(4, 3, padding="same", name="c1")(inp)
    x = L.BatchNormalization(name="bn1")(x)
    x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, L.Dense(NC, activation="softmax", name="head")(x))


def _dataset(seed=0, n=32):
    r = np.random.default_rng(seed)
    return (tf.data.Dataset.from_tensor_slices(
        (r.normal(size=(n, IMG, IMG, 3)).astype("float32"),
         r.integers(0, NC, n).astype("int32")))
        .shuffle(n).batch(BS, drop_remainder=True))


def _client(cid=0, malicious=False, cls=HierFedRepClient):
    c = compose_client_class(cls, None, BadPFLMixin if malicious else None)(
        client_id=cid, dataset=_dataset(cid), model=_model(), config=_config())
    c.is_malicious = malicious
    return c


def _maxdiff(a, b):
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(a, b))


def test_global_and_edge_weights_roundtrip(tmp_path):
    gw = [w.copy() for w in _model().get_weights()]
    ew = {0: [w.copy() for w in _model().get_weights()],
          3: [w.copy() for w in _model().get_weights()]}
    ck.save_checkpoint(tmp_path / "t", round_idx=7, global_weights=gw,
                       edge_weights=ew, clients=[])
    loaded = ck.load_checkpoint(tmp_path / "t")
    assert loaded["manifest"]["round"] == 7
    assert _maxdiff(gw, ck.global_weights(loaded)) == 0.0
    assert _maxdiff(ew[3], ck.edge_weights(loaded, 3)) == 0.0
    assert loaded["manifest"]["edges"] == [0, 3]


def test_client_model_and_private_head_roundtrip(tmp_path):
    """FedRep 的私有 head 必须往返 —— 丢了它，复现出来的起点就不是第 t 轮的起点。"""
    src = _client(cid=5)
    th = [w.copy() for w in _model().get_weights()]
    src.set_weights(th, th)
    src.local_train(1)                     # 让私有 head 真的被训出来
    head_before = [w.copy() for w in src._head_weights]
    model_before = [w.copy() for w in src.model.get_weights()]

    ck.save_checkpoint(tmp_path / "t", round_idx=3, global_weights=th,
                       edge_weights={0: th}, clients=[src])

    dst = _client(cid=5)                   # 全新对象，状态与 src 无关
    assert ck.apply_to_clients(ck.load_checkpoint(tmp_path / "t"), [dst]) == [5]
    assert _maxdiff(model_before, dst.model.get_weights()) == 0.0
    assert _maxdiff(head_before, dst._head_weights) == 0.0


def test_private_head_is_actually_nontrivial_in_that_test(tmp_path):
    """反向守卫：若 head 从未被训出来，上一条会在比较两份初始化上平凡通过。"""
    c = _client(cid=5)
    th = [w.copy() for w in _model().get_weights()]
    c.set_weights(th, th)
    before = [w.copy() for w in c._head_weights]
    c.local_train(1)
    assert _maxdiff(before, c._head_weights) > 0.0


def test_client_rng_state_roundtrip(tmp_path):
    src = _client(cid=5)
    src.rng.integers(0, 100, 17)           # 推进随机流
    state = src.rng.bit_generator.state
    th = [w.copy() for w in _model().get_weights()]
    ck.save_checkpoint(tmp_path / "t", round_idx=1, global_weights=th,
                       edge_weights={0: th}, clients=[src])
    dst = _client(cid=5)
    ck.apply_to_clients(ck.load_checkpoint(tmp_path / "t"), [dst])
    assert dst.rng.bit_generator.state == state


def test_is_malicious_roundtrip(tmp_path):
    src = _client(cid=5, malicious=True)
    th = [w.copy() for w in _model().get_weights()]
    ck.save_checkpoint(tmp_path / "t", round_idx=1, global_weights=th,
                       edge_weights={0: th}, clients=[src])
    dst = _client(cid=5, malicious=True)
    dst.is_malicious = False
    ck.apply_to_clients(ck.load_checkpoint(tmp_path / "t"), [dst])
    assert dst.is_malicious is True


def test_shared_generator_and_its_adam_slots_roundtrip(tmp_path):
    """
    共享 generator 不存 → 复现出来的 poison twin 拿到**随机初始化**的生成器，
    跑的是第 0 轮的攻击而不是第 t 轮的攻击。Adam 的 slot 同理（动量是有状态的）。
    """
    gen = build_autoencoder(img_size=IMG, channels=3)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(gen(tf.zeros((2, IMG, IMG, 3)), training=True) ** 2)
    opt.apply_gradients(zip(tape.gradient(loss, gen.trainable_variables),
                            gen.trainable_variables))
    gen_before = [w.copy() for w in gen.get_weights()]
    opt_before = {v.name: v.numpy().copy() for v in opt.variables()}

    th = [w.copy() for w in _model().get_weights()]
    ck.save_checkpoint(tmp_path / "t", round_idx=1, global_weights=th,
                       edge_weights={0: th}, clients=[],
                       shared_generator=gen, shared_gen_opt=opt)

    gen2 = build_autoencoder(img_size=IMG, channels=3)
    opt2 = tf.keras.optimizers.Adam(learning_rate=0.01)
    opt2.build(gen2.trainable_variables)
    assert _maxdiff(gen_before, gen2.get_weights()) > 0.0        # 确认初值不同
    assert ck.apply_to_generator(ck.load_checkpoint(tmp_path / "t"), gen2, opt2) is True
    assert _maxdiff(gen_before, gen2.get_weights()) == 0.0
    for v in opt2.variables():
        if v.name in opt_before:
            np.testing.assert_allclose(v.numpy(), opt_before[v.name], rtol=0, atol=0)


def test_apply_to_generator_is_a_noop_when_checkpoint_has_none(tmp_path):
    th = [w.copy() for w in _model().get_weights()]
    ck.save_checkpoint(tmp_path / "t", round_idx=1, global_weights=th,
                       edge_weights={0: th}, clients=[])
    gen = build_autoencoder(img_size=IMG, channels=3)
    assert ck.apply_to_generator(ck.load_checkpoint(tmp_path / "t"), gen) is False


def test_manifest_json_is_written_separately(tmp_path):
    """回程红线：`.npz` 留集群，只有 manifest 允许进 git —— 它必须是独立文件。"""
    import json
    th = [w.copy() for w in _model().get_weights()]
    ck.save_checkpoint(tmp_path / "t", round_idx=9, global_weights=th,
                       edge_weights={0: th}, clients=[],
                       malicious_ids=[3, 1], run_info={"method": "hier_fedrep"})
    man = json.loads((tmp_path / "t.manifest.json").read_text(encoding="utf-8"))
    assert man["round"] == 9
    assert man["malicious_ids"] == [1, 3]
    assert man["run"]["method"] == "hier_fedrep"


def test_unknown_probe_state_key_is_rejected_loudly():
    """
    未实现 set_probe_state 的方法收到状态必须**报错**。忽略等于静默把个性化状态
    清零，而没有任何日志会提到它（陷阱 #7 那一类）。
    """
    c = _client(cid=0, cls=FedAvgClient)
    with pytest.raises(ValueError, match="set_probe_state"):
        c.set_probe_state({"head_weights": [np.zeros(3)]})


def test_methods_without_personal_state_accept_empty_probe_state():
    c = _client(cid=0, cls=FedAvgClient)
    assert c.get_probe_state() == {}
    c.set_probe_state({})                      # 不该报错


def test_fedrep_rejects_unknown_key():
    c = _client(cid=0)
    with pytest.raises(ValueError, match="不认识"):
        c.set_probe_state({"nope": []})


def test_apply_to_clients_skips_ids_not_present(tmp_path):
    src = _client(cid=5)
    th = [w.copy() for w in _model().get_weights()]
    ck.save_checkpoint(tmp_path / "t", round_idx=1, global_weights=th,
                       edge_weights={0: th}, clients=[src])
    other = _client(cid=99)
    assert ck.apply_to_clients(ck.load_checkpoint(tmp_path / "t"), [other]) == []
