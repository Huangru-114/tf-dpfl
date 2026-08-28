"""
tests/test_probe_end_to_end.py  —  build_world → 落 checkpoint → 跑探针 的**接线**测试（需 TF）。

CLAUDE.md：「本仓库已知的失败有一半在『接线』而不是算法本身」。上面几个 L1 文件各自
证明了一个零件是对的；这个文件证明**它们连起来还是对的** —— 尤其是：

  * `main.build_world` 抽出来之后，探针重建的世界与训练时是**同一批**客户端
  * `BackdoorCloudServer` 真的在配置的轮次落盘（而且是**进入该轮时**的状态）
  * `probe.run_probe` 能把 checkpoint 灌回一个全新建出来的世界并算出判据

数据用合成的 CIFAR 形状张量（monkeypatch keras 的下载）—— 本测试查的是接线，
不是学习效果；真数据只会让它慢十倍而断言一条都不变。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")


@pytest.fixture
def synthetic_cifar(monkeypatch):
    """替掉 keras 的数据下载，保持 load_cifar10 自己的归一化/增强管线不动。"""
    def _fake():
        r = np.random.default_rng(0)
        return ((r.integers(0, 256, (240, 32, 32, 3), dtype=np.uint8),
                 r.integers(0, 10, (240, 1)).astype(np.int64)),
                (r.integers(0, 256, (80, 32, 32, 3), dtype=np.uint8),
                 r.integers(0, 10, (80, 1)).astype(np.int64)))
    monkeypatch.setattr(tf.keras.datasets.cifar10, "load_data", _fake)


@pytest.fixture
def tiny_config(tmp_path):
    cfg = yaml.safe_load(
        (ROOT / "experiments/attack/bad-pfl/exp03/probe_smoke.yaml").read_text())
    cfg["federation"].update({"n_clients": 4, "n_edges": 2, "edge_rounds": 1,
                              "n_rounds": 2, "min_samples": 5})
    cfg["data"].update({"batch_size": 8, "shuffle_buffer": 32})
    cfg["backdoor"].update({"n_malicious": 2, "badpfl_gen_steps": 1,
                            "eval_interval": 99})     # 跳过重型后门评估
    # PM 评估要遍历每个客户端的测试集，占 run_round 的绝大部分时间。
    # 本文件查的是接线，不是评估质量 —— 关掉它把 L1 从 4 分钟压到几十秒。
    cfg["evaluation"]["eval_interval"] = 99
    cfg["training"].update({"local_epochs": 1, "plocal_epochs": 1})
    # 本文件查接线，不查模型。用最小的 arch —— BN 相关的覆盖在
    # test_probe_layermap.py（结构解析）与 test_probe_determinism.py（真训练）里。
    cfg["model"]["arch"] = "fedavg_cnn"
    cfg["probe"] = {"checkpoint_rounds": [1], "checkpoint_dir": str(tmp_path / "ck"),
                    "tag": "e2e"}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return p


def _build(cfg_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(cfg_path)])
    import main
    config = main.load_config()
    main.set_seed(config.get("seed", 42))
    return config, main.build_world(config)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """
    **共享一次训练**：run_round 里的本地训练 + 评估是本文件最贵的一步，
    每个测试各跑一次会把 L1 从几十秒推到几分钟。这些断言只读产物、互不干扰，
    共享安全。需要**不同 config** 的测试（空 checkpoint_rounds / vanilla）另跑。
    """
    mp = pytest.MonkeyPatch()
    r = np.random.default_rng(0)
    mp.setattr(tf.keras.datasets.cifar10, "load_data", lambda: (
        (r.integers(0, 256, (240, 32, 32, 3), dtype=np.uint8),
         r.integers(0, 10, (240, 1)).astype(np.int64)),
        (r.integers(0, 256, (80, 32, 32, 3), dtype=np.uint8),
         r.integers(0, 10, (80, 1)).astype(np.int64))))

    tmp = tmp_path_factory.mktemp("e2e")
    cfg = yaml.safe_load(
        (ROOT / "experiments/attack/bad-pfl/exp03/probe_smoke.yaml").read_text())
    cfg["federation"].update({"n_clients": 4, "n_edges": 2, "edge_rounds": 1,
                              "n_rounds": 2, "min_samples": 5})
    cfg["data"].update({"batch_size": 8, "shuffle_buffer": 32})
    cfg["backdoor"].update({"n_malicious": 2, "badpfl_gen_steps": 1,
                            "eval_interval": 99})
    cfg["evaluation"]["eval_interval"] = 99
    cfg["training"].update({"local_epochs": 1, "plocal_epochs": 1})
    cfg["model"]["arch"] = "fedavg_cnn"
    cfg["probe"] = {"checkpoint_rounds": [1], "checkpoint_dir": str(tmp / "ck"),
                    "tag": "e2e"}
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))

    config, world = _build(cfg_path, mp)
    gw_before = [w.copy() for w in world["global_model"].get_weights()]
    world["cloud"].run_round(1)
    yield {"config": config, "world": world, "cfg_path": cfg_path,
           "gw_before": gw_before, "tmp": tmp, "mp": mp}
    mp.undo()


def test_build_world_returns_a_complete_and_consistent_world(trained):
    """
    build_world 是从 run_experiment 里抽出来的。抽漏一个返回值不会报错 ——
    只会让探针重建出一个**看起来一样但其实不同**的世界。
    """
    world = trained["world"]
    assert set(world) >= {"config", "global_model", "clients", "edge_servers",
                          "test_clients", "cloud", "g_test_ds", "test_ds",
                          "x_test", "y_test", "malicious_ids", "eval_trigger",
                          "bd_enabled"}
    assert len(world["clients"]) == 4
    assert len(world["edge_servers"]) == 2
    assert len(world["malicious_ids"]) == 2
    # 每个客户端恰好属于一个 edge，且并集是全部客户端
    grouped = [int(c.client_id) for e in world["edge_servers"] for c in e.clients]
    assert sorted(grouped) == sorted(int(c.client_id) for c in world["clients"])


def test_build_world_is_reproducible_given_the_same_seed(synthetic_cifar,
                                                         tiny_config, monkeypatch):
    """
    探针的全部前提：同 config + 同 seed 重建出**同一批**客户端与同一套数据分区。
    不成立的话，「灌回 checkpoint 的那个 3 号客户端」根本不是训练时的 3 号客户端。
    """
    out = []
    for _ in range(2):
        _, w = _build(tiny_config, monkeypatch)
        out.append((
            sorted(int(i) for i in w["malicious_ids"]),
            [(int(c.client_id), int(c.n_samples)) for c in w["clients"]],
            [[int(c.client_id) for c in e.clients] for e in w["edge_servers"]],
        ))
    assert out[0] == out[1]


def test_checkpoint_is_written_at_the_configured_round(trained):
    ckdir = Path(trained["config"]["probe"]["checkpoint_dir"])
    assert (ckdir / "e2e_r0001.npz").exists()
    assert (ckdir / "e2e_r0001.manifest.json").exists()
    man = json.loads((ckdir / "e2e_r0001.manifest.json").read_text())
    assert man["round"] == 1
    assert len(man["clients"]) == 4
    assert man["run"]["method"] == "hier_fedrep"
    assert man["run"]["strategy"] == "badpfl"


def test_checkpoint_holds_the_state_entering_the_round_not_leaving_it(trained):
    """
    语义必须是「进入第 t 轮时」。存轮末状态的话，探针 replay 的其实是第 t+1 轮，
    与日志里的 ASR/MTA 曲线整整错开一轮 —— 而这不会报任何错。
    """
    from probe import checkpoint as ck
    loaded = ck.load_checkpoint(
        Path(trained["config"]["probe"]["checkpoint_dir"]) / "e2e_r0001.npz")
    saved = ck.global_weights(loaded)
    after = trained["world"]["global_model"].get_weights()

    def md(a, b):
        return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
                   for x, y in zip(a, b))

    assert md(saved, trained["gw_before"]) == 0.0        # 与轮首一致
    assert md(trained["gw_before"], after) > 0.0         # 且这一轮确实训练了


def test_checkpoint_gate_is_off_by_default(synthetic_cifar, tiny_config, monkeypatch):
    """
    缺省关闭 = 零开销：`probe.checkpoint_rounds` 为空时，服务器上的门必须是空集合，
    run_round 里那个 `if round_idx in self.probe_ckpt_rounds` 永不为真。
    （不跑 run_round —— 门本身就是被测对象，跑一轮只是重复上面几条的开销。）
    """
    cfg = yaml.safe_load(tiny_config.read_text())
    cfg["probe"]["checkpoint_rounds"] = []
    tiny_config.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    _, world = _build(tiny_config, monkeypatch)
    assert world["cloud"].probe_ckpt_rounds == set()


def test_checkpoint_gate_is_on_when_configured(trained):
    assert trained["world"]["cloud"].probe_ckpt_rounds == {1}


def test_probe_runs_end_to_end_and_emits_verdicts(trained, tmp_path):
    ckpt = Path(trained["config"]["probe"]["checkpoint_dir"]) / "e2e_r0001.npz"
    from probe.run_probe import main as probe_main
    trained["mp"].setattr(sys, "argv",
                          ["run_probe", "--config", str(trained["cfg_path"])])
    out = tmp_path / "out"
    assert probe_main(["--checkpoint", str(ckpt), "--out", str(out),
                       "--benign", "1", "--malicious", "1"]) == 0

    payload = json.loads((out / "probe_summary_r0001.json").read_text())
    assert payload["client_failures"] == []
    assert payload["run"]["round"] == 1
    assert payload["run"]["method"] == "hier_fedrep"
    # 判据 1：良性端两条 twin 代码路径逐字节相同 → ‖Δθ_BD‖ 必须精确为 0
    assert payload["verdicts"]["instrument_ok"]["pass"] is True
    assert payload["verdicts"]["instrument_ok"]["benign_bd_norm"]["max"] == 0.0
    # 恶意端必须真的有位移（否则开关没接上）
    assert payload["verdicts"]["signal"]["malicious_bd_over_stochastic"]["n"] == 1
    assert payload["verdicts"]["signal"]["malicious_bd_over_stochastic"]["min"] > 0.0

    import csv
    rows = list(csv.DictReader((out / "probe_rows_r0001.csv").open()))
    assert rows
    assert {"backbone", "head"} == {r["scope"] for r in rows}
    assert {"upload", "personal"} == {r["weight_space"] for r in rows}
    assert {"benign", "malicious"} == {r["role"] for r in rows}


def test_probe_refuses_dataset_baked_attacks(synthetic_cifar, tiny_config, monkeypatch):
    """
    vanilla 的投毒不经钩子 → Δθ_BD 恒为 0（假阴性）。入口必须拒绝，而不是产出一份
    看起来正常、结论是"后门无位移"的 CSV。
    """
    cfg = yaml.safe_load(tiny_config.read_text())
    cfg["backdoor"]["malicious_strategy"] = "vanilla"
    tiny_config.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    from probe.run_probe import main as probe_main
    monkeypatch.setattr(sys, "argv", ["run_probe", "--config", str(tiny_config)])
    with pytest.raises(ValueError, match="不经 on_batch"):
        probe_main(["--checkpoint", "unused.npz", "--out", "unused"])
