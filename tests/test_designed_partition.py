"""
tests/test_designed_partition.py  —  S3：改版实验 3 的新划分（data/designed_partition.py；D-062 … D-068）

本地（纯 numpy，不 import TF）：
  A. 划分的硬约束：等大小、无放回（训练 / 留出 / 干净集两两不相交）、不超过每类供给、只用自己的 rng；
  B. 设计：机构式表的结构、C1–C4 只改 airplane 一列、H 的手算值；
  C. 配置错误都被拦下（config_errors + validate_config）；
  D. 自描述：[Partition] / [PartitionEdge] 被 collect_metrics 解析（schema 6）；
  E. 旧路径逐字节不变：build_clients 的旧分支、切分语句、data/partition.py 的函数与 S3 之前的 AST 指纹相同。
集群（需要 TF）：
  F. build_clients 走 S3 分支：assignments 被 baked、每端 375 训练 / 125 留出、by_edge 布点落在 E0。
"""

import ast
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from config_validate import ConfigError, validate_config
from data import designed_partition as D

ROOT = Path(__file__).resolve().parent.parent
FEDAVG = ROOT / "fedavg"
sys.path.insert(0, str(ROOT / "harness"))
from collect_metrics import collect   # noqa: E402

LABELS = np.repeat(np.arange(D.K), D.SUPPLY_PER_CLASS)      # 合并池的类计数（顺序无关）
SEEDS = (42, 43)
CASES = ([("designed", {"condition": c, "strength": 0.25}) for c in D.CONDITIONS]
         + [("hdir", {"alpha_edge": a}) for a in (0.1, 0.3, 1.0, 10.0)]
         + [("equal_random", {})])


def _cfg(kind, design=None, **fed):
    c = {"seed": 42,
         "data": {"dataset": "cifar10", "per_client_test_ratio": 0.25},
         "federation": {"partition": kind, "n_clients": 100, "n_edges": 4,
                        "design": dict(design or {})},
         "backdoor": {"target_label": 0}}
    c["federation"].update(fed)
    return c


@pytest.fixture(scope="module")
def splits():
    return {(kind, str(d), s): D.designed_partition(LABELS, _cfg(kind, d), seed=s)
            for kind, d in CASES for s in SEEDS}


# ── A. 硬约束 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind,design", CASES)
def test_equal_size_and_disjoint(splits, kind, design):
    for s in SEEDS:
        sp = splits[(kind, str(design), s)]
        assert [len(t) for t in sp["train_indices"]] == [375] * 100
        assert [len(t) for t in sp["test_indices"]] == [125] * 100
        assert [len(c) for c in sp["clean_indices"]] == [500] * 4
        allidx = np.concatenate(sp["train_indices"] + sp["test_indices"] + sp["clean_indices"])
        assert len(np.unique(allidx)) == len(allidx) == 100 * 500 + 4 * 500      # 无放回、两两不相交
        assert allidx.min() >= 0 and allidx.max() < len(LABELS)
        assert sp["assignments"] == [e for e in range(4) for _ in range(25)]
        assert sp["info"]["max_class_use"] <= D.SUPPLY_PER_CLASS
        # 计数矩阵与索引一致
        for i in range(100):
            idx = np.concatenate([sp["train_indices"][i], sp["test_indices"][i]])
            assert (np.bincount(LABELS[idx], minlength=D.K) == sp["client_counts"][i]).all()


def test_same_seed_is_identical_and_other_seed_differs(splits):
    a = splits[("designed", str({"condition": "C3", "strength": 0.25}), 42)]
    b = D.designed_partition(LABELS, _cfg("designed", {"condition": "C3", "strength": 0.25}), seed=42)
    assert a["info"]["index_sha"] == b["info"]["index_sha"]
    assert all((x == y).all() for x, y in zip(a["train_indices"], b["train_indices"]))
    c = splits[("designed", str({"condition": "C3", "strength": 0.25}), 43)]
    assert c["info"]["index_sha"] != a["info"]["index_sha"]


def test_does_not_touch_the_global_numpy_stream():
    """旧划分（noniid 等）用全局 np.random；S3 只用自己的 default_rng([seed, RNG_TAG])。"""
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    D.designed_partition(LABELS, _cfg("hdir", {"alpha_edge": 0.3}))
    assert (np.random.get_state()[1] == before).all()


def test_equal_random_uses_every_class_equally():
    sp = D.designed_partition(LABELS, _cfg("equal_random"))
    assert sp["client_counts"].sum(0).tolist() == [5000] * D.K


def test_malicious_data_share_is_exactly_ten_percent():
    """D-027 的目的：名义 10% 的恶意端，数据占比也恰好 10%（F-028 里是 0.092–0.142）。"""
    sp = D.designed_partition(LABELS, _cfg("designed", {"condition": "C1"}))
    sizes = [len(t) for t in sp["train_indices"]]
    assert D.malicious_data_share(sizes, range(3, 13)) == 0.1
    assert D.malicious_data_share(sizes, []) == 0.0


# ── B. 设计 ─────────────────────────────────────────────────────────────────
def test_institutional_table_structure():
    T = D.base_table(0.25)
    assert np.allclose(T.sum(1), 1.0)
    assert np.allclose(T[0], 0.10)                                     # E0 均衡
    for e, pair in D.SPECIALTY.items():
        assert T[e, list(pair)].tolist() == [0.25, 0.25]
        others = [c for c in range(D.K) if c not in pair and c not in D.UNIFORM]
        assert np.allclose(T[e, others], 0.025)
    assert np.allclose(T[:, list(D.UNIFORM)], 0.10)                    # airplane/bird/ship/frog
    assert D.SPECIALTY[3] == (4, 7)                                    # E3 = deer + horse（D-067）


@pytest.mark.parametrize("cond", sorted(D.CONDITIONS))
def test_conditions_change_only_the_target_column(cond):
    T1, Tc = D.condition_table("C1"), D.condition_table(cond)
    assert np.allclose(Tc.sum(1), 1.0)
    assert np.allclose(Tc[:, D.TARGET], D.CONDITIONS[cond])
    rest = [c for c in range(D.K) if c != D.TARGET]
    for e in range(4):                                                 # 非目标部分相对组成不变
        assert np.allclose(Tc[e, rest] / Tc[e, rest].sum(), T1[e, rest] / T1[e, rest].sum())


@pytest.mark.parametrize("cond", sorted(D.CONDITIONS))
def test_realized_target_share_matches_the_table(splits, cond):
    sp = splits[("designed", str({"condition": cond, "strength": 0.25}), 42)]
    got = [e["yt_share"] for e in sp["per_edge"]]
    assert np.allclose(got, D.CONDITIONS[cond], atol=2 / 12500)


def test_h_inter_of_c1_matches_hand_computation(splits):
    """r = 0.25、q = 0.025：E0–受害 TV = (r−0.1) + 2(0.1−q) = 0.30；受害–受害 = 2(r−q) = 0.45 → 均值 0.375。"""
    assert D.h_inter(D.base_table(0.25)) == pytest.approx(0.375)
    sp = splits[("designed", str({"condition": "C1", "strength": 0.25}), 42)]
    assert sp["info"]["h_inter"] == pytest.approx(0.375, abs=1e-9)


def test_h_intra_hand_example():
    """两个 edge、每 edge 两端：edge 分布为 [.5,.5]，端为 [1,0] 与 [0,1] → 每端 TV = 0.5。"""
    counts = np.zeros((4, D.K))
    counts[0, 0] = counts[1, 1] = 10          # edge 0
    counts[2, 2] = counts[3, 2] = 10          # edge 1：两端相同
    hi, ha, pe = D.h_inter_intra(counts, [0, 0, 1, 1])
    assert hi == pytest.approx(1.0)
    assert ha == pytest.approx((0.5 + 0.0) / 2)


def test_hdir_levels_are_ordered(splits):
    """投影会让实测 H 比名义平（F-057）；四档仍要按 α_e 单调（seed 平均）。"""
    h = {a: np.mean([splits[("hdir", str({"alpha_edge": a}), s)]["info"]["h_inter"] for s in SEEDS])
         for a in (0.1, 0.3, 1.0, 10.0)}
    assert h[0.1] > h[0.3] > h[1.0] > h[10.0]


def test_lr_round_and_int_matrix():
    assert D.lr_round([0.4, 0.4, 0.2], 1).sum() == 1
    assert D.lr_round([2.6, 1.2, 0.2], 4).tolist() == [3, 1, 0]
    M = D.int_matrix(np.full((3, 2), 5.0), [10, 10, 10], [15, 15])
    assert M.sum(1).tolist() == [10, 10, 10] and M.sum(0).tolist() == [15, 15]
    with pytest.raises(ValueError):
        D.int_matrix(np.ones((2, 2)), [10, 10], [5, 5])


# ── C. 配置错误 ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mutate,needle", [
    (lambda c: c["federation"].update(n_edges=2), "4 edge"),
    (lambda c: c["federation"]["design"].update(condition="C9"), "condition"),
    (lambda c: c["federation"]["design"].update(strength=0.35), "strength"),
    (lambda c: c["backdoor"].update(target_label=1), "target_label"),
    (lambda c: c["federation"]["design"].update(n_per_client=501), "不是整数"),
    (lambda c: c["federation"]["design"].update(n_per_client=580), "超出每类供给"),   # F-057：C2 需 6017
    (lambda c: c["federation"]["design"].update(n_per_client=600), "合并池"),
    (lambda c: c["data"].update(dataset="cifar100"), "CIFAR-10"),
    (lambda c: c["federation"].update(n_clients=99), "整除"),
])
def test_designed_config_errors(mutate, needle):
    c = _cfg("designed", {"condition": "C2"})
    assert D.config_errors(c) == []
    mutate(c)
    errs = D.config_errors(c)
    assert errs and any(needle in e for e in errs), errs


def test_hdir_needs_alpha_edge_and_old_partitions_are_not_checked():
    assert any("alpha_edge" in e for e in D.config_errors(_cfg("hdir")))
    assert D.config_errors(_cfg("noniid", {"condition": "nonsense"})) == []


def _full_cfg(**fed):
    import yaml
    cfg = yaml.safe_load((ROOT / "experiments/attack/hfl-mechanism/configs/G3__C3__s42.yaml")
                         .read_text(encoding="utf-8"))
    cfg["federation"].update(fed)
    return cfg


def test_validate_config_accepts_g3_and_rejects_a_broken_design(capsys):
    validate_config(_full_cfg())
    bad = _full_cfg()
    bad["federation"]["design"]["condition"] = "C7"
    with pytest.raises(ConfigError, match="S3 划分配置不合法"):
        validate_config(bad)


# ── D. 自描述 ───────────────────────────────────────────────────────────────
def test_partition_lines_are_parsed_by_collect_metrics():
    sp = D.designed_partition(LABELS, _cfg("designed", {"condition": "C3"}))
    lines = D.data_lines(sp, malicious_ids=range(10))
    m = collect("\n".join(["[Config] loading x.yaml", *lines, "[Round 1]"]) + "\n")
    assert m["schema_version"] == 16
    run = m["run"]
    assert run["partition"] == "designed" and run["partition_condition"] == "C3"
    assert run["partition_n"] == 500 and run["partition_alpha_edge"] is None
    data = run["data"]
    assert data["malicious_data_share"] == 0.1 and data["client_size_min"] == data["client_size_max"] == 500
    assert data["index_sha"] == sp["info"]["index_sha"]
    assert [e["edge_id"] for e in data["per_edge"]] == [0, 1, 2, 3]
    assert data["per_edge"][3]["yt_share"] == pytest.approx(0.005, abs=1e-4)
    assert data["per_edge"][3]["class_counts"] == sp["client_counts"][75:].sum(0).tolist()
    assert sum(data["per_edge"][0]["clean_counts"]) == 500


def test_old_logs_without_partition_lines_give_none():
    """旧划分不打 [Partition]；dataset.py 自己的 `[Data] …` 行不能被误读成划分。"""
    log = "[Config] loading x.yaml\n[Data] CIFAR-100 | train=50000 test=10000 classes=100\n[Round 1]\n"
    run = collect(log)["run"]
    assert run["data"] is None and run["partition"] is None and run["partition_n"] is None


# ── E. 旧路径逐字节不变（指纹在改动前的 `6c152647` 上记录）────────────────────
def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _fn(path: Path, name: str):
    return next(n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(n, ast.FunctionDef) and n.name == name)


OLD_BRANCHES = {"iid": "bdb6c1f866a89472", "noniid": "43b405d102285436",
                "pathological": "c38260b9636360fe", "hierarchical": "48cb13cb81994427",
                "superclass_pathological": "854c1d257b85d15b"}
OLD_SPLIT_STMTS = "7f96f5c3f1d728f4"
OLD_PARTITION_FNS = {"make_client_dataset": "16be6450f5a44e03",
                     "split_client_train_test": "6f20a661a08f24a6",
                     "noniid_partition": "44791c689bee3e8a",
                     "iid_partition": "4bd5c363b14656fe",
                     "pathological_noniid_partition": "fe152ccacccbc263"}


def test_old_partition_branches_are_untouched():
    fn = _fn(FEDAVG / "main.py", "build_clients")
    chain = next(s for s in fn.body if isinstance(s, ast.If) and "partition" in ast.dump(s.test))
    got, node = {}, chain
    while isinstance(node, ast.If):
        comp = node.test.comparators[0] if isinstance(node.test, ast.Compare) else None
        if isinstance(comp, ast.Constant):
            got[comp.value] = _h("\n".join(ast.dump(b) for b in node.body))
        node = node.orelse[0] if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If) else None
    assert {k: got.get(k) for k in OLD_BRANCHES} == OLD_BRANCHES


def test_old_split_runs_only_when_not_s3_and_is_untouched():
    fn = _fn(FEDAVG / "main.py", "build_clients")
    guard = [s for s in ast.walk(fn) if isinstance(s, ast.If) and ast.unparse(s.test) == "s3 is None"]
    assert len(guard) == 1, "应恰有一处 `if s3 is None:` 包住旧的 train/test 切分"
    assert _h("\n".join(ast.dump(s) for s in guard[0].body)) == OLD_SPLIT_STMTS


def test_old_partition_functions_are_untouched():
    path = FEDAVG / "data" / "partition.py"
    assert {n: _h(ast.dump(_fn(path, n))) for n in OLD_PARTITION_FNS} == OLD_PARTITION_FNS


# ── F. 集群（需要 TF）：build_clients 的 S3 分支 ─────────────────────────────
def test_build_clients_s3_branch_bakes_assignments_and_equal_sizes():
    tf = pytest.importorskip("tensorflow")
    import main as M
    from attack.backdoor import assign_malicious_by_edge
    images = np.zeros((len(LABELS), 4, 4, 3), dtype=np.float32)
    cfg = _cfg("designed", {"condition": "C3"})
    cfg["data"].update(batch_size=32, img_size=4, num_classes=10)
    cfg["federation"].update(edge_assignment="block", client_fraction=0.1)
    cfg["training"] = {"drift_correction": "hierfedavg", "learning_rate": 0.1, "lr_decay": 1.0,
                       "local_epochs": 1}
    cfg["backdoor"] = {"enabled": False, "target_label": 0}
    model = tf.keras.Sequential([tf.keras.Input((4, 4, 3)), tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(10, activation="softmax")])
    out = {}
    clients, assignments, _ = M.build_clients(images, LABELS, model, copy.deepcopy(cfg), s3_out=out)
    assert assignments == [e for e in range(4) for _ in range(25)]
    assert [c.n_samples for c in clients] == [375] * 100
    assert [len(c) for c in out["clean_indices"]] == [500] * 4
    mal = assign_malicious_by_edge(assignments, [10, 0, 0, 0], seed=42)
    assert len(mal) == 10 and all(i < 25 for i in mal)


# ── G. 离线预览工具（harness/partition_preview.py = F0 的数据）──────────────
def test_partition_preview_reuses_the_training_function():
    import partition_preview as PV
    rows = PV.preview(seeds=(42,))
    assert [r["case"] for r in rows] == ["C1", "C2", "C3", "C4", "hdir-a10", "hdir-a1",
                                         "hdir-a0.3", "hdir-a0.1", "random",
                                         "C1-fixed0.2", "C1-fixed0.4", "C1-fixed0.6"]
    c1 = rows[0]
    assert c1["h_inter"] == pytest.approx(0.375) and c1["yt_share"] == [0.1] * 4
    same = D.designed_partition(LABELS, PV.config("designed", {"condition": "C1", "strength": 0.25}), seed=42)
    assert c1["index_sha"] == same["info"]["index_sha"]


def test_keras_shaped_labels_give_the_same_split():
    """keras 的标签是 (N, 1) uint8；main.py 里已经 squeeze 过，这里再兜一次底。"""
    cfg = _cfg("designed", {"condition": "C2"})
    a = D.designed_partition(LABELS, cfg)
    b = D.designed_partition(LABELS.astype(np.uint8).reshape(-1, 1), cfg)
    assert a["info"]["index_sha"] == b["info"]["index_sha"]


def test_repeated_partition_block_is_not_double_counted():
    sp = D.designed_partition(LABELS, _cfg("designed", {"condition": "C1"}))
    lines = D.data_lines(sp)
    run = collect("\n".join(["[Config] loading x.yaml", *lines, *lines, "[Round 1]"]) + "\n")["run"]
    assert len(run["data"]["per_edge"]) == 4


# ── H. 固定客户端画像（阶段三 D-130：体现「组内更同质」；F-125）──────────────────
FIXED = {"condition": "C1", "strength": 0.25, "client_profile": "fixed", "profile_lambda": 0.4}

# 反向锚点：改动前（`7c1096f`）在合成标签上记录的 index_sha —— 加固定画像之后旧路径逐字节不变
OLD_INDEX_SHA = {("designed", "C1", 42): "ed0e36c40db5", ("designed", "C1", 43): "1647f6c2ead3",
                 ("designed", "C3", 42): "4076ce940b76", ("designed", "C3", 43): "a0c64e28a750",
                 ("hdir", 0.3, 42): "063cdafd4ec0", ("hdir", 0.3, 43): "135d6befc9c0",
                 ("equal_random", None, 42): "691542207cc7", ("equal_random", None, 43): "a007ad63f569"}


@pytest.mark.parametrize("key", sorted(OLD_INDEX_SHA, key=str))
def test_old_client_layer_is_byte_identical(key):
    import partition_preview as PV
    kind, arg, seed = key
    design = ({"condition": arg, "strength": 0.25} if kind == "designed"
              else {"alpha_edge": arg} if kind == "hdir" else {})
    sp = D.designed_partition(LABELS, PV.config(kind, design), seed=seed)
    assert sp["info"]["index_sha"] == OLD_INDEX_SHA[key]
    assert "client_profile" not in sp["info"]          # 旧 [Partition] 行逐字不变：不多出字段


def test_profile_matrix_by_hand():
    pool = np.array([100, 100, 200, 0, 100, 100, 100, 100, 100, 100], dtype=float)
    Q = D.profile_matrix(pool, 6, 500, 0.4)
    assert Q.shape == (6, 10) and np.allclose(Q.sum(1), 500)
    # 第 0 端：0.6·p + 0.4·(类 0、1 各 0.5)；p_0 = 0.1
    assert Q[0, 0] == pytest.approx(500 * (0.6 * 0.1 + 0.2))
    assert Q[0, 2] == pytest.approx(500 * 0.6 * 0.2)
    assert Q[1, 3] == pytest.approx(500 * 0.2)          # 第 1 端偏重 (2, 3)；池里没有类 3 → 名义上仍是 0.2
    assert np.allclose(Q[5], Q[0])                       # c mod 5 轮换
    assert np.allclose(D.profile_matrix(pool, 3, 500, 0.0), np.tile(pool / pool.sum() * 500, (3, 1)))


@pytest.mark.parametrize("seed", SEEDS)
def test_fixed_profiles_keep_the_hard_constraints(seed):
    sp = D.designed_partition(LABELS, _cfg("designed", FIXED), seed=seed)
    cc = sp["client_counts"]
    assert cc.sum(1).tolist() == [500] * 100
    allidx = np.concatenate(sp["train_indices"] + sp["test_indices"] + sp["clean_indices"])
    assert len(np.unique(allidx)) == len(allidx)
    assert (np.bincount(LABELS[allidx], minlength=10) <= D.SUPPLY_PER_CLASS).all()
    # edge 层不动：H_inter 与 C1 的机构表相同、各 edge 的类计数与 Dirichlet 版相同
    mainline = D.designed_partition(LABELS, _cfg("designed", {"condition": "C1"}), seed=seed)
    assert sp["info"]["h_inter"] == pytest.approx(0.375)
    for e in range(4):
        assert (cc[25 * e:25 * (e + 1)].sum(0) == mainline["client_counts"][25 * e:25 * (e + 1)].sum(0)).all()


def test_clean_sets_are_identical_to_the_mainline_and_the_client_layer_uses_no_randomness():
    """同 seed：干净集逐个索引相同（在客户端层之前切）；客户端计数与 seed 无关（固定画像不抽随机数）。"""
    a = D.designed_partition(LABELS, _cfg("designed", FIXED), seed=42)
    b = D.designed_partition(LABELS, _cfg("designed", {"condition": "C1"}), seed=42)
    c = D.designed_partition(LABELS, _cfg("designed", FIXED), seed=43)
    assert all(np.array_equal(x, y) for x, y in zip(a["clean_indices"], b["clean_indices"]))
    assert (a["client_counts"] == c["client_counts"]).all()
    assert a["info"]["index_sha"] != c["info"]["index_sha"]          # 图片的选取仍随 seed 变


def test_lambda_controls_within_edge_heterogeneity():
    vals = []
    for lam in (0.0, 0.2, 0.4, 0.6):
        sp = D.designed_partition(LABELS, _cfg("designed", {**FIXED, "profile_lambda": lam}), seed=42)
        vals.append((sp["info"]["h_pair_same"], sp["info"]["h_pair_cross"]))
    assert vals[0][0] < 0.01                                          # λ = 0：组内同分布（只剩取整误差）
    assert all(a[0] < b[0] for a, b in zip(vals, vals[1:]))            # λ 单调抬高组内异质
    assert all(s < c for s, c in vals)                                 # 每一档都是组内 < 跨组
    # 本会话的两个水平（F-125）：λ = 0.4 的组内 / 跨组 ≈ 0.54，现主线 ≈ 0.79
    lam04 = vals[2][0] / vals[2][1]
    m = D.designed_partition(LABELS, _cfg("designed", {"condition": "C1"}), seed=42)
    same, cross = D.h_pairs(m["client_counts"], m["assignments"])
    assert lam04 == pytest.approx(0.54, abs=0.02) and same / cross == pytest.approx(0.79, abs=0.02)


def test_h_pairs_by_hand():
    C = np.array([[2, 0], [2, 0], [0, 2], [1, 1]])
    same, cross = D.h_pairs(C, [0, 0, 1, 1])
    assert same == pytest.approx((0.0 + 0.5) / 2)
    assert cross == pytest.approx((1.0 + 0.5 + 1.0 + 0.5) / 4)


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.update(client_profile="profiles"), "client_profile"),
    (lambda d: d.update(profile_lambda=1.5), "profile_lambda"),
    (lambda d: d.update(profile_lambda=True), "profile_lambda"),
    (lambda d: d.pop("profile_lambda"), "profile_lambda"),
    (lambda d: d.pop("client_profile"), "只在 client_profile=fixed"),
])
def test_fixed_profile_config_errors(mutate, needle):
    d = dict(FIXED)
    assert D.config_errors(_cfg("designed", d)) == []
    mutate(d)
    errs = D.config_errors(_cfg("designed", d))
    assert errs and any(needle in e for e in errs), errs


def test_fixed_profile_is_rejected_for_equal_random_and_allowed_for_hdir():
    fx = {"client_profile": "fixed", "profile_lambda": 0.4}
    assert any("equal_random" in e for e in D.config_errors(_cfg("equal_random", fx)))
    assert D.config_errors(_cfg("hdir", {"alpha_edge": 1.0, **fx})) == []


def test_fixed_profile_partition_lines_round_trip():
    sp = D.designed_partition(LABELS, _cfg("designed", FIXED))
    m = collect("\n".join(["[Config] loading x.yaml", *D.data_lines(sp, malicious_ids=range(10)),
                           "[Round 1]"]) + "\n")
    run = m["run"]
    assert run["partition_profile"] == "fixed" and run["partition_lambda"] == 0.4
    assert run["data"]["alpha_client"] is None                        # α_c 在固定画像下不起作用，不打 0.5
    assert run["data"]["h_pair_same"] == pytest.approx(sp["info"]["h_pair_same"], abs=1e-4)
    old = collect("\n".join(["[Config] loading x.yaml",
                             *D.data_lines(D.designed_partition(LABELS, _cfg("designed", {"condition": "C1"}))),
                             "[Round 1]"]) + "\n")["run"]
    assert old["partition_profile"] is None and old["partition_lambda"] is None


def test_build_clients_with_fixed_profiles_bakes_the_profile_counts():
    """集群（需要 TF）：固定画像经 main.build_clients 落地 —— 每端 375 / 125、客户端类计数 = 划分函数给的画像。"""
    tf = pytest.importorskip("tensorflow")
    import main as M
    images = np.zeros((len(LABELS), 4, 4, 3), dtype=np.float32)
    cfg = _cfg("designed", FIXED)
    cfg["data"].update(batch_size=32, img_size=4, num_classes=10)
    cfg["federation"].update(edge_assignment="block", client_fraction=0.1)
    cfg["training"] = {"drift_correction": "hierfedavg", "learning_rate": 0.1, "lr_decay": 1.0,
                       "local_epochs": 1}
    cfg["backdoor"] = {"enabled": False, "target_label": 0}
    model = tf.keras.Sequential([tf.keras.Input((4, 4, 3)), tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(10, activation="softmax")])
    out = {}
    clients, assignments, _ = M.build_clients(images, LABELS, model, copy.deepcopy(cfg), s3_out=out)
    assert assignments == [e for e in range(4) for _ in range(25)]
    assert [c.n_samples for c in clients] == [375] * 100
    sp = D.designed_partition(LABELS, cfg)
    assert all(np.array_equal(a, b) for a, b in zip(out["clean_indices"], sp["clean_indices"]))
