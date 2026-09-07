"""
tests/test_exp3_config_invariants.py  —  Experiment 3 的「唯一自变量」必须真的唯一

`experiments/attack/hfl-propagation/README.md` 声称：
    「**跨所有拓扑格恒定的不变量**：100 client · 10 恶意(全局 10%) ·
      edge_rounds × n_rounds = 400 · badpfl · hier_fedrep · resnet10 ·
      shared_generator=true · forced_participation=false · poison_ratio=0.2。
      **唯一自变量 = 恶意端在多少个 edge、怎么分布**」

这是一句**声明**，此前没有任何东西检查它。把它变成断言 —— 15+ 个 yaml 手工维护，
改一个忘一个是必然的，而 config 漂了跑出来的差异会被当成拓扑效应读。

顺带锁住三件这一轮修掉的事：
  · `target_label` 全部为 0（与 Bad-PFL 库统一，两库结果才能同框讨论）
  · 评估密度按**有效轮**对齐（原本一律 5 个 cloud round，于是等预算 400 下
    flat 拿到 80 个点、R20 只有 4 个、R40 只有 2 个）
  · `3c_R1` 必须存在 —— 它是把「层级结构」与「聚合频率」分开的唯一对照格
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CELLS_DIR = (Path(__file__).resolve().parent.parent
             / "experiments" / "attack" / "hfl-propagation")


def _cells():
    out = {}
    for f in sorted(CELLS_DIR.glob("*.yaml")):
        out[f.stem] = yaml.safe_load(f.read_text(encoding="utf-8"))
    assert out, f"{CELLS_DIR} 下一个 yaml 都没有"
    return out


# 跨所有格必须逐字相同的键（README 声称的不变量）
SHARED = [
    ("data", "dataset"),            ("data", "batch_size"),
    ("data", "per_client_test_ratio"),
    ("federation", "n_clients"),    ("federation", "client_fraction"),
    ("federation", "partition"),    ("federation", "alpha"),
    ("federation", "edge_assignment"),
    ("training", "local_epochs"),   ("training", "learning_rate"),
    ("training", "drift_correction"),
    ("model", "arch"),
    ("backdoor", "poison_ratio"),   ("backdoor", "target_label"),
    ("backdoor", "malicious_placement"),
    ("backdoor", "forced_participation"),
    ("backdoor", "badpfl_shared_generator"),
    ("backdoor", "badpfl_epsilon"), ("backdoor", "badpfl_sigma"),
    ("backdoor", "asr_max_samples"),
]


def test_declared_invariants_are_actually_invariant():
    cells = _cells()
    ref_name = "3c_R5"                      # exp007 锚点
    ref = cells[ref_name]
    problems = []
    for sect, key in SHARED:
        want = ref[sect][key]
        for name, c in cells.items():
            got = c.get(sect, {}).get(key, "<缺失>")
            if got != want:
                problems.append(f"{sect}.{key}: {name}={got!r} 但 {ref_name}={want!r}")
    assert not problems, (
        "以下键在各格之间不一致，而 README 声称它们是不变量：\n  "
        + "\n  ".join(problems)
        + "\n拓扑/频率之外的任何差异都会被当成拓扑效应读出来。")


def test_effective_budget_is_equal_across_all_cells():
    """edge_rounds × n_rounds = 400。等有效预算是跨格比较的前提。"""
    for name, c in _cells().items():
        f = c["federation"]
        prod = f["edge_rounds"] * f["n_rounds"]
        assert prod == 400, f"{name}: edge_rounds×n_rounds = {prod}，应为 400"


def test_target_class_is_zero_everywhere():
    """
    与 Bad-PFL 库统一（那边 main.py --ba_target_label 默认 0 = airplane）。
    此前本库是 9(truck)、那边是 0(airplane)，两套 ASR 数字不能同框而报告里没说。
    """
    for name, c in _cells().items():
        assert c["backdoor"]["target_label"] == 0, \
            f"{name}: target_label={c['backdoor']['target_label']}，应为 0"


def test_malicious_count_is_ten_everywhere():
    """全局恶意端比例恒为 10%，无论怎么布点。"""
    for name, c in _cells().items():
        per_edge = c["backdoor"]["malicious_per_edge"]
        assert sum(per_edge) == 10, f"{name}: malicious_per_edge={per_edge}，和应为 10"
        assert len(per_edge) == c["federation"]["n_edges"], \
            f"{name}: malicious_per_edge 长度 {len(per_edge)} != n_edges {c['federation']['n_edges']}"


def test_eval_density_is_comparable_in_effective_rounds():
    """
    `eval_interval` 数的是 **cloud round**，而有效轮 = cloud × edge_rounds。
    一律设 5 时，等预算 400 下各格拿到的评估点数是 80 / 40 / 20 / 16 / 8 / 4 / 2 ——
    任何「谁涨得快」的轨迹比较在大 R 的格子上根本没有分辨率。

    要求：评估间隔 ≤ 10 个有效轮；除非已经受 cloud 轮数下限所限
    （eval_interval 不能小于 1），那种情况是设计固有的，只要求 eval_interval == 1。
    """
    for name, c in _cells().items():
        er = c["federation"]["edge_rounds"]
        ev = c["backdoor"]["eval_interval"]
        spacing = er * ev
        if spacing > 10:
            assert ev == 1, (
                f"{name}: 每 {spacing} 个有效轮才评一次，而 eval_interval={ev} 还能再调小。"
                " 只有 eval_interval 已经是 1（受 cloud 轮数下限所限）才允许 >10。")
        assert c["evaluation"]["eval_interval"] == ev, \
            f"{name}: backdoor.eval_interval={ev} 与 evaluation.eval_interval 不一致"


def test_every_cell_has_at_least_ten_eval_points():
    """少于 10 个点的曲线没法读趋势。"""
    for name, c in _cells().items():
        pts = c["federation"]["n_rounds"] // c["backdoor"]["eval_interval"]
        assert pts >= 10, f"{name}: 只有 {pts} 个评估点"


def test_the_frequency_control_cell_exists():
    """
    3c_R1（n_edges=2, edge_rounds=1）是把「层级结构」与「cloud 聚合频率」分开的
    唯一对照格：flat_baseline 同时改了 n_edges 2→1 **和** edge_rounds 5→1，
    没有这一格就无法归因。README 早就设计了它，但文件一直没被创建。
    """
    cells = _cells()
    assert "3c_R1" in cells, "缺 3c_R1.yaml —— flat vs 层级的差异无法归因"
    c = cells["3c_R1"]
    assert c["federation"]["n_edges"] == 2, "3c_R1 必须保留 edge 层（n_edges=2）"
    assert c["federation"]["edge_rounds"] == 1, "3c_R1 的 edge_rounds 必须是 1"
    flat = cells["flat_baseline"]
    assert flat["federation"]["n_edges"] == 1 and flat["federation"]["edge_rounds"] == 1, \
        "flat_baseline 应当是 n_edges=1 且 edge_rounds=1（两个变量同时变，故需要 3c_R1）"


def test_3c_cells_only_vary_frequency():
    """3C 轴的唯一自变量是 edge_rounds —— 拓扑必须全部是 2edge [5,5]。"""
    for name, c in _cells().items():
        if not name.startswith("3c_"):
            continue
        assert c["federation"]["n_edges"] == 2, f"{name}: 3C 轴拓扑应固定为 2 edge"
        assert c["backdoor"]["malicious_per_edge"] == [5, 5], \
            f"{name}: 3C 轴布点应固定为 [5,5]"
