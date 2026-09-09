"""
tests/test_calibration.py  —  Stage B 标定的配置不变量 + 读数判据

两件分开测：
1. **配置**：6 个格子除 `local_epochs` / `seed` / 预算 / 评估密度外，必须与锚点
   `2edge_distributed` 逐字相同。标定的整个意义就是「单变量」——多漂一个键，
   得出的 (epoch, rounds) 建议就是错的，而且事后无法从 metrics.json 分辨。
2. **读数**：plateau 的判据必须**可计算且保守**。判不出来要报 None，
   不能挑一个看着像的轮次 —— 那是把「150 有效轮还没收敛」偷换成一个数字。

纯 stdlib，本地秒级。
"""

import json
import re
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "experiments" / "calibration"
ANCHOR = ROOT / "experiments" / "attack" / "hfl-propagation" / "2edge_distributed.yaml"

sys.path.insert(0, str(CALIB))
from read_calibration import (plateau_round, cost_up_to,   # noqa: E402
                              load_cells, CELL_RE)

EPOCHS = (1, 3, 5)
SEEDS = (42, 43)
# 允许与锚点不同的键。多一个都算漂移。
ALLOWED_DRIFT = {"seed", "n_rounds", "local_epochs", "eval_interval"}

# 锚点（2edge_distributed）后来加了 `stopping` 块（自适应轮数）。标定这六格是
# **已经跑完的固定轮数 run**，结果就在 results/ 里 —— 它们**不能**跟着改，
# 否则盘上的数字与配置对不上。所以锚点独有这几个键是**设计如此**，不是漂移。
# 反向锚点见 test_calibration_cells_have_no_stopping_block。
ANCHOR_ONLY_KEYS = {"criteria", "floor_effective", "cap_effective",
                    "thetas", "debounce", "pm_window", "pm_slope_tol"}


def _kv(path: Path) -> dict:
    """把 YAML 压成 {叶子键: 值}，注释与空行剔除。

    不用 yaml 库（本地可能没有）。这里只需要「哪些键的值变了」，
    重复的叶子键（如两处 eval_interval）合并成一个条目即可 —— 正因如此
    下面的 ALLOWED_DRIFT 里 eval_interval 只写一次。
    """
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^(\s*)([\w_]+):(.*)$", line)
        if not m:
            continue
        key, val = m.group(2), m.group(3).split("#")[0].strip()
        if val == "":                     # 段头（federation: / training: …）
            continue
        out.setdefault(key, set()).add(val)
    return out


# ══════════════════════════════════════════════════════════════════════════
# 1. 配置：只差该差的
# ══════════════════════════════════════════════════════════════════════════
def test_all_six_cells_exist():
    missing = [f"local_epochs{e}_seed{s}.yaml" for e in EPOCHS for s in SEEDS
               if not (CALIB / f"local_epochs{e}_seed{s}.yaml").exists()]
    assert not missing, f"缺配置：{missing}"


@pytest.mark.parametrize("epochs", EPOCHS)
@pytest.mark.parametrize("seed", SEEDS)
def test_cell_differs_from_anchor_only_in_intended_keys(epochs, seed):
    anchor = _kv(ANCHOR)
    cell = _kv(CALIB / f"local_epochs{epochs}_seed{seed}.yaml")
    extra = set(cell) - set(anchor)
    missing = set(anchor) - set(cell) - ANCHOR_ONLY_KEYS
    assert not extra and not missing, (
        f"键集合不同：多 {extra}，少 {missing}"
        f"（锚点独有的 stopping 块不算，见 ANCHOR_ONLY_KEYS）")
    anchor = {k: v for k, v in anchor.items() if k not in ANCHOR_ONLY_KEYS}
    drift = {k for k in anchor if anchor[k] != cell[k]}
    assert drift <= ALLOWED_DRIFT, (
        f"local_epochs{epochs}_seed{seed} 与锚点多漂了 {sorted(drift - ALLOWED_DRIFT)} "
        f"—— 标定必须是单变量")


@pytest.mark.parametrize("epochs", EPOCHS)
@pytest.mark.parametrize("seed", SEEDS)
def test_cell_values_are_what_the_filename_claims(epochs, seed):
    """文件名撒谎是最难查的一类错：提交脚本按文件名分格，读数脚本按文件名归类。"""
    cell = _kv(CALIB / f"local_epochs{epochs}_seed{seed}.yaml")
    assert cell["local_epochs"] == {str(epochs)}
    assert cell["seed"] == {str(seed)}


def test_budget_and_eval_density_are_uniform_across_cells():
    """预算与评估密度必须**跨格恒定** —— 它们不是自变量。

    若某一格偷偷多跑了轮数，「谁先到平台」的比较立刻失效。
    """
    budgets = {}
    for e in EPOCHS:
        for sd in SEEDS:
            kv = _kv(CALIB / f"local_epochs{e}_seed{sd}.yaml")
            budgets[f"local_epochs{e}_seed{sd}"] = (
                frozenset(kv["n_rounds"]), frozenset(kv["eval_interval"]))
    assert len(set(budgets.values())) == 1, f"预算/评估密度不统一：{budgets}"
    (n_rounds, eval_iv), = set(budgets.values())
    assert n_rounds == frozenset({"30"}), n_rounds
    assert eval_iv == frozenset({"1"}), eval_iv


def test_local_epochs_is_the_only_axis_that_varies():
    """反向锚点：固定 seed 时，跨三个格子真的只有 local_epochs 在变。

    注意比的是**每个键在各文件里的值集合**，不是把所有值并起来 ——
    `_kv` 按叶子键名压平，而 `enabled` 在 backdoor 段是 true、wandb 段是 false，
    单个文件里就已经是 {true, false}。并起来看会把它误判成「在变」。
    """
    per_file = [_kv(CALIB / f"local_epochs{e}_seed42.yaml") for e in EPOCHS]
    keys = set(per_file[0])
    changed = {k for k in keys if len({frozenset(f[k]) for f in per_file}) > 1}
    assert changed == {"local_epochs"}, \
        f"除 local_epochs 外还有 {sorted(changed - {'local_epochs'})} 在变"


# ══════════════════════════════════════════════════════════════════════════
# 2. 读数：plateau 判据
# ══════════════════════════════════════════════════════════════════════════
def _pts(key, values, start=1):
    return [{"round": start + i, key: v} for i, v in enumerate(values)]


def test_plateau_found_when_curve_flattens():
    # 0.10 0.30 0.50 0.70 0.70 0.70 0.70 → final=0.70，从第 4 轮起都在带内
    pts = _pts("pm_acc", [0.10, 0.30, 0.50, 0.70, 0.70, 0.70, 0.70])
    assert plateau_round(pts, "pm_acc", tol=0.01) == 4


def test_plateau_is_none_when_still_climbing():
    """还在爬 → None。**不能**给一个数 —— 「150 有效轮还没收敛」是结论本身。"""
    pts = _pts("pm_acc", [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
    assert plateau_round(pts, "pm_acc", tol=0.01) is None


def test_plateau_requires_staying_inside_the_band():
    """中途回到带内又跑出去，不算平台 —— 平台轮必须是「此后再没离开」。"""
    pts = _pts("pm_acc", [0.70, 0.40, 0.70, 0.70, 0.70])
    assert plateau_round(pts, "pm_acc", tol=0.01) == 3


def test_plateau_is_none_with_too_few_points():
    assert plateau_round(_pts("pm_acc", [0.7, 0.7, 0.7]), "pm_acc", 0.01) is None
    assert plateau_round([], "pm_acc", 0.01) is None


def test_plateau_skips_none_values_instead_of_treating_them_as_zero():
    """无定义的点（非 IID 下客户端分片可能一个非目标类样本都没有 → ASR=None）
    必须被跳过。当成 0 会把平台推到最后，或伪造一个假的爬升。"""
    pts = _pts("local_benign_asr", [0.70, None, 0.70, 0.70, 0.70, 0.70])
    assert plateau_round(pts, "local_benign_asr", tol=0.01) == 1


def test_cost_sums_measured_values_not_estimates():
    metrics = {
        "acc_rounds": [{"round": r, "round_time": 10.0} for r in range(1, 6)],
        "timing_rounds": [{"round": r, "total_s": 2.0} for r in range(1, 6)],
    }
    c = cost_up_to(metrics, 3)
    assert c["train_s"] == pytest.approx(30.0)
    assert c["eval_s"] == pytest.approx(6.0)
    assert c["total_s"] == pytest.approx(36.0)
    assert c["n_rounds"] == 3
    full = cost_up_to(metrics, None)
    assert full["total_s"] == pytest.approx(60.0)


def test_cost_is_none_not_zero_when_timing_absent():
    """老日志没有 [Timing] 行 → eval_s 是 None。0 会被读成「评估不花时间」，
    而那正是这批标定要测的东西。"""
    metrics = {"acc_rounds": [{"round": 1, "round_time": 10.0}], "timing_rounds": []}
    c = cost_up_to(metrics, None)
    assert c["train_s"] == pytest.approx(10.0)
    assert c["eval_s"] is None
    assert c["total_s"] is None


def test_load_cells_parses_epoch_and_seed_from_filename():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "local_epochs3_seed43.metrics.json").write_text(json.dumps({"exit_code": 0}))
        (d / "unrelated.metrics.json").write_text(json.dumps({}))
        cells = load_cells(d)
        assert len(cells) == 1
        assert cells[0]["local_epochs"] == 3 and cells[0]["seed"] == 43


def test_cell_regex_does_not_match_neighbours():
    assert CELL_RE.search("local_epochs10_seed42.metrics.json")
    assert not CELL_RE.search("2edge_distributed_seed42.metrics.json")


def test_calibration_cells_have_no_stopping_block():
    """
    反向锚点：标定六格**必须**保持固定轮数（无 `stopping`）。
    它们的结果已经在 results/ 里 —— 给它们加自适应轮数会让盘上的数字
    与配置对不上，而 metrics.json 事后分辨不出来跑的是哪一套。
    """
    for e in EPOCHS:
        for sd in SEEDS:
            txt = (CALIB / f"local_epochs{e}_seed{sd}.yaml").read_text(encoding="utf-8")
            assert "stopping:" not in txt, (
                f"local_epochs{e}_seed{sd}.yaml 被加了 stopping 块 —— "
                f"标定是已完成的固定轮数批次，不能改")
