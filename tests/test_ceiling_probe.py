"""
tests/test_ceiling_probe.py  —  天花板判定 run 的配置不变量

## 这个 run 要判什么

Stage B 标定（seed42，三格全 `exit_code: 0`）给出：

| local_epochs | 末 10 点 ASR | 末 10 点 MTA | 全长格成本 |
|---|---|---|---|
| 1 | 0.660 ± 0.092 | 0.740 | 1.81 GPU-h（80 cloud 轮） |
| 5 | 0.934 ± 0.054 | 0.740 | 0.97 GPU-h（30 cloud 轮） |

**MTA 三档一样**，而 `ep5 × 30 轮` 比 `ep1 × 80 轮` 便宜 46%、本地训练量近 2 倍
（750 vs 400 client-epoch）—— 因为每个 cloud round 有约 56 s 固定开销
（广播 / 聚合 / clone_model / dataset 搭建），只有 8.75 s/epoch 是真训练，
`local_epochs=1` 下 **85% 的墙钟花在与训练无关的开销上**。

切 ep5 唯一的顾虑是**天花板**：ASR 0.934 之上只剩 0.066 的余量。
它只威胁**拓扑轴**（3A/3B，Experiment 3 的正题）—— 防御轴、持久性、比例轴低端
都是把 ASR 往**下**压，0.934 往下全是余量。

所以判据是布点最稀释的一端还落在哪里：

  对照 `local_epochs5_seed42`（n_edges=2，HHI=0.50）→ 末 10 点 ASR = 0.934
  本格 `ceiling_10edge_ep5_seed42`（n_edges=10 全分散，HHI=0.10）

  · 明显更低 → 拓扑轴在 ep5 下跨度更宽，**整个矩阵切 (ep=5, n_rounds=30)**
  · 也在 0.93+ → ep5 压平了拓扑轴，另议

## 为什么必须锁配置

这个 run 的**全部价值**在于「与 local_epochs5_seed42 只差拓扑」。多漂一个键，
两个数就不可比，而事后从 metrics.json 分辨不出来（陷阱 #7 / #14 同类）。
下面逐条锁住它相对**两个**锚点的差异。

纯 stdlib，本地秒级。
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "experiments" / "calibration"
EXP3 = ROOT / "experiments" / "attack" / "hfl-propagation"

PROBE = CALIB / "ceiling_10edge_ep5_seed42.yaml"
TOPO_ANCHOR = EXP3 / "10edge_distributed.yaml"        # 拓扑来自它
BUDGET_ANCHOR = CALIB / "local_epochs5_seed42.yaml"   # 预算/epoch 来自它

sys.path.insert(0, str(ROOT / "tests"))
from test_calibration import _kv                       # noqa: E402  单一事实来源


# 相对拓扑锚点，允许差的键 = 标定那四处改动（seed 两边都是 42，不在其中）
BUDGET_DRIFT = {"n_rounds", "local_epochs", "eval_interval"}
# 相对预算锚点，允许差的键 = 拓扑本身
TOPO_DRIFT = {"n_edges", "malicious_per_edge"}


def test_probe_exists():
    assert PROBE.exists(), f"判定 run 的配置不在：{PROBE}"


def test_differs_from_topology_anchor_only_in_budget_keys():
    """与 10edge_distributed 相比，只准差预算/epoch/评估密度。"""
    anchor, probe = _kv(TOPO_ANCHOR), _kv(PROBE)
    assert set(anchor) == set(probe), (
        f"键集合不同：多 {set(probe) - set(anchor)}，少 {set(anchor) - set(probe)}")
    drift = {k for k in anchor if anchor[k] != probe[k]}
    assert drift <= BUDGET_DRIFT, (
        f"相对 10edge_distributed 多漂了 {sorted(drift - BUDGET_DRIFT)} —— "
        f"这个 run 的价值全在「只差拓扑」，多一个变量就白跑了")


def test_differs_from_budget_anchor_only_in_topology_keys():
    """
    **这条才是判据成立的前提**：与 local_epochs5_seed42 相比只准差拓扑。
    两个 ASR 数要直接相减，任何别的差异都会混进去。
    """
    anchor, probe = _kv(BUDGET_ANCHOR), _kv(PROBE)
    assert set(anchor) == set(probe), (
        f"键集合不同：多 {set(probe) - set(anchor)}，少 {set(anchor) - set(probe)}")
    drift = {k for k in anchor if anchor[k] != probe[k]}
    assert drift <= TOPO_DRIFT, (
        f"相对 local_epochs5_seed42 多漂了 {sorted(drift - TOPO_DRIFT)} —— "
        f"0.934 那个数就不能拿来比了")


def test_values_are_what_the_filename_claims():
    """文件名撒谎最难查：提交命令按文件名给 exp_id，结果按 exp_id 归档。"""
    kv = _kv(PROBE)
    assert kv["local_epochs"] == {"5"}, kv["local_epochs"]
    assert kv["seed"] == {"42"}, kv["seed"]
    assert kv["n_edges"] == {"10"}, kv["n_edges"]


def test_budget_matches_the_calibration_cells_exactly():
    """预算与评估密度必须与标定六格**逐字相同**，否则 0.934 不可比。"""
    kv, anchor = _kv(PROBE), _kv(BUDGET_ANCHOR)
    assert kv["n_rounds"] == anchor["n_rounds"] == {"30"}
    assert kv["eval_interval"] == anchor["eval_interval"] == {"1"}
    assert kv["edge_rounds"] == anchor["edge_rounds"], "edge_rounds 变了，有效轮就不等"


def test_malicious_budget_is_still_ten_percent():
    """恶意端总数跨拓扑恒定 = 10（全局 10%）。变了就不是「只差布点」。"""
    per_edge, = _kv(PROBE)["malicious_per_edge"]
    counts = [int(x) for x in re.findall(r"\d+", per_edge)]
    assert len(counts) == 10, f"malicious_per_edge 长度 {len(counts)} != n_edges 10"
    assert sum(counts) == 10, f"恶意端总数 {sum(counts)} != 10"
    assert set(counts) == {1}, f"10edge_distributed 应当每 edge 恰好 1 个：{counts}"


def test_probe_is_not_one_of_the_six_calibration_cells():
    """
    反向锚点：它**不能**被 run_calibration.sh 或 test_calibration.py 收进标定批。
    那六格是「只有 local_epochs 在变」的单变量设计，这一格换了拓扑 ——
    混进去会让 test_local_epochs_is_the_only_axis_that_varies 变红，
    或者更糟：悄悄污染「谁先到平台」的比较。
    """
    from test_calibration import EPOCHS, SEEDS
    six = {f"local_epochs{e}_seed{s}.yaml" for e in EPOCHS for s in SEEDS}
    assert PROBE.name not in six
    # run_calibration.sh 按名字模式枚举而不是扫目录 —— 锁住这条性质
    sh = (CALIB / "run_calibration.sh").read_text(encoding="utf-8")
    assert "local_epochs${ep}_seed${seed}" in sh, (
        "run_calibration.sh 不再按名字枚举了？若改成扫目录，这个判定 run 会被"
        "当成标定格提交，且 read_calibration.py 会把它混进平台比较。")


def test_probe_is_submittable_by_the_generic_cell_script():
    """它靠通用的 calib_cell.sbatch 提交，所以那个脚本必须仍然吃任意 config。"""
    sb = (CALIB / "calib_cell.sbatch").read_text(encoding="utf-8")
    assert "CONFIG_REL=" in sb and '--config "$CONFIG_ABS"' in sb, (
        "calib_cell.sbatch 不再接受任意 config 了 —— 判定 run 没法提交")
