"""
tests/test_exp3_matrix.py  —  Experiment 3 矩阵（T1/T2/T3）的配置不变量

## 这批格子的预算是怎么定的

`experiments/calibration/RESULTS.md`：

- Stage B 标定 → **MTA 与 `local_epochs` 无关**（0.740/0.739/0.740），而每个 cloud
  round 有约 56 s 固定开销、只有 8.75 s/epoch 是真训练 → `local_epochs=1` 时
  85% 的墙钟花在与训练无关的开销上。**取 ep=5**。
- 天花板判定 → **ASR 不收敛到内点，而是往 1.0 饱和**（末段仍以 +0.017/轮 在爬）。
  所以「跑到收敛再比终值」= 保证拿到 null。
  **主指标改 `T_θ`（首次达到 θ 的 cloud round），终值只作瞬态快照。**
- 预算 = **200 有效轮**（`n_rounds × edge_rounds`），最慢的 10edge 在此已越过
  θ=0.75（round 27），`pm_acc` 距渐近 0.001。

## 本文件锁什么

1. **有效轮跨格恒定** —— 它不是自变量。某格偷偷多跑，跨拓扑比较立刻失效，
   而事后从 metrics.json 分辨不出来（这正是判定 run 栽过的坑：`n_edges`
   影响收敛速度，等预算下比终值 = 在比「谁离收敛更近」）。
2. **评估密度按有效轮对齐** —— 每 5 个有效轮一个点。`eval_interval` 数的是
   cloud round，而有效轮 = cloud × `edge_rounds`，直接写同一个数会让 R40 只拿到
   几个点（`a66da67` 修过一次同样的错）。
3. **每格与它的基座只差声明的那一个自变量**。
4. **防御格的 `num_malicious` 不能是 0** —— `defense/factory.py:99` 是
   `if "num_malicious" not in params` 才自动估算，写着 `0` 会让 multi_krum 拿到
   f=0 → `n_select = m` → **全部接纳，防御是个 no-op**，而日志只打
   `[Defense] enabled: multi_krum`，看不出来。
5. **每一格都能通过 `config_validate`** —— 本地就能跑（它不 import TF），
   不必等 GPU 排到才发现配置不兼容。

纯 stdlib + pyyaml + numpy，本地秒级。
"""

import math
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
EXP3 = ROOT / "experiments" / "attack" / "hfl-propagation"
sys.path.insert(0, str(ROOT / "fedavg"))

EFFECTIVE_ROUNDS = 200        # n_rounds × edge_rounds，跨格恒定
LOCAL_EPOCHS = 5
EFF_PER_EVAL = 5              # 每 5 个有效轮一个评估点

CELLS = sorted(p.stem for p in EXP3.glob("*.yaml"))


def cfg(name):
    return yaml.safe_load((EXP3 / f"{name}.yaml").read_text(encoding="utf-8"))


def test_the_matrix_is_not_empty():
    """反向锚点：别哪天 glob 什么都没扫到却「全部通过」。"""
    assert len(CELLS) >= 20, f"只扫到 {len(CELLS)} 格：{CELLS}"
    for must in ("2edge_distributed", "rho0_2edge_distributed",
                 "def_multi_krum_2edge_distributed", "rho20_2edge_collocated"):
        assert must in CELLS, f"缺 {must}"


@pytest.mark.parametrize("name", CELLS)
def test_effective_rounds_are_constant(name):
    """有效轮不是自变量。"""
    c = cfg(name)["federation"]
    eff = int(c["n_rounds"]) * int(c["edge_rounds"])
    assert eff == EFFECTIVE_ROUNDS, (
        f"{name}: n_rounds={c['n_rounds']} × edge_rounds={c['edge_rounds']} = {eff}"
        f" ≠ {EFFECTIVE_ROUNDS} —— 预算不齐，跨格比较无效")


@pytest.mark.parametrize("name", CELLS)
def test_local_epochs_is_uniform(name):
    assert int(cfg(name)["training"]["local_epochs"]) == LOCAL_EPOCHS


@pytest.mark.parametrize("name", CELLS)
def test_eval_density_is_aligned_by_effective_rounds(name):
    """
    `eval_interval` 数的是 cloud round，而有效轮 = cloud × edge_rounds。
    直接跨格写同一个数，大 edge_rounds 的格子会稀到没有分辨率。
    """
    c = cfg(name)
    er = int(c["federation"]["edge_rounds"])
    want = max(1, round(EFF_PER_EVAL / er))
    for where, sub in (("backdoor", c["backdoor"]), ("evaluation", c["evaluation"])):
        got = int(sub["eval_interval"])
        assert got == want, (
            f"{name}: {where}.eval_interval={got}，按 edge_rounds={er} 应为 {want}"
            f"（每 {want*er} 个有效轮一个点）")


@pytest.mark.parametrize("name", CELLS)
def test_both_eval_intervals_agree(name):
    """两个 eval_interval 不等 → ASR 点与 acc 点无法逐点配对。"""
    c = cfg(name)
    assert int(c["backdoor"]["eval_interval"]) == int(c["evaluation"]["eval_interval"])


@pytest.mark.parametrize("name", CELLS)
def test_enough_eval_points_for_the_tail_statistic(name):
    """
    官方统计量是**末 10 个评估点的均值**（逐点 σ≈0.09，与要测的效应同量级）。
    点数不足 12 的格子必须在文件头**显式声明**它算不了 T_θ，
    不能悄悄混进表里被当成同等证据。
    """
    c = cfg(name)
    n_pts = int(c["federation"]["n_rounds"]) // int(c["backdoor"]["eval_interval"])
    if n_pts < 12:
        head = (EXP3 / f"{name}.yaml").read_text(encoding="utf-8")[:2000]
        assert "T_θ 不可算" in head, (
            f"{name} 只有 {n_pts} 个评估点，但文件头没有声明 T_θ 不可算")


@pytest.mark.parametrize("name", CELLS)
def test_malicious_count_matches_the_placement_vector(name):
    """`n_malicious` 与 `malicious_per_edge` 求和必须一致 —— 两处都进 run 块。"""
    bd = cfg(name)["backdoor"]
    if str(bd.get("malicious_placement", "")).lower() != "by_edge":
        pytest.skip("非 by_edge 布点")
    per_edge = bd["malicious_per_edge"]
    assert sum(int(x) for x in per_edge) == int(bd["n_malicious"]), (
        f"{name}: sum({per_edge}) ≠ n_malicious={bd['n_malicious']}")
    assert len(per_edge) == int(cfg(name)["federation"]["n_edges"])


@pytest.mark.parametrize("name", [c for c in CELLS if c.startswith("def_")])
def test_defense_cells_do_not_leave_num_malicious_zero(name):
    """
    **防御轴的静默失效**：`defense/factory.py` 是 `if "num_malicious" not in params`
    才自动估算。params 里写着 `0` → multi_krum 拿到 f=0 → n_select=m → 全部接纳，
    防御是个 no-op，而日志照样打印 `[Defense] enabled: multi_krum`。
    """
    c = cfg(name)
    f = int(c["defense"]["params"]["num_malicious"])
    assert f > 0, f"{name}: defense.params.num_malicious=0 → multi_krum 会变成 no-op"
    want = math.ceil(int(c["backdoor"]["n_malicious"]) / int(c["federation"]["n_edges"]))
    assert f == want, f"{name}: num_malicious={f}，按 ceil(n_mal/n_edges) 应为 {want}"


@pytest.mark.parametrize("name", [c for c in CELLS if c.startswith("def_")])
def test_defense_name_matches_the_filename(name):
    """文件名撒谎最难查：run_exp3.sh 按文件名给 exp_id，结果按 exp_id 归档。"""
    got = str(cfg(name)["defense"]["name"]).lower()
    assert name.startswith(f"def_{got}_"), f"{name} 的 defense.name={got!r} 对不上文件名"


@pytest.mark.parametrize("name", [c for c in CELLS if not c.startswith("def_")])
def test_non_defense_cells_are_undefended(name):
    assert str(cfg(name)["defense"]["name"]).lower() == "none"


def test_rho0_control_keeps_the_backdoor_pipeline_enabled():
    """
    ρ=0 必须是 `enabled=true` + 空布点，**不是** `enabled=false`。
    后者走朴素 CloudServer，一个 ASR 字段都没有 —— 出来的 json 与「跑挂了」
    长得一模一样，事后无法判读。
    """
    bd = cfg("rho0_2edge_distributed")["backdoor"]
    assert bd["enabled"] is True, "ρ=0 用了 enabled=false，会没有任何 ASR 字段"
    assert int(bd["n_malicious"]) == 0
    assert sum(int(x) for x in bd["malicious_per_edge"]) == 0


def test_ratio_axis_covers_the_intended_points():
    """比例轴改的是恶意端**个数**，不是 poison_ratio。"""
    for topo in ("2edge_collocated", "2edge_distributed"):
        got = {int(cfg(f"rho{p:02d}_{topo}")["backdoor"]["n_malicious"]) for p in (2, 5, 20)}
        assert got == {2, 5, 20}, f"{topo} 的比例轴取值 {got}"
        for p in (2, 5, 20):
            assert float(cfg(f"rho{p:02d}_{topo}")["backdoor"]["poison_ratio"]) == \
                   float(cfg(topo)["backdoor"]["poison_ratio"]), \
                   "比例轴动了 poison_ratio —— 那是每端投毒样本比例，不是攻击者比例"


def test_shrunk_3c_axis_is_out_of_the_scan_path():
    """
    3C 缩到 {R1, R5, R40}。**被砍掉的格子必须移出扫描路径** ——
    `run_exp3.sh` 是 `ls *.yaml`，留在原地就会照样提交。
    """
    assert not (EXP3 / "3c_R2.yaml").exists(), "3c_R2 还在扫描路径里"
    for kept in ("3c_R1", "3c_R5", "3c_R40"):
        assert kept in CELLS
    moved = EXP3 / "unused-3c"
    assert moved.is_dir() and list(moved.glob("*.yaml")), "被砍的 3C 格没有留档"


@pytest.mark.parametrize("name", CELLS)
def test_cell_passes_config_validate(name):
    """
    每一格都能通过启动校验 —— 本地就跑得了（config_validate 不 import TF）。
    不做这条，配置不兼容要等 GPU 排到才暴露（陷阱 #16 就是这么烧掉两批作业的）。
    """
    pytest.importorskip("numpy")
    import io
    from contextlib import redirect_stdout
    from config_validate import validate_config
    buf = io.StringIO()
    with redirect_stdout(buf):
        validate_config(cfg(name))
    out = buf.getvalue()
    assert "[配置校验] 通过" in out, out[-800:]
