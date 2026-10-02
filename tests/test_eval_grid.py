"""
tests/test_eval_grid.py  —  S5：统一评估网格（D-055 / D-084）的纯 python 部分

  · 网格规则（fedavg/server/eval_grid.py）：R ∈ {1, 2, 5, 10, 20}、G = 5 → 全部评估点（全量 + 轻）
    都落在 5 的倍数有效轮上、每格 300 有效轮内 60 个点；R5 与 flat 的点与改动前**完全相同**；
  · 开关在 EXTRA_SWITCHES（预算旋钮，不进模板），缺省 None = 旧行为；
  · config_validate §4b'：每一条拒绝 / 警告都实测一次；G2 / S5P 的声明配置全部通过；
  · 不写网格时什么都不变：已登记的每个配置，当前声明的 config_sha = INDEX.tsv 里的（status 的 done 83 靠它）。

纯标准库 + pyyaml（config_validate 本地可跑）。TF 侧（真跑 cloud / edge / client，开 / 关网格
checksum 相同、轻评估前后状态不变）在 tests/test_eval_grid_tf.py。
"""

import copy
import io
import contextlib
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import alignment as AL                                          # noqa: E402
import registry as R                                            # noqa: E402
from config_validate import ConfigError, validate_config        # noqa: E402
from server import eval_grid as EG                              # noqa: E402
from utils.kvline import format_kv, parse_kv                    # noqa: E402
from utils.provenance import config_sha                         # noqa: E402

MECH = ROOT / "experiments" / "attack" / "hfl-mechanism"
G = 5


def _reg():
    return R.Registry(MECH / "registry.yaml")


def _cfg(run_id):
    reg = _reg()
    run = next(r for r in reg.runs() if r["run_id"] == run_id)
    return copy.deepcopy(reg.declared_config(run))


def _validate(cfg):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        w = validate_config(cfg)
    return w, out.getvalue()


# ══════════════════════════════════════════════════════════════════════════
# 1. 网格规则
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("R", [1, 2, 5, 10, 20])
def test_every_eval_point_is_on_the_grid_and_there_are_60(R):
    pts = EG.eval_points(G, R, 300 // R)
    assert pts and all(eff % G == 0 for _, _, eff, _ in pts)
    assert [eff for _, _, eff, _ in pts] == list(range(5, 301, 5))     # 60 个点，一个不缺、不重
    for g, er, eff, kind in pts:
        assert eff == (g - 1) * R + er
        assert (kind == "full") == (er == R)                          # 全量 = 云轮末
        assert 1 <= er <= R


@pytest.mark.parametrize("R,full,light", [(1, 60, 0), (2, 30, 30), (5, 60, 0),
                                          (10, 30, 30), (20, 15, 45)])
def test_full_and_light_counts_match_the_plan_table(R, full, light):
    pts = EG.eval_points(G, R, 300 // R)
    assert sum(k == "full" for *_, k in pts) == full
    assert sum(k == "light" for *_, k in pts) == light


def test_r5_points_equal_the_old_points_exactly():
    """反向锚点：R5 + eval_interval 1（base）就是改动前的网格 —— 开网格不该多一个、少一个点。"""
    assert EG.eval_points(G, 5, 60) == EG.legacy_eval_points(5, 1, 60)
    assert EG.full_interval(G, 5) == 1


def test_flat_points_equal_the_old_eval_interval_5():
    """flat（R=1）没有轻评估点；全量每 5 云轮一次 = 旧的 eval_interval 5（G2 flat / G8F）。"""
    assert EG.eval_points(G, 1, 300) == EG.legacy_eval_points(1, 5, 300)


def test_old_grid_of_r20_is_what_f046_complains_about():
    """反向锚点（F-046）：旧网格下 R20 每 20 有效轮一点 → T50 在 20–45 的格子只有 1–2 个点。"""
    old = EG.legacy_eval_points(20, 1, 15)
    assert [eff for *_, eff, _ in old] == list(range(20, 301, 20))
    assert len([p for p in old if p[2] <= 45]) == 2


@pytest.mark.parametrize("R,need", [(1, 5), (2, 5), (5, 1), (10, 1), (20, 1), (3, 5), (4, 5)])
def test_full_interval_is_lcm_over_r(R, need):
    assert EG.full_interval(G, R) == need
    assert EG.period_eff(G, R) == need * R


def test_light_edge_rounds_sit_strictly_inside_the_cloud_round():
    assert EG.light_edge_rounds(5, 20, 1) == [5, 10, 15]
    assert EG.light_edge_rounds(5, 10, 3) == [5]
    assert EG.light_edge_rounds(5, 2, 3) == [1]                       # eff 5 = 第 3 云轮第 1 个 edge 轮
    assert EG.light_edge_rounds(5, 2, 5) == []                        # eff 10 = 云轮末 → 全量
    assert EG.light_edge_rounds(5, 5, 7) == []
    assert EG.light_edge_rounds(5, 1, 7) == []


def test_grid_x_equals_the_cloud_round_when_r_equals_g():
    """停止判据的横轴（网格序号）在 R5 下就是云轮号 → R5 的停轮逐位不变（F-052 的反向锚点在 test_stopping）。"""
    assert [EG.grid_x(g * 5, G) for g in range(1, 61)] == list(range(1, 61))
    assert EG.grid_x(15, G) == 3
    with pytest.raises(ValueError):
        EG.grid_x(12, G)


def test_describe_without_and_with_the_grid():
    assert EG.describe({}) == {"eval_grid": None, "period_eff": None, "full_every": None,
                               "light_per_period": None, "slope_axis": "cloud",
                               "gm_em": "every_round"}
    d = EG.describe({"evaluation": {"eval_grid": 5}, "federation": {"edge_rounds": 20}})
    assert d == {"eval_grid": 5, "period_eff": 20, "full_every": 1, "light_per_period": 3,
                 "slope_axis": "grid", "gm_em": "grid"}
    # [设定8] 往返（kvline）：打印与解析同源
    assert parse_kv(format_kv("[设定8]", d), "[设定8]") == d
    assert parse_kv(format_kv("[设定8]", EG.describe({})), "[设定8]") == EG.describe({})


# ══════════════════════════════════════════════════════════════════════════
# 2. 开关
# ══════════════════════════════════════════════════════════════════════════
def test_switch_is_extra_and_defaults_to_old_behaviour():
    extra = {s.key: s for s in AL.EXTRA_SWITCHES}
    assert EG.KEY in extra and EG.KEY not in {s.key for s in AL.SWITCHES}
    assert extra[EG.KEY].legacy is None and extra[EG.KEY].p2 is None
    assert AL.get_switch({}, EG.KEY) is None
    assert EG.KEY not in AL.load_template()


def test_base_yaml_does_not_turn_the_grid_on():
    """网格只在组的 set: 里开（交接「不要做的」）：写进 base.yaml 会让所有 P2 组变 stale。"""
    base = yaml.safe_load((MECH / "base.yaml").read_text(encoding="utf-8"))
    assert "eval_grid" not in (base.get("evaluation") or {})


# ══════════════════════════════════════════════════════════════════════════
# 3. config_validate §4b'
# ══════════════════════════════════════════════════════════════════════════
def _g2_runs():
    return [r for r in _reg().runs() if r["group"] == "G2" and r["seed"] == 42]


def test_every_g2_cell_and_s5p_validates():
    reg = _reg()
    runs = _g2_runs() + [r for r in reg.runs() if r["group"] == "S5P"]
    assert len(runs) == 11 + 4
    for run in runs:
        _validate(copy.deepcopy(reg.declared_config(run)))           # 不抛 = 通过


def test_every_g2_cell_is_on_the_5_round_grid():
    """「各格评估网格（有效轮）一致」（registry.yaml 头注释的三件之一）：网格开着、两个 interval = lcm/R。"""
    reg = _reg()
    for run in _g2_runs():
        cfg = reg.declared_config(run)
        R_ = int(cfg["federation"]["edge_rounds"])
        assert cfg["evaluation"]["eval_grid"] == G, run["run_id"]
        assert int(cfg["evaluation"]["eval_interval"]) == EG.full_interval(G, R_), run["run_id"]
        assert int(cfg["backdoor"]["eval_interval"]) == EG.full_interval(G, R_), run["run_id"]


def test_validate_prints_settings8():
    _, out = _validate(_cfg("G2__e2-R20__s42"))
    line = next(ln for ln in out.splitlines() if ln.startswith("[设定8]"))
    assert parse_kv(line, "[设定8]") == {"eval_grid": 5, "period_eff": 20, "full_every": 1,
                                        "light_per_period": 3, "slope_axis": "grid",
                                        "gm_em": "grid"}
    _, out = _validate(_cfg("G6__a__s42"))
    line = next(ln for ln in out.splitlines() if ln.startswith("[设定8]"))
    assert parse_kv(line, "[设定8]")["slope_axis"] == "cloud"


@pytest.mark.parametrize("bad", [True, 0, -5, "5", 5.0])
def test_grid_value_must_be_a_positive_int(bad):
    cfg = _cfg("G2__e2-R5__s42")
    cfg["evaluation"]["eval_grid"] = bad
    with pytest.raises(ConfigError, match="eval_grid 必须是"):
        _validate(cfg)


@pytest.mark.parametrize("sec", ["backdoor", "evaluation"])
def test_eval_interval_must_equal_lcm_over_r(sec):
    cfg = _cfg("G2__e2-R2__s42")
    cfg[sec]["eval_interval"] = 1                                     # R2 要 5
    with pytest.raises(ConfigError, match=f"{sec}.eval_interval"):
        _validate(cfg)


def test_last_round_must_be_on_the_grid():
    cfg = _cfg("G2__e2-R2__s42")
    cfg["federation"]["n_rounds"] = 151
    cfg["stopping"] = None                                            # 只测网格这一条
    with pytest.raises(ConfigError, match="末轮评估会落在网格外"):
        _validate(cfg)


@pytest.mark.parametrize("key,val,msg", [
    (("federation", "edge_schedule"), "sequential", "edge_schedule = interleaved"),
    (("evaluation", "pm_model"), "stale", "pm_model = fresh"),
    (("data", "batch_pipeline"), "legacy", "batch_pipeline = per_epoch"),
    (("backdoor", "eval_xi_model"), "victim", "eval_xi_model = fixed_attacker"),
    (("backdoor", "badpfl_shared_generator"), False, "badpfl_shared_generator = true"),
    (("defense", "name"), "simple_tuning", "后处理防御"),
])
def test_grid_is_narrowed_to_the_p2_code_path(key, val, msg):
    cfg = _cfg("G2__e2-R10__s42")
    cfg.setdefault("meta", {})["protocol"] = "P1"                     # 绕开 P2 模板核对，只测网格这条
    cfg[key[0]][key[1]] = val
    with pytest.raises(ConfigError, match=msg):
        _validate(cfg)


def test_flat_with_sequential_schedule_is_fine():
    """R = 1 没有轻评估点，调度无所谓。"""
    cfg = _cfg("G2__flat__s42")
    cfg.setdefault("meta", {})["protocol"] = "P1"
    cfg["federation"]["edge_schedule"] = "sequential"
    _validate(cfg)


def test_floor_not_on_the_period_warns_once():
    cfg = _cfg("G2__e2-R2__s42")                                      # period 10，floor 150 → 整除
    w, _ = _validate(cfg)
    assert not [x for x in w if "实际地板推迟" in x]
    cfg["stopping"]["floor_effective"] = 145
    w, _ = _validate(cfg)
    assert len([x for x in w if "实际地板推迟到 150" in x]) == 1
    # R20：period = R → §3b 的老警告已经说了，网格这条不重复
    w, _ = _validate(_cfg("G2__e2-R20__s42"))
    assert len([x for x in w if "160" in x]) == 1


def test_side_columns_warn_that_they_stay_off_the_grid():
    w, _ = _validate(_cfg("G2__e2-R10__s42"))
    assert any("stale_asr_every, stale_pm_every 只按全量点计数" in x for x in w)


def test_point_count_estimate_uses_the_grid():
    """§3b：网格下 pm_acc 平台判据的点数 = cap ÷ G（轻评估点也喂），R20 不再报「永远算不出来」。"""
    cfg = _cfg("G2__e2-R20__s42")
    cfg["stopping"]["pm_window"] = 61                                 # 60 个点 < 61
    w, _ = _validate(cfg)
    assert any("最多 60 个评估点" in x for x in w)


# ══════════════════════════════════════════════════════════════════════════
# 4. 不写网格时什么都不变
# ══════════════════════════════════════════════════════════════════════════
def test_existing_config_shas_are_unchanged():
    """INDEX.tsv 里每一行（含 S5 之前的 89 行）：当前声明的配置 sha = 登记时的 sha。"""
    reg = _reg()
    index = R.read_index(reg)
    assert len(index) >= 93
    runs = {r["run_id"]: r for r in reg.runs()}
    for run_id, row in index.items():
        cfg = yaml.safe_load(yaml.safe_dump(reg.declared_config(runs[run_id]), sort_keys=False,
                                            allow_unicode=True))
        assert config_sha(cfg) == row["config_sha"], run_id
