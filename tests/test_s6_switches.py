"""
tests/test_s6_switches.py  —  S6a：记录开关的配置侧（D-085；不 import TF，本地秒级）

  · 缺省关；不写 = 旧值（已有配置的 config_sha 不变 —— 没有任何已 materialize 的配置里出现这些键）；
  · config_validate：类型、前提（网格 / 交错调度 / hier_fedrep / Bad-PFL 固定攻击者）、防呆 ⑤；
  · G1P 登记：开 / 关两格只差三个记录开关；G1 的 set 已补齐（布点长度、300 有效轮、网格一致）。
"""

import copy
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import alignment as AL                                          # noqa: E402
import registry as R                                            # noqa: E402
from config_validate import ConfigError, validate_config        # noqa: E402

MECH = ROOT / "experiments" / "attack" / "hfl-mechanism"
KEYS = ("evaluation.update_geometry", "evaluation.update_sketch_dim",
        "evaluation.post_agg_eval", "evaluation.frozen_trigger")


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


def _cfg(cell):
    return yaml.safe_load((MECH / "configs" / f"G1P__{cell}__s42.yaml").read_text(encoding="utf-8"))


def _fails(cfg, match):
    with pytest.raises(ConfigError, match=match):
        validate_config(cfg)


def test_defaults_are_off_and_in_the_extra_table():
    assert [AL.get_switch({}, k) for k in KEYS] == [False, 4096, False, False]
    assert all(k in AL.ALL and AL.ALL[k].p2 is None for k in KEYS)     # 不进模板（记录开关，不是对齐项）
    assert set(KEYS).isdisjoint(AL.load_template())


def test_no_existing_config_mentions_the_new_keys():
    """已 materialize 的 89 个配置的 config_sha 不变 ⇐ 它们根本没有这几个键。"""
    for f in (MECH / "configs").glob("*.yaml"):
        if f.name.startswith(("G1P__", "G1__", "G1R5__")):
            continue
        text = f.read_text(encoding="utf-8")
        for k in KEYS:
            assert k.split(".")[-1] not in text, (f.name, k)


def test_g1p_on_and_off_differ_only_in_the_three_record_switches():
    on, off = _cfg("coll-on"), _cfg("coll-off")
    a, b = _flat(on), _flat(off)
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - {"meta.run_id", "meta.cell"}
    assert diff == {"evaluation.update_geometry", "evaluation.post_agg_eval",
                    "evaluation.frozen_trigger"}, diff
    assert all(on["evaluation"][k.split(".")[1]] is True for k in
               ("evaluation.update_geometry", "evaluation.post_agg_eval", "evaluation.frozen_trigger"))


@pytest.mark.parametrize("cell", ["coll-on", "coll-off", "dist-on"])
def test_g1p_configs_validate(cell, capsys):
    validate_config(_cfg(cell))
    out = capsys.readouterr().out
    assert "[设定9]" in out


def test_settings9_line_reflects_the_switches(capsys):
    validate_config(_cfg("coll-on"))
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[设定9]")][0]
    assert "update_geometry=true" in line and "post_agg_eval=true" in line
    assert "frozen_trigger=true" in line and "update_sketch_dim=4096" in line
    validate_config(_cfg("coll-off"))
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[设定9]")][0]
    assert "update_geometry=false" in line and "frozen_trigger=false" in line


def test_rejects_non_bool_and_bad_sketch_dim():
    c = _cfg("coll-on")
    c["evaluation"]["update_geometry"] = 1                           # Python 里 1 == True，必须显式拦
    _fails(c, "必须是 bool")
    c = _cfg("coll-on")
    c["evaluation"]["update_sketch_dim"] = 0
    _fails(c, "update_sketch_dim")
    c["evaluation"]["update_sketch_dim"] = True
    _fails(c, "update_sketch_dim")


def test_rejects_switch_without_grid_or_without_interleaved():
    c = _cfg("coll-on")
    del c["evaluation"]["eval_grid"]
    _fails(c, "要求 evaluation.eval_grid 已开")
    c = _cfg("coll-on")
    c["federation"]["edge_schedule"] = "sequential"
    _fails(c, "interleaved")


def test_rejects_geometry_on_other_methods_and_post_agg_without_badpfl():
    c = _cfg("coll-on")
    c["training"]["drift_correction"] = "hierfedavg"
    _fails(c, "hier_fedrep")
    c = _cfg("coll-on")
    c["backdoor"]["malicious_strategy"] = "vanilla"
    _fails(c, "badpfl|Bad-PFL")


def test_guard_stopping_needs_grid_5():
    """⑤：停止判据的容差按 5 有效轮标定（F-052）→ 开着 stopping 时网格只能是 5（或不开）。"""
    base = yaml.safe_load((MECH / "base.yaml").read_text(encoding="utf-8"))
    for g in (1, 10):
        c = _cfg("coll-off")
        c["stopping"] = copy.deepcopy(base["stopping"])
        c["evaluation"]["eval_grid"] = g
        c["federation"]["n_rounds"] = 30
        with pytest.raises(ConfigError, match="不能同开|eval_interval|标定"):
            validate_config(c)
    c = _cfg("coll-off")
    c["stopping"] = copy.deepcopy(base["stopping"])
    c["federation"]["n_rounds"] = 30
    validate_config(c)                                               # G=5 + stopping：照常通过


def test_g1_set_is_complete_and_consistent():
    reg = R.Registry(MECH / "registry.yaml")
    runs = [r for r in reg.runs() if r["group"] in ("G1", "G1R5")]
    assert len(runs) == 27 and sum(r["group"] == "G1" for r in runs) == 24
    for r in runs:
        cfg = reg.declared_config(r)
        fed, ev, bd = cfg["federation"], cfg["evaluation"], cfg["backdoor"]
        assert len(bd["malicious_per_edge"]) == fed["n_edges"] == 4
        assert sum(bd["malicious_per_edge"]) == 10
        assert fed["n_rounds"] * fed["edge_rounds"] == 300           # 固定 300 有效轮
        assert cfg.get("stopping") is None and ev["eval_grid"] == 5
        assert bd["eval_interval"] == ev["eval_interval"] == 1       # = lcm(5, R) / R
        # S6b：几何 / 云聚合后评估点 / 在线 c_k 开，frozen 不开（D-087）
        assert ev["update_geometry"] is True and ev["post_agg_eval"] is True and ev["update_ck"] is True
        assert (ev["update_ck_every"], ev["update_ck_n"], ev["update_ck_steps"]) == (5, 64, 5)
        assert "frozen_trigger" not in ev
        validate_config(cfg)
    r5 = [r for r in runs if r["group"] == "G1R5"]
    assert sorted(r["seed"] for r in r5) == [42, 43, 44]
    for r in r5:                                                     # R5 桥：= G3-C1 的配置 + 记录
        cfg = reg.declared_config(r)
        assert cfg["federation"]["edge_rounds"] == 5 and cfg["federation"]["partition"] == "designed"
        assert cfg["federation"]["design"]["condition"] == "C1"
        assert cfg["backdoor"]["malicious_per_edge"] == [10, 0, 0, 0]
    assert reg.unmet_requires("G1") == []                            # S6b 完成 + N-007 用户已确认（2026-10-03）→ 放行


# ══════════════════════════════════════════════════════════════════════════
# S6b（D-087）：在线 c_k 的开关
# ══════════════════════════════════════════════════════════════════════════
CK_KEYS = ("evaluation.update_ck", "evaluation.update_ck_every", "evaluation.update_ck_n",
           "evaluation.update_ck_steps")


def _ck_cfg():
    c = _cfg("coll-on")
    c["evaluation"].update(update_ck=True, update_ck_every=5, update_ck_n=64, update_ck_steps=5)
    return c


def test_ck_defaults_off_and_table_registration():
    assert [AL.get_switch({}, k) for k in CK_KEYS] == [False, 5, 64, 5]
    assert all(k in AL.ALL and AL.ALL[k].p2 is None for k in CK_KEYS)
    assert set(CK_KEYS).isdisjoint(AL.load_template())


def test_ck_config_validates_and_prints_settings10(capsys):
    validate_config(_ck_cfg())
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[设定10]")][0]
    assert "update_ck=true" in line and "update_ck_every=5" in line
    assert "update_ck_n=64" in line and "update_ck_steps=5" in line
    validate_config(_cfg("coll-off"))
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[设定10]")][0]
    assert "update_ck=false" in line


def test_ck_rejects_bad_types_and_missing_prerequisites():
    c = _ck_cfg()
    c["evaluation"]["update_ck"] = 1
    _fails(c, "必须是 bool")
    for k in ("update_ck_every", "update_ck_n", "update_ck_steps"):
        c = _ck_cfg()
        c["evaluation"][k] = 0
        _fails(c, k)
        c["evaluation"][k] = True
        _fails(c, k)
    c = _ck_cfg()
    del c["evaluation"]["eval_grid"]
    _fails(c, "eval_grid")
    c = _ck_cfg()
    c["federation"]["edge_schedule"] = "sequential"
    _fails(c, "interleaved")
    c = _ck_cfg()
    c["training"]["drift_correction"] = "hierfedavg"
    _fails(c, "hier_fedrep")
    c = _ck_cfg()
    c["federation"]["partition"] = "noniid"
    _fails(c, "S3 划分")
    c = _ck_cfg()
    c["federation"]["design"]["clean_per_edge"] = 0
    _fails(c, "clean_per_edge")
    c = _ck_cfg()
    c["evaluation"]["update_ck_n"] = 501
    _fails(c, "超过")


def test_no_existing_config_mentions_the_ck_keys():
    for f in (MECH / "configs").glob("*.yaml"):
        if f.name.startswith(("G1__", "G1R5__")):
            continue
        text = f.read_text(encoding="utf-8")
        assert "update_ck" not in text, f.name
