"""
tests/test_tier_split.py  —  S8：3-E 三层个性化（纯 python，不 import TF）

原文 §8 的三种划分（ResNet-10）：(a) FedRep 基线 / (b) 最后 1 个残差块只在 edge 内共享 /
(c) 最后 2 个。`federation.edge_shared_blocks` = 0 / 1 / 2；DECISIONS D-057 … D-059。

这里守的是**不需要 TF 就能断言**的部分（TF 那一半 —— 真 ResNet 的索引、cloud 不聚合、
广播保留、fresh-PM —— 在 tests/test_three_tier_personalization.py）：

  1. 规则：取值校验、块 → 层名前缀、按层名选索引、段复原（= compose_pm）；
  2. config_validate：四种「静默不生效 / 静默泄漏」的写法都拒绝；
  3. 自描述：`[设定6]` 打得出来、解析得回去，老日志 → None；
  4. 分析侧：runs_table 把 (a)/(b)/(c) 分成三格，而老文件与 (a) 同格；status 对账核对 k；
  5. 登记表 G6：9 个 run、布点 / 轮数 / 评估网格三件核对、停止判据关（D-058）；
  6. 接线（AST）：cloud 的 aggregate_edges 与 FedRep edge 的 set_weights 都真的走 keep_segment。
"""

import ast
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from config_validate import ConfigError, validate_config
from utils import tier_split as TS
from utils.kvline import format_kv, parse_kv

ROOT = Path(__file__).resolve().parent.parent
FEDAVG = ROOT / "fedavg"
sys.path.insert(0, str(ROOT / "harness"))
import registry as R                     # noqa: E402
import runs_table as RT                  # noqa: E402
from collect_metrics import collect      # noqa: E402

V2 = ROOT / "experiments/attack/hfl-mechanism/registry.yaml"


# ══════════════════════════════════════════════════════════════════════════
# 1. 规则
# ══════════════════════════════════════════════════════════════════════════
def test_missing_key_is_the_fedrep_baseline():
    assert TS.edge_shared_blocks({}) == 0
    assert TS.edge_shared_blocks({"federation": {}}) == 0
    assert TS.edge_shared_blocks({"federation": {"edge_shared_blocks": None}}) == 0


@pytest.mark.parametrize("k", [0, 1, 2])
def test_valid_values(k):
    assert TS.edge_shared_blocks({"federation": {"edge_shared_blocks": k}}) == k


@pytest.mark.parametrize("bad", [True, False, 3, -1, 1.0, "1"])
def test_invalid_values_raise(bad):
    """True == 1 在 Python 里成立 —— 不显式挡 bool，`edge_shared_blocks: true` 会被当成 (b)。"""
    with pytest.raises(ValueError):
        TS.edge_shared_blocks({"federation": {"edge_shared_blocks": bad}})


def test_prefixes_are_the_last_k_stages():
    assert TS.edge_shared_prefixes(0) == ()
    assert TS.edge_shared_prefixes(1) == ("stage4_",)
    assert TS.edge_shared_prefixes(2) == ("stage3_", "stage4_")
    with pytest.raises(ValueError):
        TS.edge_shared_prefixes(5)


# ResNet-10 的 get_weights 所属层（顺序只是示意；按名字选，与顺序无关）
OWNERS = ["stem_conv", "stem_bn", "stem_bn", "stem_bn", "stem_bn",
          "stage3_conv1", "stage3_bn1", "stage3_sc_conv",
          "stage4_conv1", "stage4_bn1", "stage4_sc_conv", "stage4_sc_bn",
          None, "stage40_imposter", "head", "head"]


def test_indices_owned_by_selects_by_layer_name_prefix():
    assert TS.indices_owned_by(OWNERS, ()) == []
    assert TS.indices_owned_by(OWNERS, TS.edge_shared_prefixes(1)) == [8, 9, 10, 11]
    assert TS.indices_owned_by(OWNERS, TS.edge_shared_prefixes(2)) == [5, 6, 7, 8, 9, 10, 11]
    # 前缀带下划线：stage40_… 不会被当成 stage4；没有所属层的变量（None）跳过


def test_keep_segment_restores_exactly_those_indices_and_mutates_nothing():
    new = [np.full((2, 2), 1.0), np.full(3, 2.0), np.full(1, 3.0)]
    old = [np.full((2, 2), -1.0), np.full(3, -2.0), np.full(1, -3.0)]
    new_c, old_c = copy.deepcopy(new), copy.deepcopy(old)
    out = TS.keep_segment(new, old, [1])
    assert np.array_equal(out[0], new[0]) and np.array_equal(out[2], new[2])
    assert np.array_equal(out[1], old[1])
    assert all(np.array_equal(a, b) for a, b in zip(new, new_c))
    assert all(np.array_equal(a, b) for a, b in zip(old, old_c))
    assert TS.keep_segment(new, old, []) == list(new)          # k=0：原样


def test_keep_segment_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        TS.keep_segment([np.zeros(3)], [np.zeros(4)], [0])


def test_describe_counts_elements_including_bn_stats():
    w = [np.zeros((3, 3)), np.zeros(4), np.zeros(5)]
    d = TS.describe(1, [1, 2], w)
    assert d == {"edge_shared_blocks": 1, "edge_shared_prefixes": "stage4",
                 "n_edge_tensors": 2, "n_edge_params": 9, "n_params": 18}
    assert TS.describe(2, [], w)["edge_shared_prefixes"] == "stage3,stage4"
    assert TS.describe(0, [], w)["edge_shared_prefixes"] == "none"


# ══════════════════════════════════════════════════════════════════════════
# 2. config_validate
# ══════════════════════════════════════════════════════════════════════════
def _g6(label="b", seed=42):
    reg = R.Registry(V2)
    run = next(r for r in reg.runs() if r["run_id"] == f"G6__{label}__s{seed}")
    return reg.declared_config(run)


@pytest.mark.parametrize("label", ["a", "b", "c"])
def test_g6_configs_validate_without_warnings(label):
    assert validate_config(_g6(label)) == []


def test_non_fedrep_method_is_rejected():
    """别的方法的 edge server 会被 cloud 广播整体覆盖 → edge 段每轮被冲掉，静默不生效。"""
    cfg = _g6()
    cfg["training"]["drift_correction"] = "hierfedavg"
    cfg["evaluation"]["pm_model"] = "fresh"
    for k in ("fedrep_poison_phases", "fedrep_bn_stats"):
        cfg["training"].pop(k, None)
    cfg.pop("meta")                       # 只测 S8 这一条，不让 P2 模板核对先报
    with pytest.raises(ConfigError, match="edge_shared_blocks"):
        validate_config(cfg)


def test_arch_without_stage_names_is_rejected():
    cfg = _g6()
    cfg["model"]["arch"] = "cifar_cnn_3conv"
    cfg.pop("meta")
    with pytest.raises(ConfigError, match="stage1_"):
        validate_config(cfg)


def test_cloud_layer_defense_is_rejected():
    cfg = _g6()
    cfg["defense"] = {"name": "median", "layers": ["edge", "cloud"], "params": {}}
    with pytest.raises(ConfigError, match="cloud 层防御"):
        validate_config(cfg)


def test_edge_layer_defense_is_fine():
    cfg = _g6()
    cfg["defense"] = {"name": "median", "layers": ["edge"], "params": {}}
    validate_config(cfg)


@pytest.mark.parametrize("bad", [True, 3, "1"])
def test_invalid_value_is_rejected_at_startup(bad):
    cfg = _g6()
    cfg["federation"]["edge_shared_blocks"] = bad
    with pytest.raises(ConfigError, match="edge_shared_blocks"):
        validate_config(cfg)


def test_single_edge_warns_that_the_arms_coincide():
    cfg = _g6()
    cfg["federation"]["n_edges"] = 1
    cfg["backdoor"]["malicious_per_edge"] = [10]
    assert any("三臂没有区别" in w for w in validate_config(cfg))


def test_other_groups_are_untouched_by_s8():
    """不写这个键 = 0：G7（与所有 S8 之前的配置）照过，没有新警告。"""
    reg = R.Registry(V2)
    run = next(r for r in reg.runs() if r["group"] == "G7")
    cfg = reg.declared_config(run)
    assert "edge_shared_blocks" not in cfg["federation"]
    assert validate_config(cfg) == []


# ══════════════════════════════════════════════════════════════════════════
# 3. 自描述：[设定6]
# ══════════════════════════════════════════════════════════════════════════
def _line(k, n_tensors=15):
    w = [np.zeros(2)] * 40
    return format_kv("[设定6]", TS.describe(k, list(range(n_tensors if k else 0)), w))


def test_settings6_roundtrip():
    d = parse_kv(_line(1), "[设定6]")
    assert d["edge_shared_blocks"] == 1 and d["edge_shared_prefixes"] == "stage4"
    assert d["n_edge_tensors"] == 15


def test_collect_puts_edge_shared_blocks_into_run_block():
    m = collect("[Config] loading x.yaml\n" + _line(2, 30) + "\n[Round 1]\n")
    assert m["run"]["edge_shared_blocks"] == 2
    assert m["run"]["tier_split"]["edge_shared_prefixes"] == "stage3,stage4"
    m0 = collect("[Config] loading x.yaml\n" + _line(0) + "\n[Round 1]\n")
    assert m0["run"]["edge_shared_blocks"] == 0            # 0 是「基线」，不是 None


def test_old_log_without_settings6_is_none_not_zero():
    m = collect("[Config] loading x.yaml\n[Round 1]\n")
    assert m["run"]["edge_shared_blocks"] is None and m["run"]["tier_split"] is None


# ══════════════════════════════════════════════════════════════════════════
# 4. 分析侧
# ══════════════════════════════════════════════════════════════════════════
def _block(k):
    b = {"method": "hier_fedrep", "n_edges": 4, "edge_rounds": 5,
         "malicious_per_edge": [10, 0, 0, 0]}
    if k is not None:
        b["edge_shared_blocks"] = k
    return b


def test_runs_table_separates_the_three_arms():
    """不加这个因素键，G6 的 (a)/(b)/(c) 会被当成同一格的三个重复。"""
    keys = {RT.factor_key(_block(k)) for k in (0, 1, 2)}
    assert len(keys) == 3


def test_runs_table_groups_old_files_with_the_baseline():
    """S8 之前的文件没有 [设定6]（None）—— 那时的代码也不存在 edge 段，与 k=0 是同一种 run。"""
    assert RT.factor_key(_block(None)) == RT.factor_key(_block(0))
    assert "edge_shared_blocks" not in RT._fmt_key(RT.factor_key(_block(0)))
    assert "edge_shared_blocks=1" in RT._fmt_key(RT.factor_key(_block(1)))


def test_status_reconciles_the_declared_k():
    reg = R.Registry(V2)
    exp = {r["run_id"]: reg.expected_run_block(r) for r in reg.runs()
           if r["group"] in ("G6", "G7")}
    assert exp["G6__a__s42"]["edge_shared_blocks"] == 0
    assert exp["G6__c__s44"]["edge_shared_blocks"] == 2
    assert exp["G7__std__s42"]["edge_shared_blocks"] is None       # 没声明 → 不核对


# ══════════════════════════════════════════════════════════════════════════
# 5. 登记表 G6（D-058）
# ══════════════════════════════════════════════════════════════════════════
def _g6_cfgs():
    reg = R.Registry(V2)
    return {r["run_id"]: reg.declared_config(r) for r in reg.runs() if r["group"] == "G6"}


def test_g6_has_nine_runs_and_the_three_splits():
    cfgs = _g6_cfgs()
    assert len(cfgs) == 9
    for rid, c in cfgs.items():
        want = {"a": 0, "b": 1, "c": 2}[rid.split("__")[1]]
        assert c["federation"]["edge_shared_blocks"] == want


def test_g6_placement_has_clean_victim_edges():
    """Q1：E0 放全部 10 个攻击者，E1–E3 干净 —— 3-E 测的是跨 edge 迁移。"""
    for rid, c in _g6_cfgs().items():
        mpe = c["backdoor"]["malicious_per_edge"]
        assert len(mpe) == c["federation"]["n_edges"] == 4, rid
        assert mpe == [10, 0, 0, 0], rid
        assert c["backdoor"]["malicious_placement"] == "by_edge", rid


def test_g6_is_fixed_length_at_the_cap_and_on_one_eval_grid():
    """Q2：固定 300 有效轮（= base 的 cap_effective），停止判据关；三臂评估网格相同。"""
    base_cap = R.Registry(V2).declared_config(
        next(r for r in R.Registry(V2).runs() if r["group"] == "G7"))["stopping"]["cap_effective"]
    grids = set()
    for rid, c in _g6_cfgs().items():
        f = c["federation"]
        assert f["n_rounds"] * f["edge_rounds"] == base_cap == 300, rid
        assert c["stopping"] is None, rid
        grids.add((f["edge_rounds"] * c["backdoor"]["eval_interval"],
                   f["edge_rounds"] * c["evaluation"]["eval_interval"]))
    assert grids == {(5, 5)}


def test_materialized_g6_configs_match_the_registry():
    """configs/ 里入库的 G6 配置与登记表现算的 sha 一致（改了 base / 登记表忘了 materialize 会红）。"""
    reg = R.Registry(V2)
    index = R.read_index(reg)
    for run in (r for r in reg.runs() if r["group"] == "G6"):
        row = index.get(run["run_id"])
        assert row is not None, f"{run['run_id']} 不在 configs/INDEX.tsv —— 重新 materialize"
        cfg = R.yaml.safe_load((ROOT / row["config"]).read_text(encoding="utf-8"))
        assert cfg == reg.declared_config(run), run["run_id"]


# ══════════════════════════════════════════════════════════════════════════
# 6. 接线（AST）：两处改动真的走 keep_segment
# ══════════════════════════════════════════════════════════════════════════
def _method(path, cls, name):
    tree = ast.parse((FEDAVG / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return f
    raise AssertionError(f"{path}: {cls}.{name} 不存在")


def _calls(fn):
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None))
    return out


def test_cloud_aggregate_edges_restores_the_edge_segment():
    assert {"robust_mean", "keep_segment"} <= _calls(
        _method("server/server.py", "CloudServer", "aggregate_edges"))


def test_fedrep_edge_keeps_its_segment_on_broadcast():
    assert "keep_segment" in _calls(
        _method("server/hier_fedrep.py", "HierFedRepEdgeServer", "set_weights"))


def test_tier_methods_are_exactly_the_edge_servers_that_keep_the_segment():
    """TIER_METHODS 放行的方法，其 edge server 必须覆写 set_weights 走 keep_segment。
    以后给别的方法开 S8，先在它的 edge server 里接线，再把它加进 TIER_METHODS。"""
    assert TS.TIER_METHODS == frozenset({"hier_fedrep"})
