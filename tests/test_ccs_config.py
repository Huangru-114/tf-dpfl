"""
阶段三 SA-C（D-109）：CCS 的配置侧 + 登记 + 回程解析。不需要 TF。

  · CCSF 两格（登记表展开结果）都通过 config_validate、打印 [设定12]；两格只差 defense 块；
  · config_validate 拒绝「静默不是 CCS」的写法（方法 / 层 / 模型 / cloud 层）；
  · resnet10_torch_proj 与 P2 模板等价（[设定4] 仍照实打印），其余 arch 不受影响；
  · 打印 ↔ 解析同源：[设定12] → run.ccs_*；[CCS] → ccs_decisions[]；schema 13；
  · 登记表的 run 块核对含 ccs_at / ccs_cluster / ccs_space / ccs_pool。
"""

import copy
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "fedavg"))

import alignment as AL                                          # noqa: E402
import collect_metrics as CM                                    # noqa: E402
import registry as R                                            # noqa: E402
from config_validate import ConfigError, validate_config        # noqa: E402
from defense.ccs_spec import SETTINGS_KEYS, settings_row        # noqa: E402
from utils.kvline import fmt_list, format_kv                    # noqa: E402

REG = ROOT / "experiments" / "defense" / "edge-native" / "registry.yaml"


@pytest.fixture(scope="module")
def ccsf():
    reg = R.Registry(REG)
    runs = {r["cell"]: r for r in reg.runs() if r["group"] == "CCSF"}
    return reg, {c: reg.intended_config(r) for c, r in runs.items()}, runs


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


@pytest.mark.parametrize("cell", ["off", "ccs-full"])
def test_ccsf_cells_validate_and_describe_themselves(ccsf, cell, capsys):
    _, cfgs, _ = ccsf
    validate_config(copy.deepcopy(cfgs[cell]))
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith("[设定12]")][0]
    assert ("ccs=true" in line) == (cell == "ccs-full")
    tpl = [ln for ln in out.splitlines() if ln.startswith("[设定4]")][0]
    assert "template=p2" in tpl and "arch=resnet10_torch_proj" in tpl      # 等价于模板，但照实打印


def test_ccsf_cells_differ_only_in_the_defense_block(ccsf):
    _, cfgs, _ = ccsf
    a, b = _flat(cfgs["off"]), _flat(cfgs["ccs-full"])
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - {"meta.run_id", "meta.cell"}
    assert diff == {"defense.name", "defense.layers", "defense.ccs.at", "defense.ccs.cluster",
                    "defense.ccs.space", "defense.ccs.pool"}, diff
    assert cfgs["ccs-full"]["model"]["arch"] == "resnet10_torch_proj" == cfgs["off"]["model"]["arch"]
    assert cfgs["off"]["federation"]["partition"] == "noniid"               # D-098
    assert cfgs["off"]["evaluation"]["eval_interval"] == 5 == cfgs["off"]["backdoor"]["eval_interval"]


def test_ccsf_expected_run_block_checks_the_ccs_switches(ccsf):
    reg, _, runs = ccsf
    exp = reg.expected_run_block(runs["ccs-full"])
    keys = {k for k, _ in R.EXPECT_KEYS}
    assert {"ccs_at", "ccs_cluster", "ccs_space", "ccs_pool"} <= keys
    assert exp["defense"] == "ccs"
    assert (exp["ccs_at"], exp["ccs_cluster"], exp["ccs_space"], exp["ccs_pool"]) == (True, True, "input", "round")


def _fails(cfg, match):
    with pytest.raises(ConfigError, match=match):
        validate_config(cfg)


def test_validation_rejects_silent_non_ccs(ccsf):
    _, cfgs, _ = ccsf
    base = cfgs["ccs-full"]
    c = copy.deepcopy(base)
    c["training"]["drift_correction"] = "hierfedavg"
    _fails(c, "hier_fedrep")
    c = copy.deepcopy(base)
    c["defense"]["layers"] = ["edge"]
    _fails(c, "必须含 client")
    c = copy.deepcopy(base)
    c["defense"]["layers"] = ["client"]
    _fails(c, "必须含 edge")
    c = copy.deepcopy(base)
    c["model"]["arch"] = "resnet10_torch"
    _fails(c, "resnet10_torch_proj")
    c = copy.deepcopy(base)
    c["defense"]["layers"] = ["client", "edge", "cloud"]
    _fails(c, "cloud")
    c = copy.deepcopy(base)
    c["defense"]["ccs"] = {"spaec": "pixel"}
    _fails(c, "不认识的键")


def test_cluster_only_arm_needs_neither_client_layer_nor_projection(ccsf, capsys):
    c = copy.deepcopy(ccsf[1]["ccs-full"])
    c["defense"]["layers"] = ["edge"]
    c["defense"]["ccs"] = {"at": False}
    c["model"]["arch"] = "resnet10_torch"
    validate_config(c)
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[设定12]")][0]
    assert "at=false" in line and "cluster=true" in line


def test_proj_arch_is_p2_equivalent_only_for_the_template_check():
    cfg = {"model": {"arch": "resnet10_torch_proj"}}
    assert ("model.arch", "resnet10_torch", "resnet10_torch_proj") not in AL.p2_mismatches(cfg)
    assert ("model.arch", "resnet10_torch", "resnet10") in AL.p2_mismatches({"model": {"arch": "resnet10"}})
    assert AL.describe_fields(cfg)["arch"] == "resnet10_torch_proj"


# ── 回程解析 ───────────────────────────────────────────────────────────────
def test_settings12_and_ccs_lines_roundtrip():
    cfg = {"defense": {"name": "ccs", "layers": ["client", "edge"], "ccs": {"pool": "edge_cache"}}}
    row = settings_row(cfg)
    lines = [format_kv("[设定12]", row),
             "    " + format_kv("[CCS]", {"cur": fmt_list([3, 7, 12]), "n_pool": 3, "status": "degenerate",
                                         "rejected": fmt_list([])}, edge_id=0),
             "    " + format_kv("[CCS]", {"cur": fmt_list([5]), "n_pool": 10, "status": "ok",
                                         "rejected": fmt_list([5])}, edge_id=2)]
    m = CM.collect("\n".join(lines))
    assert m["schema_version"] == 15
    assert m["run"]["ccs"] is True and m["run"]["ccs_pool"] == "edge_cache" and m["run"]["ccs_kl"] == 0.01
    assert all(f"ccs_{k}" in m["run"] for k in SETTINGS_KEYS)
    assert CM.CCS_SETTINGS_KEYS == SETTINGS_KEYS                     # 回程解析器里抄的那一份没走样
    # D-116 之前的 [CCS] 行没有 n_valid / nonfinite 两列 → 解析成 None / []（老日志照样能读）
    assert m["ccs_decisions"] == [
        {"edge": 0, "cur": [3, 7, 12], "n_pool": 3, "status": "degenerate", "rejected": [],
         "n_valid": None, "nonfinite": []},
        {"edge": 2, "cur": [5], "n_pool": 10, "status": "ok", "rejected": [5], "n_valid": None, "nonfinite": []}]


def test_old_logs_have_no_ccs():
    m = CM.collect("[Round 1]\n")
    assert m["run"]["ccs"] is None and m["ccs_decisions"] == []
    m = CM.collect(format_kv("[设定12]", settings_row({"defense": {"name": "none"}})))
    assert m["run"]["ccs"] is False and m["run"]["ccs_at"] is None


def test_stat_indices_are_injected_by_production_code():
    """F-107：CCSDefense 要 BN 统计量索引，SA-C 只在测试里手工注入 → CCSF ccs-full 在集群上第 1 个 edge 轮就崩。
    守卫：`defense/ccs.py` 以外的 fedavg/ 非测试代码里必须有人调 `set_stat_indices`（AST，本地秒级）。"""
    import ast
    callers = []
    for p in sorted((ROOT / "fedavg").rglob("*.py")):
        rel = p.relative_to(ROOT / "fedavg").as_posix()
        if rel == "defense/ccs.py" or "test" in p.name:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:                       # 死文件（test_main_names_are_bound 的 DEAD_FILES）
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "set_stat_indices"):
                callers.append(rel)
    assert "server/edge_server_base.py" in callers, callers


def test_settings13_fedper_roundtrip_and_validation(ccsf, capsys):
    """D-114：CCSF（FedPer）打 [设定13] fedper=true | bn_stats=shared → run.fedper / fedper_bn_stats；老日志 → None。"""
    cfg = copy.deepcopy(ccsf[1]["ccs-full"])
    assert cfg["training"]["drift_correction"] == "hier_fedper"
    warns = validate_config(cfg)
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("[设定13]"))
    m = CM.collect(line)
    assert m["run"]["fedper"] is True and m["run"]["fedper_bn_stats"] == "shared"
    assert any("hier_fedper" in w and "plocal_epochs" in w for w in warns)
    assert any("fedrep_bn_stats" in w for w in warns)                 # P2 模板里的 private 对 FedPer 无效 → 给警告
    m = CM.collect("[Round 1]\n")
    assert m["run"]["fedper"] is None and m["run"]["fedper_bn_stats"] is None
    cfg["training"]["drift_correction"] = "hier_fedrep"
    validate_config(cfg)
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[设定13]"))
    assert CM.collect(line)["run"]["fedper"] is False
