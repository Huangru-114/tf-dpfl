"""
阶段三 SA0：`fedavg/analysis/p0_snapshot.py` 的静态守卫（纯 AST，不 import TF，本地秒级）。

- 驱动重建世界时调用 main.py 的同一批 helper、**同一顺序**、`BackdoorCloudServer` 同一组参数（不同就复原不了 run 的状态；
  GPU 上由 V0 兜底，这里让错误在本地就暴露）；
- 云聚合后点的列序抄自 harness/collect_metrics.LIGHT_EDGE_COLUMNS，两份必须相同；
- 作业脚本只传参数（陷阱 #19）。
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "fedavg/analysis/p0_snapshot.py"
MAIN = ROOT / "fedavg/main.py"
HELPERS = ("load_config", "set_seed", "build_clients", "build_edge_servers", "merge_test_datasets",
           "get_malicious_ids", "build_trigger", "build_eval_trigger", "select_eval_attacker", "BackdoorCloudServer")


def _func(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(fn):
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in HELPERS:
                out.append((n.lineno, n.col_offset, name, n))
    return sorted(out)


def _first_order(calls):
    seen = []
    for _, _, name, _ in calls:
        if name not in seen:
            seen.append(name)
    return seen


def test_driver_rebuilds_the_world_with_mains_helpers_in_mains_order():
    drv = _calls(_func(DRIVER, "build_world"))
    mn = _calls(_func(MAIN, "run_experiment"))
    assert _first_order(drv) == list(HELPERS)
    assert _first_order(mn) == list(HELPERS)


def test_backdoor_cloud_server_gets_the_same_keywords():
    kw = lambda calls: sorted(k.arg for _, _, n, c in calls if n == "BackdoorCloudServer" for k in c.keywords)
    assert kw(_calls(_func(DRIVER, "build_world"))) == kw(_calls(_func(MAIN, "run_experiment")))


def test_post_edge_columns_match_collect_metrics():
    def literal(path, name):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == name)
        return ast.literal_eval(node.value)
    assert literal(DRIVER, "POST_EDGE_COLUMNS") == literal(ROOT / "harness/collect_metrics.py", "LIGHT_EDGE_COLUMNS")


def test_job_script_passes_arguments_only():
    src = (ROOT / "experiments/defense/edge-native/p0.sbatch").read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert 'source "$ROOT/cluster_env.sh"' in code and 'cd "$ROOT/fedavg"' in code
    assert "--override" not in code and "--defense" not in code and "--seed" not in code
    assert "$PY -m analysis.p0_snapshot" in code
