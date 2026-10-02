"""
tests/test_collect_eval_grid.py  —  S5：统一评估网格的回程（collect_metrics schema 9 + harness）

**两侧分开测**（陷阱 #10：解析器认得格式 ≠ 代码会打印它）：
  · 上游：`CloudServer._emit_light` 打印的键 = `collect_metrics.LIGHT_COLUMNS`（AST，不 import TF）；
  · 下游：`[设定8]` → run.eval_grid / run.grid；`[Light]` → light_rounds[]；`[LightEdge]` → 紧凑行；
    网格下非全量轮的 `[Cloud] GM=n/a` 行照样解析（含 round_time），老日志一字不变。
  · runs_table：没有轻评估点时 grid_series == effective_round_series（逐位）；有轻评估点时 T_θ / 末 10 点
    用合并后的网格序列；series.csv 的轻评估行带自己的有效轮；因素键多了 eval_grid（老文件 = None）。
  · instrumentation_check --grid：GM / EM 网格下不评不算分歧，其他照旧逐位比。
  · status：声明了网格、日志里却没有 → mismatch。
纯标准库，本地秒级。
"""

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import collect_metrics as CM                                    # noqa: E402
import instrumentation_check as IC                              # noqa: E402
import runs_table as T                                          # noqa: E402
import status as ST                                             # noqa: E402
from analyze_exp3 import effective_round_series                 # noqa: E402
from utils.kvline import format_kv                              # noqa: E402

HEAD = ["[Config] loading experiments/attack/hfl-mechanism/configs/S5P__R20-on__s42.yaml",
        "[Config] run_name = s5p_seed42",
        "[配置校验] 通过 | method=hier_fedrep | defense=none | attack=badpfl | 0 个警告",
        "[设定] client_fraction=0.1 | poison_ratio=0.2 | n_clients=100 | n_edges=2 | "
        "edge_rounds=20 | n_malicious=10 | forced_participation=False | arch=resnet10_torch"]
GRID8 = "[设定8] eval_grid=5 | period_eff=20 | full_every=1 | light_per_period=3 | slope_axis=grid | gm_em=grid"
OFF8 = "[设定8] eval_grid=n/a | period_eff=n/a | full_every=n/a | light_per_period=n/a | slope_axis=cloud | gm_em=every_round"


def _light(r, er, eff, asr, pm=0.5, em=0.4, s=1.25):
    return format_kv("[Light]", {
        "edge_round": er, "effective_round": eff, "pm_acc": pm, "em_acc": em,
        "edge_asr": asr, "local_benign_asr": asr, "same_edge_asr": asr, "diff_edge_asr": None,
        "local_malicious_asr": 0.9, "eval_s": s}, round_idx=r)


def _light_edge(r, er, eff, e, asr):
    return format_kv("[LightEdge]", {
        "edge_round": er, "effective_round": eff, "edge_asr": asr, "client_benign": asr,
        "client_malicious": 0.9, "pm_acc": 0.5, "em_acc": 0.4}, round_idx=r, edge_id=e)


def _bd(r, asr):
    return (f"[Backdoor] Round {r} | GM_ASR={asr:.3f} | EM_ASR={asr:.3f} | "
            f"local_benign={asr:.3f} (same_edge={asr:.3f}, diff_edge=n/a) | local_malicious=0.900")


def _grid_log(n=2, R=20, G=5):
    """R20、G5、n 个云轮：每云轮 3 个轻评估点（er 5 / 10 / 15）+ 云轮末全量。ASR = eff / 100。"""
    L = HEAD + [GRID8]
    for g in range(1, n + 1):
        L.append(f"[Round {g:>3}] Broadcasting to 2 edges...")
        for er in (5, 10, 15):
            eff = (g - 1) * R + er
            L.append(_light(g, er, eff, eff / 100))
            L += [_light_edge(g, er, eff, e, eff / 100) for e in (0, 1)]
        L.append(f"[Checksum] Round {g} | global=abc{g:09d}")
        L.append(f"  [Cloud] GM=0.1000 | EM=0.2000 PM=0.3000 | loss=2.0000 | time=100.0s | "
                 f"comm=1.0MB (total=1MB)")
        L.append(_bd(g, g * R / 100))
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════
# 上游 ↔ 下游同源
# ══════════════════════════════════════════════════════════════════════════
def _server_ast():
    src = (ROOT / "fedavg" / "server" / "server.py").read_text(encoding="utf-8")
    return ast.parse(src)


def _const_tuple(name):
    for n in _server_ast().body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name:
            return [e.value for e in n.value.elts]
    raise AssertionError(f"server.py 没有常量 {name}")


def _emit_keys(var):
    """_emit_light 里把字段塞进 format_kv 的那个 dict 字面量（S6a 起写成 fields / ef 两个变量）。"""
    fn = next(n for n in ast.walk(_server_ast())
              if isinstance(n, ast.FunctionDef) and n.name == "_emit_light")
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == var
                and isinstance(node.value, ast.Dict)):
            return [k.value for k in node.value.keys]
    raise AssertionError(f"_emit_light 没有字典 {var}")


def test_light_line_keys_are_the_metrics_field_names():
    assert ["round", *_emit_keys("fields"), *_const_tuple("LIGHT_DETAIL_FIELDS")] \
        == list(CM.LIGHT_COLUMNS)
    edge = _emit_keys("ef")
    assert edge[:2] == ["edge_round", "effective_round"]
    assert ["edge_id", *edge[2:], *_const_tuple("LIGHT_DETAIL_EDGE_FIELDS")] \
        == list(CM.LIGHT_EDGE_COLUMNS)


# ══════════════════════════════════════════════════════════════════════════
# collect_metrics
# ══════════════════════════════════════════════════════════════════════════
def test_schema_is_11():
    assert CM.SCHEMA_VERSION == 11
    assert CM.collect("\n".join(HEAD))["schema_version"] == 11


def test_settings8_round_trips_into_the_run_block():
    run = CM.collect(_grid_log())["run"]
    assert run["eval_grid"] == 5
    assert run["grid"] == {"period_eff": 20, "full_every": 1, "light_per_period": 3,
                           "slope_axis": "grid", "gm_em": "grid"}
    run = CM.collect("\n".join(HEAD + [OFF8]))["run"]
    assert run["eval_grid"] is None and run["grid"]["slope_axis"] == "cloud"


def test_old_log_without_settings8_gives_none():
    run = CM.collect("\n".join(HEAD))["run"]
    assert run["eval_grid"] is None and run["grid"] is None
    assert run["edge_rounds"] == 20                                   # [设定] 照常


def test_light_points_have_their_own_table_and_stay_out_of_rounds():
    m = CM.collect(_grid_log())
    assert [r["round"] for r in m["rounds"]] == [1, 2]               # 全量点不受影响
    assert [r["round"] for r in m["acc_rounds"]] == [1, 2]
    lr = m["light_rounds"]
    assert [(r["round"], r["edge_round"], r["effective_round"]) for r in lr] == [
        (1, 5, 5), (1, 10, 10), (1, 15, 15), (2, 5, 25), (2, 10, 30), (2, 15, 35)]
    assert lr[0]["local_benign_asr"] == pytest.approx(0.05) and lr[0]["diff_edge_asr"] is None
    assert set(lr[0]) == set(CM.LIGHT_COLUMNS)
    pel = m["per_edge_light_rounds"]
    assert m["per_edge_light_columns"] == list(CM.LIGHT_EDGE_COLUMNS)
    assert sorted(pel) == [5, 10, 15, 25, 30, 35]
    assert [row[0] for row in pel[25]] == [0, 1] and pel[25][0][1] == pytest.approx(0.25)
    assert m["timing_summary"]["n_light_evals"] == 6
    assert m["timing_summary"]["light_eval_total_s"] == pytest.approx(7.5)


def test_no_grid_log_has_empty_light_tables():
    m = CM.collect("\n".join(HEAD + ["[Round   1] Broadcasting to 2 edges...",
                                     "  [Cloud] GM=0.1 | EM=0.5 PM=0.6 | loss=2.0 | time=10.0s | "
                                     "comm=1.0MB (total=1MB)"]))
    assert m["light_rounds"] == [] and m["per_edge_light_rounds"] == {}
    assert m["timing_summary"]["light_eval_total_s"] is None


def test_cloud_line_with_na_gm_still_parses_including_round_time():
    """网格下非全量轮：GM / EM 不评。不放宽正则的话，这一行（连同 round_time）会被静默丢掉。"""
    log = "\n".join(HEAD + [
        "[Round   1] Broadcasting to 2 edges...",
        "  [Cloud] GM=n/a | EM=n/a | loss=n/a | time=12.5s | comm=1.0MB (total=1MB)",
        "[Round   2] Broadcasting to 2 edges...",
        "  [Cloud] GM=0.1000 | EM=0.2000 PM=0.3000 | loss=2.0000 | time=20.0s | comm=1.0MB (total=2MB)"])
    acc = CM.collect(log)["acc_rounds"]
    assert acc[0] == {"round": 1, "gm_acc": None, "em_acc": None, "pm_acc": None,
                      "round_time": 12.5, **{k: None for k in acc[0] if k.startswith("acc_")
                                             or k == "pm_acc_stale"}}
    assert acc[1]["gm_acc"] == 0.1 and acc[1]["round_time"] == 20.0
    assert CM.collect(log)["timing_summary"]["round_time_total_s"] == 32.5


def test_light_payload_fits_the_size_budget():
    """最坏的 G2 格 e10-R20：45 个轻评估点 × 10 edge。新增字段 ≤ 60 KB（S5 时 30 KB；S6a ④ 给轻评估行加了 6 + 2 列，null 也占键名，真值再多几位数字）。"""
    L = HEAD + [GRID8]
    for g in range(1, 16):
        L.append(f"[Round {g:>3}] Broadcasting to 10 edges...")
        for er in (5, 10, 15):
            eff = (g - 1) * 20 + er
            L.append(_light(g, er, eff, 0.123456))
            L += [_light_edge(g, er, eff, e, 0.123456) for e in range(10)]
    m = CM.collect("\n".join(L))
    size = len(json.dumps({k: m[k] for k in ("light_rounds", "per_edge_light_rounds",
                                             "per_edge_light_columns")}))
    assert len(m["light_rounds"]) == 45 and size < 60_000, size


# ══════════════════════════════════════════════════════════════════════════
# runs_table / figures / verdicts
# ══════════════════════════════════════════════════════════════════════════
def test_grid_series_equals_the_old_series_when_there_are_no_light_points():
    for f in sorted((ROOT / "experiments/attack/hfl-mechanism/results/P2").rglob("*.metrics.json"))[:20]:
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("light_rounds"):           # 本条只管「没有轻评估点」的老文件（S5P / G1P 有）
            continue
        er = m["run"].get("edge_rounds") or 1
        for k in ("local_benign_asr", "edge_asr", "global_asr"):
            assert T.grid_series(m, k) == effective_round_series(m["rounds"], er, k), f.name
        assert T.grid_series(m, "pm_acc") == effective_round_series(m["acc_rounds"], er, "pm_acc")


def test_t50_and_final_window_use_the_light_points():
    m = CM.collect(_grid_log(n=3))
    s = T.grid_series(m, "local_benign_asr")
    assert [e for e, _ in s] == list(range(5, 61, 5))                 # 12 个点：9 轻 + 3 全量
    row = T.summarize_run(m, name="x", source="x")
    # ASR = eff / 100 → 0.5 在有效轮 50（轻评估点 45 与全量点 60 之间插值），旧网格（只有 20/40/60）也是 50
    assert row["t0.5_local_benign_asr"] == pytest.approx(50.0)
    assert row["t0.25_local_benign_asr"] == pytest.approx(25.0)
    assert row["local_benign_asr_n"] == 10                            # 末 10 个网格点
    assert row["local_benign_asr_last10"] == pytest.approx(sum(range(15, 61, 5)) / 10 / 100)
    assert row["global_asr_n"] == 3                                    # global 只有全量点
    assert row["eval_grid"] == 5


def test_series_rows_carry_the_light_points_on_their_own_effective_round():
    m = CM.collect(_grid_log())
    rows = [r for r in T.series_rows(m, "x") if r[4] == "local_benign_asr"]
    assert [(r[1], r[2]) for r in rows] == [(5, 1), (10, 1), (15, 1), (20, 1),
                                            (25, 2), (30, 2), (35, 2), (40, 2)]
    edge = [r for r in T.series_rows(m, "x") if r[4] == "edge.edge_asr" and r[3] == 1]
    assert [(r[1], r[2]) for r in edge] == [(5, 1), (10, 1), (15, 1), (25, 2), (30, 2), (35, 2)]


def test_eval_grid_is_a_factor_and_old_files_stay_in_their_cell():
    import figures as F
    import verdicts as V
    assert "eval_grid" in T.FACTOR_KEYS and "eval_grid" in F.FACTOR_COLUMNS
    assert "eval_grid" in V._SAME_EXCEPT_TOPOLOGY
    assert "eval_grid" not in T.FACTOR_DEFAULTS
    old = {"method": "hier_fedrep", "edge_rounds": 1}
    assert T.factor_key(old) == T.factor_key({**old, "eval_grid": None})
    assert T.factor_key(old) != T.factor_key({**old, "eval_grid": 5})


# ══════════════════════════════════════════════════════════════════════════
# instrumentation_check --grid / status
# ══════════════════════════════════════════════════════════════════════════
def _pair():
    ref = {"checksums": [{"round": 1, "global": "a"}, {"round": 2, "global": "b"}],
           "rounds": [{"round": 2, "local_benign_asr": 0.5}],
           "acc_rounds": [{"round": 1, "gm_acc": 0.1, "em_acc": 0.2, "pm_acc": None},
                          {"round": 2, "gm_acc": 0.3, "em_acc": 0.4, "pm_acc": 0.6}],
           "per_edge_acc_rounds": {"1": [{"edge_id": 0, "em_acc": 0.2, "pm_acc": None}],
                                   "2": [{"edge_id": 0, "em_acc": 0.4, "pm_acc": 0.6}]}}
    new = json.loads(json.dumps(ref))
    new["acc_rounds"][0].update(gm_acc=None, em_acc=None)
    new["per_edge_acc_rounds"]["1"][0]["em_acc"] = None
    new["light_rounds"] = [{"round": 1, "effective_round": 1}]
    return ref, new


def test_instrumentation_check_grid_mode():
    ref, new = _pair()
    res = IC.compare(ref, new, grid=True)
    assert res["diffs"] == [] and res["n_grid_skipped"] == 3 and res["n_light"] == 1
    assert IC.compare(ref, new)["diffs"]                              # 不加 --grid → 算分歧
    new["acc_rounds"][1]["pm_acc"] = 0.61                             # 全量点的数变了 → 仍是分歧
    assert IC.compare(ref, new, grid=True)["diffs"]
    ref, new = _pair()
    new["checksums"][1]["global"] = "c"
    assert any(k.startswith("checksums") for k, *_ in IC.compare(ref, new, grid=True)["diffs"])


def test_status_flags_a_grid_that_silently_did_not_happen():
    exp = {"eval_grid": 5, "edge_rounds": 20}
    assert ST.run_block_mismatches(exp, {"eval_grid": None, "edge_rounds": 20}) == [
        ["eval_grid", 5, None]]
    assert ST.run_block_mismatches(exp, {"eval_grid": 5, "edge_rounds": 20}) == []
    assert ST.run_block_mismatches({"eval_grid": None}, {"eval_grid": None}) == []
