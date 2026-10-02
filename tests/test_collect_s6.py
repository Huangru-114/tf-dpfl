"""
tests/test_collect_s6.py  —  S6a：回程（collect_metrics schema 10；D-085）

**两侧分开测**（陷阱 #10：解析器认得格式 ≠ 代码会打印它）：
  · 上游：`UpdateGeometry.flush` / `_emit_light` / `_frozen_eval` 打印的键 = collect_metrics 的列常量（AST / 实打）；
  · 下游：`[设定9]` → run 块的四个标量；`[UpdateGeo]` → 紧凑表；`[PostAgg*]` → post_agg_rounds[]（单独成表）；
    `[FrozenASR*]` → frozen_rounds[] + 紧凑行；`[Light]` 新列；`[Dump] kind=sketch` → dumps.sketch；
    老日志 → 空表 / None，已有的字段一字不变。
纯标准库 + numpy，本地秒级。
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import collect_metrics as CM                                    # noqa: E402
from aggregation.client_update import ClientUpdate              # noqa: E402
from server import update_geometry as UG                        # noqa: E402
from utils.kvline import format_kv                    # noqa: E402

HEAD = ["[Config] loading experiments/attack/hfl-mechanism/configs/G1P__coll-on__s42.yaml",
        "[配置校验] 通过 | method=hier_fedrep | defense=none | attack=badpfl | 0 个警告",
        "[设定] client_fraction=0.1 | poison_ratio=0.2 | n_clients=100 | n_edges=4 | "
        "edge_rounds=10 | n_malicious=10 | forced_participation=False | arch=resnet10_torch"]
S9 = ("[设定9] update_geometry=true | update_sketch_dim=4096 | post_agg_eval=true | "
      "frozen_trigger=true")


def _geo_lines():
    obs = UG.UpdateGeometry(malicious_ids={1}, sketch_dim=8)
    ew = [np.zeros(4, np.float32), np.zeros(1, np.float32)]

    def upd(c, w):
        return ClientUpdate([np.asarray(w, np.float32), np.zeros(1, np.float32)], 5, 0.1, 0.0,
                            client_id=c)
    obs.observe(0, 3, [upd(0, [1, 0, 0, 0]), upd(1, [0, 1, 0, 0])], ew, [0])
    obs.observe(1, 3, [upd(7, [1, 1, 0, 0])], ew, [0])
    return obs.flush(4, 3)


def _log():
    L = list(HEAD) + [S9]
    L += _geo_lines()
    L.append(format_kv("[PostAgg]", {
        "edge_round": 0, "effective_round": 10, "pm_acc": 0.31, "em_acc": 0.3, "edge_asr": 0.2,
        "local_benign_asr": 0.1, "same_edge_asr": 0.1, "diff_edge_asr": None,
        "local_malicious_asr": 0.9, "eval_s": 2.5, "margin_p10": -3.0, "margin_p50": -1.0,
        "margin_p90": 0.5, "benign_asr_p90": 0.4, "benign_asr_gt50": 0.05, "flip_other": 0.2},
        round_idx=2))
    for e in (0, 1):
        L.append(format_kv("[PostAggEdge]", {
            "edge_round": 0, "effective_round": 10, "edge_asr": 0.2, "client_benign": 0.1,
            "client_malicious": 0.9, "pm_acc": 0.3, "em_acc": 0.3, "margin_p50": -1.0,
            "benign_asr_p90": 0.4}, round_idx=2, edge_id=e))
    L.append(format_kv("[Light]", {
        "edge_round": 5, "effective_round": 15, "pm_acc": 0.5, "em_acc": 0.4, "edge_asr": 0.3,
        "local_benign_asr": 0.2, "same_edge_asr": 0.2, "diff_edge_asr": None,
        "local_malicious_asr": 0.9, "eval_s": 1.5, "margin_p10": -2.0, "margin_p50": -0.5,
        "margin_p90": 1.0, "benign_asr_p90": 0.6, "benign_asr_gt50": 0.1, "flip_other": 0.1},
        round_idx=2))
    for ph, er, eff in (("post", 0, 10), ("light", 5, 15), ("full", 10, 20)):
        L.append(format_kv("[FrozenASR]", {"phase": ph, "edge_round": er, "effective_round": eff,
                                           "local_benign": 0.05 * er, "eval_s": 1.0}, round_idx=2))
        for e in (0, 1):
            L.append(format_kv("[FrozenASREdge]", {"phase": ph, "edge_round": er,
                                                   "effective_round": eff, "client_benign": 0.1 * e},
                               round_idx=2, edge_id=e))
    L.append(format_kv("[Dump]", {"kind": "sketch", "path": "run.123/sketch_r002.npz",
                                  "bytes": 1000, "sha": "ab12"}, round_idx=2))
    return "\n".join(L)


def test_schema_is_10():
    assert CM.SCHEMA_VERSION == 10


def test_settings9_becomes_four_scalars_in_the_run_block():
    run = CM.collect(_log())["run"]
    assert (run["update_geometry"], run["update_sketch_dim"],
            run["post_agg_eval"], run["frozen_trigger"]) == (True, 4096, True, True)
    run = CM.collect("\n".join(HEAD))["run"]                         # 老日志：不知道
    assert (run["update_geometry"], run["update_sketch_dim"],
            run["post_agg_eval"], run["frozen_trigger"]) == (None, None, None, None)
    assert run["edge_rounds"] == 10 and run["n_edges"] == 4          # [设定] 照常


def test_update_geometry_table_is_compact_and_parsed():
    t = CM.collect(_log())["update_geometry"]
    assert t["columns"] == list(CM.UPDATE_GEO_COLUMNS)
    assert len(t["rows"]) == 2
    r0 = dict(zip(t["columns"], t["rows"][0]))
    assert (r0["round"], r0["edge_id"], r0["edge_round"]) == (4, 0, 3)
    assert r0["cid"] == [0, 1] and r0["mal"] == [0, 1] and r0["norm"] == [1.0, 1.0]
    r1 = dict(zip(t["columns"], t["rows"][1]))
    assert r1["cos_edge"] == [None]                                  # 只有一个更新：无定义，不是 0
    assert r1["cos_global"] == pytest.approx([1.0])                  # [1,1] vs 其余之和 [1,1]
    assert r0["cos_global"] == pytest.approx([1 / np.sqrt(5)] * 2, abs=1e-4)      # 打印 4 位小数


def test_post_agg_is_its_own_table_not_light_rounds_or_rounds():
    m = CM.collect(_log())
    assert [(r["round"], r["edge_round"], r["effective_round"]) for r in m["post_agg_rounds"]] == [(2, 0, 10)]
    assert m["post_agg_rounds"][0]["margin_p50"] == -1.0
    assert [(r["round"], r["effective_round"]) for r in m["light_rounds"]] == [(2, 15)]
    assert m["light_rounds"][0]["benign_asr_gt50"] == 0.1 and m["light_rounds"][0]["flip_other"] == 0.1
    assert m["rounds"] == []                                         # 日志里没有 [Backdoor] 行：post-agg 不会冒充全量点
    pe = m["per_edge_post_agg_rounds"]
    assert sorted(pe) == [2] and [row[0] for row in pe[2]] == [0, 1]
    assert pe[2][0][CM.LIGHT_EDGE_COLUMNS.index("margin_p50")] == -1.0


def test_frozen_columns_parse_with_phase():
    m = CM.collect(_log())
    assert [(r["phase"], r["effective_round"]) for r in m["frozen_rounds"]] == [
        ("post", 10), ("light", 15), ("full", 20)]
    t = m["frozen_edge_rounds"]
    assert t["columns"] == list(CM.FROZEN_EDGE_COLUMNS) and len(t["rows"]) == 6
    assert dict(zip(t["columns"], t["rows"][1]))["client_benign"] == pytest.approx(0.1)


def test_sketch_dumps_are_counted_and_timing_totals_exist():
    m = CM.collect(_log())
    assert m["dumps"]["sketch"] == {"count": 1, "bytes": 1000, "dir": "run.123"}
    assert m["timing_summary"]["post_agg_eval_total_s"] == 2.5
    assert m["timing_summary"]["frozen_eval_total_s"] == 3.0
    old = CM.collect("\n".join(HEAD))
    assert old["update_geometry"] == {"columns": list(CM.UPDATE_GEO_COLUMNS), "rows": []}
    assert old["post_agg_rounds"] == [] and old["frozen_rounds"] == []
    assert old["per_edge_post_agg_rounds"] == {} and old["frozen_edge_rounds"]["rows"] == []
    assert old["timing_summary"]["post_agg_eval_total_s"] is None


# ── 上游：代码打印的键 = 解析器认的列 ─────────────────────────────────────────
def _dict_keys_near(path, func, tag):
    tree = ast.parse((ROOT / "fedavg" / path).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    for call in ast.walk(fn):
        if (isinstance(call, ast.Call) and getattr(call.func, "id", "") == "format_kv"
                and isinstance(call.args[0], ast.Constant) and call.args[0].value == tag):
            return [k.value for k in call.args[1].keys]
    raise AssertionError(f"{func} 没有打印 {tag}")


def test_frozen_lines_print_the_columns_the_parser_reads():
    got = _dict_keys_near("server/backdoor_server.py", "_frozen_eval", "[FrozenASR]")
    assert ["round", *got] == list(CM.FROZEN_COLUMNS)
    e = _dict_keys_near("server/backdoor_server.py", "_frozen_eval", "[FrozenASREdge]")
    assert set(e) >= set(CM.FROZEN_EDGE_COLUMNS) - {"round", "edge_id"}


def test_update_geo_line_prints_the_columns_the_parser_reads():
    rows = CM.collect("\n".join(_geo_lines()))["update_geometry"]["rows"]
    assert len(rows) == 2 and all(len(r) == len(CM.UPDATE_GEO_COLUMNS) for r in rows)
    keys = [ln.split(" | ")[0:3] for ln in _geo_lines()][0]
    assert keys[1].startswith("edge") and "edge_round=" in keys[2]


# ── instrumentation_check：G1P 的开 / 关比较（S6a）──────────────────────────────
def test_instrumentation_check_compares_light_points_on_reference_columns():
    import instrumentation_check as IC
    base = {"checksums": [{"round": 1, "global": "aa"}, {"round": 2, "global": "bb"}],
            "rounds": [], "acc_rounds": [], "per_edge_rounds": {}, "per_edge_acc_rounds": {},
            "light_rounds": [{"round": 1, "effective_round": 5, "edge_asr": 0.25, "pm_acc": 0.5,
                              "eval_s": 1.0}]}
    new = {**base, "light_rounds": [{"round": 1, "effective_round": 5, "edge_asr": 0.25, "pm_acc": 0.5,
                                     "eval_s": 9.0, "margin_p50": -1.0}]}      # 新列 + 计时不同：不算分歧
    res = IC.compare(base, new)
    assert not res["diffs"] and not res["missing"] and res["n_fields"] >= 2
    bad = {**base, "light_rounds": [{"round": 1, "effective_round": 5, "edge_asr": 0.26, "pm_acc": 0.5}]}
    assert [d[0] for d in IC.compare(base, bad)["diffs"]] == ["light_rounds[effective_round=5].edge_asr"]
    gone = {**base, "light_rounds": []}
    assert IC.compare(base, gone)["missing"] == ["light_rounds[effective_round=5]"]
