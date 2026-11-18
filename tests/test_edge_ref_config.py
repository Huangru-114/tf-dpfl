"""
方向 2（D-129 / D-131）：参照打分的配置侧 + 登记 + 回程解析。不需要 TF。

  · ERP / ERPB 12 格（登记表展开结果）都通过 config_validate、打印 [设定14]；
  · config_validate 拒绝「静默没打分 / 打错」的写法（方法 / 调度 / 层 / 划分 / 干净集 / 未知键）；
  · 打印 ↔ 解析同源：[设定14] → run.edge_ref_*；[EdgeRefScore] / [EdgeRefNull] → 两张表；schema 16；
    回程解析器里抄的字段名与 defense/edge_ref_spec 的一致；老日志 → None / 空表；
  · metrics.json 体积：300 有效轮 × 4 edge × 每轮约 2.5 个上传的规模。
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "fedavg"))

import collect_metrics as CM                                    # noqa: E402
import registry as R                                            # noqa: E402
from config_validate import ConfigError, validate_config        # noqa: E402
from defense import edge_ref_spec as S                          # noqa: E402
from utils.kvline import fmt_list, format_kv                    # noqa: E402

REG = ROOT / "experiments" / "defense" / "edge-native" / "registry.yaml"
GROUPS = ("ERP", "ERPB")


@pytest.fixture(scope="module")
def erp():
    reg = R.Registry(REG)
    return {r["run_id"]: reg.intended_config(r) for r in reg.runs() if r["group"] in GROUPS}


def test_twelve_runs(erp):
    assert len(erp) == 12
    assert sum(k.startswith("ERP__") for k in erp) == 8 and sum(k.startswith("ERPB__") for k in erp) == 4


def test_every_cell_validates_and_describes_itself(erp, capsys):
    for rid, cfg in erp.items():
        validate_config(copy.deepcopy(cfg))
        out = capsys.readouterr().out
        line = [ln for ln in out.splitlines() if ln.startswith("[设定14]")]
        assert line and "edge_ref=true" in line[0] and "mode=observe" in line[0], rid


def _base(erp):
    return copy.deepcopy(erp["ERP__main-t0-collocated__s42"])


@pytest.mark.parametrize("mutate,needle", [
    (lambda c: c["training"].update(drift_correction="hier_fedper"), "hier_fedrep"),
    (lambda c: c["federation"].update(edge_schedule="sequential"), "interleaved"),
    (lambda c: c["defense"].update(layers=["edge", "cloud"]), "layers"),
    (lambda c: c["defense"].update(layers=["client", "edge"]), "layers"),
    (lambda c: c["federation"].update(partition="noniid"), "S3 划分"),
    (lambda c: c["federation"]["design"].update(clean_per_edge=0), "clean_per_edge"),
    (lambda c: c["defense"]["edge_ref"].update(cutoff=0.5), "不认识"),
    (lambda c: c["defense"]["edge_ref"].update(mode="filter"), "score"),
    (lambda c: c["defense"]["edge_ref"].update(mode="filter", score="kl_all"), "threshold"),
    (lambda c: c["defense"]["edge_ref"].update(threshold=0.1), "只在 mode = filter"),
    (lambda c: c["defense"]["edge_ref"].update(mode="quarantine"), "mode"),
])
def test_config_validate_rejects(erp, mutate, needle):
    c = _base(erp)
    mutate(c)
    with pytest.raises(ConfigError, match=needle):
        validate_config(c)


def test_stray_edge_ref_block_warns(erp):
    c = _base(erp)
    c["defense"]["name"] = "none"
    warns = validate_config(c)
    assert any("defense.edge_ref" in w for w in warns)


# ── 回程解析 ───────────────────────────────────────────────────────────────
def _score_line(g, e, er, cid, rng):
    f = {"effective_round": (g - 1) * 5 + er, "edge_round": er, "cid": cid, "n": 375}
    for k in S.SCORE_FIELDS[4:]:
        f[k] = int(rng.integers(0, 2)) if k in ("ccs_rej", "mix_argmax") else float(rng.random())
    return "    " + format_kv("[EdgeRefScore]", f, round_idx=g, edge_id=e, digits=6)


def _null_line(g, e, er, rng, n_fake=8):
    f = {"effective_round": (g - 1) * 5 + er, "edge_round": er, "n_clean": 500, "ccs_pool": 12,
         "ccs_status": "ok"}
    for k in S.NULL_FIELDS[5:]:
        f[k] = fmt_list(rng.random(n_fake).tolist(), digits=6)
    return "    " + format_kv("[EdgeRefNull]", f, round_idx=g, edge_id=e)


def test_settings14_and_tables_roundtrip():
    rng = np.random.default_rng(0)
    cfg = {"defense": {"name": "edge_ref", "edge_ref": {"mode": "observe"}}}
    lines = [format_kv("[设定14]", S.settings_row(cfg)), _score_line(3, 1, 2, 47, rng), _null_line(3, 1, 2, rng),
             "    " + format_kv("[TimingEdgeRef]", {"effective_round": 12, "edge_round": 2, "n_updates": 1,
                                                    "edge_ref_s": 1.25}, round_idx=3, edge_id=1, digits=2)]
    m = CM.collect("\n".join(lines))
    assert m["schema_version"] == 16
    assert m["run"]["edge_ref"] is True and m["run"]["edge_ref_mode"] == "observe"
    assert m["run"]["edge_ref_n_fake"] == 8 and m["run"]["edge_ref_fake_n"] == 375
    sc = m["edge_ref_scores"]
    assert sc["columns"][:2] == ["round", "edge_id"] and len(sc["rows"]) == 1
    row = dict(zip(sc["columns"], sc["rows"][0]))
    assert row["round"] == 3 and row["edge_id"] == 1 and row["effective_round"] == 12 and row["cid"] == 47
    nu = dict(zip(m["edge_ref_null"]["columns"], m["edge_ref_null"]["rows"][0]))
    assert nu["ccs_status"] == "ok" and len(nu["kl_all"]) == 8 and len(nu["mix_res_base"]) == 8
    assert m["timing_summary"]["edge_ref_total_s"] == pytest.approx(1.25, abs=0.051)


def test_copied_field_names_match_the_spec():
    assert CM.EDGE_REF_SETTINGS_KEYS == S.SETTINGS_KEYS
    assert CM.EDGE_REF_SCORE_COLUMNS == ("round", "edge_id") + S.SCORE_FIELDS
    assert CM.EDGE_REF_NULL_COLUMNS == ("round", "edge_id") + S.NULL_FIELDS
    assert set(CM.EDGE_REF_NULL_LISTS) == set(S.NULL_FIELDS[5:])


def test_old_logs_have_no_edge_ref():
    m = CM.collect("[Round 1]\n")
    assert m["run"]["edge_ref"] is None and m["run"]["edge_ref_mode"] is None
    assert m["edge_ref_scores"]["rows"] == [] and m["edge_ref_null"]["rows"] == []
    assert m["timing_summary"]["edge_ref_total_s"] is None
    m = CM.collect(format_kv("[设定14]", S.settings_row({"defense": {"name": "none"}})))
    assert m["run"]["edge_ref"] is False and m["run"]["edge_ref_mode"] is None


def test_registry_checks_the_edge_ref_mode(erp):
    assert ("edge_ref_mode", "defense.edge_ref.mode") in R.EXPECT_KEYS


def test_metrics_growth_for_a_full_run_stays_bounded():
    """300 有效轮 × 4 edge = 1200 个边缘轮，每轮 3 个上传（偏多：实际平均 2.5 → 约 1.9 MB）；
    indent=2 的 JSON 增量 < 2.5 MB / run（实测 2.29 MB；D-131 记了这个体积，12 个 run 约 23 MB）。"""
    rng = np.random.default_rng(1)
    lines = []
    for g in range(1, 61):
        for er in range(1, 6):
            for e in range(4):
                for j in range(3):
                    lines.append(_score_line(g, e, er, 25 * e + j, rng))
                lines.append(_null_line(g, e, er, rng))
    grow = len(json.dumps(CM.collect("\n".join(lines)), indent=2, ensure_ascii=False)) - \
        len(json.dumps(CM.collect("[Round 1]\n"), indent=2, ensure_ascii=False))
    assert grow < 2_500_000, grow



def test_second_stage_configs_validate_and_print_the_exact_threshold(capsys):
    """ERF / ERC 36 个配置都过校验；[设定14] 的 threshold 回读与配置逐位相等（缺省 4 位小数会打成 0.0041）。"""
    reg = R.Registry(REG)
    runs = [r for r in reg.runs() if r["group"] in ("ERF", "ERC")]
    assert len(runs) == 36
    for r in runs:
        cfg = reg.intended_config(r)
        validate_config(copy.deepcopy(cfg))
        line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[设定14]")][0]
        run = CM.collect(line)["run"]
        er = cfg["defense"]["edge_ref"]
        assert run["edge_ref_mode"] == er["mode"]
        assert run["edge_ref_threshold"] == er.get("threshold") and run["edge_ref_score"] == er.get("score")
