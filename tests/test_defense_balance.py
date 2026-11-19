"""
tests/test_defense_balance.py  —  harness/defense_balance.py（论文整合：阶段三各防御臂的统一对比）

  · 灌入 / 洗掉 / 净增量是手算值（前半程 g = 2 … ⌊n/2⌋、后半程其余；受害边缘三台均值；E0 单独）；
  · 净增量 = 灌入 − 洗掉（逐 g 的恒等式）；margin 版按列名读（列序打乱结果不变）；
  · 云聚合后点的列序不符 → 该 run 无效、不算平衡；分散布点不算平衡；无效 run 不进汇总；
  · 臂按 run 块归类（edge_harden 的 "none" 字符串 = 不设防；observe = 不设防）；
  · 去重按 GROUPS 顺序；不投毒的开发实验不收；Δ = 对照 − 本臂（逐种子）；参照过滤的冷启动死锁标记；
  · 真实数据锚点：SNAP 集中 s42 前半程灌入 / 洗掉 = F-124 的 0.220 / 0.194，且与 g1_verdict.sawtooth 逐位一致；
    三层个性化 s42 前半程灌入 0.035；三层个性化在现主线集中布点的受害边缘下降 0.715（current-focus〇·一：0.996 → 0.281）。
"""

import copy
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import defense_balance as B                                  # noqa: E402
from collect_metrics import LIGHT_EDGE_COLUMNS               # noqa: E402

FULL = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.6, 5: 0.7, 6: 0.8}      # 受害边缘三台均值（云轮末）
POST = {2: 0.4, 3: 0.6, 4: 0.7, 5: 0.75, 6: 0.85}           # 同上（云聚合后）
OFFS = {1: 0.1, 2: 0.0, 3: -0.1}                             # 三台边缘各自偏移，均值不变
E0_FULL, E0_POST = 0.9, 0.6
DETAIL_COLS = ["edge_id", "margin_p50", "benign_asr_p90", "yt_clean_benign"]


def _m(*, seed=42, mpe=(10, 0, 0, 0), harden=None, mode=None, tier=0, profile=None, P=0.9, MTA=0.86,
       detail_cols=DETAIL_COLS, light_cols=tuple(LIGHT_EDGE_COLUMNS), admitted=None, poison=0.2):
    run = {"seed": seed, "malicious_per_edge": list(mpe), "edge_harden": harden, "edge_ref_mode": mode,
           "edge_shared_blocks": tier, "partition_profile": profile, "n_rounds": 6, "edge_rounds": 5,
           "poison_ratio": poison, "malicious_ids": [7]}
    per_edge_rounds, detail, post = {}, {}, {}
    for g, v in FULL.items():
        rows = [{"edge_id": 0, "client_benign": E0_FULL}] + [{"edge_id": e, "client_benign": v + OFFS[e]} for e in (1, 2, 3)]
        per_edge_rounds[str(g)] = rows
        drows = []
        for e in (0, 1, 2, 3):
            mv = 9.0 if e == 0 else 10 * (v + OFFS[e])
            d = {"edge_id": e, "margin_p50": mv, "benign_asr_p90": None, "yt_clean_benign": None}
            drows.append([d[c] for c in detail_cols])
        detail[str(g)] = drows
    for g, v in POST.items():
        rows = []
        for e in (0, 1, 2, 3):
            cb = E0_POST if e == 0 else v + OFFS[e]
            d = {"edge_id": e, "edge_asr": None, "client_benign": cb, "client_malicious": None, "pm_acc": None,
                 "em_acc": None, "margin_p50": 6.0 if e == 0 else 10 * cb, "benign_asr_p90": None}
            rows.append([d[c] for c in LIGHT_EDGE_COLUMNS])
        post[str(g)] = rows
    m = {"exit_code": 0, "client_failures": [], "n_malicious_participations": 5, "run": run,
         "rounds": [{"round": r, "local_benign_asr": P} for r in range(1, 7)],
         "acc_rounds": [{"round": r, "pm_acc": MTA} for r in range(1, 7)],
         "per_edge_rounds": per_edge_rounds,
         "per_edge_detail_columns": list(detail_cols), "per_edge_detail_rounds": detail,
         "per_edge_light_columns": list(light_cols), "per_edge_post_agg_rounds": post}
    if admitted is not None:
        cols = ["round", "edge_id", "edge_round", "cid", "admitted"]
        m["edge_ref_scores"] = {"columns": cols, "rows": [[1, 0, i, c, a] for i, (c, a) in enumerate(admitted)]}
    return m


# ── 手算值 ────────────────────────────────────────────────────────────────

def test_balance_hand_computed():
    row = B.run_row(_m())
    assert row["invalid"] == []
    v1, v2 = row["balance"]["victim_asr"]["first"], row["balance"]["victim_asr"]["second"]
    assert v1["g_range"] == [2, 3] and v1["n_used"] == v1["n_expected"] == 2
    assert v1["jump"] == pytest.approx(0.3) and v1["wash"] == pytest.approx(0.1) and v1["net"] == pytest.approx(0.2)
    assert v2["g_range"] == [4, 6] and v2["n_used"] == 3
    assert v2["jump"] == pytest.approx(0.5 / 3) and v2["wash"] == pytest.approx(0.2 / 3) and v2["net"] == pytest.approx(0.1)
    e = row["balance"]["e0_asr"]["first"]
    assert e["jump"] == pytest.approx(E0_POST - E0_FULL) and e["wash"] == pytest.approx(E0_POST - E0_FULL)
    assert e["net"] == pytest.approx(0.0)
    vm = row["balance"]["victim_margin"]["first"]
    assert vm["jump"] == pytest.approx(3.0) and vm["wash"] == pytest.approx(1.0) and vm["net"] == pytest.approx(2.0)
    q = row["quantities"]
    assert q["V"] == pytest.approx(0.5) and q["B0"] == pytest.approx(0.9)
    assert q["P"] == pytest.approx(0.9) and q["MTA"] == pytest.approx(0.86)


def test_net_is_jump_minus_wash_per_g():
    full, post = B.full_series(_m(), B.VICTIMS), B.post_series(_m(), B.VICTIMS)
    for lo, hi in B.halves(6):
        b = B.balance(full, post, lo, hi)
        assert b["net"] == pytest.approx(b["jump"] - b["wash"], abs=1e-12)


def test_missing_points_are_skipped_and_counted():
    m = _m()
    del m["per_edge_post_agg_rounds"]["3"]
    b = B.run_row(m)["balance"]["victim_asr"]["first"]
    assert b["n_used"] == 1 and b["n_expected"] == 2 and b["jump"] == pytest.approx(0.3)


def test_margin_is_read_by_column_name():
    shuffled = ["margin_p50", "yt_clean_benign", "edge_id", "benign_asr_p90"]
    a = B.run_row(_m())["balance"]["victim_margin"]
    b = B.run_row(_m(detail_cols=shuffled))["balance"]["victim_margin"]
    assert a == b


def test_light_column_mismatch_invalidates_run():
    row = B.run_row(_m(light_cols=tuple(reversed(LIGHT_EDGE_COLUMNS))))
    assert row["balance"] is None and any("per_edge_light_columns" in r for r in row["invalid"])


def test_distributed_has_no_balance():
    row = B.run_row(_m(mpe=(3, 3, 2, 2)))
    assert row["placement"] == "distributed" and row["balance"] is None and row["invalid"] == []


# ── 归类 ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw, expect", [
    ({}, ("main", "none", 0, "collocated", "不设防")),
    ({"harden": "sau-e8-train-high"}, ("main", "harden", 0, "collocated", "上传前加固")),
    ({"harden": "none", "mode": "observe", "tier": 1}, ("main", "none", 1, "collocated", "三层个性化")),
    ({"harden": "none", "mode": "filter"}, ("main", "filter", 0, "collocated", "参照过滤")),
    ({"mode": "filter", "tier": 1, "profile": "fixed"}, ("homo", "filter", 1, "collocated", "参照过滤 + 三层个性化")),
    ({"mpe": (3, 3, 2, 2)}, ("main", "none", 0, "distributed", "不设防")),
    ({"mpe": (0, 10, 0, 0)}, ("main", "none", 0, "distributed", "不设防")),
])
def test_arm_of(kw, expect):
    a = B.arm_of(_m(**kw)["run"])
    assert (a["partition"], a["defense"], a["tier"], a["placement"], a["arm"]) == expect


# ── 汇总、去重、Δ、死锁 ───────────────────────────────────────────────────

def _row(m, source):
    r = B.run_row(m)
    r["source"] = source
    return r


def test_delta_is_control_minus_arm_per_seed_and_invalid_runs_are_excluded():
    rows = [_row(_m(seed=42, P=0.9, MTA=0.86), "c42"), _row(_m(seed=43, P=0.8, MTA=0.85), "c43"),
            _row(_m(seed=42, tier=1, P=0.5, MTA=0.86), "t42"), _row(_m(seed=43, tier=1, P=0.6, MTA=0.84), "t43")]
    bad = _m(seed=44, tier=1, P=0.0)
    bad["client_failures"] = [{"client_id": 3}]
    rows.append(_row(bad, "t44"))
    res = B.assemble(rows)
    cell = next(c for c in res["cells"] if c["arm"] == "三层个性化")
    assert cell["seeds"] == [42, 43]
    d = cell["delta_vs_control"]["P"]
    assert d["per_seed"] == {42: pytest.approx(0.4), 43: pytest.approx(0.2)} and d["mean"] == pytest.approx(0.3)
    assert cell["delta_vs_control"]["MTA"]["mean"] == pytest.approx(0.005)
    assert [x["source"] for x in res["invalid"]] == ["t44"]
    ctrl = next(c for c in res["cells"] if c["arm"] == "不设防")
    assert ctrl["delta_vs_control"] == {}
    assert cell["mean"]["victim_asr.first.jump"] == pytest.approx(0.3)


def test_deadlock_flag():
    dead = B.run_row(_m(mode="filter", admitted=[(7, 0), (1, 0), (2, 0)]))["online"]
    assert dead["deadlock"] and dead["all_rejected_frac"] == 1.0 and dead["benign_reject"] == 1.0
    live = B.run_row(_m(mode="filter", admitted=[(7, 0), (1, 1), (2, 1)]))["online"]
    assert not live["deadlock"] and live["attacker_pass"] == 0.0 and live["benign_reject"] == 0.0
    assert live["all_rejected_frac"] == round(1 / 3, 4)        # 每个边缘轮一个上传：只有攻击者那一轮全被剔；输出保留 4 位


def test_load_dedupes_by_group_order_and_skips_dev_runs(tmp_path):
    for g, name, m in (("SNAP", "SNAP__collocated__s42", _m()),
                       ("ERP", "ERP__main-t0-collocated__s42", _m(harden="none", mode="observe")),
                       ("ERC", "ERC__dev__s41", _m(seed=41, poison=0.0))):
        (tmp_path / g).mkdir(exist_ok=True)
        (tmp_path / g / f"{name}.metrics.json").write_text(json.dumps(m), encoding="utf-8")
    rows, dups = B.load(tmp_path, groups=("SNAP", "ERP", "ERC"))
    assert [r["source"] for r in rows] == ["SNAP/SNAP__collocated__s42"]
    assert dups == [{"key": ["main", "collocated", "不设防", 42], "kept": "SNAP/SNAP__collocated__s42",
                     "dropped": "ERP/ERP__main-t0-collocated__s42"}]


def test_markdown_renders_every_cell():
    rows = [_row(_m(seed=42), "c"), _row(_m(seed=42, tier=1, P=0.5), "t"), _row(_m(seed=42, mpe=(3, 3, 2, 2)), "d")]
    res = B.assemble(rows)
    res["duplicates"], res["note"] = [], "n"
    md = B.markdown(res)
    assert "三层个性化" in md and "分散布点" in md and "集中布点" in md


# ── 真实数据锚点 ──────────────────────────────────────────────────────────

SNAP42 = B.RESULTS / "SNAP/SNAP__collocated__s42.metrics.json"
TIER42 = B.RESULTS / "ERP/ERP__main-t1-collocated__s42.metrics.json"


@pytest.mark.skipif(not SNAP42.exists(), reason="没有 SNAP 回传数据")
def test_real_snap_matches_f124_and_g1_sawtooth():
    import g1_verdict
    m = json.loads(SNAP42.read_text(encoding="utf-8"))
    b = B.run_row(m)["balance"]["victim_asr"]["first"]
    assert round(b["jump"], 3) == 0.220 and round(b["wash"], 3) == 0.194          # F-124 的数
    s = g1_verdict.sawtooth(m)
    assert b["n_used"] == s["n_g_used"]
    assert math.isclose(b["jump"], s["d_jump"], abs_tol=1e-12)
    assert math.isclose(b["wash"], s["wash_per_cycle"], abs_tol=1e-12)


@pytest.mark.skipif(not TIER42.exists(), reason="没有 ERP 回传数据")
def test_real_three_tier_cuts_injection():
    m = json.loads(TIER42.read_text(encoding="utf-8"))
    b = B.run_row(m)["balance"]["victim_asr"]["first"]
    assert round(b["jump"], 3) == 0.035


@pytest.mark.skipif(not (SNAP42.exists() and TIER42.exists()), reason="没有阶段三回传数据")
def test_real_full_table():
    res = B.analyse()
    assert res["invalid"] == []
    assert {tuple(d["key"][:3]) for d in res["duplicates"]} == {("main", "collocated", "不设防"),
                                                                ("main", "distributed", "不设防")}
    cell = next(c for c in res["cells"] if (c["partition"], c["placement"], c["arm"]) == ("main", "collocated", "三层个性化"))
    assert cell["seeds"] == [42, 43, 44]
    assert round(cell["mean"]["V"], 3) == 0.281 and round(cell["delta_vs_control"]["V"]["mean"], 3) == 0.715
    homo_filter = [c for c in res["cells"] if c["partition"] == "homo" and "参照过滤" in c["arm"]]
    assert homo_filter and all(c["deadlock"] for c in homo_filter)
    main_filter = [c for c in res["cells"] if c["partition"] == "main" and "参照过滤" in c["arm"]]
    assert main_filter and not any(c["deadlock"] for c in main_filter)
