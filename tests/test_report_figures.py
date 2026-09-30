"""
tests/test_report_figures.py  —  REPORT.md 的结果图（harness/report_figures.py）

图上的数必须与各组判定脚本的输出是**同一份**：这里在真实结果上逐项核对（结果不在盘上 → skip）。
数据整理部分纯标准库；真正出图需要 matplotlib（本地没有就 skip）。
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import decay_verdict as DV              # noqa: E402
import g3_did as G3                     # noqa: E402
import g5_verdict as G5                 # noqa: E402
import g6_verdict as G6V                # noqa: E402
import report_figures as R              # noqa: E402
import status as ST                     # noqa: E402
from registry import Registry           # noqa: E402

SRC = (ROOT / "harness" / "report_figures.py").read_text(encoding="utf-8")


def _real(prep):
    d = prep()
    if d is None:
        pytest.skip("结果不在盘上")
    return d


# ── 纯函数 ─────────────────────────────────────────────────────────────────

def _m(per_edge_rounds, edge_rounds=5):
    return {"run": {"edge_rounds": edge_rounds}, "per_edge_rounds": per_edge_rounds}


def test_edge_group_curve_averages_edges_and_skips_incomplete_rounds():
    m = _m({"1": [{"edge_id": 1, "client_benign": 0.2}, {"edge_id": 2, "client_benign": 0.4}],
            "2": [{"edge_id": 1, "client_benign": 0.6}, {"edge_id": 2, "client_benign": None}],
            "3": [{"edge_id": 1, "client_benign": 0.8}, {"edge_id": 2, "client_benign": 1.0}]})
    assert R.edge_group_curve(m, (1, 2)) == [(5, pytest.approx(0.3)), (15, pytest.approx(0.9))]
    assert R.edge_group_curve(m, (1,), shift=5) == [(0, 0.2), (5, 0.6), (10, 0.8)]


def test_benign_weights_split_by_attacker_presence():
    m = _m({"1": [{"edge_id": 0, "n_benign": 15, "has_malicious": True},
                  {"edge_id": 1, "n_benign": 25, "has_malicious": False},
                  {"edge_id": 2, "n_benign": 25, "has_malicious": False}]})
    assert R.benign_weights(m) == (15, 50)


def test_pooled_curve_is_on_effective_rounds():
    m = {"run": {"edge_rounds": 5}, "rounds": [{"round": 1, "x": 0.1}, {"round": 2, "x": None},
                                               {"round": 3, "x": 0.3}]}
    assert R.pooled_curve(m, "x", shift=5) == [(0, 0.1), (10, 0.3)]


def test_no_cjk_in_drawn_text():
    """集群容器里的 matplotlib 没有中文字体（同 test_figures.py）。只扫会被画上去的字符串。"""
    drawn_fns = {"set_xlabel", "set_ylabel", "set_title", "suptitle", "supxlabel", "supylabel",
                 "annotate", "text", "_style", "_panel_title", "_save", "set_xticklabels",
                 "set_yticklabels", "_band_line", "_floor_line"}
    texts = []
    for node in ast.walk(ast.parse(SRC)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        cands = list(node.args) if name in drawn_fns else []
        cands += [k.value for k in node.keywords if k.arg in ("label", "title")]
        for c in cands:
            for sub in ast.walk(c):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    texts.append(sub.value)
    assert len(texts) > 30, "扫到的太少，扫描器坏了"
    for t in texts:
        assert not any("一" <= ch <= "鿿" for ch in t), f"图上出现 CJK：{t}"


# ── 真实数据：图上的数 = 判定脚本的数 ──────────────────────────────────────

def test_g5_panel_b_is_the_verdict_itself():
    d = _real(R.data_g5)
    assert d["verdict"] == G5.judge(*G5.load())


def test_g5_pooled_peak_in_panel_c_equals_the_verdict_peak_excess():
    d = _real(R.data_g5)
    for p in d["verdict"]["per_run"]:
        assert d["split"][(p["t0"], p["seed"])]["pooled"] == pytest.approx(p["peak_excess"], abs=6e-5)


def test_g5_pooled_column_is_the_benign_weighted_mix_of_attacker_edge_and_victim_edges():
    """panel (c) 的读法依据：池化 = (n_E0良性 × 同 edge + n_受害 × 异 edge) / n_良性，逐轮成立（只差三位小数的舍入）。
    反向锚点：权重错一个客户端（16 / 74）就对不上 —— 证明这条恒等式不是空检验。"""
    g5, _, _ = G5.load()
    if not g5:
        pytest.skip("G5 不在盘上")
    worst, worst_wrong = 0.0, 0.0
    for m in g5.values():
        ws, wd = R.benign_weights(m)
        assert (ws, wd) == (15, 75)
        for r in m["rounds"]:
            if None in (r.get("local_benign_asr"), r.get("same_edge_asr"), r.get("diff_edge_asr")):
                continue
            mix = lambda a, b: (a * r["same_edge_asr"] + b * r["diff_edge_asr"]) / (a + b)
            worst = max(worst, abs(r["local_benign_asr"] - mix(ws, wd)))
            worst_wrong = max(worst_wrong, abs(r["local_benign_asr"] - mix(16, 74)))
    assert worst <= 0.001
    assert worst_wrong > 0.005


def test_g5_attacker_edge_is_near_ceiling_at_every_t0():
    """panel (c) 标题的依据：E0 良性端的峰值 excess 在每个 (t0, seed) 都 ≥ 0.6，受害 edge 都 ≤ 0.25。"""
    d = _real(R.data_g5)
    assert min(q["attacker_edge"] for q in d["split"].values()) >= 0.6
    assert max(q["victim_edges"] for q in d["split"].values()) <= 0.25


def test_g5_panel_d_reads_g0_pm_acc_at_effective_round_t0():
    d = _real(R.data_g5)
    g0 = G5.load()[1][42]
    acc = {r["round"]: r["pm_acc"] for r in g0["acc_rounds"]}
    assert d["pm_start"][(20, 42)] == acc[4] and d["pm_start"][(180, 42)] == acc[36]


def test_g3_bars_are_g3_did():
    d = _real(R.data_g3)
    g3, g0, _ = G3.load()
    assert d["verdict"] == G3.judge(g3, g0)


def test_decay_panels_use_decay_verdict():
    d = _real(R.data_decay)
    assert d["hfl_verdict"] == DV.judge(DV.load())
    assert d["flat_verdict"] == DV.judge_flat(DV.load_flat())
    assert d["stop_eff"] == 150


def test_g6_endpoints_are_the_report_table():
    """REPORT §5.4 / §5.5 的表（末 10 个评估点）：受害 E1–E3 与 fresh pm_acc；G6D 是池化良性端。"""
    d = _real(R.data_g6)
    got = {(r["group"], r["arm"], r["seed"]): (round(r["asr"], 3), round(r["pm_acc"], 3)) for r in d["end"]}
    want = {("G6", "a", 42): (0.794, 0.868), ("G6", "a", 43): (0.657, 0.873), ("G6", "a", 44): (0.995, 0.875),
            ("G6", "b", 42): (0.267, 0.861), ("G6", "b", 43): (0.169, 0.866), ("G6", "b", 44): (0.438, 0.870),
            ("G6", "c", 42): (0.061, 0.849), ("G6", "c", 43): (0.117, 0.853), ("G6", "c", 44): (0.078, 0.863),
            ("G6D", "a", 42): (0.999, 0.873), ("G6D", "b", 42): (0.966, 0.863), ("G6D", "c", 42): (0.975, 0.856)}
    assert got == want


def test_margin_zero_crossing_takes_the_first_drop_below_zero_after_the_stop():
    assert R.margin_zero_crossing([(-5, 2.0), (0, 1.0), (5, 0.5), (10, -0.1), (15, 0.2), (20, -1)]) == 10
    assert R.margin_zero_crossing([(0, -1.0), (5, 0.3), (10, -0.2)]) == 10      # 停手前已为负、停手后短暂回正
    assert R.margin_zero_crossing([(0, 1.0), (5, 0.5)]) is None


# ── 第二批（2026-09-30）：进度 / G3 全划分 / 3-E 判定 / G8 margin 与长尾 / pilot ──

def test_status_figure_counts_are_status_py():
    d = _real(R.data_status)
    rep = ST.classify(Registry(R.STUDY / "registry.yaml"))
    assert d["counts"] == rep["counts"]
    assert sum(c["done"] + c["stale"] + c["blocked"] + c["todo"] for c in d["groups"].values()) == len(rep["runs"])


def test_status_figure_gpu_hours_match_the_report_budget_table():
    """REPORT §8 的实测机时（包墙钟 × 1 卡；单跑取训练 + 评估墙钟）。"""
    d = _real(R.data_status)
    got = {g: round(c["gpu_h_used"], 1) for g, c in d["groups"].items()}
    want = {"G0": 12.2, "G3": 18.5, "G5": 6.4, "G6": 15.6, "FLR": 3.0, "G8": 3.1, "G6D": 3.2,
            "G5AB": 4.8, "G8F": 4.6, "G7": 11.7}
    if any(d["groups"][g]["done"] + d["groups"][g]["stale"] == 0 for g in want):
        pytest.skip("结果不全")
    assert {g: got[g] for g in want} == want
    assert round(d["pilot_gpu_h"], 1) == 16.0


def test_g3_all_partitions_panel_is_g3_did():
    d = _real(R.data_g3_all)
    g3, g0, hd = G3.load()
    v = G3.judge(g3, g0)
    by = {(r["cell"], r["seed"]): r for r in d["rows"]}
    assert len(d["rows"]) == 24 and {c for c, _ in by} == set(R.G3_ORDER)
    for r in v["per_run"]:
        assert by[(r["cell"], r["seed"])]["raw"] == {int(e): x for e, x in r["raw"].items()}
    assert d["explore_yt"] == G3.explore_yt(v["per_run"], G3.explore_hdir(hd))


def test_3e_verdict_figure_is_g6_verdict():
    d = _real(R.data_g6_verdict)
    assert d["verdict"] == G6V.judge(G6V.load())


def test_g8_margin_and_tail_are_the_numbers_in_f068_and_f071():
    d = _real(R.data_g8_margin)
    hfl, flat = d["hfl"], d["flat"]
    assert hfl["seeds"] == [42, 43, 44] and flat["seeds"] == [42, 43, 44]
    # F-068：第 70 云轮（停手后 200 有效轮）margin 中位数 −8.8 / −6.9 / −6.3；s42 第 33 轮、s44 第 39 轮过零
    assert [round(c[-1][1], 1) for c in hfl["margin_p50"]] == [-8.8, -6.9, -6.3]
    assert [c[-1][0] for c in hfl["margin_p50"]] == [200, 200, 200]
    cross = [R.margin_zero_crossing(c) for c in hfl["margin_p50"]]
    assert cross[0] // 5 + 30 == 33 and cross[2] // 5 + 30 == 39
    # F-071：flat 到第 350 有效轮 −5.3 / −4.4 / −4.0
    assert [round(c[-1][1], 1) for c in flat["margin_p50"]] == [-5.3, -4.4, -4.0]

    def win(c):                              # 停手后 105–150 有效轮 = G8 第 51–60 云轮 = G8F 第 255–300 有效轮
        v = [y for x, y in c if 105 <= x <= 150]
        return round(sum(v) / len(v), 3)
    assert [win(c) for c in hfl["benign_asr_p90"]] == [0.413, 0.324, 0.428]
    assert [win(c) for c in hfl["benign_asr_gt50"]] == [0.063, 0.017, 0.058]
    assert [round(win(c), 2) for c in flat["benign_asr_gt50"]] == [0.08, 0.16, 0.03]


def test_pilot_panel_is_pilot_a4_and_pack_test():
    d = _real(R.data_pilot)
    assert d["d029"]["overall"] == "pass" and d["d031"]["overall"] == "different"
    assert d["det"]["verdict"] == "pass" and d["pack"]["adopt_k"] == 3
    assert round(d["pack"]["per_k"][3]["speedup"], 2) == 2.86


# ── 出图 ───────────────────────────────────────────────────────────────────

def test_every_figure_renders(tmp_path):
    pytest.importorskip("matplotlib")
    assert R.main(["--out", str(tmp_path)]) == 0
    made = {p.name for p in tmp_path.glob("*.png")}
    expected = {name for name, prep, _ in R.FIGURES.values() if prep() is not None}
    assert expected and made == expected
    assert all((tmp_path / n).stat().st_size > 20_000 for n in made)
