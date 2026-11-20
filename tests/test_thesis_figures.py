"""
tests/test_thesis_figures.py  —  论文图（harness/thesis_figures.py + harness/thesis_style.py）

  · 图上的数与已有判定 / 分析输出是同一份（划分度量 = F-125 的表；锯齿的洗掉速率 = 3-C 判定；分解示意的灌入 / 洗掉 =
    defense_balance 的定义；可观测性的判定标签 = g1_verdict.json）；
  · 渲染出的全部文字（含刻度、图例）不许出现内部名词：组名、F- / N- / D- 编号、口径版本、内部变量名、判定标签、中文
    （反向锚点：故意写进一个组名，扫描必须抓到）；
  · 画图代码里不直接写颜色值（颜色只经 thesis_style 的实体表）；
  · 每张图宽 = 正文宽度、没有总标题；保存出的 PDF 嵌入了 TrueType 字体；
  · 有序色阶 ramp() 的取值。
结果不在盘上 → 相应的测试 skip；渲染需要 matplotlib（没有就 skip）。
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import thesis_style as TS                  # noqa: E402
import thesis_figures as T                 # noqa: E402

SRC = (ROOT / "harness" / "thesis_figures.py").read_text(encoding="utf-8")

BANNED = [r"\bG\d", r"G1R5", r"\bSNAP\b", r"\bER[CFP]B?\b", r"\bCCS", r"\b[FND]-\d{3}\b", r"\bP[012]\b",
          r"fresh-?PM", r"r_down", r"d_jump", r"\bself_cleaning\b", r"\bno_edge_gain\b", r"\bundetectable\b",
          r"\bnot_gated\b", r"\buser_decides\b", r"\bcollocated\b", r"\bE[0-3]\b", r"norm_w", r"s_ck", r"neg_cos",
          r"\brho\b", r"[一-鿿]"]


def banned_hits(texts) -> list:
    return [(t, p) for t in texts for p in BANNED if re.search(p, t)]


def figure_texts(fig) -> list:
    import matplotlib.text
    fig.canvas.draw()
    return [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]


@pytest.fixture(scope="module")
def rendered():
    pytest.importorskip("matplotlib")
    out = {}
    for key in T.FIGURES:
        name, fig = T.render(key)
        if fig is not None:
            out[key] = fig
    if not out:
        pytest.skip("结果不在盘上")
    yield out
    import matplotlib.pyplot as plt
    plt.close("all")


# ── 数据与判定同源 ──────────────────────────────────────────────────────────

def _need(d):
    if d is None:
        pytest.skip("结果不在盘上")
    return d


def test_partition_numbers_match_the_f125_table():
    cases = {c["case"]: c for c in T.data_partition()["cases"]}
    assert round(cases["C1"]["h_pair_same"], 3) == 0.549 and round(cases["C1"]["h_pair_cross"], 3) == 0.696
    assert round(cases["C1-fixed0.4"]["h_pair_same"], 3) == 0.296 and round(cases["C1-fixed0.4"]["h_pair_cross"], 3) == 0.547
    assert round(cases["random"]["h_pair_same"] / cases["random"]["h_pair_cross"], 2) == 1.00
    assert [c["used"] for c in T.data_partition()["cases"]].count(True) == 2


def test_decomposition_matches_defense_balance_definitions():
    import defense_balance as DB
    d = _need(T.data_decomposition())
    m = T._load(T.DECOMP_RUN)
    g = T.DECOMP_CYCLE
    full, post = DB.full_series(m, T.VICTIMS), DB.post_series(m, T.VICTIMS)
    c = d["asr_cycle"]
    assert c["jump"] == pytest.approx(post[g] - full[g - 1]) and c["wash"] == pytest.approx(post[g] - full[g])
    assert c["net"] == pytest.approx(c["jump"] - c["wash"])
    assert (round(c["jump"], 3), round(c["wash"], 3), round(c["net"], 3)) == (0.274, 0.165, 0.109)
    assert c["net"] > 0                                   # 标注的是净增量为正的典型周期
    assert max(x for x, _ in d["asr"]) <= T.DECOMP_MAX_CLOUD * d["edge_rounds"]


def test_floor_reuses_report_data_and_attacked_runs_exist():
    import report_figures as R
    d = _need(T.data_floor_thesis())
    assert d["floor_main"] == R.data_floor(T.P2_MECH)["curves"]["C1"]
    assert len(d["attacked_main"]) == 3 and max(p["floor"] for p in d["scatter"]) < d["criterion"]


def test_gating_verdict_and_split():
    d = _need(T.data_gating())
    assert d["verdict"] == "not_gated"
    assert all(min(d["split"]["attacker_edge"][t]) > 0.6 for t in d["t0s"])
    assert all(max(d["split"]["victim_edges"][t]) < 0.25 for t in d["t0s"])


def test_sawtooth_washing_rates_are_the_verdict_numbers():
    d = _need(T.data_sawtooth())
    assert {k: round(v, 4) for k, v in d["r_down"].items()} == {5: 0.0333, 10: 0.0223, 20: 0.0069}
    assert sorted(d["phase"]) == [5, 10, 20]


def test_decay_and_views_labels():
    d = _need(T.data_decay_thesis())
    assert d["hfl_verdict"] == "user_decides" and d["flat_verdict"] == "user_decides"
    v = _need(T.data_views())
    assert {r["score"]: r["verdict"] for r in v["rows"]} == {
        "norm_w": "no_edge_gain", "neg_cos_global_w": "undetectable", "neg_cos_edge_w": "undetectable",
        "s_ck": "undetectable"}
    assert round(v["bn_stats_global_mean"], 3) == 0.905 and v["detect_min"] == 0.7


# ── 图上的文字、尺寸、颜色 ─────────────────────────────────────────────────

def test_no_internal_names_in_any_rendered_text(rendered):
    for key, fig in rendered.items():
        hits = banned_hits(figure_texts(fig))
        assert hits == [], f"{key}: {hits}"


def test_banned_scan_catches_a_planted_internal_name():
    pytest.importorskip("matplotlib")
    fig, axes = TS.figure(1, 1)
    axes[0][0].set_title("G1R5 victims, F-124")
    assert {p for _, p in banned_hits(figure_texts(fig))} >= {r"G1R5", r"\b[FND]-\d{3}\b"}
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_every_figure_is_text_width_without_suptitle(rendered):
    for key, fig in rendered.items():
        assert fig.get_size_inches()[0] == pytest.approx(TS.TEXTWIDTH), key
        assert fig._suptitle is None, key


def test_plot_code_has_no_hard_coded_colors():
    assert not re.findall(r"#[0-9a-fA-F]{6}\b", SRC)
    assert re.findall(r"#[0-9a-fA-F]{6}\b", (ROOT / "harness" / "thesis_style.py").read_text(encoding="utf-8"))


def test_save_writes_pdf_with_embedded_truetype(tmp_path):
    pytest.importorskip("matplotlib")
    fig, axes = TS.figure(1, 1)
    axes[0][0].plot([0, 1], [0, 1], label="victim edges")
    pdf, png = TS.save(fig, "probe", tmp_path)
    assert pdf.stat().st_size > 0 and png.stat().st_size > 0
    assert b"/FontFile2" in pdf.read_bytes()


def test_ramp():
    assert TS.ramp("blue", 3) == [TS.BLUE_RAMP[0], TS.BLUE_RAMP[2], TS.BLUE_RAMP[4]]
    assert TS.ramp("gray", 5) == list(TS.GRAY_RAMP)
    with pytest.raises(ValueError):
        TS.ramp("gray", 6)
