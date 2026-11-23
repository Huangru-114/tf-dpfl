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


# ── 第 6 章：数据与判定 / 分析同源 ─────────────────────────────────────────

def _need_ch6():
    if not (T.P2_EDGE / "SNAP/SNAP__collocated__s42.metrics.json").exists():
        pytest.skip("阶段三结果不在盘上")


def test_digests_cover_every_arm_with_three_seeds():
    _need_ch6()
    D = T.digests()
    for arm in T.ARM_ORDER:
        assert sorted(k[3] for k in D if k[:3] == ("main", "collocated", arm)) == [42, 43, 44], arm
    assert sorted(k[3] for k in D if k[:3] == ("main", "distributed", "harden")) == [42]   # 分散布点的加固只有单种子
    assert D[("main", "collocated", "none", 42)]["source"].startswith("SNAP/")             # 去重：对照取 SNAP


def test_tier_and_comparison_match_defense_balance():
    _need_ch6()
    c = T.data_comparison()
    cells = {(x["partition"], x["placement"], x["arm"]): x for x in T._analyse()["cells"]}
    tier = cells[("main", "collocated", "三层个性化")]
    assert c["cols"]["victim"]["tier"] == {s: q["V"] for s, q in tier["per_seed"].items()}
    assert round(sum(c["cols"]["victim"]["tier"].values()) / 3, 3) == 0.281             # current-focus 〇·一
    assert c["cols"]["compute"]["tier"] == {42: 0.0, 43: 0.0, 44: 0.0}                  # 共享块不加计算
    assert 3.5 < sum(c["cols"]["compute"]["harden"].values()) / 3 < 4.0                 # 每次加固约 56 s × 240 次


def test_hardening_screen_matches_the_stage3_report():
    _need_ch6()
    d = _need(T.data_hardening())
    ch = sorted((p["snapshot"], round(p["r_h"], 2), round(p["d_pool"], 3)) for p in d["points"] if p["cfg"] == T.HARDEN_CHOSEN)
    assert ch == [(6, 0.58, 0.032), (15, 0.44, 0.023)]                                  # REPORT §4.2 的表


def test_flat_clustering_matches_the_verdicts():
    d = _need(T.data_flat_clustering())
    a, b = T._load(T.CCSD5), T._load(T.CCSDP)
    cl = {p["seed"]: round(p["dP"], 4) for p in d["points"] if p["variant"] == "cluster"}
    assert cl == {int(s): v for s, v in a["headline"]["delta_P"].items()}
    ca = {p["seed"]: p["dP"] for p in d["points"] if p["variant"] == "cache"}
    assert ca == {int(s): q["dP"] for s, q in b["delta"]["cache"].items()}


def test_scoring_and_filtering_anchors():
    _need_ch6()
    s = _need(T.data_scoring())
    assert s["main"]["selected"] == "mix_res" and s["homo"]["selected"] == "kl_all"
    assert round(min(s["main"]["auroc"]["mix_res"].values()), 3) == 0.847                # F-126
    f = _need(T.data_filtering())
    assert {k: round(v["r"], 2) for k, v in f["scatter"].items()} == {"observe": 0.07, "filter": 0.57}   # F-129
    assert f["threshold"] == pytest.approx(0.119039)


def test_misjudgement_uses_the_same_records_as_the_filter_verdict():
    _need_ch6()
    import edge_ref_filter_verdict as V
    rows = T.digests()[("main", "collocated", "filter", 42)]["benign_reject"]
    m = T._load(T.P2_EDGE / "ERF/ERF__main-t0-collocated__s42.metrics.json")
    pooled = sum(r["rate"] * r["n"] for r in rows) / sum(r["n"] for r in rows)
    assert pooled == pytest.approx(V.online_rates(m)["benign_reject"], abs=1e-12)
    d = _need(T.data_misjudgement())
    assert set(d["rates"]) == {("collocated", a, r) for a in ("filter", "filter+tier") for r in ("victim", "compromised")} \
        | {("distributed", a, "all") for a in ("filter", "filter+tier")}


def test_balance_values_are_defense_balance_per_seed():
    _need_ch6()
    d = _need(T.data_balance())
    assert round(d["vals"][("victim_asr", "jump")]["tier"][42], 3) == 0.035
    assert round(d["vals"][("victim_asr", "jump")]["none"][42], 3) == 0.220


def test_pdf_output_is_byte_reproducible(tmp_path):
    """重画同一张图，PDF 必须逐字节相同（否则每次重画都在 git 里冒出假改动）。"""
    pytest.importorskip("matplotlib")
    import time
    out = []
    for i in range(2):
        if i:
            time.sleep(1.1)                     # 时间戳精度是秒：隔开一秒，否则不修也会碰巧相同（反向锚点）
        fig, axes = TS.figure(1, 1)
        axes[0][0].plot([0, 1], [0, 1])
        pdf, _ = TS.save(fig, f"probe{i}", tmp_path)
        out.append(pdf.read_bytes())
    assert out[0] == out[1]


# ── 补充实验（N-015，F-130 … F-132）：6.2 / 6.3 与判定同源，6.9 加中位数基线 ─────────────────

def _need_supp():
    if not (T.P2_EDGE / "MED/MED__cloud-collocated__s44.metrics.json").exists():
        pytest.skip("补充实验结果不在盘上")


def test_sharing_depth_is_the_d1_report_and_reproduces_g6():
    _need_supp()
    import supp_verdict as SV
    d = _need(T.data_sharing_depth())
    rep = SV.run_all(only="d1")["d1"]["report"]
    for b in T.DEPTH_LEVELS:
        k = int(b[2:])
        assert {s: (q["V"], q["MTA"], q["frac"]) for s, q in d["main"][k].items()} == \
            {int(s[1:]): (q["V"], q["MTA"], q["cloud_fraction"]) for s, q in rep[b].items()}, b
    mv = {k: sum(q["V"] for q in per.values()) / 3 for k, per in d["old"].items()}
    assert mv[0] - mv[1] == pytest.approx(0.524, abs=1e-3) and mv[0] - mv[2] == pytest.approx(0.730, abs=1e-3)   # G6（F-077）
    assert all(q["acc_cost"] == 0 for q in d["main"][0].values())
    assert round(sum(q["acc_cost"] for q in d["main"][4].values()) / 3, 3) == 0.022                          # F-130


def test_threat_boundary_is_the_d2_report():
    _need_supp()
    import supp_verdict as SV
    d = _need(T.data_threat_boundary())
    rep = SV.run_all(only="d2")["d2"]["report"]
    for key, lab in T.BOUNDARY_ROWS:
        assert d["rows"][lab]["none"] == {int(s[1:]): q[key] for s, q in rep["B=0"].items()}
        assert d["rows"][lab]["tier"] == {int(s[1:]): q[key] for s, q in rep["B=1"].items()}
    assert min(d["rows"]["edge with 1 attacker"]["tier"].values()) > 0.9                                      # F-131：一个攻击者就失守


def test_comparison_adds_the_median_baselines_from_defense_balance():
    _need_supp()
    c = T.data_comparison()
    assert c["arms"] == list(T.COMPARE_ARMS) and {"median_edge", "median_cloud"} <= set(c["arms"])
    cells = {(x["partition"], x["placement"], x["arm"]): x for x in T._analyse()["cells"]}
    for key, lab in (("median_edge", "边缘层中位数"), ("median_cloud", "云层中位数")):
        assert c["cols"]["victim"][key] == {s: q["V"] for s, q in cells[("main", "collocated", lab)]["per_seed"].items()}
        assert c["cols"]["dispersed"][key] == {s: q["P"] for s, q in cells[("main", "distributed", lab)]["per_seed"].items()}
        assert c["cols"]["compute"][key] == {42: 0.0, 43: 0.0, 44: 0.0}                                       # 中位数不加计算
    assert T.ARM_ORDER == ("none", "tier", "harden", "filter", "filter+tier")                                # 别的图的行序不变


# ── 逐图说明（experiments/thesis/FIGURE-NOTES.md）与登记表同步 ───────────────

NOTES = ROOT / "experiments" / "thesis" / "FIGURE-NOTES.md"


def notes_problems(figures: dict, text: str, fig_dir: Path) -> list:
    """登记表里每张图都要在说明里出现（`文件名` 形式）；说明里提到的每个图文件都要在登记表里、且 PDF / PNG 都存在。"""
    names = {v[0] for v in figures.values()}
    problems = [f"缺说明：{n}" for n in sorted(names) if f"`{n}`" not in text]
    for ref in sorted(set(re.findall(r"`(fig\d+_\d+_[a-z0-9_]+)`", text))):
        if ref not in names:
            problems.append(f"说明里有、登记表里没有：{ref}")
        elif not ((fig_dir / f"{ref}.pdf").exists() and (fig_dir / f"{ref}.png").exists()):
            problems.append(f"图文件不存在：{ref}")
    return problems


def test_every_figure_has_a_notes_section():
    text = NOTES.read_text(encoding="utf-8")
    assert notes_problems(T.FIGURES, text, TS.OUT) == []
    assert text.count("\n## 图 ") == len(T.FIGURES)


def test_notes_check_catches_a_missing_section():
    figs = dict(T.FIGURES)
    figs["fig9_9"] = ("fig9_9_planted", None, None)                 # 反向锚点：登记了却没写说明
    assert notes_problems(figs, NOTES.read_text(encoding="utf-8"), TS.OUT) == ["缺说明：fig9_9_planted"]
