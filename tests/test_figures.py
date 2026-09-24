"""
tests/test_figures.py  —  由因素驱动的画图基元

数据整理部分纯标准库，本地秒级；真正出图需要 matplotlib（容器里有，本地没有就 skip）。
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import figures as F                      # noqa: E402
import runs_table as T                   # noqa: E402

SRC = (ROOT / "harness" / "figures.py").read_text(encoding="utf-8")


def test_band_keeps_only_x_every_curve_has():
    a = [(5, 0.1), (10, 0.3), (15, 0.5)]
    b = [(5, 0.3), (10, 0.5)]
    assert F.band([a, b]) == [(5, pytest.approx(0.2), 0.1, 0.3),
                              (10, pytest.approx(0.4), 0.3, 0.5)]
    assert F.band([]) == []


def test_spread_labels_keeps_order_and_min_gap():
    assert F.spread_labels([0.96, 0.97, 0.5], gap=0.05) == [0.96, pytest.approx(1.01), 0.5]
    assert F.spread_labels([0.2, 0.8], gap=0.05) == [0.2, 0.8]


def test_groups_sort_numerically():
    runs = [{"n_edges": v} for v in ("10", "2", "4")]
    assert [k for k in F.group_runs(runs, ["n_edges"])] == [("2",), ("4",), ("10",)]


def test_heterogeneous_group_is_detected():
    """按 n_edges 分组却没固定布点 → 同一条带子里混了不同的格子（F-010 那类错）。"""
    runs = [{"n_edges": "2", "malicious_per_edge": "[5, 5]", "seed": "42"},
            {"n_edges": "2", "malicious_per_edge": "[10, 0]", "seed": "42"},
            {"n_edges": "4", "malicious_per_edge": "[3, 3, 2, 2]", "seed": "43"},
            {"n_edges": "4", "malicious_per_edge": "[3, 3, 2, 2]", "seed": "42"}]
    got = F.heterogeneous_factors(F.group_runs(runs, ["n_edges"]))
    assert got == {("2",): ["malicious_per_edge"]}


def test_palette_is_the_validated_fixed_order():
    """dataviz 参考调色板（light），validate_palette.js 全部 PASS；顺序固定、不循环。"""
    assert F.SERIES == ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                        "#008300", "#4a3aa7")


def test_no_cjk_in_drawn_text():
    """集群容器里的 matplotlib 没有中文字体。只扫会被画上去的字符串。"""
    drawn_fns = {"set_xlabel", "set_ylabel", "set_title", "suptitle", "supxlabel",
                 "supylabel", "annotate", "_style"}
    texts = []
    for node in ast.walk(ast.parse(SRC)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        cands = list(node.args[:3]) if name in drawn_fns else []
        cands += [k.value for k in node.keywords if k.arg in ("label", "title")]
        for c in cands:
            for sub in ast.walk(c):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    texts.append(sub.value)
    assert texts, "一条都没扫到"
    for t in texts:
        assert not any("一" <= ch <= "鿿" for ch in t), f"图上出现 CJK：{t}"


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    d = tmp_path_factory.mktemp("tables")
    runs, series = T.build([ROOT / "experiments/attack/hfl-propagation/results"],
                           legacy_protocol="P1")
    T.write_tables(runs, series, d)
    return d


def test_trajectory_refuses_mixed_groups(tables):
    pytest.importorskip("matplotlib")
    runs, series = F.load_tables(tables)
    with pytest.raises(SystemExit, match="malicious_per_edge"):
        F.trajectory(runs, series, metric="local_benign_asr", group_by=["n_edges"],
                     filters=[("edge_rounds", "5")], out=str(tables / "x.png"))


def test_trajectory_and_per_edge_render(tables):
    pytest.importorskip("matplotlib")
    runs, series = F.load_tables(tables)
    out = tables / "t.png"
    F.trajectory(runs, series, metric="local_benign_asr", group_by=["malicious_per_edge"],
                 filters=[("n_edges", "4")], out=str(out))
    assert out.stat().st_size > 10_000
    out2 = tables / "e.png"
    F.per_edge(runs, series, run="10edge_mixed_seed42", metric="edge.client_benign",
               out=str(out2))
    assert out2.stat().st_size > 10_000
