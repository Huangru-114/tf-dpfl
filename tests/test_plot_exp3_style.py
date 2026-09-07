"""`plot_exp3.py` 的出版级约束（回应导师意见 #3）。

**验证范围如实**：本机没有 matplotlib/numpy，这里是**源码级**约束检查，
不是渲染检查。它挡的是三类会静默复现的回归：

  1. 无定义的指标被 `np.nan_to_num` 画成 0（报告 Figure 11 里
     `10edge_collocated` 的 E0 就是这么画成 ASR=0 的）。陷阱 #13 在数据侧
     修好了（metrics.json 里是 null），但**修不到画图这一步** —— 绘图代码
     自己又把 nan 变回了 0。
  2. 共享 x 轴的图里，上 panel 没有横轴（报告 Figure 12 的上 panel）。
     这两个 panel 常被分别截进正文，各自都得读得懂。
  3. 曲线只靠颜色区分 → 数值贴近时后画的把先画的完全盖住
     （Figure 12 里蓝色 GM ASR 被橙红 EM ASR 盖没）。

真正「有没有重叠」要在集群上跑完 plot_exp3.py 之后目检 PNG。

纯 stdlib（ast + 正则），本地秒级。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLOT = ROOT / "experiments" / "attack" / "hfl-propagation" / "plot_exp3.py"
SRC = PLOT.read_text(encoding="utf-8")


def _func_src(name: str) -> str:
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(SRC, node)
            assert seg, name
            return seg
    raise AssertionError(f"plot_exp3.py 里找不到 {name}")


def _code_lines(src: str) -> str:
    """去掉注释行 —— 注释里提到 nan_to_num 不算调用它。"""
    return "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))


# ══════════════════════════════════════════════════════════════════════════
# 1. 无定义 != 0
# ══════════════════════════════════════════════════════════════════════════
def test_bar_heights_are_never_nan_to_num():
    """柱高绝不能过 nan_to_num：那把「无定义」和「后门完全没传过去」
    画成同一根高度为 0 的柱。"""
    code = _code_lines(_func_src("fig_topology_summary"))
    assert "nan_to_num(means)" not in code, \
        "柱高又被 nan_to_num 了 —— 无定义会被画成 0"
    assert re.search(r"np\.isfinite\(means\)", code), \
        "没有筛掉无定义的格子"


def test_yerr_may_still_use_nan_to_num():
    """反向：误差须长度用 nan_to_num 是**对的** —— 单 seed 时 lo==hi==mean，
    差值是 0 不是「无定义」。这条防止上一条被过度修正成一刀切。"""
    code = _code_lines(_func_src("fig_topology_summary"))
    assert "nan_to_num(yerr)" in code


def test_undefined_cells_are_labelled_not_silently_dropped():
    """跳过不画还不够：空着看起来像「忘了画」。必须标出来。"""
    for fn in ("fig_topology_summary", "fig_per_edge"):
        code = _code_lines(_func_src(fn))
        assert '"n/a"' in code, f"{fn} 没有标注无定义的点"


# ══════════════════════════════════════════════════════════════════════════
# 2. 每个 panel 自带 x 轴
# ══════════════════════════════════════════════════════════════════════════
def test_shared_x_figures_label_every_panel():
    """`sharex=True` 只给最下面那个 panel 刻度标签。上 panel 被单独截进
    报告时就完全没有横轴 —— Figure 12 正是这样。"""
    for fn in ("fig_3c", "_fig_timeseries"):
        code = _code_lines(_func_src(fn))
        assert "labelbottom=True" in code, f"{fn} 的上 panel 没有 x 刻度标签"
        assert code.count("set_xlabel") >= 2, f"{fn} 只有一个 panel 有 x 轴标签"


# ══════════════════════════════════════════════════════════════════════════
# 3. 不能只靠颜色
# ══════════════════════════════════════════════════════════════════════════
def test_series_are_distinguished_by_more_than_colour():
    """GM 与 EM 的 ASR 数值本来就贴得极近（这本身是个发现，不该被画没）。
    linestyle + marker 都不同，重合时两条线都还看得见。"""
    code = _code_lines(_func_src("fig_3c"))
    assert "linestyle" in code and "marker" in code, "系列只靠颜色区分"
    # 只看**上 panel 那一组**。下 panel 是另一个 axes，跨 axes 复用样式没问题；
    # 第一版把整个函数一起扫，于是把合法的复用判成了重复。
    block = re.search(r'\("global_asr".*?\]:', code, re.S)
    assert block, "找不到上 panel 的系列列表"
    # marker 字符类不能只写 [a-zA-Z]：^ v < > * + x . , 都是合法 matplotlib
    # marker，第一版就因为漏了 "^" 把 5 条线数成了 4 条。
    styles = re.findall(r'\("(-{1,2}|-\.|:)",\s*"([a-zA-Z^v<>*+x.,])"',
                        block.group(0))
    assert len(styles) == 5, f"上 panel 有 5 条线，却只找到 {len(styles)} 组样式"
    assert len(set(styles)) == 5, f"同一个 panel 里有重复样式：{styles}"


def test_overlapping_series_have_distinct_zorder():
    """GM 画在最上层且线更粗，EM 的虚线从它下面透出来 ——
    zorder 一样的话谁盖谁取决于绘制顺序，等于随机。"""
    code = _code_lines(_func_src("fig_3c"))
    zs = [int(m) for m in re.findall(r'"[a-zA-Z^v<>*+x.,]",\s*[\d.]+,\s*(\d+)\)',
                                     code)]
    assert zs, "没解析到 zorder"
    assert max(zs) > min(zs), f"所有系列 zorder 相同：{zs}"


def test_colours_stay_colourblind_safe():
    """Okabe–Ito 色板：色盲安全。改样式时别顺手把色板换掉。"""
    okabe_ito = {"#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
                 "#56B4E9", "#F0E442", "#000000"}
    palette = re.search(r"^C = \{(.*?)^\}", SRC, re.S | re.M)
    assert palette, "找不到色板 C"
    used = set(re.findall(r'"(#[0-9A-Fa-f]{6})"', palette.group(1)))
    assert used and used <= okabe_ito, f"非 Okabe–Ito 色：{used - okabe_ito}"


def test_no_cjk_in_figure_text():
    """铁律 #4 的 tf-dpfl 版：图上文本一律英文。

    只扫**会被画上去**的字符串（set_xlabel / set_ylabel / set_title /
    suptitle / label= / text 的第一个字面量），注释与 docstring 不算。
    """
    drawn = re.findall(
        r'(?:set_xlabel|set_ylabel|set_title|suptitle|label)\s*=?\s*\(?\s*'
        r'(["\'])(.*?)\1', SRC)
    assert drawn, "一条都没扫到，正则失效了"
    for _, text in drawn:
        for ch in text:
            assert not ("一" <= ch <= "鿿"), f"图上出现 CJK：{text}"
