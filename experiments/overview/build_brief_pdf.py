"""把导师简报（BRIEF-advisor-*.md）排成带图的 PDF。

用法（仓库根目录）：python3 experiments/overview/build_brief_pdf.py
依赖：pandoc、Node.js 版 playwright（预装的 Chromium）、中文字体（文泉驿正黑）。
图取自 experiments/thesis/figures/（由 harness/thesis_figures.py 从判定输出同源生成）。
"""
import pathlib
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = ROOT / "experiments" / "thesis" / "figures"
SRC = HERE / "BRIEF-advisor-2026-11-23.md"
OUT = HERE / "BRIEF-advisor-2026-11-23.pdf"

# (插在哪一行之后：该行的开头文字, 图文件, 图注)
FIGURES = [
    ("2. **灌入 vs 洗掉**", "fig5_3_sawtooth.png",
     "图 1　云周期内的锯齿。(a) 每 10 个边缘轮一次云聚合（集中布点，3 个种子，带子 = 最小–最大）："
     "橙 = 攻击者所在边缘的良性端，蓝 = 受害边缘；每次云聚合受害边缘向上跳（灌入），周期内往下降（洗掉），"
     "洗掉追不上灌入 → 走向饱和。(b)(c) 一个云周期内的相位剖面（每 5 / 10 / 20 个边缘轮一次云聚合），"
     "分别用攻击成功率与 margin（目标类对数概率 − 其余最大）量。"),
    ("4. **边缘视角没有检测优势**", "fig5_5_observability.png",
     "图 2　边缘视角的可观测性（分散布点）。(a) 各分数区分攻击者与良性端的 AUROC：实心 = 边缘视角，"
     "空心 = 全局视角（同样大小的参照池）；两者几乎重合，只有更新范数越过 0.7；批归一化统计量部分约 0.90，但可伪造。"
     "(b) 更新范数的 ROC 曲线：边缘视角与全局视角几乎重合。"),
    ("不分层设置（1 台服务器", None, None),  # 占位：图 3 放在防御对比表之后（见下）
    ("- **威胁边界**（布点 [9,1,0,0]）", "fig6_2_sharing_depth.png",
     "图 4　三层个性化的代价对照。横轴 = 跨边缘聚合的参数比例（对数轴），数字 = 只在边缘内平均的残差块数。"
     "(a) 受害边缘良性端攻击成功率；(b) 相对完全共享的精度代价（按种子配对），虚线 = 门槛 0.02。"
     "绿 = 机构式划分（现主线），灰 = 早期的 Dirichlet 划分；小点 = 单个种子。"),
    ("- **威胁边界**（布点 [9,1,0,0]）", "fig6_3_threat_boundary.png",
     "图 5　威胁边界：布点 [9,1,0,0]（边缘依次有 9 / 1 / 0 / 0 个攻击者）。灰 = 不设防，绿 = 三层个性化（末 1 块）；"
     "竖线 = 3 个种子均值，符号 = 单个种子。两台干净边缘受保护，只有 1 个攻击者的边缘照样失守 → 隔离是按边缘的。"),
]
TABLE_FIG = ("fig6_9_comparison.png",
             "图 3　防御的综合比较（现主线划分，3 个种子；分散布点的上传前加固只有 1 个种子）。"
             "(a) 受害边缘、(b) 攻击者所在边缘的良性端、(c) 分散布点全部良性端的攻击成功率；"
             "(d) 精度下降（虚线 = 门槛 0.02）；(e) 每个 run 的额外 GPU 小时。竖线 = 均值，符号 = 单个种子。")

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: 'WenQuanYi Zen Hei', sans-serif; font-size: 10pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 16pt; margin: 0 0 6pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
blockquote { margin: 4pt 0; padding: 2pt 8pt; border-left: 3px solid #ccc; color: #555; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; margin: 6pt 0; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 2pt 4pt; vertical-align: top; }
th { background: #f0f0f0; }
code { font-size: 8.5pt; }
figure { margin: 8pt 0 10pt; page-break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figcaption { font-size: 8.5pt; color: #444; text-align: left; margin-top: 3pt; }
li { margin: 2pt 0; }
"""


def fig_html(name, caption):
    return (f'\n<figure><img src="{(FIG / name).as_uri()}">'
            f"<figcaption>{caption}</figcaption></figure>\n")


def main():
    raw = SRC.read_text(encoding="utf-8").split("\n")
    # pandoc 要求列表前有空行；md 原文里「**小标题**：」后直接跟列表 → 补空行
    lines = []
    for ln in raw:
        is_item = ln.startswith("- ") or (ln[:2].rstrip(".").isdigit() and ln[1:3] == ". ")
        prev = lines[-1] if lines else ""
        prev_item = prev.startswith(("- ", "  ")) or (prev[:2].rstrip(".").isdigit() and prev[1:3] == ". ")
        if is_item and prev.strip() and not prev_item:
            lines.append("")
        lines.append(ln)
    out = []
    i = 0
    in_defense_table = False
    while i < len(lines):
        line = lines[i]
        out.append(line)
        # 图 3：防御对比表（以「| 做法 |」开头的表）结束之后
        if line.startswith("| 做法 |"):
            in_defense_table = True
        elif in_defense_table and not line.startswith("|"):
            out.insert(len(out) - 1, fig_html(*TABLE_FIG))
            in_defense_table = False
        for anchor, name, cap in FIGURES:
            if name and line.startswith(anchor):
                out.append(fig_html(name, cap))
        i += 1
    for _, name, _ in FIGURES:
        if name and not (FIG / name).exists():
            sys.exit(f"缺图：{name}")
    md = "\n".join(out)
    html = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "html5", "-s",
         "--metadata", "title=导师讨论简报", "--metadata", "pagetitle=导师讨论简报"],
        input=md, capture_output=True, text=True, check=True).stdout
    # pandoc 的 title 块会重复 h1，去掉；换成我们的样式
    html = html.replace('<header id="title-block-header">\n<h1 class="title">导师讨论简报</h1>\n</header>', "")
    html = html.replace("</head>", f"<style>{CSS}</style></head>")
    tmp = HERE / ".brief_tmp.html"
    tmp.write_text(html, encoding="utf-8")
    js = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const b = await chromium.launch();
  const pg = await b.newPage();
  await pg.goto({tmp.as_uri()!r}, {{ waitUntil: 'networkidle' }});
  await pg.pdf({{ path: {str(OUT)!r}, format: 'A4', printBackground: true,
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#888;">'
      + '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: {{ top: '16mm', bottom: '16mm', left: '15mm', right: '15mm' }} }});
  await b.close();
}})();
"""
    subprocess.run(["node", "-e", js], check=True)
    tmp.unlink()
    print(OUT)


if __name__ == "__main__":
    main()
