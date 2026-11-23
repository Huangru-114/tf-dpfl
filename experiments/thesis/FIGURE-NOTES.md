# 论文图说明（2026-11-23）

> 每张图一节：文件、论文位置、一句话结论、各面板含义、关键数字（带出处）、数据来源、证据强度与注意事项、英文图注草稿、改图入口。
> 图文件都在 `experiments/thesis/figures/`（`.pdf` 矢量版放进 LaTeX；`.png` 是 300 dpi 预览），由 `python3 harness/thesis_figures.py` 生成
> （`--only fig6_7` 只重画一张）。外观（宽度、字号、颜色、线型）统一在 `harness/thesis_style.py`。图表总方案见 `FIGURES.md`，章节见 `OUTLINE.md`。
>
> **几个全书通用的约定**（图注里不再重复时，正文方法节要交代一次）：
> - **攻击成功率**：良性客户端在自己的「评估用个性化模型」（当前边缘主干 + 自己的分类头与批归一化统计量）上，加触发器后被判成目标类的比例；只数非目标类样本、测在留出数据上。终值 = 最后 10 个评估点的平均。
> - **margin**：触发样本上「目标类对数概率 − 其余类最大对数概率」的中位数；> 0 表示多数触发样本被判成目标类。攻击成功率饱和时它仍有变化。
> - **有效轮** = 云轮 × 每云轮的边缘轮数（主设定每云轮 5 个边缘轮、60 云轮 = 300 有效轮）。
> - **集中布点** = 10 个攻击者全在一台边缘服务器（「被攻陷的边缘」），其余三台是「受害边缘」；**分散布点** = 每台边缘都有攻击者（3 / 3 / 2 / 2）。
> - 曲线 = 种子均值，浅色带 = 种子间最小—最大；空心标记 = 单个种子（圆 42、方 43、三角 44、菱形 45、倒三角 46），竖短线 = 种子均值。
> - 「出处」一栏里的 F-编号是发现记录（`experiments/*/*/FINDINGS.md`），只供查证，**不进论文正文**；内部实验组名放在括号里「备查」，同样不进正文。

---

## 图 4.1 数据划分的异质性 — `fig4_1_partition`

- **论文位置**：4.5 数据划分（回答旧稿公式 2.17：边缘内是否确实更同质 = 假设 H1 的前提）。
- **一句话**：本文的划分里，同一边缘内的两个客户端比跨边缘的两个客户端更相似；「组内更同质」的变体差距更明显。随机分配则没有差别。
- **面板**：
  - (a) 横轴 = 边缘之间的差异，纵轴 = 客户端与自己所在边缘的差异（都用类别分布的总变差，0 = 相同、1 = 完全不重叠）。
  - (b) 横轴 = 跨边缘两个客户端的平均差异，纵轴 = 同边缘两个客户端的平均差异；点落在虚线（相等）以下 = 边缘内更同质。
  - 实心黑点 = 实验里真正用到的两种划分（主划分、λ = 0.4 的组内更同质划分）；空心灰点 = 其他参照划分。
- **关键数字**（3 个种子均值）：主划分同边缘 0.549 vs 跨边缘 0.696（比值 0.79）；组内更同质 0.296 vs 0.547（0.54）；随机分配 0.610 vs 0.608（1.00）。出处：F-125。
- **数据来源**：`harness/partition_preview.py`（与训练用同一个划分函数，只依赖类计数，离线可算，不需要 GPU）；数据函数 `data_partition`。
- **强度与注意**：描述性；只看标签分布，不涉及特征分布。主划分的「组内更同质」体现得较弱（每台边缘内部仍是一个高度异质的联邦学习问题），所以才另做了 λ = 0.4 的变体。
- **英文图注草稿**：*Label heterogeneity of the data partitions (mean total-variation distance, three seeds). (a) Between-edge versus client-to-own-edge heterogeneity. (b) Pairs of clients within the same edge versus across edges; points below the dotted line indicate that clients of the same edge are more alike. Filled markers: the two partitions used in the experiments (the institution-style partition and its more homogeneous variant, λ = 0.4); open markers: reference partitions.*
- **改图入口**：`plot_partition`；标注位置在 `PARTITION_CASES` 的偏移量里改。

## 图 4.2 灌入—洗掉分解示意 — `fig4_2_decomposition`

- **论文位置**：4.3 分析工具（灌入—洗掉分解的定义）。
- **一句话**：每次云聚合把后门「灌」进受害边缘（竖直跳升），两次云聚合之间边缘内训练「洗掉」一部分（斜坡下降）；一个周期的净增量 = 灌入 − 洗掉，净增量为正就一步步走向饱和。
- **面板**：(a) 攻击成功率、(b) margin；横轴 = 有效轮（前 15 个云轮）。每个竖直线段 = 一次云聚合；圆点 = 云轮末的评估点；灰底 = 被标注的那个周期（第 11 云轮）。
- **关键数字**（被标注周期）：攻击成功率口径灌入 +0.274、洗掉 −0.165、净增量 +0.109；margin 口径 +2.33 / −1.48 / +0.85。全程前半程均值见图 6.10。
- **数据来源**：主设定、不设防、集中布点、种子 42 的受害边缘三台平均（备查：SNAP 组）；数据函数 `data_decomposition`，定义与 `harness/defense_balance.py` 相同。
- **强度与注意**：单种子的示意图，只用来定义量；结论性的数字在图 5.3 与 6.10。小数位（3 位 / 2 位）是为了让「灌入 − 洗掉 = 净增量」在舍入后仍对得上。
- **英文图注草稿**：*Injection–washing decomposition on one run (no defense, concentrated attackers, seed 42; mean over the three victim edges). Each vertical segment is a cloud aggregation (injection); the slope between aggregations is intra-edge training (washing). The highlighted cycle has injection +0.274, washing −0.165 and net change +0.109 in attack success rate (a), and +2.33 / −1.48 / +0.85 in trigger margin (b).*
- **改图入口**：`plot_decomposition`；标注哪个周期改 `DECOMP_CYCLE`，显示多少云轮改 `DECOMP_MAX_CLOUD`。

## 图 5.1 触发器底线 — `fig5_1_floor`

- **论文位置**：5.1。
- **一句话**：攻击者不投毒时，同一个触发器只有约 0.04–0.10 的成功率，所以有攻击时的高平台期不是触发扰动本身的对抗效应；各边缘的底线随本边缘目标类占比升高。
- **面板**：(a) 主划分、全部良性端：有投毒（黑实线）vs 不投毒（灰虚线 = 底线）；点线 = 事先定的判据 0.3。(b) 每台边缘的底线 vs 该边缘目标类占比（对数横轴；各划分、各种子都画在一起）。
- **关键数字**：底线 0.04–0.10（判据 0.3，判定「平台期不是对抗效应」）；单台边缘底线 0.001–0.20，目标类占比 0.3 的边缘最高。出处：F-065、F-073。
- **数据来源**：不投毒组（备查：G0，5 种划分 × 3 种子）与同配置有投毒组（备查：G1R5）；数据函数 `data_floor_thesis`（复用 `report_figures.data_floor`）。
- **强度与注意**：预注册判定、3 个种子。底线不可忽略（随目标类占比变化），所以后面各处比较都在同一划分内做。
- **英文图注草稿**：*Trigger floor. (a) Attack success rate of benign clients with poisoning (solid) and with the attackers present but not poisoning (dashed, the floor); the dotted line is the pre-registered criterion of 0.3 for an adversarial-perturbation explanation of the plateau. (b) Floor of each edge against the share of the target class in that edge (all partitions and seeds).*
- **改图入口**：`plot_floor_thesis`。

## 图 5.2 植入不受收敛门控 — `fig5_2_gating`

- **论文位置**：5.2（假设 H2 被否定）。
- **一句话**：无论训练进行到哪一步才开始投毒，被攻陷边缘的良性端都立刻被攻破；没有「训练后期更安全」的时段。
- **面板**：(a) 投毒窗口（20 个有效轮，灰底）内外的全部良性端攻击成功率，5 条线 = 5 个投毒起点（灰阶由浅到深 = 起点由早到晚），对齐到窗口起点。(b) 窗口内峰值超出底线的量 vs 投毒起点：橙 = 被攻陷边缘的良性端，蓝 = 受害边缘；浅色带 = 种子范围。
- **关键数字**：被攻陷边缘峰值 0.62–0.96（各起点、各种子），受害边缘 0.02–0.20；预注册判定 `not_gated`（秩相关均值 −0.07）。出处：F-076。
- **数据来源**：投毒起点扫描组（备查：G5，5 个起点 × 3 种子）+ 不投毒对照（备查：G0-random）；数据函数 `data_gating`（复用 `report_figures.data_g5`）。
- **强度与注意**：预注册判定、3 个种子。这组用「随机分配」划分（边缘不对应机构），与主划分不同。
- **英文图注草稿**：*Implantation is not gated by convergence. (a) Attack success rate of all benign clients around a 20-round poisoning window (grey band), aligned at the window start, for five start points (light to dark: earlier to later). (b) Peak rise above the floor inside the window for benign clients of the compromised edge (orange) and of the victim edges (blue); bands span three seeds.*
- **改图入口**：`plot_gating`。

## 图 5.3 云周期内的锯齿 — `fig5_3_sawtooth`

- **论文位置**：5.3（扩散）。
- **一句话**：受害边缘在两次云聚合之间确实会自己「洗」掉一部分后门，但每次云聚合灌进来的更多，所以周期末的值一个周期比一个周期高，最终饱和。
- **面板**：(a) 每 10 个边缘轮一次云聚合时的完整轨迹：蓝 = 受害边缘，橙 = 被攻陷边缘的良性端（云聚合时被三台干净边缘稀释、往下掉）。(b)(c) 一个云周期内的相位剖面（前半程各周期平均）：攻击成功率与 margin，3 条线 = 周期长度 5 / 10 / 20 个边缘轮。
- **关键数字**：每次云聚合灌入均值 0.197 / 0.268 / 0.180（周期 5 / 10 / 20）；每周期洗掉 0.166 / 0.223 / 0.138；每有效轮洗掉速率 0.0333 / 0.0223 / 0.0069；预注册判定 4 / 5 格「有自清洁」。出处：F-084 … F-086。
- **数据来源**：锯齿组（备查：G1 + G1R5，主划分、集中布点、3 种子）；数据函数 `data_sawtooth`（复用 `report_figures.data_g1_sawtooth`）。
- **强度与注意**：预注册判定、3 个种子；周期 20 的两个标签在噪声范围内（应读作「洗掉速率约 0.005」）。(b)(c) 三条线的高低不能直接比：前半程覆盖的训练阶段不同。
- **英文图注草稿**：*Sawtooth dynamics within cloud cycles. (a) Attack success rate with one cloud aggregation every 10 edge rounds: victim edges (blue) and benign clients of the compromised edge (orange); bands span three seeds. (b, c) Mean profile within a cloud cycle over the first half of training, in attack success rate and trigger margin, for cycles of 5, 10 and 20 edge rounds. Victim edges wash out part of the backdoor within each cycle, but each aggregation injects more.*
- **改图入口**：`plot_sawtooth`；(a) 画哪种周期长度改 `data_sawtooth` 里的 `main_R`。

## 图 5.4 攻击者离开后的衰减 — `fig5_4_decay`

- **论文位置**：5.4。
- **一句话**：攻击者停止投毒后，后门大约在 200 个有效轮内褪回底线附近，留少数长尾客户端；不分层的单服务器设定也一样，所以衰减不是分层结构造成的。
- **面板**：(a) 分层设定下受害边缘（蓝）与被攻陷边缘良性端（橙），灰虚线 = 底线；(b) 全部良性端攻击成功率，分层（黑实线）vs 不分层（灰虚线）；(c) 同样比较 margin。横轴 = 攻击者停止后的有效轮（0 = 停止时刻）。
- **关键数字**：分层设定最后窗口超出底线 0.092 / 0.037 / 0.140（预注册判定 `user_decides`，用户判「退回底线」）；不分层末窗口 0.194 / 0.286 / 0.170。出处：F-068、F-071、F-074。
- **数据来源**：衰减组（备查：G8，旧的逐类 Dirichlet 划分，第 31 云轮停止）、其不分层对照（备查：G8F）、底线（备查：FLR）；数据函数 `data_decay_thesis`。
- **强度与注意**：预注册判定落在「由用户决定」分支；这组用旧划分（不是主划分）。另一个仓库的官方代码在不分层设定下 200 轮后约 0.35，比本管线高 0.15–0.23（原因没有逐项拆开）。
- **英文图注草稿**：*Decay after the attackers stop (time 0). (a) Hierarchical setting: victim edges (blue), benign clients of the compromised edge (orange) and the trigger floor (dashed). (b, c) Attack success rate and trigger margin of all benign clients, hierarchical (solid) versus flat single-server training (dashed). The backdoor fades towards the floor within about 200 effective rounds in both settings.*
- **改图入口**：`plot_decay_thesis`；图例在图上方一行（`fig.legend`）。

## 图 5.5 边缘视角的可观测性 — `fig5_5_observability`

- **论文位置**：5.5（假设 H1 被否定：边缘内更同质并没有让检测变强）。
- **一句话**：对单个上传更新，边缘服务器用本边缘做参照，与用同样大小的全局随机样本做参照，分辨攻击者的能力几乎一样；能分开攻击者的只有更新范数，而最强的信号来自批归一化统计量（可以零代价伪造）。
- **面板**：(a) 各检测量的 AUROC（攻击者 vs 良性）：实心 = 边缘视角，空心环 = 全局视角，每个种子一行（同一种子两点重合时成环套点）；最后一行 = 批归一化统计量的范数（只有全局视角的均值）；点线 = 0.7（事先定的「可检测」下限）。(b) 权重更新范数的 ROC，边缘视角（实线）vs 全局视角（虚线）。
- **关键数字**（分散布点、每 10 个边缘轮一次云聚合）：权重更新范数 边缘 0.87 / 0.74 / 0.74 vs 全局 0.85 / 0.71 / 0.74；两种余弦与功能分数都在 0.51–0.62；批归一化统计量 0.905。预注册判定：范数 `no_edge_gain`，其余 `undetectable`。出处：F-085。
- **数据来源**：锯齿组的分散布点格（备查：G1，C1 × 分散 × R10，3 种子）+ 判定文件 `experiments/attack/hfl-mechanism/analysis/g1_verdict.json`；数据函数 `data_views`。
- **强度与注意**：预注册判定、3 个种子；判定脚本写于部分数据之后（规则在前），两份独立实现结果相同。
- **英文图注草稿**：*Observability at the edge. (a) AUROC for separating attackers' updates from benign ones, using the edge's own updates as reference (filled) or a size-matched global sample (open rings), one row per seed; the last row is the norm of the batch-normalisation statistics (global view, mean over seeds); dotted line: pre-registered detectability threshold of 0.7. (b) ROC of the weight-update norm in the two views. The edge view gives no detection advantage.*
- **改图入口**：`plot_views`；种子行间距改 `(j - 1) * 0.26`。

## 图 6.1 边缘级共享块：轨迹 — `fig6_1_edge_shared_block`

- **论文位置**：6.1。
- **一句话**：把最后一个残差块只在边缘内平均（约占 75% 参数）后，攻击者集中在一台边缘时，受害边缘的后门被压到约 0.3，两种划分结果一致；被攻陷边缘的良性端完全不受保护。
- **面板**：2 × 2；行 = 主划分 / 组内更同质划分，列 = 受害边缘 / 被攻陷边缘的良性端；灰实线 = 不设防，绿虚线 = 边缘级共享块；集中布点。
- **关键数字**：主划分受害边缘 0.996 → 0.281、被攻陷边缘 0.999 → 0.988、精度降 0.002；组内更同质划分 0.911 → 0.302、1.000 → 1.000。出处：`analysis/defense_balance.md`（统一对比表）。
- **数据来源**：不设防对照（备查：SNAP、ERP / ERC 的只打分对照，已证与不设防逐位相同）与共享块组（备查：ERP / ERC main-t1 / homo-t1），各 3 种子；数据函数 `data_tier`。
- **强度与注意**：3 个种子、两种划分；早期在旧划分上有预注册判定（隔离 1 / 2 块降 0.52 / 0.73，F-077）。主划分曲线在最后阶段有回升（约 0.35），不能说「已压到底」。
- **英文图注草稿**：*Edge-shared block with concentrated attackers. Attack success rate of benign clients in the victim edges (left) and in the compromised edge (right), for the institution-style partition (top) and its more homogeneous variant (bottom). Keeping the last residual block within each edge (green, dashed) suppresses the cross-edge spread; benign clients of the compromised edge remain unprotected.*
- **改图入口**：`plot_tier`。

## 图 6.2 共享深度的权衡 — `fig6_2_sharing_depth`

- **论文位置**：6.1（排除「只是少共享」的平凡解释）。
- **一句话**：上云的参数越少，受害边缘越安全；只隔离最后一块就压到 0.28，精度几乎不变；隔离更多块压得更低，但精度代价逐步上升，「几乎完全不上云」的代价约 0.02。
- **面板**：横轴 = 跨边缘聚合的参数比例（对数刻度）；(a) 受害边缘攻击成功率（数字 = 留在边缘内的残差块数），(b) 相对完全共享的精度代价（按种子配对），点线 = 0.02 门槛。绿实线 = 主划分（隔离 0 / 1 / 2 / 4 块），灰虚线 = 旧划分（0 / 1 / 2 块，复现序列）；空心标记 = 单个种子。
- **关键数字**（主划分，3 种子均值）：上云比例 100% / 25.1% / 6.4% / 0.14%；受害边缘 0.996 / 0.281 / 0.046 / 0.027；精度代价 0 / 0.002 / 0.010 / 0.022。隔离 1 块比「几乎不上云」多保住的精度 0.020 / 0.022 / 0.018（预注册判定 `inconclusive`：种子 44 差一点没到 0.02）。出处：F-130。
- **数据来源**：块数扫描组（备查：TBS，3 种子）+ 不设防对照（SNAP）+ 旧划分的三臂（备查：G6）；数据函数 `data_sharing_depth`（数 = `harness/supp_verdict.py` 的报告）。
- **强度与注意**：预注册判定 `inconclusive` → 正文写数字、不写方向（「多保住 0.018–0.022」，不说「有意义」也不说「隔离是免费的」）。精度只在客户端本地分布上测，可能低估共享的价值（按类精度没有记录）。
- **英文图注草稿**：*Trade-off of the sharing depth. Horizontal axis: share of parameters aggregated across edges (log scale); numbers in (a) give the residual blocks kept within each edge. (a) Attack success rate in the victim edges; (b) accuracy cost relative to full sharing, paired by seed (dotted: 0.02 threshold). Green: institution-style partition; grey dashed: earlier Dirichlet partition. Open markers are individual seeds.*
- **改图入口**：`plot_sharing_depth`；划分的颜色线型在 `thesis_style.PARTITION`。

## 图 6.3 威胁边界：一个内鬼 — `fig6_3_threat_boundary`

- **论文位置**：6.1（有效边界）。
- **一句话**：边缘级共享块是「按边缘」的隔离：最后一块只由良性端训练的边缘受保护；只要边缘内有一个攻击者参与训练，这台边缘的良性端就和被攻陷边缘一样失守。
- **面板**：行 = 布点 [9, 1, 0, 0] 下的四台边缘（9 个攻击者、1 个攻击者、两台干净边缘）；横轴 = 良性端攻击成功率；灰 = 不设防，绿 = 边缘级共享块；空心 = 种子，竖短线 = 均值。
- **关键数字**：干净边缘 不设防 0.89–1.00 → 共享块 0.25–0.50（下降 0.61–0.68）；有 1 个攻击者的边缘在共享块下仍 0.93–1.00（与干净边缘差 0.57–0.73）；预注册判定 `clean_protected` + `e1_breached`。出处：F-131。
- **数据来源**：威胁边界组（备查：TBH，3 种子）；数据函数 `data_threat_boundary`（数 = `harness/supp_verdict.py` 的报告）。
- **强度与注意**：预注册判定、3 个种子；「信任域」解释由此从「和数据一致」升为「事先预测并被检验」。25 个客户端里 1 个攻击者、参与频率约 10% 就足以让该边缘失守。
- **英文图注草稿**：*Threat boundary of the edge-shared block. Attackers placed 9 / 1 / 0 / 0 across the four edges. Attack success rate of benign clients per edge without defense (grey) and with the edge-shared block (green); open markers are seeds, bars are means. Edges whose last block is trained only by benign clients are protected; a single attacker inside an edge is enough to break the protection of that edge.*
- **改图入口**：`plot_threat_boundary`；行名在 `BOUNDARY_ROWS`。

## 图 6.4 上传前加固 — `fig6_4_hardening`

- **论文位置**：6.2。
- **一句话**：离线试的对抗训练配置里，能明显削减云聚合灌入的都过不了 0.02 的精度门槛；在线时选代价最小的那个配置，受害边缘被压到 0.5–0.7 之间来回振荡（平均降 0.41），被攻陷边缘不受影响，额外计算约 3.7 GPU 小时 / 实验。
- **面板**：(a) 离线筛选：每个点 = 一个配置在一个快照上；横轴 = 干净精度下降，纵轴 = 灌入削减率（1 = 全削掉）；形状 = 方法族；橙色实心 = 在线用的配置；点线 = 精度门槛 0.02 与判据 0.5。(b)(c) 在线轨迹：受害边缘、被攻陷边缘良性端；灰 = 不设防，琥珀色点划线 = 上传前加固。
- **关键数字**：在线配置离线削减率 0.58 / 0.44（第 6 / 15 云轮快照）、精度降 0.032 / 0.023；离线判定 `accuracy_bound`。在线受害边缘 0.996 → 0.587（三个种子降 0.476 / 0.522 / 0.228）、被攻陷边缘 0.999 → 0.982、精度降 0.012。出处：F-100、F-104；阶段三报告 §4.2 的表。
- **数据来源**：离线筛选 `experiments/defense/edge-native/analysis/p0/SNAP__collocated__s42.screen.json`（单种子）；在线组（备查：P1 / H1，3 种子）+ 不设防对照（SNAP）；数据函数 `data_hardening`。
- **强度与注意**：离线判定由种子 42 决定（单种子）；在线判定预注册、3 个种子（受害端 `protects_victims`、被攻陷边缘 `no_effect`）。没有做不投毒的下限实验 → 只能说「压低」，不能说「清除」。
- **英文图注草稿**：*Pre-upload hardening at the edge. (a) Offline screening: each point is one configuration on one snapshot; drop in clean accuracy versus reduction of the injection (1 = fully removed); shapes denote families; filled orange: the configuration used online; dotted lines: accuracy threshold 0.02 and criterion 0.5. (b, c) Online runs: attack success rate in the victim edges and in the compromised edge, without defense (grey) and with hardening (amber).*
- **改图入口**：`plot_hardening`；方法族的形状与名字在 `HARDEN_FAMILY`。

## 图 6.5 不分层的统计量聚类 — `fig6_5_flat_clustering`

- **论文位置**：6.3（检测这条线的第一步；对照文献方法）。
- **一句话**：在单服务器设定下，只按上传的批归一化统计量聚类剔除，精度代价小而稳，但防御效果随种子大起大落（5 个种子里 2 个失守）；加缓存池修好了失守的两个种子，代价约 0.027 精度；带对抗训练的设置精度损失很大。
- **面板**：(a) 全范围：横轴 = 精度下降，纵轴 = 攻击成功率下降；紫色实心圆 = 只开聚类（5 种子），紫色菱形 = 聚类 + 缓存池（2 种子），空心三角 / 方块 = 只开对抗训练 / 两个都开；灰底 = (b) 的放大区域。(b) 放大精度下降 < 0.035 的部分，箭头 = 缓存池在两个失守种子上的效果；点线 = 判据 0.5 与精度门槛 0.02。
- **关键数字**：只开聚类 攻击成功率下降 0.614 / 0.139 / 0.762 / 0.753 / 0.345（均值 0.523，判定「单独就够」但刚过线），精度降 0.008–0.016；缓存池 下降 0.779 / 0.794、精度降 0.026 / 0.028（判定 `no_fix`，用户事后接受代价）。出处：F-119、F-120。
- **数据来源**：不分层设定（1 台服务器 × 100 客户端）的组件拆分各组（备查：CCSD / CCSDA3 / CCSDC5 / CCSB3 / CCSDP，对照 CCSOFF5 / BNS）；判定文件 `analysis/ccsd5_verdict.json`、`analysis/ccsdp_verdict.json`；数据函数 `data_flat_clustering`。
- **强度与注意**：预注册判定；这组是**不分层**设定、旧划分、统计量共享，与主设定不可直接比。统计量聚类已有先例（BNGuard、CCS），在论文里是对照而不是贡献。
- **英文图注草稿**：*Clustering on batch-normalisation statistics in a flat (single-server) setting. Drop in attack success rate versus drop in clean accuracy per seed for clustering only, clustering with a cache pool, adversarial training only, and both components. (b) zooms into accuracy drops below 0.035; arrows show the effect of the cache pool on the two seeds where clustering alone failed. Dotted lines: criterion 0.5 and accuracy threshold 0.02.*
- **改图入口**：`plot_flat_clustering`；方法与标记在 `FLAT_VARIANTS`。

## 图 6.6 参照打分的可分性 — `fig6_6_reference_scoring`

- **论文位置**：6.3。
- **一句话**：边缘服务器用自己的干净图做参照给每个上传打分时（只打分、不剔除），集中布点下攻击者几乎完全分得开，分散布点明显弱一些；不同分数之间差别很大。
- **面板**：(a) 主划分、(b) 组内更同质划分；每行 = 一个候选分数，横轴 = AUROC；实心圆 = 集中布点，空心方 = 分散布点；灰 = 不开共享块，绿 = 开共享块；灰底行 = 事先规则选中的分数；点线 = 可行性门槛 0.75。
- **关键数字**：主划分选中「类混合残差」，AUROC 共享块关：集中 0.999、分散 0.847；共享块开：集中 0.998、分散 0.904；四格最小 0.847；组内更同质划分选中「全部层的参照差异」，最小 0.895；两个划分都判「可行」。出处：F-126。
- **数据来源**：只打分组（备查：ERP，种子 42）与不投毒开发组（备查：ERPB，种子 41，定阈值）；`experiments/defense/edge-native/analysis/edge_ref_probe.json`；数据函数 `data_scoring`。
- **强度与注意**：预注册判定，但每格只有一个种子；只打分阶段的可分性会**高估**在线过滤的效果（见图 6.7）。
- **英文图注草稿**：*Separability of edge-reference scores (scoring only, no filtering). AUROC of attackers' versus benign uploads for each candidate score, for concentrated (filled circles) and dispersed (open squares) attackers, without (grey) and with (green) the edge-shared block; (a) institution-style partition, (b) more homogeneous variant. The shaded row is the score selected by the pre-registered rule; dotted line: feasibility threshold 0.75.*
- **改图入口**：`plot_scoring`；分数的英文名在 `SCORE_NAMES`。

## 图 6.7 在线过滤与反馈循环 — `fig6_7_filtering_feedback`

- **论文位置**：6.3。
- **一句话**：真正开始丢弃可疑上传后，后门只被削弱到约 0.5–0.6；原因是一过滤，攻击者的分数就降到和良性端一样，分数跟着「被攻陷边缘的后门强度」走 —— 它测的是「模型已被植入后门」，而不是「这个客户端正在投毒」，于是过滤与后门之间形成反馈循环。
- **面板**：(a)(b) 受害边缘 / 被攻陷边缘良性端的轨迹，四种设置（不设防、共享块、参照过滤、过滤 + 共享块）；(c) 种子 42、过滤开时，被攻陷边缘里攻击者上传（紫实线）与良性上传（灰虚线）的分数随轮次，点线 = 阈值；(d) 横轴 = 上一个云轮被攻陷边缘的攻击成功率，纵轴 = 本云轮攻击者分数（中位数）；空心 = 只打分，实心紫 = 过滤开。
- **关键数字**：受害边缘 不设防 0.996、共享块 0.281、过滤 0.536、过滤 + 共享块 0.184；被攻陷边缘 过滤 0.781、过滤 + 共享块 0.800（其余约 1.0）；(d) 的相关系数 只打分 0.07、过滤开 0.57；阈值 0.119。出处：F-128、F-129。
- **数据来源**：过滤组（备查：ERF，3 种子）、对照（备查：SNAP / ERP / ERC）；(c)(d) 只用种子 42；数据函数 `data_filtering`（相关系数的口径 = 每云轮取中位数，从原始打分记录复现）。
- **强度与注意**：(a)(b) 的标签是预注册判定（只有「共享块开 + 分散」一格过全部门槛）；(c)(d) 是**单种子的相关性读数**，正文用「观察到」，不用「表明」。
- **英文图注草稿**：*Online filtering with edge-reference scores. (a, b) Attack success rate in the victim edges and in the compromised edge for four settings. (c) Seed 42 with filtering: median suspicion score per cloud round of attackers' and benign uploads in the compromised edge; dotted line: threshold. (d) Attackers' median score against the attack success rate of the compromised edge in the previous cloud round, with scoring only (open, r = 0.07) and with filtering (filled, r = 0.57).*
- **改图入口**：`plot_filtering`；四种设置的图例在图上方一行。

## 图 6.8 良性端的误判分布 — `fig6_8_benign_misjudgement`

- **论文位置**：6.3（公平性）。
- **一句话**：参照过滤的误判集中在少数良性端身上：多数良性端很少或从未被丢弃，但有一小部分被丢弃的上传高达一半以上；分散布点下误判更多。
- **面板**：三个面板 = 集中布点的受害边缘 / 集中布点的被攻陷边缘 / 分散布点；横轴 = 一个良性端被丢弃的上传比例，纵轴 = 良性端的累计比例（经验分布）；紫点线 = 参照过滤，粉点划线 = 过滤 + 共享块；点线 = 名义误剔率 0.10（阈值按不投毒实验的 90% 分位定）。
- **关键数字**（3 种子合并）：从未被丢弃的良性端 —— 受害边缘 55% / 62%、被攻陷边缘 60% / 67%、分散布点 36% / 57%（过滤 / 过滤 + 共享块）；被丢弃不超过一成的 80% / 84%、82% / 96%、66% / 79%；单个良性端被丢弃的最高比例：受害边缘 0.70 / 0.77、被攻陷边缘 0.38 / 0.41、分散布点 0.79 / 0.86。
- **数据来源**：过滤组（备查：ERF main，3 种子）里逐上传的接纳记录；数据函数 `data_misjudgement`（与过滤判定用同一份记录，逐上传合并后的误剔率与判定一致，测试钉住）。
- **强度与注意**：描述性（新补的分析，不是预注册判定）；只做了主划分（组内更同质划分的过滤陷入冷启动死锁、全部被丢弃，不提供信息）。没有分析被误判的客户端有什么共同特征（例如数据分布少见）—— 要写成局限或补一个分析。
- **英文图注草稿**：*Who is wrongly filtered. Empirical distribution over benign clients of the share of their uploads discarded by edge-reference filtering (dotted) and by filtering with the edge-shared block (dash-dotted), for benign clients in the victim edges and in the compromised edge under concentrated attackers, and under dispersed attackers. Dotted vertical line: nominal false-rejection rate of 0.10. Most benign clients are rarely discarded, but a minority loses up to half of their uploads or more.*
- **改图入口**：`plot_misjudgement`。

## 图 6.9 综合比较 — `fig6_9_comparison`

- **论文位置**：6.4。
- **一句话**：把全部做法放在一起：边缘级共享块在集中布点下效果最好、几乎零代价；过滤与共享块叠加最低；只有参照过滤碰得到被攻陷边缘；已知的逐坐标中位数（在边缘层或云层）几乎无效；上传前加固最贵。
- **面板**：五列 = 受害边缘、被攻陷边缘、分散布点的全部良性端（三列都是攻击成功率）、精度下降、每个实验的防御计算量（GPU 小时）；行 = 七种设置（不设防、边缘层中位数、云层中位数、边缘级共享块、上传前加固、参照过滤、过滤 + 共享块）；空心 = 种子，竖短线 = 均值；精度列点线 = 0.02。
- **关键数字**（种子均值）：

  | 设置 | 受害边缘 | 被攻陷边缘 | 分散布点 | 精度下降 | GPU 小时 |
  |---|---|---|---|---|---|
  | 不设防 | 0.996 | 0.999 | 0.986 | 0 | 0 |
  | 边缘层中位数 | 0.976 | 1.000 | 0.950 | 0.002 | 未单独计时（可忽略） |
  | 云层中位数 | 0.935 | 0.998 | 0.991 | 0.001 | 未单独计时（可忽略） |
  | 边缘级共享块 | 0.281 | 0.988 | 0.956 | 0.002 | 0 |
  | 上传前加固 | 0.587 | 0.982 | 0.713（单种子） | 0.012 | 3.7 |
  | 参照过滤 | 0.536 | 0.781 | 0.607 | 0.021 | 0.24 |
  | 过滤 + 共享块 | 0.184 | 0.800 | 0.622 | 0.015 | 0.24 |

  出处：`analysis/defense_balance.md`；中位数臂 F-132。
- **数据来源**：阶段三全部臂（备查：SNAP / P1 / H1 / ERP / ERC / ERF / MED）；数据函数 `data_comparison`（数 = 统一对比表）；计算量 = 防御自身的计算时间（加固的逐次耗时合计、打分的总耗时），不是整个实验的时长；中位数没有单独计时，图上画在 0。
- **强度与注意**：各臂各自有预注册判定，这张表本身是描述性汇总；中位数臂按事先规则判 `blocks` 只因方向一致，效应量很小（正文必须带效应量）；分散布点的加固只有种子 42。
- **英文图注草稿**：*Overview of all defenses. Columns: attack success rate in the victim edges, in the compromised edge (concentrated attackers) and over all benign clients (dispersed attackers); drop in clean accuracy (dotted: 0.02); defense compute per run in GPU-hours. Open markers are seeds, bars are means. The coordinate-wise median baselines are shown in grey.*
- **改图入口**：`plot_comparison`；列定义在 `COMPARE_COLS`，行顺序在 `COMPARE_ARMS`，中位数的灰阶在 `thesis_style.ARM`。

## 图 6.10 灌入—洗掉按防御分解 — `fig6_10_injection_washing`

- **论文位置**：6.4（用第 4.3 节的分析工具解释各防御的作用方式）。
- **一句话**：边缘级共享块主要靠减少每次云聚合的灌入；上传前加固按攻击成功率看几乎不减少灌入（按 margin 看减少约三分之一），主要作用在周期净增量上；参照过滤前半程能减少灌入，后半程回升。
- **面板**：上行 = 攻击成功率口径，下行 = margin 口径；三列 = 每周期灌入、洗掉、净增量（训练前半程平均）；行 = 五种设置；空心 = 种子，竖短线 = 均值。
- **关键数字**（集中布点、受害边缘，三个种子）：灌入 不设防 0.220 / 0.222 / 0.148，共享块 0.035 / 0.072 / 0.071，加固 0.174 / 0.223 / 0.142；margin 口径灌入 不设防 2.00 / 2.00 / 1.75、共享块 0.94 / 1.20 / 1.01、加固 1.37 / 1.44 / 1.07。出处：`analysis/defense_balance.md`。
- **数据来源**：同图 6.9 的集中布点各臂；数据函数 `data_balance`（定义见 `harness/defense_balance.py` 头注释）。
- **强度与注意**：**描述性分析，不是预注册判定**，部分数据在写分析规则前已经看过；两个口径结论不一致的地方（加固）正文要明说。按全书规则，标题数字用攻击成功率（前半程对照还没饱和），margin 并列。
- **英文图注草稿**：*Injection–washing decomposition per defense (victim edges, concentrated attackers, first half of training). Per-cycle injection, washing and net change in attack success rate (top) and trigger margin (bottom); open markers are seeds, bars are means. The edge-shared block mainly reduces injection; pre-upload hardening mainly lowers the net change.*
- **改图入口**：`plot_balance`；坐标范围在函数里的 `lims`。

---

## 改图时怎么和我交互（建议）

1. **按「图号 + 面板」点名，一次列全**，例如：

   ```
   图 6.7：
   - (c) 纵轴从 0.06 开始
   - 图例挪到图下方
   图 5.3：去掉 (b)，只留 (a)(c)
   全局：字号 +1；正文宽度 15.9 cm
   ```

   我一轮改完、只重画涉及的图，把改前 / 改后并排发给你，确认后一次提交。
2. **三类改动**：全局样式（字号、宽度、颜色、线型、字体）→ `harness/thesis_style.py` 一处，全部图一起变；单张图排版（面板、坐标范围、图例、轴标题、去掉某个面板）→ 该图的画图函数（每节「改图入口」）；内容（换数据、加臂或种子、换口径）→ 数据函数，测试锚点会检查数字没变错。
3. **截图圈画也可以**：在 PNG 上圈出来、写几个字上传。
4. **越早给越省返工的参数**：论文模板的 `\the\textwidth`（现在暂按 15 cm）；正文字体（图里要不要换成同一字体）；哪些图做全宽 / 半宽；是否把面板拆成单独文件给 LaTeX 子图用。
5. **可选**：把全部图 + 本说明 + 图注放进一个可评论的私有网页，你在每张图下面留评论，我读评论逐条改（需要你点头再做）。
