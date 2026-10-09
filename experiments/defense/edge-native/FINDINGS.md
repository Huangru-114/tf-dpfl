# 结论台账 —— 阶段三（edge 原生防御）

> 每条结论 = 声明 + 证据（能重跑的命令或 文件:行）+ 状态。编号接续 `experiments/attack/hfl-mechanism/FINDINGS.md`（F-086 之后从 F-087 起；D-090）。
> 状态：`confirmed`（证据是代码、数据或原文本身）· `provisional`（单 seed / 推断，当假设用）· `retracted`（撤回，写原因）。
> **数字只从脚本产物、复核命令或原文表格里来，不凭记忆写**（陷阱 #14）。预注册的判读规则写成 `N-` 条目，冻结后才交作业。

## 2026-10-09（阶段三计划定稿会话）

### F-087 `confirmed`（代码与配置核对）—— G8 / G6 / G6D 用的旧 `noniid` 划分没有 edge 干净集 → P0 不能用 G8 存盘，改用新组 SNAP

- `experiments/attack/hfl-mechanism/base.yaml:42`：`partition: "noniid"`；G8（`registry.yaml:266-269`）、G6（`:228-234`）、G6D（`:280-286`）的 `set:` 都没有改划分。
- edge 干净集只在 S3 划分（designed / hdir / equal_random）下产生：`fedavg/main.py:506` 只在 S3 路径写 `s3_out["clean_indices"]`，`main.py:774-775` 其余情况下 `edge.clean_indices = None`。
- `fedavg/analysis/ck_snapshot.py` 的说明里写明：G8 没有 edge 干净集，F-081 的 c_k 预检拿**良性端留出分片**当干净数据 —— 只读评估没问题，
  但拿它做**加固**（训练）就是在评估探针上训练（陷阱 #11）。
- 后果：P0 的离线反事实需要「S3 划分 + edge 干净集 + 上传前 body」的快照 → `PLAN.md` §4 的 **SNAP**（C1 × {集中, 分散} × R5 × s42–44，快照第 6 / 15 / 60 轮）。
  集中布点的 SNAP 与 G1R5 只差快照开关，按 F-084 / D-073 应逐位相同（顺带复验）；SNAP 同时是全部主检验的无防御对照臂。
- G8 存盘（集群 `tfdpfl-dumps/G8__a__s4x.*`，约 0.81 GB）因此不再是阶段三的前置；删不删由用户定（D-073）。

### F-088 `confirmed`（原文核对，用户上传的 PDF）/ 一处出处待确认 —— Bad-PFL 的触发器是两个 ℓ∞ 对抗扰动之和；flat 文献里已知防御几乎都挡不住它，对抗训练没被测过；CCS 是唯一的正面证据，但由两个组件组成

数字与原文表号见 `LITERATURE.md` 第 2 部分。对计划有直接影响的六点：

1. **触发器结构**（Bad-PFL Eq. 5–7）：δ = ε·G_w(x) 定向推向 y_t、ξ = σ·sign(∇L) 非定向（FGSM），ε = σ = 4/255 → ‖δ + ξ‖∞ ≤ 8/255。
   → 对抗训练的扰动集要覆盖 8/255，且要对定向扰动鲁棒（`PLAN.md` §3.2 的 ε 网格与 AT-tgt）。
2. **δ 是主体，且有剂量反应**（表 4 / 18，FedRep）：去掉 δ → 12.82%、去掉 ξ → 79.32%；δ 的预算 0 / 1 / 2 / 3 / 4（/255）→ 12.82 / 54.92 / 79.68 / 86.79 / 97.95%。
   → 加固不必做到完全鲁棒（**推断**：这条曲线是攻击者改 ε 测的，不是防御方改鲁棒性测的）。
3. **已知防御在 flat 下几乎都失败**（FedRep，ASR）：FT-15 / 30 / 45 97.31 / 97.76 / 97.01、NAD 86.25、I-BAU 76.58、Simple-Tuning 88.82、BAERASER 91.54、MAD 90.74、
   ClipAvg 97.28、Multi-Krum 96.15、Median 77.21；Sign 20.32 但 Acc 34.49（训练崩）。检测也弱：Neural Cleanse 异常指数 2.2、STRIP 熵 0.77（干净 0.92）。
   → 干净微调降为归因对照、删 NAD 式蒸馏、删 Simple-Tuning（D-094）；I-BAU 的通用扰动遗忘失效 → 改成逐样本的定向 AT；H5 的多数基线预期失败 → s42 先筛。
4. **Bad-PFL 原文没有测对抗训练**；附录 D 认为唯一可想到的对策是「不含目标类数据地微调」。
5. **CCS**（ICASSP 2026，flat、**FedPer**、ResNet-18）：Bad-PFL 94.42% → 8.54%（MTA 79.46 → 79.76），α 0.1–1 都约 10%。组成 = 客户端对抗训练（CE + 0.01·KL + 0.01·MMD）
   + server 对 BN running 统计量做 HDBSCAN 剔除；威胁模型明写攻击者不知道防御。CCS 的 ASR 不过滤目标类（无攻击时约 9–10.5%），8.54% 即无攻击水平。
6. **CCS 的消融出处待确认**：用户记得原文做过消融、对抗训练约占 85%；上传的这一版**没有**（`pdftotext <CCS PDF> - | grep -a -i -E 'ablat|w/o|85%'` 无结果，
   实验部分只有表 1 与图 2）。而本仓库 F-085 测到 BN 统计量通道单独 AUROC 0.90、且在 FedRep 下可零代价伪造 → 本设定下聚类的份额可能比原文大（推断）
   → CCSP 三臂复核组件贡献 + AA-S（伪造统计量）检验聚类部分；拿到消融后把「ΔV(CCS-AT) ≈ 0.85 × ΔV(CCS-full)」写成预注册预测。

### F-089 `confirmed`（环境）—— 本地用 Python ≥ 3.12 跑 L1 会多出与代码无关的红；用 Python 3.11 与交接基线逐项相同

| 解释器（numpy） | L1（`bash run_l1.sh`，本地无 TF） |
|---|---|
| Python 3.11.17（numpy 2.4.6） | **1599 passed / 45 skipped / 3 xfailed → PASS**（= 交接记录的基线） |
| Python 3.12.3（numpy 2.4.6，只跑下面两个文件） | `test_designed_partition.py` 1 条红 + `test_report_figures.py` 1 条红 |
| Python 3.13.16（numpy 2.5.3，本容器默认 `python3`） | **4 failed** / 1595 passed / 45 skipped / 3 xfailed → FAIL |

- **AST 指纹**（`tests/test_designed_partition.py` 的 `test_old_partition_*` 三条）：测试对 `ast.dump()` 的输出取哈希。Python 3.12 给每个 `FunctionDef` 加了 `type_params=[]` 字段
  （函数级指纹那 1 条变红）；Python 3.13 的 `ast.dump()` 默认省略空列表与 None 字段（3 条全红）。复现：
  `for v in 3.11 3.12 3.13; do python$v -c "import ast; print(ast.dump(ast.parse('def f(a):\n    if a:\n        return 1').body[0]))"; done`。
- **第 4 位小数的舍入**（`tests/test_report_figures.py::test_g1_views_figure_reads_the_committed_verdict_and_it_reproduces`）：重算 `g1_verdict.judge_3c` 与入库的
  `analysis/g1_verdict.json` 只差 C1·R20 的两个 `d_jump`（s43 0.1706 vs 0.1707、s44 0.202 vs 0.2021），判定标签不变。同一 numpy 2.4.6 下 3.11 过、3.12 红 →
  是 Python 版本造成的：3.12 起内置 `sum()` 对浮点用补偿求和（`python3.11 -c "print(sum([0.1]*10))"` → 0.9999999999999999，3.12 → 1.0），
  `harness/g1_verdict.py:82` 用内置 `sum` 求均值，落在舍入边界上的值就翻了。
- **代码与数据都没有问题**：集群容器是 Python 3.10.12（CLAUDE.md 的实测记录）；交接记录的 1599 / 45 / 3 与本会话 Python 3.11 的结果逐项相同。
- 做法（本会话起）：本地用 3.11 的 venv 跑 L1 —— `uv venv -p python3.11 <dir>`、装 `pytest numpy pyyaml matplotlib`，再 `TFDPFL_PY=<dir>/bin/python bash run_l1.sh`
  （`cluster_env.sh` 会打印 `mode=override`）。要不要把这几条测试改成与解释器版本无关（例如 AST 指纹先做与版本无关的规范化、均值改用 `math.fsum`），由用户定；本会话没有改。
