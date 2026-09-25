# Experiment 3（改版）—— HFL 中个性化后门的机制研究：执行计划

> 原文：`PLAN-original-2026-09-24.md`（@Gao，2026-09-24，逐字保存，不改）。
> 本文件 = 原文 + 2026-09-24 评审 + 与代码的映射 + 预注册判定 + 执行路线。
> **范围**：只做改版实验 3（原文 §3 的 3.1–3.3 与 §4–§8 的 3-A–3-E）。
> 原文 §9 阶段三与 §11 算力方案**不在本计划内**（DECISIONS D-008）。
>
> 相关文件：`AUDIT.md`（对齐审计，**开跑门槛**）· `DECISIONS.md` · `FINDINGS.md` ·
> `current-focus.md` · `README.md`（工作流）。

---

## 0. 口径版本（不是训练 epoch）

| 版本 | 含义 | 位置 | 能否进结论 |
|---|---|---|---|
| **P0** | 探针修正之前的归档 | `../hfl-propagation/results/archive-pre-fix/` | 否 |
| **P1** | 统一标准后重跑的 seed42 批次（`d8c8d0b`，26 个文件） | `../hfl-propagation/results/` | 否，只作试点 |
| **P2** | `AUDIT.md` 全部关闭之后的正式批次 | `results/P2/<group>/` | **是，唯一可进结论的版本** |

代码里的常量是 `fedavg/utils/provenance.py:PROTOCOL_VERSION`，每个 run 通过 `[Provenance]` 行把它写进 metrics.json。
分析工具拒绝混用不同版本。

P1 的用途（FINDINGS F-002/F-008/F-009）：同 seed 噪声的实测、效应量的量级（决定 seed 数）、待检验假设、S1 工具的真实数据锚点。

---

## 1. 评审结论

**总评：方向对，值得执行。** 原文对旧实验 3 的核心批评被代码证实：25 个旧 yaml 全是
`partition: noniid` + `edge_assignment: block`，edge 的数据构成与 edge 编号无关
（`inter_edge`/`intra_edge` 两个键没人读，FINDINGS F-005）。

**修正与补充**（每条的证据见 FINDINGS）：

1. **3.1 是其他结论的前提。** 旧的 rho0 格回退到静态 BadNet 触发器（`main.py:171`），测出的 0.037 **不是** Bad-PFL 的下限（F-006）。
   - 真正的下限只要改配置：`poison_ratio: 0` + 恶意端照常布点 → 一个样本都不投毒，而 `on_round_start` 仍在干净模型上训练生成器（`client_badpfl.py:120-147`）。A03 对齐成伯努利后，ρ=0 时同样不投毒。
   - 主指标用 **excess ASR = ASR − floor**。
2. **3-A 混入了学习率日程**（AUDIT A08）。LR 按云轮衰减 → 同一有效轮上 flat 与 HFL 的 lr 不同。
   - A2 会话定为 D-023：保留调过参的 0.992 / head 0.005，改成**按有效轮衰减**，flat 逐字节不变。
   - 不采用官方的常数 LR（D-022：训练协议不为对齐而对齐）。
   - 要先过可行性实验 D-029。
3. **同 seed 不可复现**（F-002）。主结论配置 5 个 seed；按 seed 配对只锁得住划分与布点。
4. **3-C：FedRep 下 edge / global 模型的 head 是初始化 head，`client.model` 是陈旧的**（F-007）。
   - 主曲线用 **fresh-PM ASR** = 当前 edge body ⊕ 每个良性端的私有 head；陈旧 PM 作对照。
   - 逐 edge 轮评估挂在 `edge.run()` 的循环里就行：一个云周期内各 edge 互相独立。
5. **3-D 的样本量**：每 edge 每轮只有 B/n_edges 个更新（AUDIT D02）。
   - 功能分数 s_i 对类别 k 自归一化，不需要同伴。
   - 几何分数的参照 = 本轮下发的 edge 权重，在最近 W 个 edge 轮里滑窗、用 median/MAD。
   - edge 视角与全局视角**按相同池大小比较**。
   - 结论只对 edge 内恶意比例 < 50% 的配置成立（collocated 下 E0 是 40%，`10edge_collocated` 是 100%）。
   - 只用 body 索引：FedRep 的上传里带着私有 head（陷阱 #9 同类）。
6. **R_edge ≥ 10 时，云轮粒度的 T50 太粗**（`grid_too_coarse`）→ benign / edge 的 T50 改用逐 edge 轮评估。global ASR 本来就只有云轮粒度。
7. **3.3 / G5 不需要 checkpoint 分叉**：从头跑时间窗 [t0, t0+20) 再观察 50 轮，只需加 `attack_start_round`。
8. **P1 已在质疑「HFL 延迟」**：T_0.5(benign) 为 flat 57、2edge 47、4edge 23 有效轮。只有一个 seed，而且不符合审计要求 → 只当假设（F-009）。

---

## 2. 子实验 → 需要的功能 → 会话

| 子实验 | 需要的功能 | 现状 | 会话 |
|---|---|---|---|
| 全部 | 对齐审计关闭 | — | A1–A4 |
| 3.1 对抗下限 | ρ=0 影子攻击者；ξ-only 下限；（可选）迁移 baseline 需要保存生成器 | 前者只需配置 + L1；后两者缺 | S4 |
| 3.2 私有头吸收 | FedRep ↔ FedAvg、ρ=1；恶意端干净精度（已算、没解析）；body 更新范数 | 部分已有 | S6 |
| 3.3 收敛门控 | `attack_start_round`；固定长度 run | 缺起始轮 | S4 |
| 3-A 公平对照 | flat / HFL / R_edge / 参与配额 / T_θ（都已有）；逐 edge 轮评估（高 R） | 大部分已有 | S5 |
| 3-B 目标类分布 | 固定比例表 C1–C4；层级 Dirichlet p_e~Dir(α_e p)；H_inter/H_intra；逐 edge 下限 | 缺 | S3 |
| 3-C 锯齿 | 逐 edge 轮评估 + fresh-PM ASR；攻击停止（已有） | 缺 | S5 |
| 3-D 可观测性 | 逐更新几何分数日志；周期性整包转储；离线 c_k（PGD 代价）；edge 干净集 | 缺 | S3 + S6 |
| 3-E 三层个性化 | `get_base_head_indices` 加第三组；cloud / edge 各自只聚合对应层 | 缺 | S8（可选） |

---

## 3. 预注册判定（数据回来之前定下，由 `harness/verdicts.py` 执行）

> 阈值带「⚠待确认」的由你在 A4 之前拍板；拍板后写进 DECISIONS，改阈值要留记录。
> 通用规则：
> - 终值 = 末 10 个评估点的均值；
> - 「显著」= 按 seed 配对的 bootstrap 95% CI 不含零（10000 次重采样，固定种子）；
> - seed 数低于要求 → 判定为 `insufficient`，**不报方向**；
> - 删失（跑完也没越过阈值）单独报，不当成数值参与平均。

| 子实验 | 量 | 判定规则 | seed |
|---|---|---|---|
| 3.1 | floor_gen = ρ=0 影子攻击者下 benign ASR 的终值 | floor_gen ≥ 0.3 → 「平台期主要来自对抗脆弱性」（原文阈值） | ≥3 |
| 3-A | 比值 r = T50(HFL)/T50(flat)，benign，有效轮，按 seed 配对 | log r 的 CI 全 > 0 → 结构性延迟；CI 落在 [log 0.9, log 1.1] 内 → 约等于 1（⚠待确认 ±10% 等效区间）；其余 → 不确定 | 5 |
| 3-B | C3 下 edge3 与其他受害 edge 的 excess ASR 之差 | CI 全 < 0 → 迁移依赖目标类的自然特征；CI 含 0 → 无差异 | 3 |
| 3-C | r_down = 干净 edge 在一个云周期内 fresh-PM ASR 的逐 edge 轮斜率（取负） | CI 全 > 0 → edge 级隔离可用；CI 含 0 → 后门进入 body 后冲不掉 | 3 |
| 3-D | ΔAUROC = AUROC(edge 视角) − AUROC(全局视角)，**等池大小** | CI 全 > 0 → edge 的价值包含检测 | 3 |
| 3-E | 受害 edge 的 excess ASR：(b)/(c) 相对 (a)；MTA 损失 | ASR 下降的 CI 全 > 0，且 MTA 下降 ≤ 0.02（⚠待确认） | 3 |
| 3.2 | ρ=1 时恶意端自身干净精度、body 更新范数；FedRep vs FedAvg 的 benign ASR | 按原文 §3.2 的三条预测逐条判定 | 3 |
| 3.3 | 窗口内的峰值 ASR（植入）、窗口后第 50 轮的 ASR（稀释），随 t0 的变化 | 峰值随 t0 单调上升的秩相关 CI > 0 → 植入受收敛门控 | 3 |

---

## 4. 运行组（P2；预算在 A4 复核后填入）

| 组 | 配置 | 服务 | 依赖 |
|---|---|---|---|
| G0 下限主干 | 划分 {随机, C1, C2, C3, C4} × ρ=0 影子攻击者 × 3 seed | 3.1 下限、3-B 逐 edge 下限、3.3 对照 | A4 + S3 + S4 |
| G1 主攻击（详细记录） | 4 edge；划分 {随机, C1} × 放置 {collocated, distributed} × R_edge {10, 20} × 3 seed；逐 edge 轮评估；更新日志 | 3-C、3-D、3-A 的一部分 | A4 + S3 + S5 + S6 |
| G2 结构扫描 | flat + edge {2, 4, 10} × R_edge {2, 5, 10, 20}，去掉 G1 已覆盖的格子 × 5 seed | 3-A | A4 + S5 |
| G3 目标类条件 | C2/C3/C4（collocated）+ 层级 Dirichlet α_e {0.1, 0.3, 1, 10} × 3 seed | 3-B | A4 + S3 |
| G4 私有头 | ρ {0.25, 1.0} × {FedRep, FedAvg} × 3 seed | 3.2 | A4 + S6 |
| G5 时间窗 | t0 {20, 60, 100, 140, 180} × 20 轮投毒 + 50 轮观察 × 3 seed，从头跑 | 3.3 | A4 + S4 |
| G6（可选） | 3-E 的三种划分 × 3 seed | 3-E | S8 |
| G7 预处理对比 | 主配置 × 「官方预处理」（无标准化、无增强）× 3 seed，与主协议同 seed 配对 | 检验「官方设定降低了攻击难度」（D-025） | A4（ε 换算跟随开关，F-027）；在 A4 登记进 `registry.yaml` |

**全部组共用同一条参与量匹配规则（D02）**，否则 T50 不能跨组比较。

---

## 5. 执行路线（一会话 = 一模块 = 一分支）

| 会话 | 内容 | 门槛 / 解锁 |
|---|---|---|
| **S1**（2026-09-24） | 结果管理 + 分析底座；本计划、AUDIT、DECISIONS、FINDINGS 入库 | — |
| S1b | 修 `exp3_cell.sbatch` 写死的 `--defense none` + 守卫（只有你还要旧方案的防御格时才需要） | — |
| **A1** | 审计：攻击（A01–A06、A14） | 你逐行拍板 |
| **A2**（2026-09-25） | 审计：训练协议（A07–A11、A13、A15、A16、A20、A22、A23，新增 A25）。原则 D-022：攻击定义必须对齐，训练协议不为对齐而对齐；决定 D-022 … D-029 | 已拍板；A08 待 D-029 |
| **A3** | 审计：FedRep 实现细节（A12 已由 D-024 定为维持 P1 现状，取代 D-012；剩「head 是哪几层、BN 算不算 body」）/ ResNet-10（A17）/ A18 / HFL 形式化（A21）+ D01–D06 签字 | 论文 PDF 需重新上传（D-013） |
| **A4**（实现会话；≠ AUDIT 行 A04） | 按拍板改代码 + L1 测试；跑 D-029 可行性实验（A08 按有效轮衰减 + A25 取数修复），通过后 A08 → `deviate`；登记 G7；`PROTOCOL_VERSION` 升 P2；2 个 smoke 复核标定 | AUDIT 全部关闭（A08 由 D-029 关闭） |
| S3 | 划分：C1–C4 比例表、层级 Dirichlet、H_inter/H_intra（打 `[Data]` 行）、edge 干净集（500/edge）、划分 seed 分离。**硬要求：客户端等大小**（D-027，F-028） | → G0 / G3 |
| S4 | ρ=0 影子攻击者 L1、ξ-only 下限、`attack_start_round` | → G0 / G5 |
| S5 | 逐 edge 轮评估 + fresh-PM ASR | → G1 / G2 |
| S6 | 更新日志、几何分数（body-only、滑窗）、周期转储、离线 c_k、恶意端干净精度 | → G1 / G4 |
| S7 | 各子实验的判定代码补全 + 出图 | — |
| S8（可选） | 3-E 三层个性化 | → G6 |

---

## 6. 图目录（出图规则见 README）

| ID | 内容 | 子实验 |
|---|---|---|
| F0 | 各划分的 (H_inter, H_intra) 散点 | §2 |
| F1 | ASR 与 floor_gen / floor_ξ 随有效轮的曲线，三层 | 3.1 |
| F2 | T50(HFL)/T50(flat) 森林图，按 (n_edges, R_edge) 分组 | 3-A |
| F3 | 逐 edge × 有效轮的 fresh-PM ASR 热条，标出云聚合时刻，叠加锯齿 | 3-C |
| F4 | C1–C4 逐 edge 的 excess ASR 与 MTA | 3-B |
| F5 | edge 视角 vs 全局视角的 ROC（等池大小） | 3-D |
| F6 | ρ × {FedRep, FedAvg}：benign ASR、恶意端干净精度、body 更新范数 | 3.2 |
| F7 | t0 窗口网格：峰值 ASR 与 +50 轮的 ASR | 3.3 |
