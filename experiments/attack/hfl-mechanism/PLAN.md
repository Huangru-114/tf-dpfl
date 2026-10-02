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
   - **A3 推广到全部子实验（D-033）**：主 ASR 与主 pm_acc 必须在同一个个性化模型上测，两者都用 fresh-PM = [当前 edge body, 自己的 head, 自己的 BN 统计量]（统计量私有见 D-032）；陈旧 PM 的两者作副列。实现从 S5 挪到 A4。
   - ~~逐 edge 轮评估挂在 `edge.run()` 的循环里就行：一个云周期内各 edge 互相独立。~~ **A3 更正**：共享生成器让各 edge 在云周期内**不**独立（F-038）。D-036 改为按 edge 轮交错执行后，逐 edge 轮评估挂在 cloud 驱动的 edge 轮循环里，所有 edge 停在同一有效轮。
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
| 3.1 对抗下限 | ρ=0 影子攻击者；~~ξ-only 下限~~（**不做**，D-078）；迁移 baseline（**推迟**：S9 快照已存生成器，干净模型的快照可带开关重跑补） | ✅ 影子攻击者的 L1 守卫已补（S4） | S4 ✅ |
| 3.2 私有头吸收 | FedRep ↔ FedAvg、ρ=1；恶意端干净精度（已算、没解析）；body 更新范数 | 部分已有 | S6 |
| 3.3 收敛门控 | `attack_start_round` + `generator_schedule`（A = window / B = always）；固定长度 run | ✅ 已有（S4，D-078 / D-079）；G5AB 判 A（D-080）；G5 判 `not_gated`（F-076） | S4 ✅ |
| 3-A 公平对照 | flat / HFL / R_edge / 参与配额 / T_θ（都已有）；逐 edge 轮评估（高 R） | 大部分已有 | S5 |
| 3-B 目标类分布 | 固定比例表 C1–C4；层级 Dirichlet p_e~Dir(α_e p)（按社区口径 = 每类 α_e，D-063）；H_inter/H_intra；逐 edge 下限 | ✅ 划分已有（S3）；逐 edge 下限等 FLR / G0 | S3 |
| 3-C 锯齿 | 逐 edge 轮评估 + fresh-PM ASR；攻击停止（已有） | 缺（fresh-PM 与交错执行在 A4，D-033 / D-036） | A4 + S5 |
| 3-C 攻击停止版（G8，D-075） | `attack_stop_round`（已有）+ margin / 逐客户端 / logits / 快照（S9，D-072 / D-073） | ✅ 已有 | S9 |
| 3-D 可观测性 | 逐更新几何分数日志；周期性整包转储；离线 c_k（PGD 代价）；edge 干净集 | 缺 | S3 + S6 |
| 3-E 三层个性化 | `get_base_head_indices` 加第三组；cloud / edge 各自只聚合对应层 | ✅ 已有（S8，D-057：`federation.edge_shared_blocks`） | S8（可选） |

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
| 3.1 | floor_gen = ρ=0 影子攻击者下 benign ASR 的终值 | floor_gen ≥ 0.3 → 「平台期主要来自对抗脆弱性」（原文阈值）。**先由 FLR（D-061）给出量级**：FLR 判 `negligible` 时 3.1 直接由它回答 | ≥3 |
| 3-A | 比值 r = T50(HFL)/T50(flat)，benign，有效轮，按 seed 配对 | log r 的 CI 全 > 0 → 结构性延迟；CI 落在 [log 0.9, log 1.1] 内 → 约等于 1（⚠待确认 ±10% 等效区间）；其余 → 不确定 | 5 |
| 3-B | **差中差**（D-062，机构式比例表下受害 edge 不再完全可交换）：[E3 − E1/E2 均值]_{C3} − [同]_{C1}，按 seed 配对；ASR 用 excess（FLR 判 `negligible` 时用原始 ASR 并注明上界，D-061） | CI 全 < 0 → 迁移依赖目标类的自然特征；CI 含 0 → 无差异（**2026-09-29 结果**：CI 全 > 0、天花板下的机械结果，两支都没落上，F-073 / F-075；`harness/g3_did.py`，脚本写于数据之后） | 3 |
| 3-C | r_down = 干净 edge 在一个云周期内 fresh-PM ASR 的逐 edge 轮斜率（取负） | CI 全 > 0 → edge 级隔离可用；CI 含 0 → 后门进入 body 后冲不掉 | 3 |
| 3-D | ΔAUROC = AUROC(edge 视角) − AUROC(全局视角)，**等池大小** | CI 全 > 0 → edge 的价值包含检测 | 3 |
| 3-E | 受害 edge（E1–E3）的**原始** benign ASR（fresh-PM）：(b)/(c) 相对 (a)，按 seed 配对（D-059：不等 S4 的 floor；「三臂 floor 相同」无证据，注明）；MTA 损失（**判定用 fresh 列**，D-071；陈旧列另报） | ASR 下降的 CI 全 > 0，且 MTA 下降 ≤ 0.02（D-071：维持 0.02，另报 ΔASR–ΔMTA 权衡；更高门槛只能事后标注） | 3 |
| 3.2 | ρ=1 时恶意端自身干净精度、body 更新范数；FedRep vs FedAvg 的 benign ASR | 按原文 §3.2 的三条预测逐条判定 | 3 |
| 3.3 | 良性端池化 fresh-PM ASR：窗口内 4 个评估点的最大值（植入峰值）、t0 + 65 / 70 / 75 有效轮三点的均值（稀释），都减同有效轮的 G0-random floor（D-078），随 t0 的变化 | 峰值随 t0 单调上升的秩相关 CI > 0 → 植入受收敛门控（口径细节 D-081：逐 seed Spearman、按 seed bootstrap、`anti_gated` 单独报；`harness/g5_verdict.py`） | 3；✅ **收尾：`not_gated`**（F-076 / D-082） |
| 3-C 衰减（G8，D-075） | 受害 edge 良性 ASR 在第 51–60 轮（= FLR 的 floor 窗口，同有效轮同 lr）减 floor；`harness/decay_verdict.py` | 3 seed 全部 ≥ 0.10 → `persists`（需主动清除）；全部 ≤ 0.05 → `decays_to_floor`（踢出攻击者可能就够）；其余用户定（阈值沿用 D-068） | 3 |

---

## 4. 运行组（P2；预算见本节末，D-047）

| 组 | 配置 | 服务 | 依赖 |
|---|---|---|---|
| FLR floor 验证（D-061） | G6 臂 (a) 原样 × ρ=0 影子攻击者 × 3 seed，与 G6(a) 按 seed 配对；预注册判定 `harness/flr_verdict.py` | 定 G0 的规模；兼作 3-E 臂 (a) 的 floor | A4（只改配置）→ **可交** |
| G0 下限主干 | 划分 {随机, C1, C2, C3, C4} × ρ=0 影子攻击者 × 3 seed；4 edge 集中 [10,0,0,0]、R5（D-066）；**固定 60 云轮**（D-078）。~~规模等 FLR~~ **FLR 判 floor 不可忽略（F-065）→ 逐划分测** | 3.1 下限、3-B 逐 edge 下限、3.3 对照 | ✅ 已回传：floor 跟着 y_t 占比走、floor_gen < 0.3（F-073） |
| G1 主攻击（详细记录；**按「退回 floor」一支重新规划**，D-076） | 4 edge；划分 {随机, C1} × 放置 {collocated, distributed} × R_edge {10, 20} × 3 seed；逐 edge 轮评估；更新日志 | 3-C（**保留，主量用 margin，另报尾部**，D-076；**只当副产品**、加冻结触发器列，D-085）、3-D、3-A 的一部分 | A4 + S3 + S5 + S6。**D-085**：先 S6a + 探路 G1P（约 1 GPU-h），规模等它回来再定（`S6a-PLAN.md`） |
| G2 结构扫描（规模未定，D-056；S5 已完成 → 挂 `g2-scale`，D-084） | flat + edge {2, 4, 10} × R_edge {2, 5, 10, 20}，去掉 G1 已覆盖的格子 × 5 seed（布点 / 轮数 2026-09-27 补齐，D-047）；**先跑 G2P 一致性复测**（pilot 表，seed42 的 4 格），结果回来再定 G2 规模 —— G2P 已回来（`consistent`，F-049），**规模尚未定**，由用户定 | 3-A | A4 + S5 ✅ + `g2-scale` |
| G3 目标类条件 | **C1**（D-062 新增，差中差的基准）/ C2 / C3 / C4 + 层级 Dirichlet α_e {0.1, 0.3, 1, 10}（社区口径，D-063）× 3 seed；4 edge 集中 [10,0,0,0]、R5；比例表 r = 0.25、E3 = deer + horse（D-067） | 3-B | A4 + S3 ✅ → **可交**（24 run） |
| G4 私有头（**搁置**，D-047） | ρ {0.25, 1.0} × {FedRep, FedAvg} × 3 seed。用户：FedAvg 臂会被立刻攻陷、给不出结论；等 3.2 按 N-003 重新表述时一起重设计对照臂 | 3.2 | A4 + S6 + 重新表述 |
| G5 时间窗 | t0 {20, 60, 100, 140, 180} × 20 轮投毒 + 50 轮观察 × 3 seed，从头跑；G0-random 配置，固定到 t0+75（D-078）；生成器语义 A（D-080） | 3.3 | ✅ 已回传（2026-09-30，15 run，6.4 GPU-h）：**`not_gated`**（ρ 0.1 / 0.0 / −0.3，F-076）；攻击者所在 edge 在每个 t0 都饱和；**3.3 收尾**（D-082） |
| G6（可选） | 3-E 的三种划分 × 3 seed；4 edge 集中 [10,0,0,0]、R5、固定 300 有效轮（停止判据关）（D-058） | 3-E | ✅ 已回传；判定 `harness/g6_verdict.py` → b、c 两臂都 **`blocks`**（F-077） |
| G8 衰减（D-075，S9） | G6(a) + 第 1–30 轮投毒（`attack_stop_round: 31`）+ 70 轮（攻击者走后 200 有效轮）× 3 seed；logits 每点、快照 30 / 70 | 3-C 攻击停止版 = 1B-2 的 HFL 复现；floor = FLR 同 seed；决定 G1 怎么改（D-074） | ✅ 已回传：判 `user_decides`（F-068） |
| G6D 探针（D-075，S9） | G6 三臂 × 分散布点 [3,3,2,2] × s42 | 3-E 在没有干净 edge 时还有没有用；go / no-go（低 ≥ 0.15 → 扩 3 seed） | ✅ 已回传：**止步**（只低 0.033 / 0.024，F-069） |
| G5AB 生成器语义（D-079，S4） | G5 的 t20 / t140 × {A = window, B = always} × seed {42, 43} | 定 G5 用哪种生成器语义；预注册 `harness/g5ab_verdict.py` | ✅ 已回传：`insensitive` → A（F-072 / D-080） |
| G8F flat 对照（D-077） | G8 的 flat 版：1 edge、每云轮 1 个 edge 轮、350 轮、第 1–150 轮投毒、每 5 轮评估 × 3 seed | 衰减来自 HFL 结构还是训练协议（对照 1B-2 的 0.45 平台；**更正：约 0.35**，F-074）；预注册 `decay_verdict.py --flat` | ✅ 已回传：flat 也衰减，判 `user_decides`（F-071） |
| G7 预处理对比 | 主配置 × 「官方预处理」（无标准化、无增强）× 3 seed，与主协议同 seed 配对 | 检验「官方设定降低了攻击难度」（D-025） | A4（ε 换算跟随开关，F-027）；在 A4 登记进 `registry.yaml` |

**全部组共用同一条参与量匹配规则（D02）**，否则 T50 不能跨组比较。

**划分（D-065）**：G0-random、G1-random、G2、G5 用**等大小版 random**（每端 Dir(0.5,…)、n 张、无放回、block）；写在各组 `set:` 里，**不改 `base.yaml`**。G6 / G7 保留旧 noniid（组内比较，G6 的 s44 要与 s42 / s43 配对）。

**预算（D-047，2026-09-27）**：上限约 100 GPU 小时 / 周（按 GPU 小时计费）。pilot 实测一个整 run 约 2.3 GPU-h，其中评估约 36%（flat 实测）；run 长度由 pm_acc 平台判据决定（F-046）。在用的杠杆：① 一卡多跑，**先测后用**（`pack.sbatch` + `harness/pack_test.py`，checksum 必须不变、加速比 ≥ 1.5 才采用）；② 副列（白盒 / 陈旧 ASR）降频，**先量后定**（`[TimingASR]` 分项计时）；③ 砍规模：G2 先做一致性复测、G4 搁置。其余组的排期等复测结果回来再定。
**2026-09-27 回收后**（F-048 … F-051）：一卡三跑加速比 2.86 → 每 run 约 0.35 × 单跑机时（D-048）；白盒关、陈旧 ASR 隔点（D-050）。剩余约 139 个 run（去掉已跑的 G7 与搁置的 G4）估约 125 GPU-h ≈ 1.3 周（按 pilot 两格外推，10edge / R20 格没有实测）。

---

## 5. 执行路线（一会话 = 一模块 = 一分支）

| 会话 | 内容 | 门槛 / 解锁 |
|---|---|---|
| **S1**（2026-09-24） | 结果管理 + 分析底座；本计划、AUDIT、DECISIONS、FINDINGS 入库 | — |
| S1b | 修 `exp3_cell.sbatch` 写死的 `--defense none` + 守卫（只有你还要旧方案的防御格时才需要） | — |
| **A1** | 审计：攻击（A01–A06、A14） | 你逐行拍板 |
| **A2**（2026-09-25） | 审计：训练协议（A07–A11、A13、A15、A16、A20、A22、A23，新增 A25）。原则 D-022：攻击定义必须对齐，训练协议不为对齐而对齐；决定 D-022 … D-029 | 已拍板；A08 待 D-029 |
| **A3**（2026-09-25/26） | 审计：FedRep 实现细节 / ResNet-10 / A18 / HFL 形式化 + D01–D06 签字。决定 D-030 … D-038：head = 末层 Dense（D-030）；FedRep 训练顺序两种都跑 pilot（A26，D-031）；BN γ/β 共享、统计量私有（A27，D-032）；主 ASR 与 pm_acc 同模型、都用 fresh-PM（A28，D-033）；ResNet-10 三项对齐（A29，D-034）；A17 / A18 `done`、A21 `deviate`（D-035）；D01 交错执行、D02 按有效轮轮转（D-036）；D03–D06 签字（D-037）；D-029 扩大范围（D-038） | 已拍板；A26 待 pilot |
| **A4**（实现会话；≠ AUDIT 行 A04；2026-09-26 代码 + L1 完成，pilot 待集群） | 按拍板改代码 + L1 测试（D-039：每行一个开关、默认旧行为，对齐项收成一套模板 `fedavg/config/alignment_p2.yaml`）（A1–A3 的全部 `align` 行，含 A26 的顺序开关、A27 统计量私有、A28 fresh-PM、A29 `resnet10_torch`、D01 交错执行、D02 按有效轮轮转）；跑 D-029 可行性实验（D-038：A08 + A25 + A27 + A29 一起开，2edge 那格带上 D01 / D02），通过后 A08 → `deviate`；同批跑 A26 的 2 个 body_first run，按 D-031 的判据关 A26；登记 G7；`PROTOCOL_VERSION` 升 P2；2 个 smoke 复核标定 | AUDIT 全部关闭（A08 由 D-029、A26 由 pilot 关闭） |
| **S3**（2026-09-28 ✅，D-062 … D-068） | 划分：C1–C4 **机构式**比例表（只改 y_t 列，D-062）、层级 Dirichlet（社区口径，D-063）、**等大小 random**（D-065）、H_inter/H_intra（打 `[Partition]` 行）、edge 干净集（500/edge，按 p_e，D-064）、划分 seed 分离。**硬要求：客户端等大小**（D-027，F-028）；无放回；n 默认 500（F-057）。实现：`fedavg/data/designed_partition.py`（不 import TF）、`[Partition]` / `[PartitionEdge]` → `run.data`（schema 6）、`harness/partition_preview.py`（F0 数据，F-058） | → G0 / G3 |
| **S4**（2026-09-29 ✅，D-078 / D-079） | ρ=0 影子攻击者 L1（生成器确实在训、评估触发器用生成器）、`attack_start_round` + `generator_schedule`（`attack/attack_window.py`，不 import TF；`[设定7]`，schema 8）；~~ξ-only 下限~~ 不做（D-078；δ-only / ξ-only 逐点消融列也不做，D-051） | → G0 / G5AB（G5 等 G5AB） |
| ~~S5~~ ✅（2026-10-01，D-084；预案 D-055） | 统一评估网格 `evaluation.eval_grid: G`（`fedavg/server/eval_grid.py`）：全量评估在网格上的云轮末（两个 eval_interval = lcm(G,R)/R），其余网格点在 edge 轮之间做**轻评估**（只算主列：fresh-PM 的 local / edge ASR + pm_acc + EM 精度）；停止判据横轴 = eff / G（R5 逐位不变）；GM 只在全量点。collect_metrics schema 9（`light_rounds[]` / `per_edge_light_rounds`）。G2 写好网格但挂 `g2-scale`；G1 的网格留给 S6；探路组 **S5P** 已回传：GPU 上开 / 关网格逐位相同、轻评估点约 35 s（F-079） | → G1 / G2 |
| S6 → ~~S6a~~ ✅（2026-10-02，D-085 / D-086）/ **S6b**（待 G1P 回传再定） | 原定：更新日志、几何分数（body-only、滑窗）、周期转储、离线 c_k、恶意端干净精度（后者 S9 已做）。**D-085 拆分**：S6a = 逐更新几何日志（标量进 metrics.json、草图进 dumps）、云聚合后评估点、冻结触发器列、轻评估点带 margin / 尾部、停止横轴防呆、G8 快照上的 c_k 预检 + 登记 G1P —— **已实现**（collect_metrics schema 10；`[设定9]`；GPU 待 G1P）；S6b = 在线 c_k（G1P 回来再定）。方案 `S6a-PLAN.md` | → G1P → G1 / G4 |
| S7（**部分完成**，2026-09-30） | 各子实验的判定代码补全 + 出图。✅ 3-E 判定 `harness/g6_verdict.py`（F-077，两臂 `blocks`）；✅ 已回传各组的结果图 `harness/report_figures.py`（REPORT §0）。剩下：等 G1 / G2 / G4 的数据（F2 / F3 锯齿 / F5 / F6） | — |
| **S9**（2026-09-28 ✅，D-071 … D-075） | 评估仪表（常开：逐客户端、margin、y_t 偏置、按类 ASR、非目标翻转率；schema 7）+ 两个存盘开关（logits / 快照，默认关）+ `harness/decay_verdict.py` / `instrumentation_check.py`；登记 G8 / G6D | → G8 / G6D；G1 待 FLR + G8 重新规划（D-074） |
| **S8**（2026-09-27 ✅，D-057 … D-059） | 3-E 三层个性化 | → G6 |

---

## 6. 图目录（出图规则见 README）

| ID | 内容 | 子实验 | 状态（`figures/final/`，`harness/report_figures.py`） |
|---|---|---|---|
| F0 | 各划分的 (H_inter, H_intra) 散点 | §2 | ✅ `F0_partitions.png`（`partition_preview.py --plot`） |
| F1 | ASR 与 floor_gen 随有效轮的曲线，三层（~~floor_ξ~~ 不做，D-078） | 3.1 | ✅ `F1_floor_G0_FLR.png`（池化 floor 曲线 + 逐 edge floor vs y_t + FLR） |
| F2 | T50(HFL)/T50(flat) 森林图，按 (n_edges, R_edge) 分组 | 3-A | 等 G2（pilot 版：`G2P_T50_ratio.png`，单 seed） |
| F3 | 逐 edge × 有效轮的 fresh-PM ASR 热条，标出云聚合时刻，叠加锯齿 | 3-C | 衰减版 ✅ `F3_decay_G8_G8F.png`（G8 / G8F）；锯齿热条等 G1 |
| F4 | C1–C4 逐 edge 的 excess ASR 与 MTA | 3-B | ✅ `F4_3B_G3.png`（C1 / C3 逐 edge 原始 vs floor + 差中差分解；MTA 未画） |
| F5 | edge 视角 vs 全局视角的 ROC（等池大小） | 3-D | 等 G1（S6） |
| F6 | ρ × {FedRep, FedAvg}：benign ASR、恶意端干净精度、body 更新范数 | 3.2 | 等 G4（搁置） |
| F7 | t0 窗口网格：峰值 ASR 与 +50 轮的 ASR | 3.3 | ✅ `F7_G5_convergence_gating.png` |

目录外另有 `3E_G6_G6D.png` / `3E_verdict_G6.png`（3-E）、`F3b_decay_margin_tail.png`（3-C 的 margin 与长尾）、`F4b_3B_all_partitions.png`（3-B 全部 24 run）、`G5AB_generator_semantics.png`、`G7_preprocessing.png`、`pilot_A4.png`、`status_progress.png`（进度与机时）（2026-09-30）；每张图的读法见 `REPORT.md` §0 / §5 / §9.1。
