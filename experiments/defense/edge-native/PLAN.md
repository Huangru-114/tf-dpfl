# 阶段三（edge 原生防御）实验计划 —— 定稿 v1（2026-10-09）

> **状态**：定稿（本目录 `DECISIONS.md` D-089 … D-095）。**D0 已完成（2026-10-09）**：登记表 `registry.yaml`（SNAP 可交；CAL / CCSF 占位）、
> 提交脚本 `submit.sh`、功效分析 F-090、SNAP 核对 F-091、文献核对 F-092、预注册草案 N-008（待用户确认）。未实现、没有交任何作业。交接见 `current-focus.md`。
> 台账：本目录 `DECISIONS.md` / `FINDINGS.md`（编号接续 `experiments/attack/hfl-mechanism/`）；文献：`LITERATURE.md`。
> `PLAN-draft.md`（2026-10-03）已被本文件取代。

## 来历

输入：`PLAN-draft.md`（三份方案 + 两个评审）、`experiments/attack/hfl-mechanism/REPORT.md` §1 / §9.4 / §10、FINDINGS F-061 … F-086、
原始规划 `PLAN-original-2026-09-24.md` §9 / §10、2026-10-09 会话的代码与数据核对，以及用户提供的 **Bad-PFL 原文**（ICLR 2025）、
**CCS 原文**（Cui et al., ICASSP 2026）、**用户的文献综述**（2026-09-30，全文收在 `LITERATURE.md`）。
讨论过程中用户的批注与处理见附 A；拍板见 `DECISIONS.md`：

- D-089：解除 D-008，阶段三 = 本计划；
- D-090：台账放本目录、编号接续；
- D-091：默认 3 seed + 预注册的加 seed 规则（§3.5）；
- D-092：可选组 CAL / FLAT / DATA 与第二攻击（CerP + IBA）全部纳入，不加 PFedBA；
- D-093：基线 = Median / Trimmed-mean、Multi-Krum、FLAME（按文献修好）、FedBAP-ℓ∞、SHIELD、RLR；CCS = 独立主线 + 消融；
- D-094：加固候选按文献改为对抗训练家族；删 NAD 式蒸馏；干净微调降为归因对照；Simple-Tuning 删除；
- D-095：3.2「私有头吸收」不进阶段三的证据范围。

## 0. 范围与固定项

- **威胁情形**：真正开放的是 ① 分散布点、② 攻击者所在 edge 的良性端。只在「集中 × 受害 edge」上有效的东西与 3-E 重复。
- **防御方**：edge 可信、持 500 张干净图（S3 `clean_per_edge`，与客户端不相交）、有算力；cloud 只看到 edge 聚合（PriRoAgg 的 aggregated privacy）。
- **主攻击**：Bad-PFL（ε = σ = 4/255、ρ = 0.2、10% 恶意端），评估用固定攻击者（`[设定5]`）。主线非自适应；AA 组自适应；ATK2 换攻击。
- **口径**：P2 模板全开；主列 fresh-PM（D-033），陈旧列只报告；网格 5 有效轮；停止判据关、固定 300 有效轮。
- **D-008 已解除**（D-089）。

## 1. 决定性的事实

### 1.1 文献（读原文后）

1. **Bad-PFL 的触发器 = 两个 ℓ∞ 对抗扰动之和**（原文 Eq. 5–7）：δ = ε·G_w(x) 是**定向**对抗扰动（推向 y_t，ε = 4/255），ξ = σ·sign(∇L) 是**非定向** FGSM（σ = 4/255）
   → ‖δ + ξ‖∞ ≤ 8/255。消融（表 4 / 表 18，FedRep）：去掉 δ → 12.82%，去掉 ξ → 79.32%，两者 → 97.95% —— **δ 是主体**。
   → 对抗训练的扰动集必须覆盖 8/255（综述：SAU 在触发器范数超出扰动集时失效），且要对**定向**扰动鲁棒，不只是非定向。
   **剂量反应**（表 18，FedRep）：δ 的预算 ε = 0 / 1 / 2 / 3 / 4（/255）→ ASR 12.82 / 54.92 / 79.68 / 86.79 / 97.95%；σ = 0 → 79.32%。
   → 加固**不必**做到完全鲁棒：只要把 δ 的「有效预算」从 4/255 压到约 1/255，按这条曲线 ASR 就会降到约 55%（推断：曲线是攻击者改 ε 测的，不是防御方改鲁棒性测的）。
   **代价风险**：ε = 8/255 的完整对抗训练在 CIFAR-10 上通常要付出明显的干净精度（鲁棒性–精度权衡，Tsipras 2019 / TRADES，凭记忆）→ ΔMTA ≤ 0.02 的门槛可能卡住，`costly` 是一个现实的结果。
2. **I-BAU 也失效（76.58%）**：它反学的是**一个通用扰动**，而 Bad-PFL 的触发器是逐样本的 → 草案的 O3（类级通用扰动遗忘）文献不支持，改成**逐样本**的定向 PGD 对抗训练（AT-tgt）。
3. **CCS 是两个组件的组合**：CCS = 客户端对抗训练（CE + β·KL + γ·MMD，**β = γ = 0.01**，对抗样本用可变形 patch 初始化 + max-margin 精炼）
   + server 对客户端上传的 **BN running_mean / running_var** 做 HDBSCAN 聚类剔除，再无权 FedAvg。
   **消融的出处待确认**：用户记得原文做过消融、对抗训练约占 85% 的效果；2026-10-09 会话对上传的 ICASSP 2026 版本（约 4 400 词）做了全文检索 ——
   没有「ablation / w/o / 85%」，实验部分只有表 1（防御对比）与图 2（α 敏感性），**这一份里没有消融**。可能在别的版本（期刊扩展版 / 补充材料）里 → 请上传；
   在拿到之前，计划按「组件贡献未知」设计（§4 CCSP 三臂），拿到之后把它写成 CCSP 的**预注册预测**（CCS-AT 的 ΔV ≈ 0.85 × CCS-full）。
   而我们 F-085 测到：BN 统计量通道单独 AUROC 0.90（`norm_s`，TPR@5% 约 0.68），并且 FedRep 下统计量私有、上传值不进任何受害者模型 → **攻击者可零代价伪造**。
   CCS 的威胁模型明写「攻击者不知道防御」。→ 即使原文（flat、FedPer）里对抗训练占大头，**本设定（FedRep、统计量私有）下聚类的份额可能更大**，
   而聚类是可伪造的 —— 这是 CCSP 三臂与 AA-S 要回答的。
4. **flat 文献里几乎所有已知防御都挡不住 Bad-PFL**（表 2 / 3 / 21，FedRep）：ClipAvg 97.28、Multi-Krum 96.15、Median 77.21、Sign 20.32（但 Acc 34.49，训练崩）、
   FLAME 79.05（CCS 表 1，FedPer）、BAERASER 91.54、MAD 90.74。→ H5 的大部分基线**预期失败**；用 s42 筛选省机时（§4 H5）。
   检测类也弱（表 22）：Neural Cleanse 异常指数 2.2（其余攻击 4.9–5.8）、STRIP 熵 0.77（干净样本 0.92）—— 与我们 3-D「edge 视角没有检测优势」同向，支持不做模块 B。
   原文附录 D 认为「唯一可想到的对策是不含目标类数据地微调」（会伤目标类精度）；**原文没有讨论对抗训练**。

### 1.2 仓库（代码与数据核对）

1. **P0 不能用 G8 快照**：G8 / G6 / G6D 用旧 `noniid` 划分（`base.yaml:42`），**没有 edge 干净集**（`clean_indices` 只在 S3 划分下才有，`main.py:506 / 774`）；
   F-081 拿留出分片当干净数据 = 在探针上训练（陷阱 #11）→ 新开 **SNAP**，兼作所有主检验的对照臂。
2. **对照臂与 floor 大部分现成**：G1R5（C1 × 集中 × R5 × 300 有效轮，s42–44）与 G0-C1（同配置 ρ=0）；记录 / 快照开关已证明不改变训练（F-084 / D-073）。
3. **动态范围**：G1R5 受害 edge 良性 ASR 第 2–12 云轮 0.03–0.48、第 17 轮 0.66–0.95、第 30 轮后 ≥ 0.95；分散（G6D(a) s42）第 1–12 轮 0.27–0.70、第 13 轮起 ≥ 0.87 → 快照取 **6 / 15 / 60**。
4. **第二攻击接线**：P2 `per_epoch` 下静态投毒被拒；CerP 是动态投毒 mixin，但触发器训练仍读 tf.data（AUDIT A25）；IBA 无实现；
   `post_agg_eval` / `frozen_trigger` 只支持 Bad-PFL（`config_validate.py:511`）。

---

## 2. 研究问题 → 证据 → 文献

### 2.1 研究问题

| RQ | 问题 | 阶段二证据 | 文献 |
|---|---|---|---|
| D1 | edge 上传前用 500 张干净图做**对抗训练**，能否让每次云聚合的灌入（Δ_jump）小于周期内洗掉的量？ | 灌入 0.13–0.27 / 周期、净升 0.02–0.05（F-085 / F-086）→ 只需砍掉 Δ_jump 的约 20–40% | 正面：CCS（客户端 AT，flat）；SAU（小干净集、逐样本对抗反学习）；FLIP、FedBAP（FL 内对抗扰动）。反面：Bad-PFL 表 3 / 21（FT / NAD / I-BAU / ST / BAERASER 全失效） |
| D2 | 一直开着的在线加固能否让干净 edge 保持干净（集中布点）？ | 没有安全时段（F-076）；3-E 的门槛（F-077，D-071） | HFL 内没有用优化型触发器评估过的防御（综述方向二：SHIELD / RoHFL / DARCS / FLVaccin 都是投毒或固定触发器） |
| D3 | 能否动分散布点、攻击者 edge 内的良性端？ | 分散时 3-E 无效、ASR 0.999（F-069）；E0 良性端 0.92–1.0（F-077） | 空白（综述方向三） |
| D4 | 对抗训练放在哪里：客户端（CCS）/ edge / cloud / flat server，效果与代价各多少？「edge 原生」有没有增益？ | cloud 只看到聚合（REPORT §10.4）；flat 与 HFL 衰减相同（F-071） | CCS（客户端）、FLTrust / FeRA / ADFL（防御方持少量干净数据是惯例） |
| D5 | 在 HFL × FedRep 下，CCS 的两个组件（客户端对抗训练 / edge 侧 BN 统计量聚类）各贡献多少？与原文消融（用户记得对抗训练约占 85%，出处待确认）一致吗？ | `norm_s` AUROC 0.90（F-085）→ 本设定下聚类单独可能就很强 | CCS（上传版本无消融）|
| D6 | 对每个在非自适应攻击下有效的防御臂：换成**知道这个防御**的攻击者（AA1 / AA2 / AA-S，定义见 §4 AA）后，防御带来的受害端 ASR 下降还**保留多少**（保留率 = ΔV_adapt / ΔV_nonadapt）？ | 阶段二全部非自适应（REPORT §10.5）；BN 统计量可零代价伪造（F-085） | A3FL（NeurIPS 2023，综述：自适应攻击者的直接模板）、EOT（Athalye 2018）、DPOT（更新隐匿） |
| D7 | 与 3-E edge 段组合：叠加还是冗余？3-E 面对分段感知攻击者是否仍成立？ | 3-E 集中时 `blocks`（F-077），旧划分 | Simple-Tuning（KDD 2023，部分共享阻断传播）、关键层投毒（LP，ICLR 2024） |
| D8 | 效果是否只对 Bad-PFL 的 ℓ∞ 对抗式触发器成立？ | δ / ξ 都是 ℓ∞ 对抗扰动（Bad-PFL Eq. 5–7） | CerP（AAAI 2023，可训练 patch 式触发器）、IBA（NeurIPS 2023，生成器 ℓ∞ 噪声触发器 + 低重要性参数；Bad-PFL 表 21 中 IBA 在 FedRep 下 ST 后 49.31%） |
| D9 | 比已知防御好吗？ | 陷阱 #2 / #3 / #9 未修 | Bad-PFL 表 2 / 21、CCS 表 1：Multi-Krum / Median / FLAME / ClipAvg / Sign 在 flat 下都挡不住或训练崩 |

### 2.2 关键文献的数字（2026-10-09 读原文 / 用户综述；摘录见 `LITERATURE.md` 第 2 部分）

| 文献 | 设定 | 对 Bad-PFL 的结果（ASR，CIFAR-10） | 对本计划的含义 |
|---|---|---|---|
| Bad-PFL（ICLR 2025）表 2 / 3 / 21 | flat、FedRep、ResNet-10、100 端 / 10 恶意、Dir 0.5、15 步、1000 轮 | 无防御 97.95；FT-15/30/45 97.31 / 97.76 / 97.01；NAD 86.25；I-BAU 76.58；ST 88.82；BAERASER 91.54；MAD 90.74；ClipAvg 97.28；Multi-Krum 96.15；Median 77.21；Sign 20.32（Acc 34.49） | 干净数据微调 / 蒸馏 / 通用扰动反学习 / 鲁棒聚合在 flat 下都不够；**对抗训练没测过** |
| CCS（ICASSP 2026）表 1 / 图 2 | flat、**FedPer**、ResNet-18、100 端 / 10%、Dir 0.5、2 步、1000 轮；β = γ = 0.01 | FedPer 无防御 94.42 → CCS **8.54**（MTA 79.76）；同表 Multi-Krum 80.78、FLAME 79.05、ST 81.87、FLIGHT 84.17、SARS 82.87、PFL-ALB 86.69 | 唯一的正面证据；两个组件；上传版本没有消融（用户记得有：对抗训练约占 85%，出处待确认）；威胁模型非自适应 |
| SAU（NeurIPS 2023，综述已读全文） | 集中式，5% 干净数据，L∞ ≤ 0.2 的 5 步 PGD | 未测 Bad-PFL；对逐样本触发器（WaNet / SSBA）有效；**触发器范数超出扰动集即失效** | edge-AT 的 ε 必须覆盖 8/255；SAU 作候选之一 |
| FedBAP（ACM MM 2025） | 客户端良性对抗扰动；MaskGen 假设局部 patch | 未测 Bad-PFL；其实验里 FLAME / FLTrust / FLIP 面对 A3FL 接近 100% | 基线需改写成 L∞ 球内 PGD（综述第 2 条） |
| IBA（NeurIPS 2023） | 生成器 L∞ 噪声触发器 + 只毒化低重要性参数 | Krum / RLR 能部分压制（CIFAR-10 27% / 64%，综述） | 第二攻击；RLR 入基线 |
| SHIELD（TDSC 2025，综述已读全文） | 三层、逐层递归 HDBSCAN、上传多个簇均值 | 只测标签翻转 + 强度型模型投毒 | HFL 原生基线 |

**定位**（综述结论）：HFL 安全方向几乎都用弱威胁模型、不带个性化；「HFL + PFL + 优化型 / 自然特征触发器」是空白。离本项目最近的 FLVaccin（层级 FedPer）只报精度、没有 ASR；
ACISP 2026 的「edge」实为两层 FL 的客户端。模块 A 要与 FedBAP / CCS 正面区分三点：**执行位置在可信 edge、客户端零开销；用 edge 的干净数据；扰动模型匹配 Bad-PFL 的 ℓ∞ 触发器。**

### 2.3 阶段二证据的采纳与弃用（证据 → 设计的追溯）

#### (a) 作为科学证据采纳的结论 → 导向的设计与组别

| # | 阶段二结论（强度） | 关键数字 | 导向的设计 | 组 |
|---|---|---|---|---|
| E1 | 3-C 锯齿：后门只经云聚合跨 edge 传播，edge 内自清洁追不上灌入（✅ F-085；探索性 F-086） | Δ_jump 0.13–0.27（R5 0.197）；净升 0.02–0.05 / 周期；margin 每周期推 1.9–3.0、洗 1.7–2.5 logit | ① 干预点在**云聚合前**（`A-pre`）；② P0 主量 = 灌入削减率 R_H，门槛 0.5 / 0.2 由「只需砍掉 20–40%」推出；③ P1 的 `go_main`（Δ_jump 减半、Δmargin ≥ 2 logit）；④ `D-base` 开 `post_agg_eval` | P0、P1、H1、H3、CLD |
| E2 | 3-C 的稳健性：R5 / R10 标签离阈值 2.6–7 SE，R20 在噪声里（✅ F-085 统计复核） | — | `D-base` 用 **R5**：G1R5 有 3 seed 现成对照；300 有效轮里 60 次云聚合 = 灌入事件最多（推理：最严的检验） | 全部 |
| E3 | 攻击者 edge 的反向锯齿：每次云聚合被稀释、周期内以 0.033–0.041 / 有效轮爬回（探索性 F-086）；攻击者 edge 在任何 t0 都一开窗就饱和（✅ F-076，峰值时 E0 良性 ≥ 0.6） | — | 只在上传前加固**碰不到 E0 的良性端**（每个周期被本 edge 攻击者重新污染）→ 加 `A-every` 与客户端侧 AT（CCS 各臂）；H3 事先写明 `A-pre` 预期 `no_effect` | P1、CCSP、H1（`treats_E0`）、H3 |
| E4 | 3.3 `not_gated`：没有安全时段（✅ F-076） | ρ(t0, 峰值 excess) 0.1 / 0.0 / −0.3 | 所有防御**从第 1 轮一直开**；不设「后期才开」或时间窗类的臂 | 全部在线组 |
| E5 | 3-D：edge 视角无检测优势；余弦、单更新 c_k 不可检测（✅ F-085） | norm_w ΔAUROC ≤ 0.031、TPR@5% 0.28–0.36；cos / c_k AUROC 0.50–0.64 | 不做依赖识别恶意端的防御（模块 B 删除）；P0 的反事实是「**所有 edge 都加固**」 | P0、全部 `A-*` |
| E6 | 3-D：最强的单更新信号来自 BN 统计量，且可零代价伪造（✅ F-085 + 代码证据） | norm_s raw AUROC 0.90–0.91、TPR@5% 约 0.68 | ① 推断：本设定下 CCS 的聚类在非自适应攻击下单独就可能有效 → 组件份额可能与原文（flat、FedPer）不同 → CCSP 三臂在本设定下复核；② **AA-S**（伪造统计量）直接检验聚类部分 | P0（离线检测）、CCSP、AA |
| E7 | body 级 c_k：c_{y_t} ≈ 0、其余类 0.1–0.5；分不出 edge（F-081 + 更正：在不同 body 上测的 → 弱反面证据） | c_0 0.000–0.032 | ① AT-tgt 以 k* = argmin c_k 定向，不需要知道 y_t；② 跨 edge 差分定位不做主线，只在 P0 用正确的干净集复测 | P0 |
| E8 | 3-C 衰减：攻击者走后约 200 有效轮褪到 floor 附近、margin 无平台、留 2–6% 长尾；flat 一样（✅ F-068 / F-071） | retention 约 0.15 | ① 干净数据训练会洗掉后门，但攻击者在场时被灌入抵消（E1）→ 光靠干净微调不够、需要**对抗**部分 → C-ft 只作归因对照；② flat 与 HFL 相同 → 需要 **FLAT** 才能说「edge 原生」；③ 逐客户端 ASR 分位数一直报告 | P0、FLAT、全部 |
| E9 | 3-E：集中时 edge 段 `blocks`，分散时无效，治不了 E0 良性端（✅ F-077 / F-069） | ΔV 0.52 / 0.73，fresh ΔMTA ≤ 0.02、陈旧 0.027–0.029；分散 Δ −0.03、P 0.999、margin 7.47 | ① 开放的威胁 = **分散 + E0 良性端** → H3 为主检验、B0 必报；② 门槛沿用（ΔV 0.15、fresh ΔMTA ≤ 0.02、陈旧只报告）；③ G6 是旧划分 → H2 先在 C1 复现，再加分段感知攻击者 | H1、H3、H2 |
| E10 | floor 不可忽略、随本 edge 的 y_t 占比走（✅ F-065 / F-073） | floor 0.04–0.13；floor_gen 0.04–0.10 | ① 报 excess；② **防御可能连 floor 一起压低**（对抗训练就是降对抗脆弱性）→ 有效臂测自己的 floor，「清除后门」须 Δexcess ≥ 0.10；③ 对照 floor 复用 G0-C1；④ 用 C1（四个 edge 的 y_t 占比都是 0.10）→ edge 间 floor 可比 | H1、H3 |
| E11 | 攻击接近饱和，3-B 因天花板判不出（✅ F-073 / F-075；G1R5 受害 0.994–0.998） | — | ① **CAL** / W2；② 天花板规则（两臂都 ≥ 0.95 改读 margin）；③ 饱和时 margin 仍在动（F-086）→ margin_p50 作后备主量 | CAL、W2、H1、H3 |
| E12 | 白盒 ASR ≈ 主列：私有 head 挡不住 ξ（✅ F-051） | 差 ≤ 0.007 | ① 防御必须作用在**共享 body**；② 预测 Simple-Tuning 无效（已被 Bad-PFL 表 21 / CCS 表 1 证实）→ 删；③ 只用主列（固定攻击者），白盒保持关 | 全部；P0 |
| E13 | fresh-PM 低估干净精度，随拓扑变（✅ F-051） | +0.011 … +0.094 | ① 精度门槛用 fresh、陈旧列必报；② P0 一次性反事实里 head 没适应新 body → 精度过滤偏保守 | P0、全部 |

#### (b) 作为协议 / 工具采纳的结论（决定「怎么跑才可信」）

| 结论 | 出处 | 用途 |
|---|---|---|
| P2 下同配置同 seed 逐位可复现（每 run 4 核） | F-045 / F-047 / F-084 | 按 seed 配对；SNAP-col 须与 G1R5 逐位相同；P1 在第一次加固前与 SNAP 相同 |
| 记录 / 快照 / 网格开关不改变训练 | F-064 / F-067 / F-079 / F-081 / F-084 | SNAP 兼作对照臂 |
| GPU 确定性 × 推理模式 BN 求梯度会崩，崩前恶意端被**静默**踢出（陷阱 #23） | F-043 / F-045 | **对抗训练正是在推理模式 BN 上求梯度** → 新代码走 `TorchBatchNorm` 推理路径、先交 10 分钟探路作业；`client_failures` 非空 = `invalid`（CCS 的客户端 AT 若被吞，良性端会被静默剔除） |
| Python random 是未播种的扰动通道 | F-078 | SA0 / SA1 / SA-C 的新代码包 random 围栏 + 反向锚点 |
| 评估约占 36% 墙钟 | F-046 | 单价与开销外推 |
| K=3 一卡三跑、显存 / 主机内存规则 | F-048 / F-055 / F-059 / F-060 | 包的大小；客户端 AT 这类新配置先单独探显存 |
| 旧方案 `def_*` 格**根本没开防御** | F-001（陷阱 #19） | 本仓库对 Median / Multi-Krum 等**没有任何有效数据** → H5 是第一次测；作业脚本只传 `--config` |
| 陷阱 #9 / #2 / #3（代码证据） | CLAUDE.md | SA3 的前置修复 |

#### (c) 没有采纳的结论及原因

| 结论 | 出处 / 强度 | 原因 |
|---|---|---|
| 3-A：HFL 结构延迟（4edge < 2edge < flat < 10edge） | F-049，单 seed 🔸 | 证据弱（单 seed、幅度随协议大变、原因未知）；阶段三固定 4 edge。「防御效果是否随 edge 数变」记为局限 |
| 3-B：目标类分布的差中差 | F-066 / F-073 / F-075，⚠ | 天花板下判不出；C2 / hdir「受害 edge 偏低」只是探索性读数 → 只用 C1；**只采纳方法论教训**（E11） |
| 3.3 的细节（稀释量、各 t0 的数） | F-076 / F-072 | 稀释值贴着 floor、没有动态范围；只采纳 `not_gated` |
| G5AB：生成器语义无差别 | F-072 | 阶段三不设投毒窗口 → 不相关；只借用「ρ=0 影子攻击者」的语义定义 floor |
| G7：官方预处理让攻击更强 | F-050，事后判据 | 规则写于数据之后、精度差 0.10 混杂 → 保持 P2 标准预处理 |
| 3-C 衰减的数值细节（与外部 1B-2 的差、停手后首轮回升） | F-068 / F-074 | 协议差异没有逐项证据；与防御设计无关 |
| R20 与 random·R20 的 3-C 标签 | F-085 统计复核 | 在噪声里 → 只用 R5 |
| lr 衰减对 r_down 的贡献 | F-085 / F-086 | 分不开；各臂 lr 日程相同、按 seed 配对 → 不影响阶段三的比较 |
| 几何分数（norm_w）、余弦、单更新在线 c_k 及参数敏感性 | F-085 | 无检测优势 → 模块 B 删除；`update_geometry` / `update_ck` 在 `D-base` 关掉；MARS 不加 |
| C1 与 random 的对比 | F-085 | 两种划分在 3-C / 3-D 上定性相同；阶段三的前提是「edge = 机构、持本机构分布的干净数据」→ 只用 C1；非机构式 edge 记为局限 |
| G6 / G6D / G8 的数值本身 | F-061 / F-069 / F-068 | 旧 `noniid` 划分、无 edge 干净集 → 不能当对照，只作先验与效应量参照；G8 存盘不用于 P0 |
| G6(a) 受害 edge 到 300 有效轮仍在上升 | F-062 | 旧划分；C1 × R5 在 150–200 有效轮已饱和 |
| 旧方案 P1 / P0 的全部结果 | F-002 … F-011 | 协议已被 P2 取代，只作背景 |
| AUDIT 对齐细节 | F-012 … F-040 | 已吸收进 P2 模板，不单独产生设计 |
| 官方 ρ=1 时 PGD 与生成器抵消 | F-013 | 只与 ρ=1 有关；阶段三 ρ = 0.2 |

（3.2「私有头吸收」不在阶段三的证据范围内：按用户意见忽略这一组，相关结论见 Simple-Tuning（KDD 2023）与 Bad-PFL §2.2 / 表 12。）

#### (d) 计划里没有阶段二证据支撑的部分（只靠文献或新假设）

- 对抗训练在本设定下是否有效、精度代价多大（只有 CCS 的 flat 结果与 Bad-PFL 攻击者侧的剂量反应）→ P0 → P1 / CCSP 是第一次测；
- cloud 侧 vs edge 侧的差别、客户端 AT 的开销、500 张干净集够不够；
- CCS 聚类在 C1 下的误报（edge 内 Dir 0.5 的异质性会不会让良性端被当成离群；REPORT §9.3：y_t 富集良性端的误报没测过）→ P0 离线先看 FPR；
- 任何自适应攻击者下的读数；CerP / IBA 在 FedRep × HFL 下能否植入；已知防御在本管线里的表现（F-001）。

#### (e) 反查：每组的依据

| 组 | 阶段二依据 | 文献依据 |
|---|---|---|
| SNAP | F-081 更正、F-084、G1R5 轨迹（仓库事实：G8 无干净集） | — |
| P0 | E1 E5 E6 E7 E8 E12 E13 | Bad-PFL 表 3 / 18、SAU、CCS |
| CAL / W2 | E11 | Bad-PFL 图 4 |
| P1 | E1 E3 E4、陷阱 #23 | — |
| CCSF / CCSP | E3 E6 | CCS 表 1、Bad-PFL 表 21 |
| H1 | E1 E3 E9 E10 E11 E13 | CCS |
| H3 | E3 E9 E11 | — |
| CLD | 无实验证据（REPORT §10.4 的推断） | FLTrust / FeRA |
| FLAT | E8 | CCS（flat） |
| DATA | 无 | SAU、FLTrust |
| AA | E6、REPORT §10.5 | A3FL、综述 |
| H2 | E9 | Simple-Tuning（KDD 2023）、LP |
| ATK2 | E12 | Bad-PFL 表 21、IBA、CerP |
| H5 | F-001、陷阱 #9 / #2 / #3 | Bad-PFL 表 2、CCS 表 1、IBA |

## 3. 共用协议

### 3.1 配置基线 `D-base`

= G1R5 的 `set:` 去掉 `update_geometry` / `update_ck`，加 `post_agg_eval: true`、`frozen_trigger: true`：
4 edge × 25 端，C1 设计划分（strength 0.25，每端 500 = 375 / 125，`clean_per_edge` 500），R5 × 60 云轮 = 300 有效轮，网格 5，`stopping: null`。
布点：**col** = `[10,0,0,0]`（E0 = 攻击者 edge，E1–E3 受害），**dist** = `[3,3,2,2]`。

### 3.2 对抗训练家族（P0 只在 s42 上筛，冻结后再读 s43 / s44）

edge 侧：作用在**上云的可训练 body**（不碰 head、不碰私有 BN moving 统计量）；edge 没有 head → 每次在冻结 body 上拟合临时线性探针 head，用完丢弃。

| 代号 | 做法 | 依据 |
|---|---|---|
| AT-pgd | Madry 式 PGD 对抗训练，非定向，ε ∈ {4, 8}/255 | 覆盖 ξ（4/255）与 δ + ξ（8/255） |
| AT-trades | TRADES，ε = 8/255，β ∈ {1, 6} | 显式权衡干净精度 |
| AT-tgt | **逐样本**定向 PGD：推向 body 级 c_k 最小的类 k*（ε = 4/255，模拟 δ）+ 非定向一步（σ = 4/255，模拟 ξ），在扰动样本上训正确标签 | §1.1 事实 1–2；F-081：body 级 c_{y_t} ≈ 0、其余类 0.1–0.5 → 不需要知道 y_t |
| AT-sau | SAU：在「进来的 edge 模型」与「正在净化的模型」之间找共享对抗样本并反学习，ε = 8/255、5 步 | SAU（小干净集、逐样本触发器有效）；官方 BackdoorBench 配置 `trigger_norm: 0.2`、5 步（F-092）→ ε 的差别进 SA0 语义 diff |
| AT-ccs | CCS 的目标函数搬到 edge：max-margin 对抗样本 + β·KL + γ·MMD（投影网络） | 检验「CCS 的对抗训练部分」放在 edge 上是否成立 |
| C-ft | 干净微调（同数据、同步数、无对抗部分）—— **归因对照，不是候选** | Bad-PFL 表 3：FT 无效 |

× `bn` {train, frozen} × 预算 {低, 高}（epochs × PGD 步数）→ 约 24 个配置（离线筛选，零 run 成本）。

### 3.3 臂（在线）

| 臂 | 谁做对抗训练 | 数据 | 客户端开销 |
|---|---|---|---|
| `A-pre` | edge，上传 cloud 前 | edge 的 500 张 | 0 |
| `A-every` | edge，每个 edge 轮聚合后、下发前（唯一能碰到 E0 良性端的 edge 侧做法） | 同上 | 0 |
| `A-cloud` | cloud，聚合后（**对照**：cloud 持原始数据违背部署设定） | 4 × 500 = 2000 张 | 0 |
| `CCS-full` | 良性客户端 AT（CCS 目标）+ **edge** 对本 edge 客户端的 BN 统计量做 HDBSCAN 剔除（cloud 看不到单个客户端，只能放 edge） | 客户端本地数据 | 高（外推 ×2.5–4 run 墙钟） |
| `CCS-AT` | 只有客户端 AT | 同上 | 高 |
| `CCS-clu` | 只有 edge 侧 BN 统计量聚类剔除 | — | 0 |

新开关（SA1 / SA-C 实现，名字暂定；缺省关 = 逐字节不变）。

### 3.4 指标（collect_metrics 现成）

- **V**：受害 edge（E1–E3）良性 ASR，末 10 个评估点均值再对三个 edge 平均（同 `g6_verdict.py`）；**B0**：E0 良性端 ASR；**P**：分散布点全部良性端池化 ASR；
- **margin_p50**（饱和时的连续量）、**Δ_jump / r_down**（`post_agg_rounds[]`，N-007）、冻结触发器列（拆开「body 变了」与「触发器漂移」）；
- **MTA**：fresh pm_acc 末 10 点（主），陈旧列只报告；
- **excess** = ASR − 同臂 ρ=0 floor（对抗训练本来就降低对抗脆弱性 → 可能连 floor 一起压低，D-059 的教训）；
- CCS-clu / CCS-full：`admitted[]` / `rejected_ids`（`[Decision]` 行，陷阱 #10 已统一）→ 剔除的 TPR / FPR；
- 开销：`timing_summary`（客户端 / edge / cloud 分项）—— 「客户端零开销」要有数字。

### 3.5 统计、加 seed 规则、有效性闸

- 按 seed 配对（P2 下划分、恶意端 id、评估攻击者只由 seed 决定，F-045）。筛选 = s42 单 seed；**其余一律先 3 seed（s42–44）**。
- **3 seed 判定**：最小 seed 值（= bootstrap CI 下界，REPORT §2）+ 均值。
- **加 seed 规则（预注册，防「加到显著为止」）**：
  1. 只在两种情况下加到 5 seed（s45、s46，**只加一次**）：(a) 3 seed 标签 ∈ {`partial`, `inconclusive`, `degraded`, `user_decides`} 且该组承载结论（H1 / H3 / CLD / AA / ATK2）；
     (b) 用户在看**标签之前**指定该组为论文头条结论。
  2. 是否加 seed 只看标签的**类别**，不看方向与数值；触发原因写进 FINDINGS。
  3. 5 seed 判定：均值 + 配对 percentile bootstrap 95% CI（固定 RNG），门槛见各组；两个标签都报告。
  4. (b) 情形下 5 seed 只能**确认或降级**原标签，不能升级。
  5. 加 seed 时对照臂同步补 SNAP s45 / s46。
- 功效先验（G6）：配对 ΔV 的 seed 间 SD 0.035（b 臂）/ 0.19（c 臂）。D0 已正式算（F-090，`harness/d0_power.py`）：假阳性很低、σ 小时 μ = 0.20 几乎必过；**`no_effect` 对真零效应也常判不出**（σ = 0.035 时 62%）→ 改不改由用户定（§8 第 1 条）；margin 2 logit 维持。
- 每组的闸：exit 0、`client_failures == []`、攻击者每轮参与、config_sha 一致（`status.py`）；判定脚本**先冻结进 git 再打开数据**（同 G1，D-088）。

### 3.6 单价（GPU-h / run，K=3 包；除标「实测」外都是外推，第一次用时实测改写）

| 类型 | 单价 | 依据 |
|---|---|---|
| R5 × 300 有效轮，无记录 | 0.8 | 实测：G0 12.2 / 15、G3 18.5 / 24 |
| + post_agg + 冻结列 + 快照 | ≈ 1.0 | G1P 的记录开销外推（F-081） |
| + edge 上传前 AT | × 1.05–1.2 | 每次 4 edge × 500 × E × (k+1) 次前向反向，约为训练量的 5–20% |
| + edge 每轮 AT | × 1.25–1.7 | 同上 × 5 |
| + 客户端 AT（CCS-AT / CCS-full） | × 2.5–4 | 训练约占墙钟 64%（F-046）；良性端每批 k 步对抗样本 → 客户端训练量 × (k+1)；**CCS 原文没写 k** |
| flat 300 轮 | ≈ 1.3 | 实测 G8F 4.6 / 3（350 轮） |
| 自适应攻击者 / CerP / IBA | × 1.1–1.5 | 外推 |

---

## 4. 实验组

记号：`组__单元__s<seed>`；「条件」= 只有前置组给出相应标签才交。

### SNAP —— 快照 + 对照臂（不改代码，D0 之后即可交）

- **目的**：P0 的原料（S3 划分上的上传前 body + edge 干净集 + 各端私有 BN 统计量）；所有主检验的无防御对照。
- **证据**：§1.2 仓库事实 1–3；快照 = 各 edge 自己的上传前 body（F-081 更正）。
- **配置**：`D-base` + `evaluation.snapshot_rounds: "6/15/60"`。run：`SNAP__{col,dist}__s{42,43,44}` = **6 run**（加 seed 时补 s45 / s46）。
- **机时**：约 6 GPU-h；盘约 1.8 GB（≤ 20 GB，D-073）。
- **有效性**：col 的 s42–44 全部 `[Checksum]` = G1R5 同 seed（`instrumentation_check`）；不等就停下查，不读 P0。

### P0 —— 零训练离线反事实（SA0）

- **问题**：一次云聚合里，「所有 edge 都对抗训练后再聚合」能把受害 edge 的灌入砍掉多少？精度代价多少？哪种 AT 目标、什么 ε？
- **文献**：§2.2；SAU 的前提（扰动集覆盖触发器范数）。
- **做法**（快照 t ∈ {6, 15}，配置 H ∈ §3.2）：G = 快照里的全局模型；G'_H = FedAvg(H(w_e))，**所有 edge 都加固**（不能假设知道攻击者 edge）；
  用 run 的评估代码算受害端 fresh-PM ASR / margin / pm_acc。
  - 灌入削减率 R_H = [ASR_v(G) − ASR_v(G'_H)] / [ASR_v(G) − ASR_v(t 末全量点)]（基线 jump < 0.05 的快照不计）；dist 用池化良性 ASR 之差 D_H。
  - 附带（零训练）：
    - **先量触发器范数**：探针上 ‖δ‖∞ / ‖ξ‖∞ / ‖δ+ξ‖∞（像素空间与模型输入空间各一）→ 核对 ε 网格覆盖它（SAU 的前提）；
    - C-ft 归因对照（AT 的效果减去它 = 对抗部分的贡献）；
    - 阻尼 w_e + λ(G − w_e)，λ ∈ {0.5, 0.25}；**oracle 隔离**（G 去掉 E0 = 灌入削减上界）；**cloud 侧** H_cloud(G)（2000 张，D4 离线版）；
    - `n_clean` ∈ {100, 200, 500}（DATA 离线版）；单次 AT 能否洗掉 E0 自己的模型（`A-every` 的离线代理）；加固前后各 edge 的 body 级 c_k（用正确的干净集复测 F-081 的 edge 级对比）；
    - **CCS-clu 离线检测**：快照里各端私有 BN 统计量 → 每 edge HDBSCAN → 剔除集合对恶意端的 TPR / FPR（零成本先看 CCS 聚类在本设定下分不分得开）。
- **机时**：0 训练 run；离线作业约 4–6 GPU-h（s42 全配置约 3–4 h；s43 / s44 只跑冻结的前 3 名 + 对照约 1.5 h）。
- **判定**（选配置只用 s42 → 冻结 → 读 s43 / s44）：
  - 闸 V0：离线重算的 ASR_v(G) 与 run 记录的下一轮 `post_agg` 值差 ≤ 0.01、重算的 FedAvg 与快照的 G 一致 → 否则 `invalid`；
  - 精度过滤：fresh ΔMTA ≤ 0.02（池化）且任一 edge ≤ 0.04（一次性反事实里 head 来不及适应，偏保守）；
  - `go_online`：最佳配置在两个快照、三个 seed 上 R_H ≥ 0.5（F-086：只需约 0.2–0.4，留一倍余量）；
  - `kill_pre`：全部 AT 配置、全部 seed 的 R_H < 0.2 → `A-pre` 止步（**`A-every` 不由 P0 判死**，只报告单次洗 E0 的代理）；
  - `accuracy_bound`：只有超预算配置达标 → 用户定；其余 `inconclusive`（P1 照做，只作探路）；
  - 另报：最佳 AT 配置的 R_H − C-ft 的 R_H（对抗部分的贡献）。

### CAL —— 不饱和工作点 W2

- **问题**：饱和区里「部分有效」看不见；找 V 末值 0.3–0.8 的攻击强度，用来解释 `partial` / `no_effect`。**证据**：多处天花板、3-B 判不出（REPORT §1 / §6）。
  文献：Bad-PFL 图 4（FedRep 下 ρ 0.1–0.5 ASR 都接近 100%）→ **降 ρ 不一定能脱离饱和**，所以同时降恶意端比例。
- **配置**：`D-base` × {ρ 0.05, ρ 0.10, 5% 恶意端（col `[5,0,0,0]` / dist `[2,1,1,1]`）} × {col, dist} × s42 = **6 run**，约 5 GPU-h（与 P1 同批）。
- **判定（选择规则）**：取使 col 的 V 末值 ∈ [0.3, 0.8] 且 E0 良性 ≥ 0.7 的**最强**设置；都不满足 → 不设 W2，主检验改读 margin。
- **条件后续 W2**：H1 / H3 判 `partial` / `no_effect` 时，W2 × {对照, 最佳臂} × 该布点 × s42–44 = 6 run / 布点，约 7 GPU-h / 布点。判定规则同 H1。

### P1 —— 模块 A 在线单 seed 探路（SA1）

- **问题**：一次性效果能否累积成在线效果；`A-every` 的真实开销。**证据**：F-076、F-085。
- **配置**（s42；先交约 10 分钟的短作业验证 GPU 确定性路径，陷阱 #23）：`P1__{col,dist}-{pre,every,cloud}__s42` = **6 run**。
  `A-every` 的预算按 P0 最佳配置缩到「墙钟开销 ≤ +50%」（预注册换算）。对照 = SNAP s42。
- **机时**：约 7–9 GPU-h。
- **判定**（筛选，不进结论）：`go_main`：col ΔV ≥ 0.15 或前半程 Δ_jump 减半；dist ΔP ≥ 0.10 或 Δmargin_p50 ≥ 2 logit（F-086：一次云聚合推 margin 1.9–3.0）；
  且 fresh ΔMTA ≤ 0.02。`stop`：ΔV < 0.05 且 Δ_jump 减少 < 20%。
- **有效性**：第 1 轮上传之前的训练与 SNAP s42 相同；`[Harden]` 每次加固都有一行。

### CCSF —— CCS 的 flat 复现闸（SA-C）

- **问题**：我们的 CCS 实现在 flat 下能否复现原文的量级？复现不了，HFL 里的任何 CCS 结论都无法归因。
- **文献**：CCS 表 1（FedPer：94.42 → 8.54）。**已知偏差**（写进语义 diff 表）：本仓库是 FedRep 不是 FedPer、ResNet-10 不是 ResNet-18、本地 5 epoch 不是 2 步、300 有效轮不是 1000 轮。
- **配置**：flat 版 `D-base`（1 edge × 100 端、R=1、300 轮；划分 D0 定：Dir 0.5 贴近原文，或 equal_random 贴近 FLAT 组）× {off, CCS-full} × s42 = **2 run**，约 5 GPU-h。
- **判定**：`ccs_reproduces`：CCS-full 的良性 ASR 比 off 低 ≥ 0.5 且 ΔMTA ≤ 0.02；`ccs_not_reproduced`：降幅 < 0.2 → 先按语义 diff 表排查实现，**不读 CCSP**，由用户定；其余 `partial_repro`（照做 CCSP，结论里注明）。

### CCSP —— CCS 三臂在 HFL 里的筛选（SA-C）

- **问题**：D5 —— HFL × FedRep 下 CCS 两个组件各贡献多少。若拿到原文消融（对抗训练约占 85%），先把「ΔV(CCS-AT) ≈ 0.85 × ΔV(CCS-full)」写成预注册预测再交。
  `CCS-clu` 臂便宜（不带客户端 AT），无论如何都跑：它也是 AA-S 的检验对象。
- **配置**：{`CCS-full`, `CCS-AT`, `CCS-clu`} × {col, dist} × s42 = **6 run**，约 13–16 GPU-h（4 个客户端 AT run 是大头）。
- **判定**：每臂按 P1 的 `go_main` / `stop`；另报 `CCS-clu` / `CCS-full` 每个 edge 的剔除 TPR / FPR；
  分解读法（只报告）：ΔV(full) ≈ ΔV(clu) ≫ ΔV(AT) → 「效果来自聚类」；ΔV(full) ≈ ΔV(AT) → 「来自对抗训练」。
- **信息**：哪个 CCS 臂进主检验；与 `A-*` 在同一 seed 上的直接对比（客户端 AT vs edge AT 的效果与开销）。

### H1 —— 集中布点主检验

- **问题**：D2 + D3 的一半（E0 良性端）+ D4（同场比较各臂）。**门槛**沿用 3-E（D-071）。
- **配置**：P1 / CCSP 判 `go_main` 的臂（≤ 3 个：例如 `A-pre` / `A-every` / 最好的 CCS 臂）× s43–44（s42 = 筛选 run）；对照 = SNAP-col；
  条件 floor：判 `protects` / `partial` 的臂 × ρ=0 × s42–44（对照 floor = G0-C1 现成）。
- **run / 机时**：≤ 6 + 9 = 15 run，约 8–30 GPU-h（含 CCS 臂时偏上）。
- **判定**（3 seed）：
  - `protects_victims`：ΔV 均值 ≥ 0.15 且最小 seed ≥ 0.10，fresh ΔMTA 均值 ≤ 0.02、每个 seed ≤ 0.025；
  - `costly`：ΔV 达标、精度超标；`no_effect`：三个 seed 都 |ΔV| < 0.05；其余 `partial`；
  - 5 seed 时：均值 ≥ 0.15 且 CI 下界 ≥ 0.10 / `no_effect` = |均值| < 0.05 且 CI ⊂ (−0.10, 0.10)；
  - `treats_E0`：同规则作用于 ΔB0；
  - **天花板规则**：对照与防御臂的 V 在全部 seed 都 ≥ 0.95 → 改读 Δmargin_p50（阈值 D0 定，草稿 2 logit）；
  - **机制**（宣称「清除后门」时必需）：Δexcess ≥ 0.10，否则只说「压低 ASR」；
  - **开销**：每臂报告客户端 / edge 的额外墙钟（不设判定，进 Pareto 表）。

### H3 —— 分散布点主检验

- **问题**：D3，真正开放的情形。**证据**：F-069（P 0.999，margin 7.5）。
- **配置**：同 H1，布点 dist；条件 floor = 臂 × ρ=0 × 3 seed + 对照 dist × ρ=0 × 3 seed（G0 没有 dist floor）。
- **run / 机时**：≤ 6 + 12 = 18 run，约 9–33 GPU-h。
- **判定**：同 H1（主量 ΔP）。**事先写明预期**：`A-pre` 应为 `no_effect`（每个 edge 被自己的攻击者重新污染）；`A-every` 与 CCS 的客户端 AT 才是真检验；
  `CCS-clu` 在分散布点下每个 edge 只有 2–3 个攻击者，HDBSCAN 的少数簇更难成形（推断，没有证据）。

### CLD —— cloud 侧对照（「edge 原生」的增益）

- **问题**：D4，同样 2000 张干净图放 cloud 聚合之后做，是否一样？**条件**：H1 或 H3 有 `A-*` 臂判 `protects` / `partial`。
- **配置**：`A-cloud` × {col, dist} × s43–44（s42 在 P1）= **4 run**，约 4.5 GPU-h。
- **判定**：`edge_native_gain`：三个 seed 的 ΔV(edge 臂) − ΔV(cloud) 都 ≥ 0.10 且精度不更差；`no_position_effect`：都 |差| < 0.05；其余 `inconclusive`。

### FLAT —— flat 对照

- **问题**：同样的对抗训练放在 flat FL 的 server 上（1 edge、R=1、`clean_per_edge` 2000）效果是否相同；CCS 在本仓库 flat 下的 3 seed 结果（CCSF 只有 s42）。
  → 论文叙述是「HFL 特有」还是「通用 + edge 是合适的部署位置」。**证据**：F-071。**条件**：H1 / H3 有臂判 `protects` / `partial`。
- **配置**：flat × {off, A-flat（server 侧 AT）, CCS-full} × s42–44；CCSF 的 s42 两个 run 复用 → **7 run**，约 15 GPU-h。
- **判定**：只报告各臂的 ΔP、ΔMTA、开销，不设标签（与 HFL 的威胁情形不可比，同 G8F）。

### DATA —— 干净集大小（在线）

- **问题**：「每 edge 500 张可信干净图」是否现实。**条件**：H1 有 `A-*` 臂判 `protects` / `partial`；先看 P0 的离线敏感性。
- **配置**：`n_clean` {100, 200} × 最佳 `A-*` 臂 × col × s42–44 = **6 run**，约 7 GPU-h。
- **判定**：`data_robust`：n=200 在三个 seed 都保留 ≥ 70% 的 ΔV；n=100 只报告。文献参照：SAU 用 5% 干净数据；FLTrust 约 100 张根数据集。

### AA —— 自适应攻击者（SA2）

- **问题**：D6 —— 对每个在 H1 / H3 判 `protects` / `partial` 的防御臂，换成「知道这个防御」的攻击者后，防御带来的受害端 ASR 下降还保留多少。
  **条件**：该臂在 H1 / H3 判 `protects` / `partial`；否则不跑（没有可攻破的东西）。
- **量**（按 seed 配对；分散布点用 P、E0 用 B0，同样算）：
  - ΔV_nonadapt = V(SNAP) − V(臂, 标准 Bad-PFL) —— H1 / H3 已有；
  - ΔV_adapt = V(SNAP) − V(臂, 自适应 Bad-PFL) —— 对照仍是 SNAP：没有防御时，攻击者的最佳回应就是标准 Bad-PFL；
  - **保留率** = ΔV_adapt / ΔV_nonadapt —— 「还剩多少」指的就是它；
  - 另报：自适应给攻击者自己带来的代价（恶意端的干净精度、生成器每轮的训练时间）。
- **攻击者的知识**（Kerckhoffs 原则）：知道防御的算法、超参（ε、步数、预算、频率）与作用时机；**不知道** edge 的干净集、其他客户端的数据与上传。
  评估用的固定攻击者也用自适应后的生成器（否则报告的 ASR 偏低）。
- **三种自适应攻击者**（各对一类防御；组合防御用组合攻击者）：

  | 代号 | 对付 | 攻击者每次被选中时多做什么 | 依据 |
  |---|---|---|---|
  | AA1（EOT；Athalye 等 ICML 2018「Synthesizing Robust Adversarial Examples」，F-092） | 对抗训练类：`A-pre` / `A-every`（edge 侧）、`CCS-AT`（客户端侧） | 复制收到的 body，用自己的 375 张干净图把**同一个**对抗训练步骤跑一遍得到 θ_H；生成器目标从「在 θ 上把 x + δ + ξ 判成 y_t」（Bad-PFL Eq. 7）改为在 θ 与 θ_H 上都判成 y_t，ξ 也在 θ_H 上求；投毒训练照旧 | EOT（Athalye 2018）；综述第 1 条 |
  | AA2（A3FL 式） | 同上，但**不假设**防御的具体算法 | 在 θ 的副本上模拟「把触发器训掉」：用 (x + δ + ξ, 真实标签) 训几步得到 θ_U，让触发器在 θ_U 上仍然有效 —— 只假设「防御会沿触发器方向反学习」 | A3FL（NeurIPS 2023） |
  | AA-S（统计量伪造） | 聚类类：`CCS-clu`、`CCS-full` 的聚类部分 | 投毒训练照旧，但**上传**的 BN running 统计量换成「看起来良性」的值（本 edge 上一次聚合的统计量，或只在自己干净数据上算的统计量）；自己的私有统计量照用 | F-085：FedRep 下统计量私有、上传值不进任何模型 → 零代价 |

  `CCS-full` 用 AA1（模拟 CCS 的客户端 AT）+ AA-S 的组合 —— 真实的自适应攻击者会两件都做。
- **配置**（s42–44 × 有效的布点）：最好的 `A-*` 臂 × {AA1, AA2}（≤ 12 run）+ 最好的 CCS 臂 × 对应的自适应攻击者（`CCS-clu` → AA-S；`CCS-AT` → AA1；`CCS-full` → AA1 + AA-S；≤ 6 run）
  = 最多 **18 run**，约 20–37 GPU-h（带客户端 AT 的偏贵）。
- **判定**（三个 seed）：`robust`：保留率 ≥ 0.5 且 ΔV_adapt ≥ 0.10；`broken`：ΔV_adapt < 0.05 → 该臂记为不稳健，**不准再调参救**；其余 `degraded`。
  AA-S 另报剔除 TPR（预期降到随机水平）；AA1 / AA2 另报哪一种更伤（决定论文里的局限写法）。

### H2 —— 与 3-E 组合 + 分段感知攻击者（SA2 同期）

- **问题**：D7。G6 的阳性结果在旧划分上，也没面对过「只往上云的块里写后门」的攻击者。
- **配置**（col，s42–44）：`k1-off`（3-E 在 C1 上复现，**不依赖 A**）、`k1-A`（条件：A 有效）、`k1-off-seg`、`k1-A-seg`（分段感知攻击者，LP 式）= 最多 **12 run**，约 13–15 GPU-h。
- **判定**：`3E_replicates`：三个 seed 的 ΔV(k1 vs SNAP) 都 ≥ 0.3；`additive`：V(k1+A) < min(V(k1), V(A)) − 0.10（三个 seed）；`redundant`：差都 < 0.05；
  `3E_broken_by_seg`：分段感知下 ΔV(k1) 都 < 0.15。

### ATK2 —— 第二攻击：CerP + IBA（SA5 / SA6）

- **问题**：D8，效果是否只对 ℓ∞ 对抗式触发器成立。
- **ATK2P 探路**（无防御）：{CerP, IBA} × {col, dist} × s42 = **4 run**，约 4–5 GPU-h。植入判据：col 的 E0 良性 ASR 末 10 点 ≥ 0.5、dist 的 P ≥ 0.5 → `implants`；
  否则 `no_implant`（记为发现，与 Bad-PFL「手工触发器在 PFL 存活不了」对照；Bad-PFL 表 21 中 IBA 在 FedRep 下本就弱于 Bad-PFL），该攻击退出。
- **ATK2 主检验**（条件：某臂在 H1 / H3 有效、且该攻击 `implants`）：攻击 × {对照, 最佳臂} × 有效布点 × s42–44 = 最多 **24 run**，约 26 GPU-h。
- **判定**：`generalizes`：满足 H1 / H3 的 `protects` 规则；`bad_pfl_specific`：Bad-PFL 下 `protects`、这里 `no_effect`；其余 `partial`。
  这两种攻击下没有 `post_agg` / 冻结列（`config_validate.py:511`），只读 V / B0 / P / MTA。

### H5 —— 已知防御作为基线（SA3 / SA4）

- **问题**：D9。A 无论成败都要有这张表（A 失败时它就是阶段三的主内容）。
- **前置修复**（SA3）：陷阱 #9（防御只看 body 索引）、#2 第六处（FLAME 的全局 np.random）、#3（FLAME 按文献：HDBSCAN、无权平均）。
- **基线**（部署位置）：H5a（SA3 后）Median / Trimmed-mean / Multi-Krum / FLAME / RLR（edge 聚合）；H5b（SA4 后）FedBAP-ℓ∞（良性客户端）/ SHIELD（edge + cloud 递归多簇）。
- **文献预测**（flat，Bad-PFL 表 2 与 CCS 表 1）：Multi-Krum 96.15、FLAME 79.05、Median 77.21 → 预期失败；Sign 20.32 但 Acc 34.49 → RLR 预期「压得住但训练崩」或部分压制（IBA 原文 64%）。
- **配置（两段，省机时）**：先 7 个基线 × {col, dist} × s42 = **14 run**（约 12–14 GPU-h）筛选；
  筛选规则：ΔV < 0.05 且 ΔP < 0.05 → `fails_screen`（与文献一致，**停在 1 seed**）；否则扩到 s43–44（最多 28 run，约 25 GPU-h）。作业脚本只传 `--config`（陷阱 #19）。
- **判定**：扩了 seed 的基线与最佳臂比：`A_preferred`：三个 seed 都 ΔV(A) − ΔV(b) ≥ 0.10 且 ΔMTA(A) ≤ ΔMTA(b) + 0.01；否则出 Pareto 表（ΔV vs ΔMTA vs 客户端开销），不下优劣结论。

---

## 5. 执行顺序、闸门、机时汇总

```
D0 ──> SNAP
  └─> SA0 ──> P0 ──┬─ A 判死 ────────────────┐
                   └─ go / inconclusive ─> SA1 ─> P1 + CAL ─┐
  └─> SA-C ──> CCSF ─(复现)─> CCSP ─────────────────────────┴─> H1 + H3 ─> 有 protects/partial ─> CLD + FLAT + DATA + floor + W2
                                                                         └─> SA2 ─> AA + H2 ─> SA5 / SA6 ─> ATK2P ─> ATK2
  └─> SA3 / SA4（可与上面交错）──> H5 筛选 ──> H5 扩 seed
```

| 阶段 | 组 | run（上限） | GPU-h（外推） |
|---|---|---|---|
| 0 | SNAP、P0（离线） | 6 + 离线 | 10–12 |
| 1 | P1、CAL、CCSF、CCSP | 6 + 6 + 2 + 6 | 30–35 |
| 2 | H1、H3（含条件 floor）、CLD、FLAT、DATA、W2 | 15 + 18 + 4 + 7 + 6 + 12 | 55–110 |
| 3 | AA、H2、ATK2P、ATK2 | 18 + 12 + 4 + 24 | 63–83 |
| 4 | H5 筛选 + 扩 seed | 14 + 28 | 12–39 |
| — | 加 seed（条件） | ≤ 25 | ≤ 30 |
| **合计** | 全部分支都放行 | **约 210** | **约 170–300** |
| 最短路径 | A 判死、CCS 不复现 → SNAP + P0 + CAL + CCSF + H2(k1-off) + H5 筛选 | 约 31 | **约 40** |

- D-047 上限约 100 GPU-h / 周：全部放行约需 2–3 周的额度，按会话节奏分摊到 **8–10 周**；最贵的是客户端对抗训练（CCS 各臂）。
- 预算紧时的砍法（按信息量从低到高）：W2 → DATA → FLAT 的 CCS 臂 → H5 的 dist 臂 → ATK2 的 dist 臂 → AA-S 对 CCS-full（保留对 CCS-clu）。

## 6. 代码会话（一个会话一个模块）

| 会话 | 内容 | 开工前 |
|---|---|---|
| D0 | 阶段三登记表（SNAP / P0 / CAL）、预注册 P0 / P1 / CCSF 的判定规则（N-008）、功效分析脚本（定门槛）、文献逐条核对（凭记忆引用、还没对过原文的条目：TRADES、I-BAU、NAD、FLTrust、A3FL、EOT、Tsipras 2019、LP 等）；D-089 与台账已在 2026-10-09 完成 | — |
| SA0 | edge 侧 AT 核心（纯算术不 import TF；TF 部分包 random 围栏，F-078）+ 快照离线探针（含触发器范数、CCS-clu 离线检测） | 语义 diff：Madry PGD-AT / TRADES / SAU 官方实现 vs 本仓库 |
| SA1 | 在线接线：`cloud_upload` 上传变换、`run_edge_round` 每轮钩子、`CloudServer` 聚合后变换；`[设定11]`、`[Harden]`；BN 用 `TorchBatchNorm` 推理路径 | 守卫：开关关逐位不变、去围栏的反向锚点、`gpu_determinism_check` |
| SA-C | CCS：客户端 AT mixin（投影网络 + KL + MMD + max-margin 对抗样本；可变形 patch 初始化按原文或记为偏差）+ edge 侧 BN 统计量 HDBSCAN（经 `robust_mean` 收口，陷阱 #10）+ flat | 语义 diff：CCS 原文（是否有官方代码待查） |
| SA2 | 自适应攻击者 AA1 / AA2 / AA-S + 分段感知攻击者（mixin） | 语义 diff：A3FL 官方实现 |
| SA3 | 修陷阱 #9 / #2 / #3；Median / Trimmed / Multi-Krum / FLAME / RLR 登记 | 语义 diff：FLAME、RLR 官方 |
| SA4 | FedBAP（改写为 L∞ 球内 PGD，客户端 mixin）+ SHIELD | 语义 diff：两篇原文 / 官方代码 |
| SA5 | CerP 接 `per_epoch` 取数，`config_validate` 放行 | 守卫：静态投毒仍被拒 |
| SA6 | IBA 移植（mixin） | 语义 diff：IBA 官方实现 |

## 7. 相对 `PLAN-draft.md` 的改动

1. P0 改用 SNAP 快照（G8 没有 edge 干净集）；SNAP 兼作对照臂。
2. **加固候选改为对抗训练家族**（PGD-AT / TRADES / 逐样本定向 AT / SAU / CCS 目标）；**删 D1（NAD）**；O1 降为归因对照；O3 从「类级通用扰动」改成逐样本（I-BAU 在 Bad-PFL 下失效）。
3. P0 的门槛改为灌入削减率 R_H ≥ 0.5（由 F-086 推出）+ 有效性闸 V0；先量触发器范数再定 ε。
4. 新增 **CCS 主线**（CCSF 复现闸 → CCSP 三臂 → 进主检验 → AA-S），对抗训练「在哪做」成为 D4 的核心比较。
5. 新增 CLD；floor 条件性、对照 floor 复用 G0-C1；Simple-Tuning 删除（文献已证无效）。
6. H2 拆出不依赖 A 的「3-E 在 C1 上复现」与分段感知攻击者。
7. 默认 3 seed + 预注册的加 seed 规则。
8. 新增 CAL / W2、FLAT、DATA、ATK2（CerP + IBA）、H5（含 RLR、FedBAP-ℓ∞、SHIELD；先 s42 筛选）。

## 8. 仍待讨论

0. **CCS 的消融出处**：上传的 ICASSP 版本里没有；若有扩展版 / 补充材料，请上传（影响 CCSP 的预注册预测）。
1. 各组门槛（0.15 / 0.10 / 0.05、R_H 0.5 / 0.2、margin 2 logit）—— D0 的功效分析见 F-090：0.15 / 0.10 与 2 logit 有余量；`no_effect`（三个 seed 都 |Δ| < 0.05）对真零效应也常判不出，是否改成「|均值| < 0.05 且最大 |Δ| < 0.10」待定；N-008 草案待确认。
2. 加 seed 规则（§3.5）(b) 款：现在就指定哪些组算「头条结论」（推荐 H1 / H3）？
3. CCS 的可变形 patch 初始化（DPR，ECCV 2022）照原文实现，还是换成 ℓ∞ PGD 并记为偏差？（SA-C 的语义 diff 时定，影响 CCSF 能否复现）
4. CCSF 的划分：Dir 0.5（贴近 CCS 原文）还是 equal_random（贴近本仓库其余 flat 组）。
5. H5 的「s42 筛选、失败即停在 1 seed」是否接受。
6. G8 存盘：本计划不再依赖它，删不删由你定（D-073）。

---

## 附 A：定稿过程中的批注与处理（2026-10-09）

第一轮批注针对讨论稿 v2，第二轮针对 v3；拍板见 `DECISIONS.md`。

| 批注 | 文献核对（读原文） | 处理 |
|---|---|---|
| O1 干净微调：类似方案在 flat Bad-PFL 原文里效果不佳 | Bad-PFL 表 3（FedRep / CIFAR-10）：FT-15 / 30 / 45 后 ASR 97.31 / 97.76 / 97.01%；表 21：Simple-Tuning 88.82%；CCS 表 1：ST 81.87% | O1 **不再是候选**，只在 P0 离线保留为「同数据、同步数、去掉对抗部分」的**归因对照**（零 run 成本）；Simple-Tuning 从 P0 删除 |
| D1（NAD 式蒸馏）在 flat Bad-PFL 原文中实验不可行 | Bad-PFL 表 3：NAD 后 86.25%；综述：F3BA 已证 FedDF / FedRAD 类蒸馏在持久触发器下失效、teacher 会连同后门一起继承 | **删除** |
| 对抗训练去哪了？已被文献证明对 flat 可能可行 | CCS 表 1（flat FedPer / CIFAR-10）：Bad-PFL 94.42% → **8.54%**（MTA 79.46 → 79.76），α ∈ {0.1 … 1} 都约 10%（图 2）；Bad-PFL 原文**没有测对抗训练** | 对抗训练（AT）改为**主候选家族**，并分「在哪做」三种：客户端（CCS，flat 文献形态）/ edge 上传前（模块 A，edge 原生）/ cloud（对照） |
| CCS 检索不到 → 用户提供原文 | 见 §2.2 | 独立主线（拍板） |
| （第二轮）CCS 原文已做过消融，对抗训练约占 85% | 上传的 ICASSP 版本全文检索：没有消融（§1.1 事实 3） | 出处待用户确认；CCSP 三臂保留，拿到消融后写成预注册预测 |
| （第二轮）「自适应攻击者下还剩多少」具体指什么 | — | D6 改写为「保留率 = ΔV_adapt / ΔV_nonadapt」，AA 组写明三种攻击者的知识、做法与对应的防御（§4 AA） |
| （第二轮）忽略 3.2「私有头吸收」 | 结论见 Simple-Tuning 与 Bad-PFL §2.2 / 表 12 | 从证据表中删去 |
