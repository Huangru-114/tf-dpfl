# 阶段三文献 —— 用户综述（原文）+ 原文核对摘录

> 第 1 部分是用户提供的综述（2026-09-30），**原文照录、不改**；第 2 部分是 2026-10-09 会话读 Bad-PFL / CCS 原文 PDF 后摘录的、`PLAN.md` 直接用到的数字。
> PDF 不进 git（Bad-PFL 10.5 MB 超过单文件 10 MB 的红线；版权）。引用时注明原文表号；数字以原文为准。
> 综述里「待核实」的条目与 `PLAN.md` 里凭记忆引用的条目，由 D0 逐条核对。

---

## 第 1 部分：用户综述（原文照录）

# HFL个性化后门攻防：相关文献整理

整理日期：2026-09-30

## 总体判断

HFL安全方向的现有工作几乎都用弱威胁模型（标签翻转、符号翻转、高斯噪声、固定patch），且几乎都不带个性化。因此"HFL + PFL + 优化型/自然特征触发器"是一个可以清楚立起的空白。

新颖性风险已基本排除。ACISP 2026中的"edge节点"就是两层FL的客户端（100个客户端、每轮10个），没有edge聚合，也没有防御。离本项目最近的只剩FLVaccin（层级FedPer，但只报告精度、没有ASR），区分点清楚。

检索覆盖四个方向，另附集中式小样本净化方法作为模块A的参照。已上传的21篇均已通读核查；仍标注"待核实"的条目只读过摘要或引用列表。

---

## 已上传文献的定位

12篇中真正能进related work主干的是Alqattan两篇、F3BA、A3FL和BackDFL，其余以概念引用或一句带过为主。

| 文献 | 核心内容 | 建议用法 |
|---|---|---|
| Alqattan et al. 2024, Security Assessment of Hierarchical Federated Deep Learning | 唯一系统比较2L/3L/4L后门TASR的工作；结论是4L最脆弱，重叠覆盖区的攻击者更强 | 重点引用并指出混淆：20个云轮内3L/4L分别有40/120个区域edge聚合轮，训练量多于2L，正是3-A要控制的参与量问题 |
| Alqattan et al. 2025 (HCC), Analysis of DL under adversarial attacks in HFL | MDS多指标度量edge级差异；4L在分布式攻击下差异模式被多层聚合抹平 | 与3-D"edge内视角更可分"是同一命题的两面，可直接对照 |
| HaghighiFard & Coleri 2024 | 车联网HFL，信誉分+相邻轮更新余弦阈值0.5 | 攻击只有高斯噪声、MNIST，一句带过；期刊扩展版DARCS见方向二 |
| Yang et al., Sensors 2023, Edge-Cloud Collaborative Defense | 边缘ABL+注意力自蒸馏，云端余弦过滤 | 实验中的edge实为训练参与方，本质是两层FL的双端防御；引用时说明它不是cloud-edge-client结构 |
| ADFL (Computers & Security 2023) | 服务端GAN从全局模型反演数据，干净模型重标注后蒸馏 | 攻击全是固定触发器（含WaNet、DBA）；可作"edge蒸馏"的思路来源，对Bad-PFL需谨慎 |
| F3BA (AAAI 2023), On the Vulnerability of Backdoor Defenses for FL | 翻转低重要性参数符号+触发器优化 | FedDF、FedRAD、FedMV剪枝等模型精炼防御全部被穿透；"低重要性参数承载后门"与实验2的休眠容量假设同构 |
| A3FL (NeurIPS 2023) | 让触发器在"被unlearn过的全局模型"上仍然有效 | 阶段三自适应攻击者的直接模板 |
| BackDFL (arXiv 2026) | DFL后门基准，15%恶意比例下多数防御失效 | "鲁棒性被高估"的论证框架可借给HFL；IBA > A3FL > Neurotoxin的排序支持把IBA作为第二攻击 |
| ALPHA (IEEE TMC 2025), From Non-IID to IID | 移动性感知的client-edge关联控制 | 收敛界分离了上行与下行分布散度，为H_inter/H_intra提供理论依据 |
| CHPFL (HCC 2026) | edge级个性化：α混合edge模型与全局模型 | 支持3-E的概念；实验只有8个客户端、MNIST类数据，概念引用即可 |
| RaSA (IEEE TIFS 2025) | HFL安全聚合，可验证检测恶意edge | 本项目假设edge可信；RaSA是放松该假设的参照，写进limitations |
| PriRoAgg (2025) | "aggregated privacy"：只泄露聚合统计量的鲁棒聚合 | 借用该定义形式化隐私主张：单个更新只在机构内检查，cloud只见edge聚合 |

---

## 方向一：强攻击基线下的FL后门防御（2023–2026）

FedBAP是离模块A最近的现有工作，必须正面区分；其余多数防御在A3FL这类自适应攻击下已被证明失效。在FedBAP的非IID CIFAR-10实验中，FLAME、FLTrust、FLIP面对A3FL的平均ASR接近或达到100%。

| 方法 | 出处 | 评估过的攻击 | 与本项目的关系 |
|---|---|---|---|
| FedBAP (arXiv:2507.21177) | ACM MM 2025 | BadNets、LP、A3FL；附录加CerP | 客户端良性对抗扰动+自适应强度调度。其MaskGen假设局部patch触发器，面对Bad-PFL需改造 |
| MARS (arXiv:2509.20383) | NeurIPS 2025（Wan et al.） | MRA、CerP、3DFed，另加针对MARS的自适应攻击；对比了BackdoorIndicator、FedCLP等8种防御 | 无需干净数据：用Lipschitz上界近似每个神经元的后门能量，思路接近CLP。edge可在客户端模型上计算，适合作3-D第三类分数；风险是自然特征触发器未必把能量集中在少数神经元 |
| FeRA (arXiv:2505.10297) | arXiv 2025 | 归一化幅度、模仿良性统计的自适应攻击 | 需要服务端小型干净参考集，论证"edge持有干净数据"是领域惯例 |
| AlignIns | CVPR 2025 | BadNets、DBA、Scaling、PGD、Neurotoxin，没有动态触发器攻击 | 方向对齐检查，讨论了非IID下的鲁棒性；只作几何类检测基线，不能说明对生成器触发器有效 |
| BackdoorIndicator | USENIX Security 2024 | 待核实 | 用OOD数据注入指示任务主动检测；edge天然适合托管指示任务 |
| FDCR | NeurIPS 2024（Huang et al.） | 固定patch触发器 | Fisher信息差异聚类+重要参数重缩放，无需代理数据；作者自认无法清除已持续存在的触发参数。攻击基线偏弱，适合在related work中代表"异质感知"一类 |
| CCS | 项目文件 | 待核实 | PFL专用：客户端对抗训练+BN统计聚类；已在基线列表 |

综述段落可一笔带过：Lockdown、FedGame（NeurIPS 2023），FLShield（S&P 2024），Snowball（AAAI 2024），RoseAgg（TIFS 2024），LeadFL（ICML 2023），FLIP，FreqFed、CrowdGuard（NDSS 2024），MESAS（CCS 2023），Scope（TIFS 2025）。基准与理论：BackFed（arXiv:2507.04903）把A3FL和IBA归为动态触发器攻击，可作代码组织参照；Hammer and Anvil（arXiv:2509.08089）尝试给出FL后门理论。

---

## 方向二：HFL中的后门/投毒防御

HFL专用防御只有少数几篇，攻击均为投毒或固定触发器；只有FLVaccin同时是层级+部分共享个性化，但它没有报告ASR。

| 方法 | 出处 | 机制 | 评价 |
|---|---|---|---|
| SHIELD | IEEE TDSC 2025 | 逐层递归的HDBSCAN聚类：低层聚合器上传多个簇均值而非单一聚合，只在顶层选最优簇；不需要参考数据集，不假设多数客户端良性 | 已读全文。三层、250客户端/25个网络；MNIST、CIFAR-10、KDD99、5G-NIDD。攻击主体是源→目标标签翻转加六种非定向攻击，另有一个只变化强度的模型投毒后门，无触发器优化，也未交代非IID程度。适合作HFL基线，其递归多簇上传也是模块B的对照设计 |
| RoHFL | IEEE TITS 24(5), 2023（Zhou et al.） | 4层结构（100车、20 RSU、4 CBS）；对数归一化压制被放大的梯度，车辆与RSU双重信誉，不需要干净数据；威胁模型包含恶意RSU | 攻击是标签翻转、Krum攻击、Trim攻击，全部非定向，没有后门；有收敛分析 |
| DARCS | Vehicular Communications 60 (2026) 101055（HaghighiFard & Coleri） | 更新范数z-score + 余弦 + EPC级跨簇一致性验证 + 可靠性加权聚合 | 攻击只有高斯噪声和梯度上升；跨簇验证的思路可引用 |
| Al-Maslamani et al. | IEEE OJ-COMS 2023 | 每个edge用FoolsGold检测+多智能体DRL信誉选客户端；相邻edge交换对共同客户端的间接信誉 | 有定向后门投毒，但只是MNIST+MLP、10个edge/300客户端、40%恶意；跨edge信誉共享与移动性和3-D相关 |
| FLVaccin | MDPI Computers 2026（preprints.org/manuscript/202607.2135） | 层级FedPer（共享backbone、私有头）；按深度和轮次自适应隔离客户端，根节点拒绝验证趋势变差的backbone | 设定最接近本项目；但后门只是3×3白块，全文只报告精度、没有ASR |
| RaSA、Sensors 2023 | 见已上传文献 | — | — |

这些工作的共同缺口是没有用优化型触发器评估，也没有利用edge持有的干净数据做主动加固。

---

## 方向三：HFL中的攻击与脆弱性分析

未找到在HFL中研究生成器或优化型触发器的工作，这是本项目最直接的空白。现有工作只有几类：

| 文献 | 出处 | 内容 | 用法 |
|---|---|---|---|
| Alqattan et al. 2024 / 2025 | 见已上传文献 | 2L/3L/4L下的标签翻转、符号翻转、目标后门；重叠区攻击者 | 对应3-A、3-D |
| Backdoor Risks in Personalized Federated Learning Under Practical Constraints | ACISP 2026，LNCS 16793（He et al.） | 所谓end–edge实为两层FL：100客户端、每轮10个，FedAvg/Ditto/pFedMe/FedRep，没有防御。固定触发器下FedRep最难攻破（ConvNet多攻击者Blended：FedRep 6.73%、Ditto 29.29%），而生成式TCT以5%投毒率在FedRep上达到72.6% | 不构成新颖性威胁。可用来支撑"固定触发器低估FedRep风险"，并提醒读者该文的edge不是HFL的edge |
| FLVaccin的无防御实验 | MDPI Computers 2026 | 层级FedPer下五类混合攻击使精度崩到约21% | 证明"个性化头不自带防御"，但攻击是非定向的 |
| BackDFL | 见已上传文献 | DFL中拓扑对攻击传播的影响 | 拓扑≈本项目的edge数与放置 |
| Backdoor Attacks in Peer-to-Peer Federated Learning | arXiv:2301.09732 | P2P FL中的攻击者放置 | 类比collocated/distributed放置，待细读 |

写法建议：把HFL攻击文献的稀疏与BackDFL"DFL鲁棒性被高估"的论点并列，论证HFL的"天然压制"同样没有在强攻击下被检验过。

---

## 方向四：生成器或持续优化触发器的攻击

Bad-PFL与IBA之外，PFedBA是最应补充的PFL专用攻击；DPOT是针对更新检测的自适应模板。

| 方法 | 出处 | 机制 | 在本项目中的角色 |
|---|---|---|---|
| Bad-PFL | ICLR 2025 | 自然特征触发器，生成器与模型互相强化 | 主攻击 |
| IBA | NeurIPS 2023（Nguyen et al.） | U-Net/自编码器生成L∞约束的噪声触发器+只毒化低重要性参数；评估了NDC、Krum、Multi-Krum、RFA、RLR、FoolsGold | 第二攻击。单独使用时Krum和RLR能部分压制（CIFAR-10 ASR分别27%、64%），故符号投票类防御仍应作基线 |
| PFedBA (arXiv:2406.06207) | USENIX Security 2024（Lyu et al.） | 在固定掩码区域内逐轮优化触发器，使后门任务与主任务的损失和梯度对齐；评估了10种PFL算法（含FedRep）、Multi-Krum/Trimmed mean/DnC/FLAME和客户端微调类缓解 | 建议作第三攻击。其掩码式触发器正好符合FedBAP的假设，可与全图触发器的Bad-PFL形成对照 |
| A3FL | NeurIPS 2023 | 对抗适应：触发器在unlearn过的模型上仍有效 | 阶段三自适应攻击者模板 |
| DPOT (arXiv:2405.06206) | arXiv 2024（Duke） | 优化触发器使后门数据对模型更新影响最小，只靠数据投毒 | 针对模块B的自适应攻击 |
| FTA (arXiv:2309.00127) | arXiv 2023，出处待核实 | 生成器为每个样本、每轮生成不可感知触发器 | 与Bad-PFL生成器机制最接近 |
| FLAT (arXiv:2508.04064) | arXiv 2025 | 条件自编码器按需生成任意目标类触发器 | 多目标扩展 |
| Mirage（Infighting in the Dark） | CVPR 2025 | 多标签后门，对抗式触发器优化建立分布内映射 | 多攻击者共存场景 |
| TCT | ACISP 2026 | 生成器只用目标类数据训练，产出clean-label触发器；离线生成后固定 | 不属于持续优化，可作受限攻击者的对照 |

次要可引：F3BA、CerP（AAAI 2023），3DFed（S&P 2023），Chameleon（ICML 2023），DarkFed（IJCAI 2024），FCBA（AAAI 2024），LP（backdoor-critical layers）。

---

## 附：集中式小样本净化（模块A参照）

edge持有约500张干净数据，本质是"用少量干净数据净化后门模型"。以下条目除SAU外来自已有知识而非本次检索，出处需复核。

| 方法 | 出处 | 与模块A的关系 |
|---|---|---|
| SAU（Shared Adversarial Unlearning） | NeurIPS 2023 | 已读全文。只用下游任务的5%干净数据，与edge的500张同量级；用L∞≤0.2的5步PGD生成共享对抗样本并unlearn，对WaNet、SSBA、LF等样本特定触发器有效，投毒率升到50%仍稳定。关键限制：触发器范数超出扰动集时失效（WaNet在Tiny ImageNet上L∞=0.348就挂了），所以模块A的ε必须覆盖Bad-PFL触发器的实际范数 |
| FST（Feature Shift Tuning） | NeurIPS 2023 | 小样本微调中主动偏移特征，防止微调后后门残留 |
| BTI-DBF | ICLR 2024 | 触发器反演+解耦净化 |
| I-BAU | ICLR 2022 | 隐式超梯度的对抗unlearning |
| NAD / ABL | ICLR 2021 / NeurIPS 2021 | 注意力蒸馏与反后门学习；Sensors 2023与ADFL的思路来源 |

---

## 对实验3与阶段三设计的影响

最需要立即落实的是前两条：自适应攻击者按A3FL范式实现，FedBAP基线按L∞改造。

1. **自适应攻击者用A3FL范式。** 攻击者在本地模拟edge加固（对自己的数据跑同样的TRADES步骤），再训练生成器使触发器在加固后的模型上仍有效。针对模块B，再加DPOT/PFedBA式的"更新隐匿"目标。

2. **FedBAP基线必须改造并正面区分。** 其MaskGen对每个目标类优化带L1稀疏惩罚的掩码，面对全图L∞触发器时稀疏假设失效，公平做法是换成L∞球内PGD。PFedBA的触发器恰好是掩码式的，所以同时跑PFedBA与Bad-PFL，就能直接展示"扰动模型选择"是否关键。论文需讲清模块A的三点不同：执行位置在可信edge、客户端零开销；用edge干净数据；扰动模型匹配自然特征触发器。

3. **蒸馏路线降级为对照，换成SAU式净化。** F3BA已证明FedDF/FedRAD类蒸馏在持久触发器下失效；edge用自身模型做teacher会连同后门一起继承。SAU的5%干净数据预算与edge的500张同量级，直接可移植；但它在触发器范数超出扰动集时会失效，所以3.1要先量出Bad-PFL触发器的实际L∞范数，再回来定模块A的ε。

4. **3-A的related work写法。** 把Alqattan 2024"4L最脆弱"与本项目"HFL延迟"放在一起，指出前者混入了训练量差异，让参与量匹配显得必要。

5. **3-D的对照点。** Alqattan 2025的"多层聚合抹平可区分性"与"edge内更可分"可用同一批G1数据检验。MARS无需干净数据和触发器，可直接在上传的body参数上算，成本低；它对自然特征触发器是否有效本身就是一个有价值的实验结论。

6. **新颖性风险已排除。** ACISP 2026是两层FL，其"FedRep在固定触发器下最难攻破、生成式触发器却能攻破"的结果正好支撑选择Bad-PFL的理由。related work中应明确区分它的end–edge用词。

7. **基线组合建议加入RLR。** PFedBA显示FLAME在FedRep下被攻破，IBA显示Krum和RLR能部分压制生成器触发器。RLR实现成本很低，应补入基线以覆盖"符号投票"这一类。

8. **隐私与假设论证。** 用PriRoAgg的aggregated privacy形式化"cloud只见edge聚合"；用FLTrust、FeRA、ADFL说明"防御方持有少量干净数据"是领域惯例。RoHFL、RaSA和SHIELD都考虑了恶意中间节点，作为放松"edge可信"的future work参照。

---

## 文献获取状态

11篇外部文献全部收齐并核查完毕，结论已写入上文各表。

| 状态 | 题目 | 作者与出处 |
|---|---|---|
| 已核查 | SHIELD – Secure Aggregation Against Poisoning in Hierarchical Federated Learning | Siriwardhana et al.；IEEE TDSC 22(2), 2025 |
| 已核查 | Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples | Wei et al.；NeurIPS 2023 |
| 已核查 | Backdoor Risks in Personalized Federated Learning Under Practical Constraints | He et al.；ACISP 2026, LNCS 16793 |
| 已核查 | Toward Robust Hierarchical Federated Learning in Internet of Vehicles | Zhou et al.；IEEE TITS 24(5), 2023 |
| 已核查 | Reputation-Aware Multi-Agent DRL for Secure Hierarchical Federated Learning in IoT | Al-Maslamani et al.；IEEE OJ-COMS 2023 |
| 已核查 | Secure Cluster-Based Hierarchical Federated Learning in Vehicular Networks（DARCS） | HaghighiFard & Coleri；Vehicular Communications 60 (2026) 101055 |
| 已核查 | Lurking in the Shadows: Unveiling Stealthy Backdoor Attacks against Personalized Federated Learning | Lyu et al.；USENIX Security 2024 |
| 已核查 | IBA: Towards Irreversible Backdoor Attacks in Federated Learning | Nguyen et al.；NeurIPS 2023 |
| 已核查 | MARS: A Malignity-Aware Backdoor Defense in Federated Learning | Wan et al.；NeurIPS 2025 |
| 已核查 | Parameter Disparities Dissection for Backdoor Defense in Heterogeneous Federated Learning | Huang et al.；NeurIPS 2024 |
| 已核查 | Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection | Xu, Zhang, Hu；CVPR 2025 |

---

## 第 2 部分：原文核对摘录（2026-10-09，读用户上传的 PDF）

> 只摘计划里直接用到的数。表号、公式号都是原文的。ASR / Acc 为百分数，CIFAR-10。

### 2.1 Bad-PFL（Fan, Hu, Wang, Chen；ICLR 2025）

**设定**（§4.1 / 附录 A）：100 客户端、1000 轮、10 个被攻陷、每轮随机选 10%；Dirichlet 0.5；本地与个性化模型都用 SGD lr 0.1、batch 32、15 步（约一个 epoch）；
ResNet-10；投毒率 α = 0.2；ε = σ = 4/255；生成器 Adam lr 0.01、30 步；目标类随机。生成器是 4 层卷积编码器 + 4 层反卷积解码器（表 5，末层 tanh）。

**触发器结构**（Eq. 5–7）：T(x) = x + δ + ξ。
- δ = ε·G_w(x)：**定向**，最小化 L(F(x + δ; θ_g), y_t)，‖δ‖∞ ≤ ε；生成器按样本产生；
- ξ = σ·sign(∇_x L(F(x; θ_g), y))：**非定向**，单步 FGSM，‖ξ‖∞ ≤ σ；
- → ‖δ + ξ‖∞ ≤ ε + σ = 8/255。原文自己说明 ξ 在技术上与 FGSM 相似（§3.2 末段）。

**消融与剂量反应**（表 4 / 表 18）：

| | FedBN Acc / ASR | FedRep Acc / ASR |
|---|---|---|
| 去掉 δ | 80.78 / 13.74 | 80.17 / 12.82 |
| 去掉 ξ | 80.56 / 68.56 | 80.20 / 79.32 |
| 两者 | 80.72 / 82.22 | 80.29 / 97.95 |

| δ 的预算 ε（/255） | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| FedRep ASR | 12.82 | 54.92 | 79.68 | 86.79 | 97.95 |
| FedBN ASR | 13.74 | 23.05 | 78.91 | 80.04 | 82.22 |

| ξ 的预算 σ（/255） | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| FedRep ASR | 79.32 | 85.38 | 88.88 | 93.86 | 97.95 |
| FedBN ASR | 68.56 | 72.84 | 77.37 | 80.52 | 82.22 |

**对鲁棒聚合**（表 2，Bad-PFL 行，Acc / ASR）：

| PFL | ClipAvg | Multi-Krum | Median | Sign |
|---|---|---|---|---|
| FedBN | 82.55 / 82.66 | 66.93 / 80.28 | 74.44 / 50.52 | 31.42 / 24.13 |
| FedRep | 81.59 / 97.28 | 70.41 / 96.15 | 70.23 / 77.21 | 34.49 / 20.32 |

原文：Sign 能压住几乎所有攻击，但梯度量化让模型很难训练（Acc 约 31–34）。

**对客户端侧清除**（表 3，Bad-PFL 行，Acc / ASR；FT-15 / 30 / 45 = 微调 15 / 30 / 45 步）：

| PFL | 清除前 | NAD | I-BAU | FT-15 | FT-30 | FT-45 |
|---|---|---|---|---|---|---|
| FedBN | 80.72 / 82.22 | 76.75 / 76.24 | 77.78 / 75.15 | 81.79 / 81.12 | 82.04 / 80.49 | 82.21 / 80.12 |
| FedRep | 80.29 / 97.95 | 76.33 / 86.25 | 79.58 / 76.58 | 80.67 / 97.31 | 80.79 / 97.76 | 81.10 / 97.01 |

原文的解释：三种方法都是在自然数据上微调，而 Bad-PFL 的后门来自自然数据本身，在自然数据上训练反而让它更持久。

**更多防御**（表 21，FedRep，Acc / ASR）：

| 攻击 | Simple-Tuning | BAERASER | MAD |
|---|---|---|---|
| Iba | 81.82 / 49.31 | 77.74 / 78.98 | 74.55 / 55.58 |
| PFedBA | 81.27 / 42.36 | 78.59 / 31.88 | 74.29 / 55.92 |
| Bad-PFL | 82.05 / 88.82 | 77.68 / 91.54 | 74.37 / 90.74 |

**检测**（表 22）：Neural Cleanse 目标类异常指数 Bad-PFL 2.2（Neurotoxin 5.8、LF-Attack 5.7、PFedBA 4.9；非目标类平均 1.9）；
STRIP 熵 Bad-PFL 0.77（干净样本 0.92；其余攻击 0.12–0.25）。

**攻击开销**（表 23，FedRep，客户端本地训练秒数）：无攻击 0.447、Bad-PFL 1.206、Iba 1.227、PFedBA 1.649。

**附录 D**：δ 是目标类的自然特征（表 24：训练时给目标类样本加 ξ，目标类精度 80.30 → 6.70；再加 δ 回到 39.10）。
原文认为「唯一可想到的对策是客户端不含目标类数据地微调个性化模型，但会严重伤害目标类精度」。

**原文没有测对抗训练。**

### 2.2 CCS —— Turning Data Heterogeneity into a Backdoor Shield for PFL（Cui, Wang, Chen, Wang, Fu；ICASSP 2026，pp. 1061–1065）

**方法**（§3）：
- 客户端：对抗样本先用可变形 patch 表示（DPR，ECCV 2022）初始化，再按 max-margin 损失精炼（Eq. 4）；
  本地目标 L = L_CE + β·L_KL + γ·L_MMD（Eq. 5–7）：KL 约束干净样本与对抗样本的输出一致，MMD 约束经**个性化投影网络** p_i 后的特征分布一致；β = γ = 0.01。
- server：用每个客户端上传编码器里 BN 层的 running_mean / running_variance 拼成客户端向量 v_c，HDBSCAN 聚类，剔除判为恶意的客户端，其余**无权** FedAvg（Eq. 8）。
  原文的理由：统计量直接由数据算出、不经梯度，攻击者无法靠操纵梯度掩盖统计偏差。
- 威胁模型（§3.1）原文：攻击者「不知道 PFL 里部署了任何防御，无论在客户端还是 server」→ **非自适应**。

**设定**（§4.1）：MNIST（ConvNet）/ SVHN / CIFAR-10（ResNet-18），分别训 200 / 500 / 1000 epoch；100 客户端、10% 恶意；Dirichlet 0.5；
每轮随机选 10%、每个客户端两步本地训练；SGD lr 0.1、batch 64。除非另说明，评估用 FedPer 下的 Blended 攻击。

**表 1（CIFAR-10，Acc / ASR）**：

| 攻击 | FedPer | Multi-Krum | FLAME | ST | FLIGHT | SARS | PFL-ALB | CCS |
|---|---|---|---|---|---|---|---|---|
| 无攻击 | 79.98 / 9.30 | 79.89 / 10.03 | 79.85 / 10.10 | 81.01 / 9.92 | 79.86 / 10.01 | 79.92 / 9.89 | 79.95 / 10.30 | 83.38 / 10.50 |
| PFedBA | 80.07 / 97.28 | 79.51 / 87.05 | 79.38 / 83.18 | 79.95 / 86.16 | 79.23 / 90.00 | 79.71 / 87.59 | 80.42 / 80.44 | 80.27 / 8.56 |
| BADPFL | 79.46 / 94.42 | 79.41 / 80.78 | 79.14 / 79.05 | 79.48 / 81.87 | 80.07 / 84.17 | 79.24 / 82.87 | 79.73 / 86.69 | 79.76 / 8.54 |

注意：「无攻击」一行 ASR 约 9–10.5% → CCS 的 ASR **不过滤目标类样本**（本仓库主列只数非目标类，AUDIT）；CCS 的 8.54% 等于无攻击时的水平。

**图 2**：α ∈ {0.1, 0.3, 0.5, 1}，CCS 对 Bad-PFL 的 ASR 都约 10%；FedPer 在 α = 1 时接近 100%。

**消融**：上传的这一版**没有**（全文约 4 400 词，`pdftotext` 检索不到 ablation / w/o / 85%；实验部分只有表 1 与图 2）。
用户记得原文做过消融、对抗训练约占 85% 的效果 —— 出处待确认（`PLAN.md` §8 第 0 条）。
