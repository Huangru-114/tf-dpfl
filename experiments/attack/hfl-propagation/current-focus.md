# current-focus —— Experiment 3（HFL 后门传播）

> 本文件是 `CLAUDE.md`「新会话开场第 3 步」要读的那一份。
> **写于 2026-09-08**，对应分支 `claude/federated-learning-experiment-review-pt5j1b`（`9310704`）。

## 本会话唯一要回答的问题

导师意见 #7：**「Exp 3 作为更重要的实验，需要更多实验和分析。」**
这句话是两件事，必须分开处理：

- **分析不够** —— 数据早就在盘上，只是没人算（见下「零机时的部分」）
- **格子不够** —— 而且缺的方向和此前补的方向不一样（见「要新格子的部分」）

判据：`RESULTS.md` 能不能报出 `README.md`「读什么（判据）」里定义的那四条，
以及能不能给出「到 ASR=0.5 的有效轮数」和「干净 edge 的爬升延迟」。
现在两样都报不出来 —— 不是缺数据，是缺代码。

---

## ⚠️ 已经做完的，不要重做

这三条我在 2026-09 各误判过一次，浪费了整轮。**动手前先核对**：

| 事 | 状态 | 证据 |
|---|---|---|
| `run` 块记 `edge_rounds` | ✅ `ac797cd`（2026-08-25）就修好了 | 归档里缺这个字段的是**那次提交之前**跑的老文件，键根本不存在。不是现在的 bug |
| `eval_interval` 按**有效轮**对齐 | ✅ `a66da67`（2026-09-07） | flat 10×1、2edge 2×5、3c_R1 10×1 都 = 10 有效轮 |
| `--bind <仓库上一级>` + `cd $ROOT/fedavg` | ✅ 8 月验证过的正确设计，已回退到位 | 陷阱 #17。**不要再拆** |

本会话新加（`9310704`）：
- `[设定2]` 自描述行 → `run` 块多出 `malicious_per_edge` / `malicious_placement` /
  `edge_assignment` / `local_epochs` / `plocal_epochs` / `seed` / 两个 `eval_interval`
- `[Acc] Round N | edge0 | em_acc=… | pm_acc=…` → `per_edge_acc_rounds` / `per_edge_acc_final`
- 两个 `eval_interval` 缺失或不等时发警告
- 守卫：`tests/test_run_self_description.py`、`tests/test_per_edge_acc.py`（19 passed）

**集群上还没验过的一条**：逐 edge 精度按 `n_samples` 加权回去应当等于
`final_acc.em_acc` / `pm_acc`。本地没有 TF 跑不了 `server.run_round`，
只在合成日志上锁了形式。**跑第一个 smoke 时先核这个数**：

```bash
sbatch run_smoke.sh attack hfl-propagation badpfl none smoke_selfdesc hier_fedavg_fedrep
python3 -c "
import json; d=json.load(open('<metrics.json>'))
pe=d['per_edge_acc_final']; tot=sum(e['n_samples'] for e in pe)
print(sum(e['em_acc']*e['n_samples']/tot for e in pe), 'vs', d['final_acc']['em_acc'])"
```

---

## 盘上的实况（2026-09-08 实测 29 个归档 metrics.json）

| 事实 | 数字 |
|---|---|
| `defense` 全是 `none` | **29/29** —— 做防御的课题，主实验里防御轴完全空白 |
| ASR 评估点数极不均衡 | flat=80，3c_R2=40，3c_R4=20，3c_R5=16，3c_R10=8，**3c_R20=4，3c_R40=2** |
| `per_edge_rounds` 是完整的逐 edge 时间序列 | 28/29 有（flat 无，n_edges=1）→ **传播分析不缺数据** |
| README 定义了 4 条判据，RESULTS.md 报了 0 条 | —— |

> Figure 12 那条「R≥4 走平」的线，右端两个点分别只有 **4 个和 2 个**评估点。
> 那两格上任何「轨迹 / 延迟 / 趋势」的说法都不成立，只能报终值。

---

## 下一步（按顺序）

### 1. 零机时：`harness/analyze_exp3.py`（**先做这个**）

纯 stdlib + numpy，**不 import TF**，读 `metrics.json` 出一张表（JSON + CSV）。
**先在 `results/archive-pre-fix/` 上调通** —— 那批数据的绝对值因探针 bug 不可用
（陷阱 #11），但**结构完全一样**，正好用来把分析代码调对。
这同时也是对陷阱 #14（凭记忆写结论）的保险：分析先写好，结论就没有临场编的空间。

要算的量：

1. **README 那四条判据**逐格逐 edge 落表。注意其中两条是同一个量的正负两面，
   报成**一个带符号的数**：`Δ_hier = final.global_asr − mean_e(edge_asr)`
   （>0 = hierarchical amplification，<0 = cross-edge cancellation）。
   另两条逐 edge：`edge_asr − client_benign`（私有头挡住多少）、
   `client_malicious − edge_asr`（聚合稀释多少）。
   **`mean_e` 是逐 edge 无权平均**（不按 `n_samples` 加权 —— 问的是 edge 之间，
   不是样本之间），任一分量 `None` 则该 edge 不参与并单独记 `n_edges_dropped`，
   全 `None` → 结果 `None` 而不是 0。
2. **到阈值的有效轮数** `T_θ`：`r_eff = round × edge_rounds`，在
   `{(r_eff, metric)}` 上线性插值，θ ∈ {0.25, 0.5, 0.75}，三层各算一次。
   三种退化各返回不同的东西：跑完没越过 → `{crossed: false, censored_at: 400}`；
   一开始就在 θ 以上 → `left_censored: true`；**评估网格太粗 → `None` + reason**
   （R20 是 20 有效轮/点、R40 是 40，插值出来的数看着精确其实是编的）。
3. **干净 edge 的爬升延迟**：`has_malicious=false` 的 edge 的 `T_θ(client_benign)`
   减去被污染 edge 的。**`*_distributed` 没有干净 edge → 必须是 `None` 不是 0**
   （这正是产生 `diff_edge_asr=0.000` 那个 bug 的同一个坑，陷阱 #13）。
4. **私有头挡住多少**的时间序列（第 1 条的逐点版）。
5. **参与度 vs ASR**：`malicious_participation_by_client` 按 edge 聚合。
   `edge_assignment=block` 下 `edge_of(cid) = cid // (n_clients // n_edges)` ——
   **这个推导现在有 `[设定2]` 的 `edge_assignment` 撑着了**，不再靠记忆。
   （已在真实数据上验过：`4edge_collocated` 的 `malicious_ids` 全 <25，落 E0，
   与 `malicious_per_edge=[10,0,0,0]` 一致。）
6. **seed 带**：所有聚合量返回 `{mean, min, max, n_seeds, seeds}`，
   `n_seeds == 1` 的格子在表里显式标记（Figure 12 的假象就是没有这条规矩）。
7. **拓扑的连续重参数化**（零机时，但显著提升说服力）：
   `HHI = Σ_e (m_e/M)²`，把 8 个类别标签变成 `HHI ∈ [0.1, 1.0]` 上的 8 个采样点，
   出 `final ASR ~ HHI` 的回归（斜率 + R² + 残差）。
   `10edge_distributed 0.10 / 4edge_distributed 0.26 / 10edge_mixed 0.30 /
   4edge_mixed 0.38 / 2edge_distributed 0.50 / *_collocated 1.00`。
   三个 `collocated` 压到同一个 x=1.0 **不是缺陷**，是天然的重复测量，
   正好估「同 HHI 下 n_edges 的剩余效应」。
   > 方法论出处：Syros et al., *Backdoor Attacks in Peer-to-Peer Federated
   > Learning*（ACM TOPS 2024, arXiv 2301.09732）用图中心性给攻击者布点。
   > 两层树上的自然类比就是集中度。

L1（`tests/test_analyze_exp3.py`，合成 fixture，断言精确值）：
插值解析解；`*_distributed` 的延迟是 `None` **且**反向锚点（有干净 edge 的 fixture
不是 `None`）；从不越过 θ → `crossed=false` 且不返回 0 也不返回 `n_rounds`；
六个 HHI 值逐个对上；`edge_of(cid)` 在**真实归档文件**上跑；
无权平均 ≠ 加权平均（构造样本数悬殊的 fixture，断言用的是无权那个）。

### 2. 待你拍板：3C 保几格

B.5 已证 3C 的效应**小于种子噪声**（跨 R 极差 0.083–0.095 vs 同 R 跨 seed
0.110–0.288），而 R20/R40 只有 4/2 个评估点。
**建议缩到 `{R1, R5, R40}`**（三个数量级的端点+中点，足够支撑「不敏感」这个负面
结论，R40 只用终值），省下 4 格 × 3 seed = 12 个 run ≈ 48 GPU-h 转给防御轴。

### 3. 重跑的格子清单（约 264 GPU-h，单格 ~4 GPU-h 待新 `timing_summary` 校正）

| 批次 | 格子 | run |
|---|---|---|
| 主干（8 拓扑 + 3C 缩到 3 + flat + **无攻击对照**）× 3 seed | 13 | 39 |
| **防御轴**（3 拓扑 × 2 防御，如 `multi_krum` + `median`）× 3 seed | 6 | 18 |
| **持久性**（Neurotoxin 式：攻击者第 T 轮退出，测衰减）× 3 拓扑 × 3 seed | 3 | 9 |

三个新方向的理由：
- **防御轴**：29/29 全是 `defense: none`。地基已经打好（`robust_mean` 收口、
  `[Decision]` 统一日志），加格子的代码成本是零。也是「edge 只有 10 个客户端、
  cloud 只有 2–10 个 edge → 经典鲁棒聚合样本量不够」这个 HFL 专属论点的实测支撑。
- **无攻击对照**：Exp 3 **没有 ρ=0 的格子**，而报告在引用「baseline」。
  没有它既给不出基线 ASR，也说不出攻击的干净精度代价。
- **持久性**：Bad-PFL 侧 Exp 1B 有，Exp 3 没有。层级结构天然提出的新问题：
  **edge 模型会不会成为后门的蓄水池，让它在攻击者走后活得更久？**
  这与「终值上层级组不比 flat 低」（B.7）不冲突 —— 强度和持久性是两件事。

**硬约束：seed 不能低于 3。** B.5 说明 2 seed 下什么都判不了；
用 2 seed 多铺格子只会生产更多没法解释的数。

### 4. 重跑前记得

陷阱 #15：`run_exp3.sh` 按 `exit_code: 0` 跳过已完成格子。
**旧的 `results/*.metrics.json` 必须先移走**，否则一个 GPU 作业都不会提交，
而输出显示「已完成=N」，看起来一切正常。

---

## 可参考的文献思路（方法论，不是背景）

| 工作 | 方法论 | 我们怎么用 |
|---|---|---|
| Liu, Zhang, Song, Letaief, *Client-Edge-Cloud Hierarchical FL*（arXiv 1905.06641 / ICC 2020） | HFL 的标准形式化，收敛性以 edge 间隔 κ₁ / cloud 间隔 κ₂ 为参数 | **3C 轴就是他们的 κ₁ 扫描**。换成他们的记号写，我们的负面结论就从孤立的「没效应」变成「干净收敛对 κ₁ 敏感（理论）而后门不敏感（实测）」的对照 |
| Syros et al., *Backdoor Attacks in P2P FL*（ACM TOPS 2024 / arXiv 2301.09732） | 学习图与通信图分开；攻击者按**图中心性**布点；跨拓扑比较 | 现存最接近的模板 → 第 1 节第 7 条的 HHI 重参数化。也提示要说清「学习拓扑=树、通信拓扑=星」 |
| Qin et al., *Revisiting Personalized FL: Robustness Against Backdoor Attacks*（KDD 2023 / arXiv 2302.01677） | 4 攻击 × 6 pFL = 600 组；**部分模型共享的 pFL 显著鲁棒，全共享的不** | 就是 B.10（FedRep 私有头挡后门，`edge_asr` 0.970 vs `client_benign` 0.859）。提示 **pFL 方法轴**是被验证过的强轴，而 hier_ditto / pfedme / ditto_rep 都现成 |
| Bagdasaryan et al., *How to Backdoor FL*（AISTATS 2020） | 模型替换/放大；**平均带来的稀释**（攻击者更新被 n 除） | HFL 里这个除法**发生两次**、分母不同 → collocated vs distributed 差异的解析钩子，值得把式子写出来当预测再对实测 |
| Sun et al., *Can You Really Backdoor FL?*（arXiv 1911.07963） | 系统扫**攻击者比例** | 我们钉死在 10%。引它说明比例轴为什么该补，并给出「布点效应应小于比例效应」的先验 |
| Zhang et al., *Neurotoxin*（ICML 2022，仓库已有实现） | 持久性的**测量协议**：攻击者第 T 轮停，测衰减轮数 | 直接搬来做第 3 节的持久性批 |

> **卷期页码投稿前逐条再核** —— 作者/venue 我用检索核对过，正文细节来自摘要，
> 引用具体数字前要读原文。另：这个环境访问不了 arxiv / openreview / mlr.press。

---

## 一句话交接

**先写 `harness/analyze_exp3.py` 并在归档数据上跑通**（零机时），
同时在第一个 smoke 上核实逐 edge 精度的加权恒等式；
然后拍板 3C 保几格；再提交重跑。
