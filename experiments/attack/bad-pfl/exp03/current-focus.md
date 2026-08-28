# current-focus：Exp 0.3 — Bad-PFL 后门的参数空间定位（地基 + Stage 1 Pilot）

> 上位实验：Exp 0.3「Bad-PFL / IBA 后门的参数—表征空间定位与 HFL 传播机制分析」。
> 本文件只覆盖**第一阶段**：把 Δθ_BD 变成一个可信的数，并用它回答 Q1/Q2。

## 现在要回答的唯一问题

**Δθ_BD 是不是一个可信的、显著超出 SGD 随机性的量？如果是，它集中在哪里？**

不是「Bad-PFL 是不是 dormant-capacity exploitation」——那是**下一步**。
在仪表被证明可信之前，任何关于「后门位于哪些参数」的话都不可读。

理由：核心量

```
Δθ_BD = (θ_poison − θ_t) − (θ_clean − θ_t)
```

是两次训练之差。两次训练里除投毒开关之外的**任何**差异都会整个进到 Δθ_BD 里。
本仓库原本有三处这样的泄漏（见 `fedavg/probe/determinism.py` 模块 docstring）。
不先堵上，clean-vs-clean control 会把结果呈现成 `‖Δθ_BD‖ ≈ ‖Δθ_stochastic‖` ——
而**「后门无法定位」和「仪表坏了」的外观完全一样**。

## 判定「完成」的客观判据（按顺序，前一条不过就停下来）

| # | 判据 | 读哪里 | 不过意味着 |
|---|---|---|---|
| **1 仪表** | 良性端 `‖Δθ_BD‖` **精确为 0** | `verdicts.instrument_ok` | 批序冻结或状态还原漏了东西。良性端没有攻击 mixin，翻 `is_malicious` 是无操作 → 两条 twin 走**逐字节相同**的代码路径，非零就只能是仪表问题。**下面所有数字不可读。** GPU 上先确认 `TF_DETERMINISTIC_OPS=1` 生效（`run_probe.sh` 已导出）；仍不为零就把这个值当**噪声地板**记录，此后所有 Δθ_BD 相对它解释 |
| **2 信号** | 恶意端 `‖Δθ_BD‖ / ‖Δθ_stochastic‖` 显著 > 1（默认阈值 3.0，取**最差**的那个恶意端） | `verdicts.signal` | BD 位移淹没在 local SGD 噪声里 → Q1 的答案是「**参数空间不可定位**」。**这是科学结论，不是 bug** —— 直接写进 notes，并把 Q3（表征空间）的优先级提到前面。不要靠调超参把比值"调"上去 |
| 3 定位 | `topk_bd_energy` 与 `bd_energy_frac_by_layer` | `summaries[]` | 判据 2 过了才读。回答 Q1「集中于少数 layer / 少数 parameter 与否」 |
| 4 占据 | `occupation_vs_bd.curve`（**分箱曲线**）+ `per_group` | `summaries[]` | 回答 Q2。**主读数是曲线不是 r** —— 见下方方法论限制 |

## ⚠️ 方法论限制（必须随任何结论一起写出去）

- **不要单看全局 Pearson/Spearman 判 dormant capacity。** resnet10 ~5M 参数，几百万坐标上的
  p 值毫无意义，r 由 bulk 主导；更要命的是**层身份同时驱动 occupation 与 BD 位移**
  （不同层的更新尺度差一两个数量级），全局 r 极容易只是个 layer-identity 伪相关。
  守卫见 `tests/test_probe_occupation.py::test_occupation_vs_bd_per_group_can_contradict_the_global_number`
  —— 那条测试里组内两层**都是正相关**，全局却是负的。
  → 主读数：occupation 十分位 → mean |Δθ_BD| 的分箱曲线，**逐层各一份**。
- **判据 1 与「良性端 Δθ_BD ≈ 0」是同一件事的两种说法**，不是两条独立证据。
  真正独立的第二路检查是 `tests/test_probe_determinism.py::test_frozen_order_is_not_a_no_op`
  （不同 order_seed 必须给出不同轨迹）——否则噪声地板会被伪造成零、比值恒为 inf。
- **Metric A（`cosine_poison_clean`）只是辅助指标。** 它反映「恶意端整体更新是否仍沿
  正常任务方向」，**不是**后门相似度。主指标是 Metric B（`cosine_bd_clean`）。

## 本会话已交付（地基）

| 件 | 位置 |
|---|---|
| checkpoint 存/载（含**私有 head** 与 **Bad-PFL 共享 generator + Adam slot**） | `fedavg/probe/checkpoint.py`；协议 `FLClientBase.get/set_probe_state` |
| 批序冻结 + 全状态快照/还原 | `fedavg/probe/determinism.py` |
| paired counterfactual runner（clean_A / poison / clean_B） | `fedavg/probe/paired.py` |
| Metric A–D（纯 numpy） | `fedavg/probe/param_metrics.py` |
| Metric E/F（纯 numpy，含手写 Spearman，不依赖 scipy） | `fedavg/probe/occupation.py` |
| 层 / BN state 索引映射 | `fedavg/probe/layermap.py` |
| 装配 + 落盘 + 判据 | `fedavg/probe/analyze.py`、`writer.py` |
| 入口 | `fedavg/probe/run_probe.py`、`run_probe.sh` |

**投毒开关 = `client.is_malicious`**（Bad-PFL 的两个钩子都以它门控），因此探针**攻击无关**：
neurotoxin / cerp 直接可用，IBA 移植后零改动接入。vanilla（badnet/blended/dba）把投毒
烘焙进数据集、不经钩子，`paired.assert_attack_is_hook_gated` 在入口**显式拒绝**
（否则 Δθ_BD 恒为 0，是最坏的一类假阴性）。

## 怎么跑

```bash
# Stage 0：先把管线跑通（不产出科学结论）
sbatch run_probe.sh experiments/attack/bad-pfl/exp03/probe_smoke.yaml

# 只重跑探针（checkpoint 已在）
PROBE_SKIP_TRAIN=1 sbatch run_probe.sh experiments/attack/bad-pfl/exp03/probe_smoke.yaml
```

Stage 0 三条判据（1、2、以及"恶意端确有位移"）都过之后，才换 exp007 锚点：
`experiments/attack/hfl-propagation/2edge_distributed.yaml` + `probe.checkpoint_rounds`，
按已有 exp007 的 ASR/MTA 曲线**人工选定** T1/T3 轮号写死进 config。
**不做运行时阈值自动选点** —— 那样不同 seed 落在不同轮，跨 seed 不可比。
Stage 1 = T1/T3 × 3 benign + 3 malicious × 1 seed。

## 明确不在本会话范围（各自另开会话，接口已留）

- **Q3 表征空间**（Metric G/H/I：repr shift / cosine / CKA）。`layermap` 已能定位任意层；
  抽取路径照抄 `attack/backdoor_eval.py:_build_feature_extractor` 推广到多层。
- **Metric J：BN state**。`layermap.weight_index_map(...)["bn_state_indices"]` 已就绪
  （moving_mean / moving_variance，**不在** trainable_variables 里）。
- **Causal ablation**（§16–19）。必须带 random-region 对照，否则「破坏任何参数 ASR 都会掉」
  这件事会被读成定位成功。
- **Q4 HFL 传播**（§20–27）。接口在 `probe/paired.py:probe_edge_round`（`NotImplementedError`
  + 设计约定）。⚠️ **`A_edge`/`M_edge` 按用户原定义恒等于 1** —— 无防御时 edge 聚合就是
  `Σ wᵢ Δθᵢ` 本身。已定改成 edge 层嵌套 counterfactual（整个 edge round 序列跑两遍）；
  偏离 1 的来源才是真机制：`edge_rounds > 1` 时良性客户端从**已污染的** edge 模型出发训练，
  他们的更新里也带 BD 分量。
- **IBA**。仓库里完全没有实现，按「一会话一方法」另开。

## 进展日志

| 日期 | 做了什么 | 证据 | 结论 |
|---|---|---|---|
| 2026-08-28 | 建 `probe/` 包 + 142 条 L1；`main.build_world` 抽取；checkpoint 钩子 | 本地（TF 2.15/Keras 2.15）`run_l1.sh`：**388 passed / 2 failed / 3 skipped / 3 xfailed**。2 红 = 陷阱 #4（Neurotoxin），与本会话无关。零回归核对：既有测试文件现为 264 passed / 2 failed，减去本会话补进 `test_config_validate.py` 的 18 条探针校验 = **246 passed / 2 failed，与改动前逐条相同** | 地基就位。CPU 上同 `order_seed` 的两条 clean twin **maxdiff = 0.000e+00**（逐比特一致）；不同 seed 有差异（冻结非平凡）；ClientSnapshot 还原 model / 共享 generator / RNG / 私有 head / 优化器 slot 全部精确。**GPU 上的判据 1 尚未验证** |
