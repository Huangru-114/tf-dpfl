# S6b 方案与实现记录：在线 c_k + G1 登记（27 run）+ 几何记录的混杂修正

> **状态：已实现（2026-10-02，D-087）。** 本文件 = 开工前用户批准的方案原文；实现取舍、预注册规则与开销见 DECISIONS D-087、FINDINGS N-007 / F-081 补注。
> G1 / G1R5 挂 `g1-prereg`：N-007 用户确认之前不放行。

## Context
用户已逐项同意（2026-10-02）：
- 决定 1 = **c**：在线 c_k 直接开进 G1（不先做离线探路）；
- 决定 2 = G1 全量 24 run **+ R5 桥 3 run**（共 27）；**关 frozen**；**不加 s45 / s46**（随 G2 一起定，G2 暂不做）；
- 决定 3 = 3-C 作 G1 的副产品；
- c_k 参数按我的建议：每 5 个有效轮评分一次、n = 64 张、5 步 PGD、ε = `backdoor.badpfl_epsilon`（4/255）。

按 CLAUDE.md 交互约定，**批准计划后才动代码**。本次会话 = S6b 实现；G1 的 27 个 run 在 S6b 验收通过、判读规则预注册之后才由用户交。

## 需要先更正的两处（我之前说错 / 漏掉的）
1. **R5 桥的对照对象**：G8 用旧 `noniid` 划分且中途停止投毒，**不是**同一配置。精确的对照是 **G3-C1**（designed C1、集中 [10,0,0,0]、R5、ρ=0.2，差别只在 G3 开自适应停止、没有网格与记录）。所以 G1R5 与 G8 只能近似比较；与 G3-C1 同 seed 的前若干轮 checksum 应逐位相同（预期，未验证）→ 免费的长程验收。
2. **S6a 的几何记录有一个混杂**：`get_base_head_indices` 的 base 索引**包含 BN 的 moving_mean / moving_variance**，FedRep 下统计量私有（A27），所以上传的 body Δ 里混着客户端自己的 BN 统计量（恶意端的投毒样本会改统计量）。F-081 的 norm AUROC 0.83 / 0.91 可能部分来自这里，**没有证据判断占多少**。S6b 顺带把几何拆成「可训练权重部分」与「BN 统计量部分」两个范数 / 余弦。

## 实现内容（语义 diff 表）

| # | 改动 | 落点 | 说明 |
|---|---|---|---|
| ① | 开关 `evaluation.update_ck`（+ `update_ck_every`=5、`update_ck_n`=64、`update_ck_steps`=5） | `fedavg/alignment.py`（EXTRA_SWITCHES，缺省关）；`config_validate.py` 新 §4h | 前提：eval_grid 已开 + 交错调度 + hier_fedrep + S3 划分（有 `clean_indices`，`federation.design.clean_per_edge` > 0）；类型显式检查；`[设定10]` |
| ② | 在线评分器 `UpdateCk`（TF） | 新 `fedavg/server/update_ck.py`；复用 `analysis/ck_snapshot.py` 的 `feature_model / extract / ck_reached` 与 `analysis/functional_score.py` | 评分点 = 有效轮 eff % every == 0 的 edge 轮；edge 收齐上传后、`robust_mean` 前**只读**：θ_before = 本 edge 轮下发的权重，θ_i = edge_w 上换入上传的**可训练权重**（γ/β/卷积核），**BN moving 统计量保持 edge_w 的**（隔离「权重变化」与「客户端数据造成的统计量差」）；每个模型用 edge 干净集的类均值做 NCM head；c_k = 前 n 张干净样本的定向 PGD（无随机起点、不碰 RNG）失败率；打 `[CkScore]`（每更新一行：cid / mal / c_0…c_9）、`[CkBefore]`、`[TimingCk]`；s_i（原文 §7 公式）**离线**由 harness 算（原始 c 全落盘，以后可换聚合方式不重跑） |
| ③ | 挂点与干净集 | `server/hier_fedrep.py:run_edge_round`（`score_observer`，同 `update_observer` 模式）；`main.py`：开关开时 `edge.clean_x/clean_y = x_all[edge.clean_indices]` | 草稿槽 `ck`（专用）；整段包 `random.getstate()/setstate()`；不动 `_eval_seq` / history |
| ④ | 几何拆分（修混杂） | `server/update_geometry.py`：`body_delta` 按 `get_bn_stat_indices` 拆成 weights 部分与 stats 部分；`[UpdateGeo]` 加 `norm_w / norm_s / cos_edge_w / cos_global_w`（旧列保留） | S6a 的行为不变（开关缺省）；老日志照读 |
| ⑤ | 回程 | `harness/collect_metrics.py` → **schema 11**：`ck_scores` 紧凑表、`ck_before`、`run.update_ck / update_ck_every / update_ck_n / update_ck_steps`（平铺标量）、`timing_summary.ck_eval_total_s`、`update_geometry` 新列；`registry.py` EXPECT_KEYS 加 4 键 | 新自描述行 `[设定10]`（不扩 `[设定9]`） |
| ⑥ | 离线读数 | 新 `harness/g1_scores.py`：s_i、AUROC（全局视角 vs edge 视角，等池大小，滑窗）、范数 / 余弦的 weights vs stats 拆分对比 | 描述性；**判定脚本 `g1_verdict.py` 要先预注册规则再写** |
| ⑦ | 登记表 | `registry.yaml`：G1 的 set 加 `update_geometry / post_agg_eval / update_ck` = true（**frozen 不写**）；`requires` 的 `S6` → `S6a, S6b`，`available` 加 `S6b`；新组 **G1R5**（C1 × 集中 × R5 × 3 seed：`edge_rounds 5, n_rounds 60`，其余同 G1）；materialize（**不带 `--group`**） | G1 = 24，G1R5 = 3，共 27 run |
| ⑧ | 文档 | DECISIONS D-087（决定 1–3 与参数、更正两条）；FINDINGS 追加「F-081 的 norm 混杂」与预注册规则 N-007；`S6b-PLAN.md`；CLAUDE.md「当前地基」S6b 一条；PLAN / README / REPORT / current-focus | 预注册规则（s_i 定义、等池大小的 edge / 全局 AUROC、阈值）**写在数据之前** |

## G1 的 run 级说明（27 run）
共享配置：hier_fedrep + ResNet-10（torch 对齐）、CIFAR-10、100 客户端（10 恶意）、4 edge × 25、client_fraction 0.1（每个 edge 轮每 edge 抽 2–3 个）、local_epochs 5、lr 0.1 按有效轮衰减 0.992、Bad-PFL ε=σ=4/255、ρ=0.2、目标类 0、共享生成器、每客户端 500 张、每 edge 干净集 500 张、固定 300 有效轮、停止判据关、网格 G=5；记录：几何（含拆分）、云聚合后评估点、在线 c_k；**frozen 关**。

| 组 / 格 | 划分 | 布点 | R_edge | 云轮 | 回答什么 | run |
|---|---|---|---|---|---|---|
| G1 C1·集中·R10 / R20 | C1 | [10,0,0,0] | 10 / 20 | 30 / 15 | 3-C 锯齿主格；跨 edge 差分定位的原料 | 2 × 3 |
| G1 C1·分散·R10 / R20 | C1 | [3,3,2,2] | 10 / 20 | 30 / 15 | 3-D 核心：edge 内恶意 + 良性混合，edge vs 全局视角 | 2 × 3 |
| G1 random·集中·R10 / R20 | equal_random | [10,0,0,0] | 10 / 20 | 30 / 15 | 3-C 在「edge 不对应机构」下的对照 | 2 × 3 |
| G1 random·分散·R10 / R20 | equal_random | [3,3,2,2] | 10 / 20 | 30 / 15 | 3-D 零对照；同时是 3-A 的 e4-R10 / R20 两格 | 2 × 3 |
| **G1R5** C1·集中·R5 | C1 | [10,0,0,0] | 5 | 60 | 桥：≈ G3-C1 + 周期内记录；与 G8 近似比较 | 3 |

seed = 42 / 43 / 44（决定模型初始化、客户端抽样、固定攻击者选取；edge 级划分设计不随 seed 变）。s45 / s46 不加。

## 开销（外推；G1P 实测单价 + 我的计算量模型，**c_k 部分没有实测**）
- 记录（几何 + 云聚合后评估点，frozen 关）：约 0.95 GPU-h / run → 27 run ≈ **26 GPU-h**；
- c_k（n=64、5 步、每 5 个有效轮、约 600 个被评分更新 / run）：整体墙钟约 **+20%**（计算量模型：PGD 一步 ≈ 训练一步）→ 约 **+5 GPU-h**；若模型低估 4 倍则约 +20；
- 合计典型 ≈ **31 GPU-h**（占周预算 31%），最坏约 46。
- 保护：`PROBE_K=2` 先交一个探路包看实价再放行；`submit.sh` 已有 held 机制。

## 验证
1. 本地无 TF：新增纯 numpy / 校验 / 回程测试；基线 1537 / 47 / 3 之上只多不少；
2. TF 2.15.1 CPU（scratch venv）：`test_update_ck_tf`（ε=0 时 c_k = 直接 NCM 预测；确定性；不碰全局 RNG；**开 / 关 checksum 逐轮相同 + 去掉 random 围栏就改训练的反向锚点**；状态清单不变）；基线 1713 / 24 / 3 / 2 failed（陷阱 #4）；
3. CPU 替身 L2（N-005：随机数据、缩小的 G1 配置）：开 / 关 `instrumentation_check`；
4. **GPU 验收（不需要额外作业）**：`G1__C1_collocated_R10__s42` 前 10 云轮 vs `G1P__coll-off__s42`；`G1R5__…__s4x` 前若干轮 vs `G3__C1__s4x`（G3 自适应停轮，取共同轮数）——都用 `instrumentation_check`；
5. `status.py`：G1 / G1R5 先 blocked（缺 S6b）→ S6b 验收后变 todo；已 materialize 的配置 sha 不变；
6. commit 前 review + 跑测试；推到 `claude/federated-learning-experiment-review-pt5j1b`，不合 main。

## 批准后的顺序
实现 S6b → L1 / TF / L2 全绿 → 写 D-087、N-007 预注册规则 → 登记 G1 / G1R5 并 materialize → push → 交付用户：先交探路包，回传后再放行其余 run。
