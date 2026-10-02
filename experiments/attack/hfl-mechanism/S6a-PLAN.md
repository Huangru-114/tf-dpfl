# S6a 方案：G1 的「便宜记录」+ 探路组 G1P + G8 快照上的 c_k 预检

> **写于 2026-10-02**（G1 / S6 规划会话；用户：「先保存计划，做好交接，我们下个会话再实现」）。
> 决定见 DECISIONS **D-085**。本文件 = 下一会话（S6a）要实现的方案 + 讨论时给用户的说明（附录）。
> **状态：方案，未实现。** 按 CLAUDE.md 交互约定，下一会话开工前先把语义 diff / 改动清单给用户过目。

## Context

**G1 是什么**：
- 服务 3-C 锯齿、3-D 可观测性，外加 3-A 的两格（e4-R10 / R20）。
- 24 run，因素为 4 edge × 划分 {random, C1} × 布点 {集中, 分散} × R{10,20} × 3 seed。
- `requires: [audit, S3, S5, S6]`，只缺 S6。
- 它的 `set:` 不完整：placement 只有 label，没有 `malicious_per_edge`（会继承 base 的 `[5,5]`，长度 ≠ 4）；`n_rounds` 继承 60（R10 → 600 有效轮，超 cap）；没有网格。

**用户复核了这两个指标的指导意义**（附录 D）：
- 旧结果帮不上：P0 / P1 全部作废，从没记过逐更新数据。
- 3-C 的先验（G8：受害 edge 每有效轮 −0.003 至 −0.008）说明答案大体可预料，并且有两个混杂（触发器漂移、lr 衰减）。
- 3-D 的 random 臂是构造出来的零对照；Bad-PFL 的更新在几何上可能根本分不出来（**没有证据**）。

**用户的选择**（D-085）：
1. **分两步**：先做 S6a + 探路 G1P（约 1 GPU-h），回来再定 c_k（S6b）、G1 规模、3-C 去留。
2. **3-C 保留，但只当副产品**：G=5 的点 + 云聚合后一点 + 冻结触发器列；r_down 只取前半程周期。
3. **G1 规模**等探路回来再定。
4. **3-A 补 seed**（s45 / s46）等 G2 定规模时一起定。
5. **逐更新日志**：标量进 metrics.json，草图进 dumps。

**S6a 会话的产出**：
- S6a 的代码与测试；
- 探路组 `G1P` 的登记与配置；
- G8 快照上 c_k 预检的脚本与作业。

**不在 S6a 里**：
- 在线 c_k（S6b，探路后定）；
- `harness/g1_verdict.py`（探路后、G1 提交前写定）；
- G1 的规模。

## 一、S6a 代码（4 个新开关 + 1 处常开改动 + 1 条防呆）

开关一律登记到 `fedavg/alignment.py` 的 `EXTRA_SWITCHES`：
- 缺省关，**只在组的 `set:` 里开**；
- 不写 = 逐字节旧行为（已有配置的 config_sha 不变）；
- 代码里只经 `get_switch(config, "字面键")` 读（`test_alignment_switches` 按字面键认「真的被读到」）。

`config_validate` 规定四个开关的前提：
- `eval_grid` 已开、`edge_schedule: interleaved`、`hier_fedrep`；
- ② ③ 另要求 badpfl + `eval_xi_model: fixed_attacker`。

| # | 开关 / 改动 | 做什么 | 落点 | 输出 |
|---|---|---|---|---|
| ① | `evaluation.update_geometry: true`，加 `evaluation.update_sketch_dim`（缺省 4096） | edge 收齐上传、`robust_mean` 之前，只读：body-only Δ_i = w_i[base] − edge_w[base]（edge_w = 本 edge 轮下发的权重，即 `run_edge_round` 里的 `self.model.get_weights()`）。在线精确算三个量：‖Δ‖；对本 edge 其他更新均值的留一余弦；对本 edge 轮全体 edge 其他更新均值的留一余弦（cloud 在交错循环里等第 er 轮所有 edge 跑完再算，算完清缓冲；10 个更新 × 约 19.6 MB 在主机内存里）。另做 CountSketch：哈希与符号由常量种子 `default_rng([0x5C7E])` 在 init 时一次生成，不碰任何训练 RNG；结果 fp16 写进 dumps | `fedavg/server/hier_fedrep.py:run_edge_round`（`client_updates` 收齐处）+ `fedavg/server/server.py:collect_and_aggregate` 的交错循环。纯算术放新模块 `fedavg/server/update_geometry.py`（**不 import TF**） | `[UpdateGeo]` 紧凑行：每 edge × edge 轮一行，"/" 连接 cid / mal / norm / cos_edge / cos_global → metrics.json 的 `update_geometry`。草图 → `tfdpfl-dumps/<run>/sketch_*.npz` + `[Dump]` manifest（约 25 MB / run，**外推**） |
| ② | `evaluation.post_agg_eval: true` | 云广播之后、第 1 个 edge 轮之前做一次轻评估（fresh-PM = 新全局 body + 各端 head）。只从第 2 个云轮起做。与上一个云轮末全量点相减 = Δ_jump。**不喂停止判据**：这个有效轮已经是上一个全量点 | `fedavg/server/server.py:run_round` 在 `broadcast_to_edges()` 之后。复用 `_light_eval` 的围栏（`random.getstate/setstate`、清 `_edge_w_cache`），另用专用 RNG 键 | `[PostAgg]` / `[PostAggEdge]` → `post_agg_rounds[]` / `per_edge_post_agg_rounds`。**单独成表**，不进 `grid_series`，否则同一有效轮会出现两次 |
| ③ | `evaluation.frozen_trigger: true` | 每个云轮开始（广播后）把固定攻击者的生成器权重与它的 fresh-PM 拷进专用草稿槽（`frozen_gen` / `frozen_attacker`）。在 post-agg / 轻评估 / 全量三类点上，对受害端另算一列「冻结触发器」ASR（ξ 在冻结的攻击者 PM 上求，δ 用冻结的生成器）。这样能把「受害 body 变了」和「触发器漂移」分开 | `fedavg/server/backdoor_server.py:_attacker_trigger` 加 `frozen=True` 分支；`fedavg/client/client_badpfl.py:eval_delta` 加可选 `generator=` 参数 | `[FrozenASR]` / `[FrozenASREdge]`（含 phase = post / light / full）→ `frozen_rounds[]` |
| ④ | 常开（无配置键，只在网格开时生效） | 轻评估点从**同一次触发前向**里取 margin 分位数、良性端 ASR 的 p90、ASR > 0.5 的比例、flip_other。不多做前向；轻评估没有干净前向，所以没有 clean_to_target | `fedavg/attack/backdoor_eval.py` 的 `light=True` 分支（现在跳过了 `_client_record`）：组一个精简记录。汇总复用 `attack/eval_detail.py:summarize` | `[Light]` / `[LightEdge]` 加字段。kv 行逐字段解析，老文件照读 |
| ⑤ | 防呆 | 停止判据开着且 `eval_grid ∉ {None, 5}` → 拒绝启动（容差 0.0010 是按 5 有效轮标定的，F-052；G = 1 时等于放宽 5 倍）。若有现存配置或测试违反，改为告警并记一条 D | `fedavg/config_validate.py` §4b' | — |

**硬要求（同 S5 / D-055）：记录不得改变训练。**
- 所有新增评估都在 `random` 围栏里，用专用草稿槽与专用 RNG 键（`clone_model` 的未播种初始化器会消耗 Python random，F-078）；
- 不动 `_eval_seq` / history / `_last_bd_metrics` / `_bd_eval_count`；
- ① 只读 `get_weights()` 的副本。

**collect_metrics → schema 10**：
- `run.s6`：四个开关的值，来自新的 `[设定9]` 行，不扩 `[设定8]`；
- 新表：`update_geometry` / `post_agg_rounds[]` / `frozen_rounds[]`；
- `light_rounds[]` 加列。

**下游跟着改**：
- `harness/registry.py` 的 EXPECT_KEYS 加 4 个键，`status` 会核对；
- `runs_table` 不把它们当因素键（它们是记录开关，不改训练）；
- `instrumentation_check --grid` 认得新开关：开 / 关比 checksum 与全量点数值。

**要复用的现有代码**：
- `CloudServer._light_eval` / `_emit_light`（`fedavg/server/server.py:567`）；
- `BackdoorCloudServer._light_rng` / `_attacker_trigger` / `_light_asr`（`fedavg/server/backdoor_server.py:148-188`）；
- `utils/dumps.py:write_npz` + `dump_line`、`BackdoorCloudServer._write_dump`（`backdoor_server.py:253`）；
- `utils/kvline.py` 的 `format_kv` / `fmt_list`；
- `attack/eval_detail.py` 的 `margins` / `summarize`；
- `ClientUpdate.client_id`（`aggregation/client_update.py`）；
- `get_base_head_indices(...)["base_weight_indices"]`（`HierFedRepEdgeServer._base_w_idx`）。

## 二、G8 快照上的 c_k 预检（只读，不训练）

**问题**：功能分数在「整个 body」这一级有没有信号？
- 第 30 轮（攻击中）：E0 的 body vs E1–E3 的 body；
- 第 30 轮 vs 第 70 轮（衰减后）；
- 各 3 seed。
- 分不开 → 单个更新更分不开 → S6b 不做 c_k，3-D 只用几何分数。

**做法**：
- 载入 `snapshot_r030/r070.npz`：格式见 `backdoor_server.py:_snapshot`；manifest 在各 G8 metrics.json 的 `dumps.snapshots`，每个约 107.5 MB；重载逐位复现 fresh-PM 已由 `test_eval_detail_tf` 守着。
- 每个 edge body 配 **NCM head**：在本 edge 良性端留出分片的并集上取倒数第二层特征，求类均值原型。G8 用旧 noniid 划分，没有 edge 干净集，所以用留出分片代替（这些图不在任何训练集里，陷阱 #11）。
- c_k = 定向 PGD 推向类 k（固定 ε、少步数）的失败率，k = 0…9，重点看 c_{y_t}。ε / 步数先取评估 ξ 的同一组（`eval_xi`）。

**代码组织**：
- 纯 numpy 部分放 `fedavg/analysis/functional_score.py`（**不 import TF**）：NCM 原型、c_k 汇总、秩 AUROC；
- TF 部分放 `fedavg/analysis/ck_snapshot.py`：载入快照、用 G8 的配置重建划分、PGD；
- 作业脚本 `experiments/attack/hfl-mechanism/ck_precheck.sbatch`：头从 `cell.sbatch` 抄，`source cluster_env.sh` + `$PY`，单卡约 10–20 分钟（**没有实测**）。

**回传**：只回 `analysis/ck_precheck.json`（每 seed × 轮 × edge 的 c_0…c_9 与样本数）。它也是 G8 存盘的一个消费者（D-073：分析做完再删存盘）。

## 三、探路组 G1P（登记表 + materialize）

**格子**：3 run，K=3 一包。
- C1 × 集中 [10,0,0,0] × R10 × s42，开 S6a 的 ① ② ③；
- 同一格，全部关（开 / 关比较，同卡）；
- C1 × 分散 [3,3,2,2] × R10 × s42，开。

**公共设置**：
- `n_rounds: 10`（100 有效轮），`stopping: null`，`eval_grid: 5`，两个 eval_interval 取 base 的 1（= lcm(5,10)/10）；
- `federation.n_edges: 4`，划分 / 设计参数沿用 G1 的 `set:`。

**`requires: [audit, S5, S6a]`**：S6a 实现完成后把 `S6a` 加进 `available`。G1 仍挂 `S6`。

**量什么**：
- 开 / 关的 `[Checksum]` 与全量点数值逐位相同 → GPU 上的「记录不改变训练」；
- ① ② ③ 的真实单价与显存；
- 几何分数在全局 / edge 视角下的 AUROC：离线读 `update_geometry`。这一步先回答「这种攻击在几何上分不分得开」。

**机时**：约 1 GPU-h（外推，见附录 E；③ 的单价没有实测）。

**顺带修 G1 的 `set:`**（不改规模，G1 仍 blocked）：
- placement 补 `malicious_per_edge`（集中 [10,0,0,0]、分散 [3,3,2,2]）；
- R10 → `n_rounds: 30`，R20 → 15（都是 300 有效轮）；
- `stopping: null`，`eval_grid: 5`。

## 四、验证

- **L1（本地，无 TF）**：
  - `tests/test_update_geometry.py`：留一余弦与范数的解析值、CountSketch 的内积无偏性（固定种子下的精确断言）、哈希不碰全局 RNG（AST）；
  - `tests/test_functional_score.py`：NCM、c_k 汇总、AUROC 的手算值；
  - `tests/test_s6_switches.py`：开关缺省不改 config_sha、`config_validate` 的拒绝规则、防呆 ⑤；
  - `tests/test_collect_s6.py`：打印 ↔ 解析同源，老日志照读；
  - schema pin 9 → 10，`test_registry` / `test_status` 的计数。
- **TF（本地 CPU，在 `git archive` 冻结副本上跑，同 S5）**：`tests/test_s6_tf.py`，用 `_prod_seeding` 复现 main.py 的播种（见 `test_eval_grid_tf.py`）：
  - 开 / 关全部 S6a 开关时 `[Checksum]` 逐轮相同、全量点数值逐位相同；
  - 一次 post-agg / 冻结触发器评估前后，状态清单不变；
  - 反向锚点：去掉 random 围栏，训练就变；
  - 正向锚点：周期内生成器不变时，冻结列与主列逐位相同。
- **CPU 替身 L2**：截短的 G1P 配置开 / 关各跑一次，`instrumentation_check --grid` 通过（N-005 的做法）。
- **基线**：开工前先实测一次 L1 记下数字，之后多出来的红才算回归。
- **GPU**：G1P 回传后跑 `instrumentation_check`，记 F-080。
- **文档**：DECISIONS（S6a 的实现取舍）、CLAUDE.md「当前地基」S6a 一条、PLAN §2 / §5、README 工作流、REPORT §8 / §9。
- commit 前 review + 跑测试；推到 `claude/federated-learning-experiment-review-pt5j1b`，不合 main。

## 五、探路回来之后（再下一会话定）

- **c_k**：预检分得开 → S6b 做在线 c_k（每 N 个有效轮、当轮全部上传、本 edge 干净集 + NCM head）；分不开 → 3-D 只用几何分数。
- **几何 AUROC**：
  - 全局视角 ≈ 0.5 → 3-D 的答案基本已知（这种攻击藏得好），G1 可以缩到只为 3-A / 3-C 服务；
  - 明显 > 0.5 → 按 C1 臂设计 ΔAUROC（等池大小，按 seed bootstrap）。
- **参数**：定 W / N / ε / 步数，写 `harness/g1_verdict.py`（**数据回来之前**，同 D-081），再定 G1 规模（附录 E）。

---

## 附录 A：编号体系

整个改版实验 3 是一条链：**问题 → 运行组 → 判定 → 证据 → 决策**，每一层有自己的编号。另见 `current-focus.md`「几套编号」。

| 编号 | 是什么 | 例子 | 在哪 |
|---|---|---|---|
| 3.1 / 3.2 / 3.3 | 机制问题：后门为什么能在 PFL 里活下来 | 3.1 对抗下限、3.2 私有头吸收、3.3 收敛门控 | `PLAN-original` §3 |
| 3-A … 3-E | HFL 结构问题：「分层」本身改变了什么 | 3-A flat 与 HFL 的速度；3-C 锯齿；3-D edge 能否认出恶意更新 | 原文 §4–§8 |
| G0 … G8；FLR / G5AB / G6D / G8F / S5P | **运行组**：一批配置（因素网格 × seed）；带字母的是小规模探路 / 对照 | G1 = 24 run | `registry.yaml` → `results/P2/<组>/` |
| S1 … S9 | **功能会话**：写代码的会话。组的 `requires` 写着需要哪些 S | G1 只差 S6 | PLAN §5 |
| A1 … A4 | 审计会话（已结束） | — | PLAN §5 |
| A01 … A29、D01 … D06 | `AUDIT.md` 的行号：A 行 = 与官方实现的差异，D 行 = 有意偏离 | AUDIT D02 = 参与配额 | `AUDIT.md`。⚠️ AUDIT 的 D02 ≠ 决策 D-002 |
| D-001 … | **决策**（只追加；推翻时新开一条） | D-076：G8 之后保留 3-C、主量 margin | `DECISIONS.md` |
| F-001 … / N-001 … | **证据**（带状态）/ 设计备注 | F-068 = G8 的衰减结果 | `FINDINGS.md` |
| P0 / P1 / P2 | 口径版本，只有 P2 进结论 | — | PLAN §0 |

一条完整链的例子（3-C 衰减）：
1. 问题：攻击者离开后，后门会不会褪去。
2. 运行组：G8（3 run）。
3. 判定：`decay_verdict.py`，数据回来之前写定。
4. 证据：F-068，判 `user_decides`。
5. 决策：D-076，用户定 G1 保留 3-C、主量改用 margin。

## 附录 B：问题 ↔ 组（2026-10-02）

| 问题 | 组（run） | 状态 | 结论 |
|---|---|---|---|
| 3.1 对抗下限 | FLR（3）、G0（15） | 完成 | floor 不可忽略、随本 edge 目标类占比变（F-065 / F-073） |
| 3.2 私有头吸收 | G4（12） | 搁置 | 等 S6 + 重新表述（N-003） |
| 3.3 收敛门控 | G5AB（8）→ G5（15） | 完成，收尾 | `not_gated`（F-076 / D-082） |
| 3-A flat 与 HFL 的速度 | **G2（55）** + **G1 的两格** | G2 挂 `g2-scale` | — |
| 3-B 目标类分布 | **G3（24）** + G0 的 floor | 完成 | 天花板下判不出（F-075） |
| **3-C 锯齿（攻击进行中）** | **G1** | 没跑 | — |
| 3-C 衰减（攻击停止后） | G8（3）、G8F（3） | 完成 | 大部分褪去，`user_decides`（F-068 / F-071） |
| **3-D 可观测性** | **G1** | 没跑 | — |
| 3-E 三层个性化 | G6（9）、G6D（3） | 完成 | 集中布点 `blocks`；分散布点无效（F-077 / F-069） |
| 背景：预处理 | G7（6） | 已回传（stale） | F-050 |
| 工具探路 | S5P（4） | 完成 | F-079 |

## 附录 C：G1 / G2 / G3 的关系

| | G3 | G2 | G1 |
|---|---|---|---|
| 回答 | 3-B | 3-A | 3-C、3-D（+3-A 两格） |
| edge 数 | 4 | flat / 2 / 4 / 10 | 4 |
| R_edge | 5 | 2 / 5 / 10 / 20 | 10 / 20 |
| 布点 | 集中 | 分散 | 集中与分散 |
| 划分 | C1–C4 + hdir 四档 | equal_random | equal_random 与 C1 |
| seed | 3 | 5 | 3 |
| run | 24（完成） | 55（11 格 × 5） | 24 |

- **G1 与 G2 只重叠一处**：G1 的「random × 分散 × R10 / R20」= G2 故意空着的 e4-R10 / e4-R20。
  - 前提是 G1 用同一网格 G=5。
  - 停止规则只决定何时结束、不改训练，所以 G2 的自适应停止和 G1 的定长 300 有效轮在 T50 上可比。
  - 但 G1 只有 3 seed，3-A 要 5 个（补不补随 G2 规模定，D-085）。
- **G1 与 G3 不重叠**：G3 只有 R5。不过 G1 的「C1 × 集中」与 G3 的 C1 合起来正好是集中布点下 R = 5 / 10 / 20 的一条扫描。

## 附录 D：两个指标的指导意义（用户质疑后的复核）

**P0 / P1 帮不上忙。**
- 数字全部作废（陷阱 #11 / #20 / #21）。
- 从没有逐更新的记录 → 3-D 没有任何旧证据。
- `def_*` 格子实际没开防御（陷阱 #19）；FLAME exp201 是错的算法（陷阱 #3）。
- 3-C 锯齿也从没测过：P1 只在云轮末评估。

**P2 里与 3-C 最接近的只有 G8**（攻击者离开后的第 32–50 云轮，R5；受害 edge E1–E3 的均值）：

| 量 | s42 | s43 | s44 |
|---|---|---|---|
| 良性 ASR 的 OLS 斜率（每有效轮） | −0.0035 | −0.0031 | −0.0077 |
| margin 中位数的 OLS 斜率（每有效轮） | −0.044 | −0.055 | −0.074 |
| 相邻两点差的 SD：ASR | 0.070 | 0.082 | 0.076 |
| 相邻两点差的 SD：margin | 0.80 | 1.10 | 0.78 |

<details><summary>重算命令（在 <code>experiments/attack/hfl-mechanism/</code> 下运行；纯标准库）</summary>

```python
import json, statistics as st
R = 5
def ols(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
for s in (42, 43, 44):
    d = json.load(open(f'results/P2/G8/G8__a__s{s}.metrics.json'))
    pe, det = d['per_edge_rounds'], d['per_edge_detail_rounds']      # det 列：edge_id, margin_p50, ...
    rs = list(range(32, 51))
    asr = [st.mean(e['client_benign'] for e in pe[str(r)] if e['edge_id'] in (1, 2, 3)) for r in rs]
    mar = [st.mean(row[1] for row in det[str(r)] if row[0] in (1, 2, 3)) for r in rs]
    da = [b - a for a, b in zip(asr, asr[1:])]; dm = [b - a for a, b in zip(mar, mar[1:])]
    print(s, ols(rs, asr) / R, ols(rs, mar) / R, st.stdev(da), st.stdev(dm))
```
</details>

> 这是会话里的临时计算，**没有进 harness 脚本，所以不进 FINDINGS**（README 规矩：数字只从脚本产物来）。
> G1 的判定脚本写好后以它为准。

**3-C 的判断：指导意义偏弱。**
1. **自清洁慢。** 折成一个云周期，R10 约 0.03–0.08、R20 约 0.06–0.15，与逐点噪声同一量级。
   - 就算 r_down > 0，也是「每个周期洗掉一点、下次云聚合又灌回来」。
   - 「隔离 / 时间窗」这条防御路线很可能弱。这是推论，**攻击进行中的数据还没有**。
2. **两个混杂可能让 r_down 虚高。**
   - 攻击进行中，共享生成器在周期内还在训练（只对着 E0），触发器漂移本身会让受害 edge 的 ASR 降。G8 停手后生成器冻结，没有这个问题。
   - lr 按 0.992^有效轮 衰减，后半程周期内本来就动得少。
   - → S6a 的 ③（冻结触发器列）+ 判定时 r_down 只取前半程周期。两条混杂有多大，**没有证据**。
3. R10 / R20 下的 floor 没测过（G0 只有 R5）。

**3-D 的判断：对防御设计更有用，但有前提。**
1. random 划分下各 edge 可交换，等池大小时 edge 视角就是全局视角的随机子集 → ΔAUROC ≈ 0 是构造出来的。random 臂只是零对照，信息只在 C1 臂。
2. Bad-PFL 的更新在几何上可能根本分不出来（两种视角的 AUROC 都 ≈ 0.5）。那样 ΔAUROC ≈ 0 说的是「这种攻击藏得好」，不是「edge 没用」。Bad-PFL 论文自称能绕过多种防御，**仓库里没有这方面的证据**。
3. 所以先要知道：全局视角的 AUROC 是否明显 > 0.5 → 这正是 G1P 要回答的。

## 附录 E：机时（外推，没有实测）

**单价假设**：
- 4 edge 训练约 20 s / 有效轮，全量评估约 80 s、轻评估约 40 s（S5P 是 2 edge 的单价，F-079）；
- c_k 每 run 约 0.3 h（**纯猜**）；
- K=3 合包，GPU-h ≈ run 小时 ÷ 3（F-059）；
- 周预算约 100 GPU-h（D-047）。

**每 run 的拆分**（方案 A）：
- R10：训练 1.67 h + 全量 30 × 80 s + 轻评估 30 × 40 s + post-agg 30 × 40 s + c_k 0.3 h ≈ **3.3 h**；
- R20：训练 1.67 h + 全量 15 × 80 s + 轻评估 45 × 40 s + post-agg 15 × 40 s + c_k 0.3 h ≈ **3.0 h**。

| 方案 | run | GPU-h |
|---|---|---|
| 预检：G8 快照上的 c_k | 0 | 约 0.1 |
| 探路 G1P | 3 | 约 1 |
| A 全量：G=5 + 云聚合后一点 | 24 | **约 25** |
| A，R10 全部 + R20 只做集中格 | 18 | 约 19 |
| A，只做 R10 | 12 | 约 13 |
| B：A + 集中格逐 edge 轮的客户端面板 | 24 | 约 28 |
| C：全部 G=1 | 24 | 约 44 |
| 任一方案去掉 c_k | — | 每 24 run 约省 2.4 |
| 3-A 补 s45 / s46 | +4 | 约 +4 |
