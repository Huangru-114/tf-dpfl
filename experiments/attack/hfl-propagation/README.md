# Experiment 3A/3B —— 恶意客户端空间分布 × edge 粒度（config 集）

层级结构如何改变后门传播。**跨所有格恒定的不变量**（对齐 exp007）：
100 client · 10 恶意（全局 10%）· `edge_rounds=5 × n_rounds=80`（=400 有效轮）·
badpfl · hier_fedrep · resnet10(BN) · `shared_generator=true` · `forced_participation=false` ·
`poison_ratio=0.2`。**唯一自变量 = 恶意端在多少个 edge、怎么分布**（`edge_assignment=block`
+ `malicious_placement=by_edge` + `malicious_per_edge`）。

> 依赖 enabling 分支（`block` 分配 / `by_edge` 布点 / per-edge 指标 / 补位修复）。本分支已
> 叠在其上；合并时先合 enabling。

## 实验格（8 个）

| config | n_edges | `malicious_per_edge` | 拓扑 | 状态 |
|---|---|---|---|---|
| `2edge_collocated`   | 2  | `[10,0]`                    | 全挤 E0 | 待跑 |
| `2edge_distributed`  | 2  | `[5,5]`                     | 均分（≈exp007） | **已完成**（留档复现） |
| `4edge_collocated`   | 4  | `[10,0,0,0]`                | 全挤 E0，E1–3 干净 | 待跑 |
| `4edge_distributed`  | 4  | `[3,3,2,2]`                 | 尽量均摊 | 待跑 |
| `4edge_mixed`        | 4  | `[5,3,2,0]`                 | 密度梯度 | 待跑 |
| `10edge_collocated`  | 10 | `[10,0,…]`                  | E0 内 100% 恶意（无良性） | 待跑 |
| `10edge_distributed` | 10 | `[1,1,…,1]`                 | 每 edge 1 个 | 待跑 |
| `10edge_mixed`       | 10 | `[4,3,2,1,0,…]`             | 梯度，6 干净 edge | 待跑 |

flat（无层级，R_edge=1）已完成，是 3C 的极端点，不在本批。

block 分配下 edge 成员是连续 id 段（100/2 → E0=0–49；100/4 → E0=0–24；100/10 → E0=0–9），
恶意端由 seed 在各 edge 内确定性抽取。

## 怎么跑（脚本）

**先 preflight（无 GPU，登录节点/容器内秒~分钟级）**：
```bash
bash experiments/attack/hfl-propagation/test_exp3.sh
```
L1（placement/forced_participation/collect_metrics）+ 8 config 校验 + 布点 dry-check
（真跑 block+resolve，断言每 edge 恶意数==malicious_per_edge）。绿了再烧 GPU。

**批量提交（每格一个独立 GPU job，断点续跑）**：
```bash
bash experiments/attack/hfl-propagation/run_exp3.sh --dry-run   # 先看清单
SEEDS="42" bash experiments/attack/hfl-propagation/run_exp3.sh  # 先 1 seed 探路
bash experiments/attack/hfl-propagation/run_exp3.sh            # 全量（8×3=24 job）
bash experiments/attack/hfl-propagation/run_exp3.sh --status   # 看进度
```
产物：`experiments/attack/hfl-propagation/results/<cell>_seed<seed>.metrics.json`。
每格独立日志（带 exp_id）——不会像直接并发 `run_full.sh` 那样互相覆盖日志。

## 怎么跑（手动，单格）

每格一条 sbatch（第 5 参 = exp_id，回传 metrics.json 的落点名；第 7 参 = config）：
```bash
sbatch run_full.sh attack hfl-propagation badpfl none 2edge_collocated  hier_fedavg_fedrep \
       experiments/attack/hfl-propagation/2edge_collocated.yaml
sbatch run_full.sh attack hfl-propagation badpfl none 4edge_collocated  hier_fedavg_fedrep \
       experiments/attack/hfl-propagation/4edge_collocated.yaml
# …4edge_distributed / 4edge_mixed / 10edge_* 同理，把两处名字换掉即可
```
产物：`experiments/hfl-propagation/<exp_id>.metrics.json`（每格一个）。

**多 seed**：改 config 里 `seed: 42`→43/44 各跑一遍（或等驱动脚本），每格 3 seed 报 mean±std。
等概率抽样有参与度方差，单 seed 不足以下结论。

## 读什么（判据）

用 `metrics.json` 的 **`per_edge_final`**（逐 edge 面板），**不要用被均值抹平的 `edge_asr_mean`**：
- Local amplification：`per_edge[e].edge_asr > per_edge[e].client_benign`
- Hierarchical amplification：`final.global_asr > mean(edge_asr over edges)`
- Dilution：`per_edge[e].edge_asr < per_edge[e].client_malicious`
- Cross-edge cancellation：各 `edge_asr` 高但 `final.global_asr` 低

**干净 edge 的 `client_benign`（collocated/mixed 里 E1… has_malicious=false）= 跨 edge 传播强度**
—— exp006/007（每 edge 都有恶意）测不到，这批第一次能测。

## Experiment 3C —— 聚合频率（6 config）

固定拓扑 = **2edge distributed `[5,5]`**（= exp007 锚点在 R_edge=5），唯一自变量 = `edge_rounds`
`R_edge`，`n_rounds` 随之使 `R_edge × R_cloud = 400`（有效 local budget 恒定）：

| config | edge_rounds | n_rounds | `eval_interval`(cloud) | 评估点数 | 含义 |
|---|---|---|---|---|---|
| `3c_R1`  | 1  | 400 | 10 | 40 | 每轮都 cloud 聚合，但**保留 edge 层** → 与 flat 的归因对照 |
| `3c_R2`  | 2  | 200 | 5  | 40 | |
| `3c_R4`  | 4  | 100 | 2  | 50 | |
| `3c_R5`  | 5  | 80  | 2  | 40 | = exp007 层级锚点 |
| `3c_R10` | 10 | 40  | 1  | 40 | |
| `3c_R20` | 20 | 20  | 1  | 20 | 强层级；`eval_interval` 已到下限 1 |
| `3c_R40` | 40 | 10  | 1  | 10 | 极端；同上，只有 10 个点，轨迹比较到此为止 |

> **`eval_interval` 按有效轮对齐**：它数的是 **cloud round**，而有效轮 =
> cloud × `edge_rounds`。此前各格一律设 5，于是等预算 400 下 flat 拿到 80 个评估点、
> R20 只有 4 个、R40 只有 2 个 —— 任何「谁涨得快」的轨迹比较在大 R 的格子上
> 根本没有分辨率。现在统一成**约每 10 个有效轮评一次**；R20/R40 受 cloud 轮数
> 下限所限（`eval_interval` 不能小于 1），是设计固有的稀疏，不是配置错误。
> 守卫：`tests/test_exp3_config_invariants.py`。

> **`3c_R1` 是归因的关键格**，README 早就设计了它但文件一直没被创建。
> `flat_baseline` 同时改了 `n_edges 2→1` **和** `edge_rounds 5→1` 两个变量，
> 所以「引入 edge 层抑制传播」这个主发现此前无法归因。三格对照：
> `flat_baseline`(1 edge, R=1) vs `3c_R1`(2 edge, R=1) vs `3c_R5`(2 edge, R=5)。

跑法（复用同一套脚本，用 `CONFIGS=` 缩到 3C）：
```bash
CONFIGS="3c_R1 3c_R2 3c_R4 3c_R5 3c_R10 3c_R20" bash experiments/attack/hfl-propagation/run_exp3.sh
```
判据：`pm_acc`(MTA) + ASR client/edge/global 随 `R_edge` 的曲线（`fig_3c_frequency.png`）。
斜率回答「edge aggregation 是放大器还是稀释器」。
> ⚠️ **parameter / representation drift 需要另加训练循环仪表，本批未含** —— 3C 现在能出
> MTA + 三层 ASR 的频率曲线（用户发现的主曲线），drift 列待补。

## flat（两层）基线 —— `flat_baseline.yaml`

`n_edges=1 + edge_rounds=1` → 退化成普通 FedAvg（cloud↔client，无 edge 中间层），
`n_rounds=400` 与层级各组**等有效预算**。放进时间序列图作参照，凸显「引入层级结构」本身
对后门植入/主任务的影响（已观察到 flat 的 benign ASR 明显更高、更接近论文）。
```bash
sbatch run_full.sh attack hfl-propagation badpfl none flat_baseline hier_fedavg_fedrep \
       experiments/attack/hfl-propagation/flat_baseline.yaml
```
> ⚠️ `n_edges=1` 是本仓库层级路径少走的角落，**第一格先单独 smoke 一下**确认能跑通
> （same/diff-edge、聚合分支在单 edge 下的行为），再纳入批量。

## 画图 —— `plot_exp3.py`

跑完（或跑到一半）后，把每格 metrics.json 聚合成图：
```bash
python3 experiments/attack/hfl-propagation/plot_exp3.py     # 读 results/，图落 results/figures/
```
- 跨 seed 同设定 → **均值 + [min, max] 界**（须/带），不是 ±std。
- 颜色用 Okabe–Ito 色盲安全色板，每个量一个独立色相（高对比、非同色系）。
- 产出五张：`fig_topology_summary.png`（3A/3B 各拓扑 global/benign/PM 柱）、
  `fig_per_edge_propagation.png`（3A 逐 edge，含恶意 edge 灰底）、`fig_3c_frequency.png`（3C 频率曲线，含漂移下panel）、
  **`fig_timeseries_topology.png` / `fig_timeseries_3c.png`**（benign ASR & MTA vs **有效轮**
  `cloud_round×edge_rounds`，各组一条线 + **flat 两层基线黑虚线**——最能体现组间差异与架构影响）。
- 只需 matplotlib+numpy，不需要 GPU/TF；缺 matplotlib 就 `pip install matplotlib` 或在容器里跑。
  单 seed 时 min–max 界退化为点，会打印提醒。

## 注意

- 10edge_collocated 的 E0 是 100% 恶意（10/10），E0 无良性端——这是最大共谋密度点，不是 bug。
- `n_malicious: 10` 在 by_edge 下不定数量（`malicious_per_edge` 说了算），保留仅作记录；
  两者和恒为 10，`config_validate` 会早筛长度/容量。
