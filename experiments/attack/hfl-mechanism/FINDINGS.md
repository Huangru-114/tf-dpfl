# 结论台账 —— Experiment 3（改版）

> 每条结论 = 声明 + 证据（能重跑的命令或 文件:行）+ 状态。
> 状态：`confirmed`（证据是代码或数据本身）· `provisional`（只有 P1 / 单 seed，当假设用）· `retracted`（撤回，写原因）。
> **数字只从脚本产物或复核命令里来，不凭记忆写**（陷阱 #14）。
> P2 数据回来后，结论性的条目由 `harness/verdicts.py` 的输出支撑，并注明 run 集合与 n_seeds。

## 2026-09-24（S1，评审时核实）

### F-001 `confirmed` —— 旧方案的 6 个 `def_*` 格没有开防御

- `experiments/attack/hfl-propagation/exp3_cell.sbatch:46` 写死 `--defense none`，而 `fedavg/main.py:225-230` 让 CLI 覆盖 yaml 的 `defense.name`（yaml 里写的是 `median` / `multi_krum`）。
- 6 个 `results/def_*_seed42.metrics.json` 全部是 `run.defense="none"`、`admitted=[]`。
- 修复：D-007 推迟；陷阱 #19。S1 起 `[Provenance]` 行会记录 `cli_overrides`，`harness/status.py` 会把这类格子标成 `mismatch`。

### F-002 `confirmed` —— 同配置、同 seed 重跑，第 1 轮就分叉

F-001 的 6 个格子等于 3 组同配置、同 seed 的重复（恶意端 id 一致）：

| 组 | 第 1 轮 gm_acc | 末 10 点 benign ASR |
|---|---|---|
| flat_baseline | 0.1144 / 0.1116 / 0.1122 | 0.755 / 0.696 / 0.685 |
| 2edge_distributed | 0.2973 / 0.3285 / 0.3331 | 0.738 / 0.723 / 0.756 |
| 4edge_distributed | 0.2167 / 0.2201 / 0.2115 | 0.972 / 0.963 / 0.910 |

按**实际因素**分组后（`harness/runs_table.py`），还多出两组之前没人注意到的同配置同 seed 重复：

- `3c_R5` 与 `2edge_distributed` 的因素完全相同 → 2edge_distributed 实际有 **4** 份：末 10 点 benign ASR 为 0.738 / 0.724 / 0.723 / 0.756；T_0.5 为 46.8 / 46.9 / 43.6 / 48.2 有效轮。
- `stoptest_10edge` 是 `10edge_distributed` 的第 2 份：末 10 点 benign ASR 为 0.729 / 0.723。

复核：`python3 harness/runs_table.py experiments/attack/hfl-propagation/results --out <dir> --legacy-protocol P1` → `runs.csv` 的 `replicate` / `n_replicates` 列。

影响：按 seed 配对只锁得住划分和布点，锁不住训练轨迹。AUDIT A15。

### F-003 `confirmed` —— 学习率按云轮衰减，flat 与 HFL 的 LR 日程不同

- `fedavg/client/client_base.py:117-126`：`lr = 0.1·0.992^cloud_round`。
- 同样 200 有效轮，末端 lr：flat（R=1）0.020 · R=5 0.073 · R=40 0.096。
- 官方是常数 lr（AUDIT A08）。

### F-004 `confirmed` —— 评估时 ξ 在受害者模型上白盒计算，官方在攻击者模型上

- TF：`main.py:170-172` → `client_badpfl.py:164-172`。
- 官方：`fba.py:53,64`。
- 已决定对齐（D-004）。
- `experiments/METRICS.md` 里「ξ 用的是 mal[0] 的模型」这句与**两边都不符**：官方是循环里最后一个恶意端，TF 是受害者模型。该文件与 Bad-PFL 库双份同步（`test_metrics_doc.py`），留给两库一起改。

### F-005 `confirmed` —— 旧方案的 edge 与数据无关

- 25 个 yaml 全是 `partition: "noniid"` + `edge_assignment: "block"`。
- `inter_edge` / `intra_edge` 只在 `partition: hierarchical` 分支被读（`main.py:373-385`），所以在这些配置下是死配置。

### F-006 `confirmed` —— 旧 rho0 格不是 Bad-PFL 的下限

- 没有恶意端时，`build_eval_trigger` 回退到静态 BadNet 触发器（`main.py:171`），0.037 测的是 BadNet。
- 真正的下限用 ρ=0 影子攻击者（PLAN §1.1）。

### F-007 `confirmed` —— FedRep 下 edge / global 模型的 head 是初始化 head；PM 是陈旧的

- `server/hier_fedrep.py:59-65` 只把 backbone 索引写回 edge 模型，head 保持下发值；cloud 平均的是各 edge 未变的 head。
- `client.model` 只在被抽中时更新（`client/hier_fedrep.py:91-109`），10% 参与率下大多数良性端是旧 body。
- 影响：edge_asr / global_asr 的含义要重新定义；3-C 主曲线用 fresh-PM ASR。

### F-008 `provisional` —— P1 中 4edge_distributed / 4edge_mixed 的 ASR 明显高于 2edge 和 10edge

- 末 10 点 benign ASR 约 0.97 / 0.94，高于 2edge 的 0.74 和 10edge 的 0.73。
- 在 3 次同 seed 重复里都出现（0.972 / 0.963 / 0.910），所以不是训练噪声。
- 候选解释是 seed42 的划分 × 布点，**没有证据**。只作假设，由 P2 检验。

### F-009 `provisional` —— P1 中 flat 并不比 HFL 快

- T_0.5(benign)：flat 57 / 2edge_dist 47 / 4edge_dist 23 有效轮（`exp3_analysis.csv`）。
- 单 seed，且有 F-003 的 LR 混淆 → 只作假设；正是 3-A 要回答的问题。

### F-010 `confirmed` —— 旧分析口径的两个问题

- `harness/analyze_exp3.py` 的终值取**单个末轮**，而标定规定用末 10 点均值（`experiments/calibration/RESULTS.md` §4.2）。
- HHI 回归把 ρ / def / 3C / stoptest 等格子也混了进去（n=25，R²≈0.02）。
- S1 的 `runs_table.py` 改用末 10 点均值，并按实际因素分组。

### F-011 `confirmed` —— 旧 metrics.json 没有溯源信息

- 没有 git commit、config hash、run_id，也没有时间戳；`run_name` 在所有 exp3 格子里都相同。
- 一个格子的身份只靠文件名。S1 加 `[Provenance]` 行。

### F-012 `confirmed` —— FedRep 的 head / body 训练量与 docstring 相反

- `client/hier_fedrep.py:21-22` 写「plocal_epochs > local_epochs」（head 多步、body 少步，FedRep 原则），而全部 exp3 配置是 `plocal_epochs: 1` / `local_epochs: 5`。
- 影响未知 → AUDIT A12。

### F-013 `confirmed`（机制）/ 无数值证据（效果）—— 官方 ρ=1 时 PGD 与生成器相互抵消，TF 版不会

- 官方生成器用 `PoisonClient.fetch_data()` 的数据训练（`fba.py:36`，`client.py:124-125`）。这些数据已按 ρ 投毒、标签为 target，于是 PGD 推离 target、生成器损失推向 target。
- TF 版用干净数据训练生成器（`client_badpfl.py:100`），所以不会抵消。
- 这证实了 METRICS.md 的推断在官方代码路径上成立，但**抵消在数值上有多大仍无证据**。AUDIT A04。

### F-014 `confirmed` —— 数据增强只采样一次

- `client/hier_fedrep.py:64` 在构造时 `list(self.dataset)`，把增强后的 batch 缓存下来，之后每个 epoch 重复使用。
- 官方不做增强。AUDIT A10。

### F-015 `confirmed` —— `training.label_smoothing: 0.1` 是死配置

- `fedavg/client` / `server` / `models` 里没有任何地方读它。AUDIT A16。

### 设计备注（不是缺陷，但解读结果时必须带上）

- **N-001**：参与配额下，10 个 edge 时每个 edge 每轮只有 1 个客户端 → edge 内聚合退化成复制这一个模型，edge 内稀释不存在。AUDIT D02。
- **N-002**：`grid_too_coarse` 有两处含义：
  - 停轮规则（`server/stopping.py:116-117`）：评估点凑不齐 `pm_window=10` 个。3c_R40 只有 8 个点，所以一直跑到上限。
  - T_θ（`harness/analyze_exp3.py:71,314`）：少于 4 个点就不插值。
