# 项目：分层联邦学习 后门攻防 移植-实验 hub

本仓库是 hub：**唯一**与集群同步的 git 仓库。活跃代码只有 `fedavg/`。
移植对象是**训练循环 / 聚合算法**（不是网络结构），所以验收信号是
「算法不变量 + 端到端 smoke」，不是「逐层 activation 对拍」。

---

## ⚠️ 集群运行环境（先看这段，否则一切都跑不起来）

**集群上所有 python 都必须在 apptainer 容器里跑。裸 `python3 xxx.py` 一个库都找不到。**

```bash
apptainer exec --nv /nobackup/proj/disk/naiss2025-22-1095/personal/ziangg/tensorflow.sif python3 ...
```

**`.sh` 脚本必须是 SLURM 格式、用 `sbatch` 提交**才会真正跑起来。完整模板：

```bash
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus 1
#SBATCH -t 24:00:00
#SBATCH -A naiss2026-4-650-gpu
#SBATCH -p gpu
#SBATCH --mem=24G

module load GPU/buildenv-nvhpc/25.9-cu13.0

apptainer exec --nv /nobackup/proj/disk/naiss2025-22-1095/personal/ziangg/tensorflow.sif python3 -m 你的模块
```

**仓库里这件事已经收口到 `cluster_env.sh`**，不要在新脚本里再硬写容器路径：

```bash
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
source "$ROOT/cluster_env.sh"     # 解析出 $PY
$PY -m pytest tests/
$PY main.py --config ...
```

`cluster_env.sh` 的行为：检测到 apptainer + 容器存在 → `module load` 并走容器；
否则回退裸 `python3`（本地开发，需要 TF 的测试自动 skip）。
它会打印 `[env] python = ... (mode=apptainer|local|override)`，
**每次跑之前扫一眼这行** —— 静默地跑在错误的环境里是最难查的一类问题。
`TFDPFL_PY` / `TFDPFL_SIF` / `TFDPFL_BIND` 可覆盖。

**⚠️ `--bind <仓库上一级>` 是这套设计的地基，不要拆（2026-09-08 定论）**

`cluster_env.sh` 里的

```bash
TFDPFL_BIND="${TFDPFL_BIND:-$(cd "$_TFDPFL_ROOT/.." && pwd)}"
PY="apptainer exec --nv --bind $TFDPFL_BIND $TFDPFL_SIF python3"
```

**是 2026-08 全部 Arrhenius run 验证过的**（exp3 的 3C 全批就是这么跑出来的，
数据提交 `20515a4` 的作者机是 `arrhenius1`）。apptainer 默认只挂 `$PWD` 和
`$HOME`，而本仓库的作业要跨目录够到三处 —— 绑定仓库上一级**一次覆盖全部**：

| 需要 | 路径 |
|---|---|
| 配置（cwd 在 `fedavg/` 时是兄弟目录） | `$ROOT/experiments/...` |
| 日志 | `$ROOT/../tfdpfl-logs/...` |
| keras 数据缓存 | `$ROOT/../data/datasets/` |

所以作业脚本一律 `cd "$ROOT/fedavg"` + `$PY main.py`，日志放 `$ROOT/../tfdpfl-logs`。
**这些都不要"优化"。**

> ### 2026-09 的三轮误判（写下来免得重演）
>
> 起因：`run_calibration.sh` 被加了 `source cluster_env.sh`，于是启动自检
> **在登录节点** arrhenius1 上跑了。自检当时用 `$PY`，而 `$PY` 带 `--nv` ——
> 登录节点没有 NVIDIA 驱动，容器根本起不来，自检却报
> 「✗ 容器里看不到仓库目录」。
>
> 那句话把人引向 `--bind`。于是连续三轮拆掉了正确的设计：
> `--bind` 改成默认不加 → `cd` 从 `fedavg` 挪到仓库根 → 再挪到上一级，
> 每一轮换一个 `FileNotFoundError`（配置 → 日志 → keras 缓存），
> 因为每一轮都只补上了 `$PWD` 恰好没盖住的那一块。
>
> **8 月之所以没暴露**：`run_exp3.sh` 在登录节点只调 `sbatch`、**不 source**
> `cluster_env.sh`，自检只在计算节点跑过。
>
> 现已修好的是自检本身：它只关心**挂载**，不需要 GPU，所以改用**不带 `--nv`**
> 的等价命令，并回显 apptainer 原话。守卫：
> `tests/test_cluster_env_usage.py::test_self_check_does_not_use_nv`
> 与 `::test_bind_defaults_to_the_repo_parent`（两个反向锚点都实测过）。
>
> **教训**：一条自检报出的原因，本身也可能是错的。它说「看不到仓库」之前，
> 先确认容器起得来 —— 这两件事的修法南辕北辙。

**`.sif` 必须写绝对路径**；脚本与数据路径可以相对（`--bind` + `$PWD` 都在）。

**Bad-PFL（torch）用的是另一个容器**：
`/nobackup/proj/disk/naiss2025-22-1095/personal/ziangg/torch_fl.sif`，
而且入口是 `python` 不是 `python3`。两个仓库的容器不要混用。

**keras 数据缓存的软链坑**：`~/.keras/datasets` 往往是指向共享盘的符号链接。
容器里看不到链接目标时它就是**悬空**的，`os.path.isdir()` 为 False，
keras 的 `os.makedirs(..., exist_ok=True)` 去 mkdir 撞上链接本身，抛出

```
FileExistsError: [Errno 17] File exists: '/home/<user>/.keras/datasets'
```

这句话和真实原因毫无关系（既不是「已存在」也不是权限）。
`cluster_env.sh` 检测到仓库上一级的 `data/datasets` 就导出 `TFDPFL_KERAS_HOME`
绕开软链；`data/dataset.py:resolve_keras_home` 另外会在悬空时**提前拦截**。
> 注意 `CIFAR_MIMER_PATH = /mimer/NOBACKUP/Datasets/CIFAR` 是 **Chalmers Mimer**
> 的路径，在别的集群上不存在，会静默走到 `~/.keras` 那条 fallback 上。

### 容器里的已知噪音（不是故障）

```
ERROR:absl:cannot import name 'runtime_version' from 'google.protobuf'
```

`google.protobuf.runtime_version` 是 protobuf ≥ 5.27 才有的模块，容器里是 4.x，
某个用新 protoc 生成的 stub 去 import 它失败后被 absl 记了一条 ERROR 就继续了。
**`ERROR:absl:` 是日志级别，不是异常，不会让进程退出。**
证据：同一个容器跑 `pytest` 时 TF 完全正常（219 passed）。
run 真的死掉时，杀死它的是别的东西 —— 去看 traceback，不要盯着这一行。

| 脚本 | 怎么跑 |
|---|---|
| `run_l1.sh` | `bash run_l1.sh`（秒级，登录节点可以跑；集群上会自动进容器） |
| `run_smoke.sh` | **`sbatch run_smoke.sh <axis> <method> <attack> [defense] [exp_id] [framework]`**（要 GPU、几分钟，别在登录节点用 bash 跑） |
| `submit_matrix.sh` | `bash submit_matrix.sh`（它自己 sbatch 一个 job array） |
| `experiment_tf.sh` | 不要直接跑，由 `submit_matrix.sh` 提交 |

---

## 新会话开场：先做这三件事

1. **确认基线并开分支**
   ```bash
   git fetch origin && git log --oneline origin/main -1   # 看 trunk 在哪
   git checkout -b claude/<axis>-<method>-<短随机> origin/main
   ```
   **永远从 `origin/main` 分出去**，不要从别的 `claude/*` 分支分，也不要在别人的
   分支上接着写。除非我明确说「叠在 X 分支上」（那种情况下 X 一定是还没合进 main
   的直接前置，比如正交化之于 Neurotoxin/Bad-PFL）。
2. **`bash run_l1.sh`** —— 基线分两种环境，对不上就先停下来问我，别在坏掉的地基上改东西。

   > ⚠️ **下表的数字已过期，开工前必须自己实测一次**，不要拿它当门禁基准。
   > 两个原因：(a) `test_badpfl_trigger` 的 4 条红已于 2026-08-21（`f76ff71`/`e34da4b`）
   > 修好，见 `experiments/attack/bad-pfl/current-focus.md:68` —— 集群实际红灯应为
   > **2 条**（`test_neurotoxin_mask`，陷阱 #4），不是 6 条；
   > (b) 本轮新增了 4 个测试模块（seeding / leakage / quota / undefined-metrics /
   > exp3-config），通过数会涨。
   > **正确做法**：`bash run_l1.sh 2>&1 | tail -3` 记下当天的数字作为基线，
   > **之后多出来的红才是回归**。

   | 环境 | 历史记录（2026-08-20 实测，**已过期**） | 说明 |
   |---|---|---|
   | **本地（无 TF）** | `172 passed / 4 skipped / 3 xfailed` → PASS（exit 0） | 4 skipped = 4 个需要 TF 的测试模块整体 skip |
   | **集群（有 TF）** | `213 passed / 3 skipped / 3 xfailed / 6 failed` → FAIL | 当时的 6 条红；其中 4 条已修，现应为 2 条 |

   > 若在容器里直接跑裸 `pytest`（不加 `tests/`），会多收 6 条
   > `fedavg/defense/test_defenses_offline.py` → **219 passed**。
   > `run_l1.sh` 只跑 `tests/`，所以是 213。两个数都对，别被吓到。
   > 已实测（2026-08-20，容器内 Python 3.10.12）：`6 failed, 219 passed, 3 skipped, 3 xfailed`。

   > **本地解释器是 Python 3.12 / 3.13 时会多出与代码无关的红**（2026-10-09，FINDINGS F-089，`experiments/defense/edge-native/`）：
   > AST 指纹测试（`ast.dump` 的输出随版本变）与一条第 4 位小数舍入（3.12 起内置 `sum()` 对浮点做补偿求和）。
   > 本地基线要用 Python 3.11 的 venv 经 `TFDPFL_PY` 跑 —— 那样是 `1599 passed / 45 skipped / 3 xfailed`（与 2026-10-03 的交接一致）。

   集群上 `run_l1.sh` 返回 FAIL 是**当前的预期状态**，不是回归：
   `test_neurotoxin_mask` ×2 是陷阱 #4（mask 语义方向未证实，测试按文献语义写、
   等实现被改过来）。
   ~~`test_badpfl_trigger` ×4~~ **已修**（2026-08-21，`f76ff71`/`e34da4b`）：
   `build_autoencoder` 加了 `assert img_size % 16 == 0`、STD 改为按 dataset 匹配、
   测试自身的 float32 舍入改用 `np.isclose`。
   （3 xfailed 是 FLAME 的已知 bug，陷阱 #3。）
3. **读 `experiments/<axis>/<method>/current-focus.md`** —— 本会话**唯一**要回答的问题
   和客观判据都在里面。没有这个文件就先和我一起写，不要直接开始改代码。

然后按下面「交互约定」走：先出语义 diff 表，我确认后再动代码。

### 分支纪律

- **trunk = `origin/main`**，集群只 pull main。
- **一条分支 = 一个会话 = 一个方法/模块**。分支不要活过一个会话。
- 会话结束、L1 绿了 → 合回 main。**不要让长链堆积**：分支一旦落后 trunk 很多，
  下一个会话就得先判断「我该从哪儿分出去」，判断错一次就是一次重复劳动。
- 并行两个方法时，先合一条，第二条合之前 `git merge origin/main` 把第一条的
  文档改动（CLAUDE.md / methods-registry.md）吃进来 —— 代码文件通常不冲突，
  冲突几乎只发生在这几个文档上。

## 当前地基（已完成，不要重复做）

- **上行聚合已统一**：所有 PFL 方法、edge 与 cloud 两层都经
  `RobustAggregationMixin.robust_mean`（`server/robust_aggregation.py`），
  defense 轴对每个方法都生效。守卫：`tests/test_defense_coverage.py`（AST 静态检查）。
- **下行广播已统一**：所有 edge server 都走 `EdgeServerBase.broadcast_to_clients`，
  主动防御的下行载荷经 `client.set_control(...)` 到达客户端。
  （此前 6 个 edge server 里只有 1 个走它，其余内联 `client.set_weights` →
  任何下行载荷在 5/6 的方法下静默送不到。）守卫：`tests/test_broadcast_coverage.py`。
- **客户端行为组合机制**：攻击/防御都是 mixin，与 PFL 方法类**组合**而非替换
  （`client/compose.py`）。钩子协议在 `FLClientBase`，默认全部无操作：
  `set_control` / `on_round_start` / `on_batch` / `on_extra_loss` / `on_upload` / `get_aux`。
  **新增攻击或防御时写 mixin + 钩子，不要写 Client 子类。**（见陷阱 #1）
- **上传带身份**：`aggregation/client_update.py` 的 `ClientUpdate` 继承 tuple，
  旧解包写法全部照旧，额外有 `.client_id` 和 `.aux`（客户端上行载荷）。
  收集顺序严格按 `selected` 顺序。
  防御的接纳/剔除经 `BaseDefense.record_decision` 翻译成 `last_admitted_ids`。
- **随机性已播种**（见陷阱 #2，**有一处漏网**）。
- **启动时配置校验**：`config_validate.py`。新增 PFL 方法必须登记到
  `METHOD_SUPPORTS_DEFENSE`；防御要在类属性 `layers` 里声明自己能作用的层，
  与 `defense.layers` 对不上就拒绝启动。
- **Hier-PerFedAvg 已移除**（聚合元梯度，与防御接口语义不兼容），
  `_collect_updates_*` 的 `meta_grad` 模式随之删除。
- **矩阵提交器**：`matrix.conf` + `submit_matrix.sh` + `experiment_tf.sh`
  + `harness/collect_matrix.py`。
- **墙钟已可拆分**：`_backdoor_eval` 分段计时（asr / feature / forgetting / drift）
  打一条 `[Timing] Round N | asr=…s | … | total=…s`，`collect_metrics` 另从
  `[Cloud]` 行抓 `time=`，`metrics.json` 出 `timing_rounds` + `timing_summary`
  （`round_time_total_s` / `bd_eval_total_s` / `bd_eval_fraction`）。
  **口径**：`round_time` 测的是 `CloudServer.run_round` 的 t0→elapsed，而
  `_backdoor_eval` 在 `super().run_round()` **返回之后**才跑 → 两者要**相加**
  才是一轮的墙钟。标定 (local_epochs, n_rounds, eval_interval) 读的就是这几个数。
  守卫：`tests/test_eval_timing.py`（上游 AST + 下游解析两侧分开测）。
- **run 块已自描述到「哪一格」的程度**：`[设定]`（`ac797cd`，含 `edge_rounds`）
  与 `[设定2]`（`<本次>`，含 `malicious_per_edge` / `malicious_placement` /
  `edge_assignment` / `local_epochs` / `plocal_epochs` / `seed` / 两个
  `eval_interval`）两条行 → `metrics.json` 的 `run` 块。
  **`[设定2]` 是独立一行不是扩 `[设定]`**：`RE_SETTINGS` 是全或无的正则，
  往里加字段一旦格式对不上，原有八个字段会**一起变 None** 而日志毫无异常。
  守卫：`tests/test_run_self_description.py`（含反向锚点
  `test_old_log_without_settings2_still_parses_the_first_line`）。
  > ⚠️ `edge_rounds` 在归档的老 `metrics.json` 里缺失，那是 `ac797cd` **之前**
  > 跑的文件，**不是现在的 bug** —— 不要再去"修"一遍。
- **逐 edge 精度已回传**：`server.run_round` 打 `[Acc] Round N | edge0 | em_acc=… |
  pm_acc=… | n_clients=… | n_samples=…`，`collect_metrics` 出
  `per_edge_acc_rounds` / `per_edge_acc_final`。此前 `em_accs[]`/`pm_accs[]` 算完
  **立刻塌成一个加权均值**，于是「被污染的 edge 精度掉了多少」问不了
  （后门的干净精度代价是逐 edge 的）。未评估轮打 `n/a` 不是 0。
  聚合表达式一字未改，守卫 `tests/test_per_edge_acc.py::test_aggregate_expressions_are_untouched`。
  末轮快照取**有 pm_acc 的最后一轮**（直接取 max(round) 会落在 pm 全 None 的轮上）。

- **攻击时间窗已实现**（Neurotoxin 式持久性协议，`<本次>`）：
  `backdoor.attack_stop_round: T` → 恶意端在第 T 个 **cloud round 及之后**停止投毒，
  之后只观察衰减。`null`（默认）= 从不停止，与加它之前逐字节一致。
  机制：`FLClientBase.attacking(round_idx)` + 基类 `on_round_start` 每轮刷新
  `self._attack_active`，三个攻击 mixin 的闸门读它。
  **闸门不是 `is_malicious`** —— 后者同时是身份标记（参与度统计 / 逐 edge 分组 /
  `build_eval_trigger` 挑生成器都依赖它），翻它会把「攻击者退出」伪装成
  「这一格没有恶意端」，而那正是「攻击完全失效」的样子。
  衰减曲线**不需要新指标**：`rounds[]` 本来就是逐轮三层 ASR，T 之后那段就是。
  ⚠️ **`malicious_strategy: vanilla` 用不了它**（静态投毒数据集，钩子拦不住），
  `config_validate` 直接拒绝；T ≥ `n_rounds` 与 T ≤ 0 同样拒绝 ——
  前者静默等于「从不停止」而 metrics.json 却写着有退出轮。
  守卫：`tests/test_attack_window.py`（含反向锚点：改动前 9 failed / 7 errors）。

- **轮数已自适应**（`<本次>`）：`stopping` 块 → **只延长，不早停**。
  `floor_effective` 之前永不停（所以同轮比较点人人都有），判据未满足才延长到
  `cap_effective`。判据两条，参数都从实测噪声标定：
  `thresholds_crossed`（三层 ASR × θ∈{0.25,0.5,0.75} 全越过，**连续 2 点**才算 ——
  ASR 逐点 σ≈0.09，单点是噪声）与 `pm_acc_plateau`（末 10 点 OLS 斜率 < 0.0010/轮；
  残差 σ≈0.0023 → 斜率 SE≈0.00024）。
  **起因**：`n_edges` 影响收敛速度，而它正是 Experiment 3 的自变量 ——
  固定轮数下跨拓扑比较其实是在比「谁离收敛更近」（实测 10edge@30 ≈ 2edge@12–18）。
  三种结束各报各的：`converged` / `cap_reached`（**是 censored**）/
  `grid_too_coarse`。自描述行 `[Stop]` + `[设定3]` → `run.stop_reason` /
  `stopped_at_round`（**没有这一行事后无法判读这一格跑了多久**）。
  > ⚠️ **不能复用 `read_calibration.plateau_round`** —— 它是回溯式的
  > （「此后再没离开 final ± tol」），跑的时候没有 final。在线判据必须前瞻式。
  守卫：`tests/test_stopping.py`（含反向锚点：两格真实轨迹回打 round 20 / 28
  与斜率 0.00036 / 0.00229）。**历史批次（标定六格、天花板判定）刻意不加
  `stopping`**，它们的结果已在盘上；守卫 `test_calibration_cells_have_no_stopping_block`。

- **结果管理：登记表 → 对账 → 整洁表 → 判定 → 出图**（2026-09-24，Exp3 改版 S1）。
  入口是 `experiments/attack/hfl-mechanism/README.md`。
  - `[Provenance]` 行（`fedavg/utils/provenance.py`）写进 `run.provenance`：git commit（直接读
    `.git` 文件）、**口径版本** `PROTOCOL_VERSION`、`config_sha`（yaml 原文的 hash）、
    run_id/group、`cli_overrides`（CLI 改掉的叶子值，**只记录不拦截**）。
  - **口径版本 P0/P1/P2 ≠ 训练 epoch**：
    - P0 = 探针修正前的归档；
    - P1 = 统一标准后的 seed42 批次，以及 P2 之前的全部 run；
    - P2 = `AUDIT.md` 全部关闭后的正式批次，**只有 P2 进结论**。
  - `harness/registry.py`（登记表 = 声明）+ `harness/status.py`（逐格对账：
    todo / blocked / failed / mismatch / stale / done + orphan）。
  - `harness/runs_table.py` 按 run 块的**实际**因素分组（不看文件名），终值取末 10 点均值，
    混合口径版本会被拒绝。
  - `harness/verdicts.py` 执行 PLAN §3 的预注册判定，seed 不够时输出 `insufficient`。
  - `harness/figures.py` 按因素出图，组内混格会被拒绝。
  - 新方案的提交走 `hfl-mechanism/submit.sh`：「完成」= exit_code 为 0 **且** config_sha 一致；
    审计没关闭就拒绝提交。
  - 守卫：`test_provenance` / `test_registry` / `test_status` / `test_runs_table` / `test_verdicts` / `test_figures`。
  - **术语：ρ 有两个意思**。旧方案文件名里的 `rho02/05/20` 指**恶意端比例**（2%/5%/20%）；
    改版规划与 Bad-PFL 里的 ρ 指**投毒率** `backdoor.poison_ratio`。新文件一律写全名。

- **对齐开关 + 「P2 对齐」模板**（2026-09-26，Exp3 改版 A4；DECISIONS D-039）。
  AUDIT 的每个对齐项是一个配置开关，**不写 = 旧行为**（冻结配置逐字节不变）。
  - 开关表 `fedavg/alignment.py`（不 import TF）；代码里**只经 `get_switch(config, "键")` 读**，
    默认值只在那里定义一次。模板 `fedavg/config/alignment_p2.yaml` = 全部 P2 值。
  - P2 = 模板全开：登记表 `overlays:` 把模板叠到 base 上；`config_validate` 对
    `meta.protocol: P2` 逐键核对，少开一项拒绝启动。消融登记在 pilot 表（P1 口径）。
  - 自描述 `[设定4]`（template=p2/legacy/mixed + 每个开关）、`[设定5]`（评估用的固定攻击者）；
    副列 `[ASR4]` `[ASRwb]` `[Stale]` `[StaleASR]`、每轮 `[Checksum]` —— 都是 key=value 行
    （`fedavg/utils/kvline.py`，逐字段解析），`collect_metrics` schema 3。
  - **新增开关**：先登记到 `alignment.py` 与模板，接好线；守卫 `tests/test_alignment_switches.py`
    （模板键集、P2 值 ≠ 旧值、每个键真的被读到）。
  - 评估的 PM：`CloudServer.main_pm(c)` 是 pm_acc 与 ASR 共用的**唯一**入口（D-033：同一个模型）。
  - 取数：`data.batch_pipeline: per_epoch` 下 FedAvg / FedRep / Bad-PFL 生成器都走
    `FLClientBase.epoch_batches`（每 epoch 重洗、drop_last、重增强）；**只接了 Bad-PFL**，
    静态投毒 + per_epoch 被 `config_validate` 拒绝（会绕过投毒数据集）。
  - L1 基线（A4 结束时）：本地无 TF 全绿；有 TF 时只有陷阱 #4 的 2 条红
    （另外 6 条自攻击时间窗起的假红已修，FINDINGS F-041）。
    **在 GPU 节点上跑是 5 条红**：另 3 条是测试默认「没有 GPU」，用户决定不改（FINDINGS F-067）。
  - **A4 收口（2026-09-27）**：pilot 第二轮 D-029 pass、A15 pass、D-031 different → head_first（D-045）；
    **AUDIT 全部关闭，`PROTOCOL_VERSION = "P2"`**（`fedavg/utils/provenance.py`）。pilot 判定带有效性闸（D-044）。
    ⚠️ `run.provenance.protocol` 是**代码**版本：此后任何 run（包括重跑冻结的 P1 配置）都记 P2。
    **这一格是不是 P2 口径的配置**要看 `run.alignment.template == "p2"`（`[设定4]`；runs_table 的
    `alignment_template` 列）—— 登记表里 `meta.protocol: P2` 的配置由 `config_validate` 逐键核对模板。
  - **机时预算（2026-09-27，D-047）**：上限约 100 GPU-h / 周（按 GPU 小时计费）；一个整 run 约 2.3 GPU-h，
    评估约占 36%，run 长度由 pm_acc 平台判据决定（F-046）。
    - `hfl-mechanism/pack.sbatch`：一张 GPU 并行 K 个 run（`TF_FORCE_GPU_ALLOW_GROWTH=true`、每个 run 绑自己的核）。
      **先测后用**：`pilot/submit_pack_test.sh` + `harness/pack_test.py`（前 5 轮 checksum 必须等于 DET 第二轮、
      加速比 ≥ 1.5）。**2026-09-27 测完：K=3 加速比 2.86、checksum 全等 → 采用 K=3**（D-048；F-048）；
      显存峰值 96 GiB 已近满 → 新配置类型先单独交一个 pack 作业看显存。
    - **已接入提交脚本**（`c88a023`，D-052 / D-053）：`PACK=3 RUN_GROUPS=… bash hfl-mechanism/submit.sh`
      （`pilot/submit_pilot.sh` 同样；共用 `submit_lib.sh`，纯 bash）。`PACK` 不设 = 一卡一跑，逐字同以前。
      **核心是不触发 OOM、尽量省机时**：同一包只放同一格子的不同 seed；格子第一次只交一个 PROBE_K 的包、其余 held
      （PROBE_K **缺省 3**，D-069：G6 已实测每 run 约 17 GiB；显存没测过的新配置类型写 `PROBE_K=2`）；
      回传后按**真实显存峰值**定 K（服务器新打的 `[GPUMem]` = TF 分配器 `get_memory_info` 峰值 → `gpu_mem.peak_mib`；
      整卡 `memory.used` 在 allow_growth 下含预留块、偏大，F-053）；OOM 过 → 降一档。
      被 `_collect_updates_*` 吞掉的 OOM 会让 run 照样 exit 0 → `pack.sbatch` 按日志判 OOM、exit_code 记 86。
      exit 0 但 config_sha 不符 = `stale`，**默认不重交**（`RESUBMIT_STALE=1` 才交）。**交完等回传再跑下一次**（不查队列）。
      `PACK_MEM_PCT=85` / `PACK_CTX_MIB=1024` **没有证据**，第一批满长包回来后校准。守卫 `tests/test_pack_submit.py`。
      **余数跨格子合包**（2026-09-28，D-060）：各格子切完满包剩下的不满 K 的那一包，若 K 来自真实峰值、无 OOM / 显存告警、
      同组格子峰值相差 ≤ `PACK_MIX_TOL_PCT`（10%）、且合后作业数变少 → 合成 `<组>__mix-a+b+…__pack-…`；`PACK_MIX=0` 关。
      G6 探路包实测：真实峰值约 17 GiB / run，整卡读数约其 2 倍（F-055）。
      **确实是并行的**（F-059）：包墙钟 ≈ 单个 run 的耗时（11.1k s），不是两个之和（21.6k s）；每有效轮的训练时间与单跑相同。
      **主机内存也要随 K 放大**（D-070 / F-060）：每 run 实测约 16.7 GiB 主机内存，K=3 在固定 48G 下被 cgroup OOM 杀掉一个（exit 137）→
      现在每个包 `--mem = PACK_MEM_PER_RUN_GB(24) × K`；exit 137 计入 n_oom、自动降档。Arrhenius 计费取各项最大值（MAX_TRES）：
      单卡作业内存 ≤ 102.6 GB、CPU ≤ 72 都只按 1 张卡计费。**防重交**：作业带 `--comment=exp3v2:<run_id,…>`，提交前查 `squeue`，
      已在队列的 run 不重交；日志名带作业号（`exp3v2_<run_id>.<job>.log`），重复提交不再互相截断。
    - `[TimingASR] Round N | main=… | whitebox=… | stale=…`（独立 kv 行，**不改 `[Timing]`**）→
      `timing_rounds[].asr_*_s` 与 `timing_summary.asr_split_total_s`。实测白盒 7.2%、陈旧 ASR 7.3% 墙钟（F-051）
      → D-050 / D-054：**白盒关、陈旧 ASR 与陈旧 pm_acc 都隔点**（同一批点，共用 `CloudServer._eval_seq`）——
      **已实现**（`68f865d`）：开关在 `alignment.EXTRA_SWITCHES`（预算旋钮，不进模板），P2 的值在 `hfl-mechanism/base.yaml`。
      副列终值用 `runs_table.window_mean`（末 10 个评估点窗口），**不要用 `last_k_mean`**（先丢 None 再往回够）。
      另有 `[TimingAcc]`（GM / EM / PM / 陈旧 PM 分项）→ `timing_summary.acc_split_total_s`。守卫 `tests/test_eval_downsampling.py`。
      **白盒 ≈ 主列是重要发现**（私有 head 挡不住 ξ）；**fresh-PM 会低估干净精度**，10edge 达 0.094（F-051）。
    - G2 先做一致性复测（pilot 表 G2P + `pilot_a4.judge_g2p`）；G4 搁置（`registry.yaml` 的 requires 含 `reformulate-3.2`）。
    - ~~**G2 与 S5 暂缓**（D-056）~~ → **S5 已完成**（2026-10-01，D-084，见「S5 统一评估网格」一条）；**G2 规模仍待用户定**（挂 `g2-scale`）。S8 已完成（见下一条），G6 已回传。
      ~~停止判据的斜率横轴是云轮号~~ → 开了网格后横轴 = 网格序号（F-052 已修；pilot D029 flat 的回放是第一条数值证据，F-078）。
    - **登记表补 `set:` 时核对三件**：`malicious_per_edge` 长度 = `n_edges`；`n_rounds × edge_rounds ≥ cap_effective`；
      各格评估网格（有效轮）一致 —— G2 当初三件都漏了（F-046）。

- **3-E 三层个性化**（2026-09-27，Exp3 改版 S8；DECISIONS D-057 … D-059）。
  `federation.edge_shared_blocks ∈ {0,1,2}`（缺省 0 = FedRep 基线，逐字节不变；`[Checksum]` 反向锚点）：
  ResNet-10 最后 k 个残差块（层名 `stage{i}_`，含 shortcut 与 BN 统计量）是 **edge 段** —— edge 内照常聚合，
  **不上云**：`CloudServer.aggregate_edges` 把它换回聚合前的值（全局模型里那一段永远是初值），
  `HierFedRepEdgeServer.set_weights` 不让 cloud 广播覆盖它（首次接收除外）。client 与 fresh-PM 都不用改
  （edge 模型自带本 edge 的 edge 段）。规则在 `fedavg/utils/tier_split.py`（不 import TF）；
  `get_base_head_indices(..., edge_shared_blocks=k)` 多返回 `edge_weight_indices` / `cloud_weight_indices`。
  - 只对 `hier_fedrep` + ResNet-10 开放、与 cloud 层防御互斥（`config_validate` §4e 拒绝）。
  - 自描述 `[设定6]`（在 CloudServer 算出索引处打）→ `run.edge_shared_blocks`（collect_metrics **schema 5**）；
    `runs_table` 以它为因素键（0 记 None，与 S8 之前的文件同格）；`status` 对账核对它。
  - **(b)(c) 下 GM 精度与 global 层 ASR 不可读**（全局模型不完整）→ G6 固定 300 有效轮、停止判据关（D-058）；
    判定读受害 edge 的原始 benign ASR（D-059，不等 S4）。
  - 守卫：`tests/test_tier_split.py`（纯 python）+ `tests/test_three_tier_personalization.py`（TF，真跑 cloud/edge/client）。
  - **判定（S7，2026-09-30）**：`harness/g6_verdict.py`（规则 D-059 / D-071 预注册，脚本写于数据之后）→ b、c 两臂都 `blocks`（FINDINGS F-077）；
    精度代价 ≤ 0.02 只对 fresh 口径成立（陈旧列 c 臂 0.027–0.029）。

- **S3 新划分**（2026-09-28，Exp3 改版 S3；DECISIONS D-061 … D-068）。`federation.partition ∈ {designed, hdir, equal_random}`
  + `federation.design.*` → `fedavg/data/designed_partition.py`（**不 import TF**）：
  - **客户端等大小**（每端 500 张 = 375 训练 / 125 留出；恶意端数据占比恰为名义值，F-028 / D-027）、**无放回**、
    每 edge 先切 500 张**干净集**（按本 edge 分布，与客户端不相交；挂在 `edge.clean_indices`，S3 不用，S6 / 阶段三用）。
  - designed = 4 edge 机构式比例表（E0 均衡，E1–E3 各偏重一对类 r = 0.25），C1–C4 只改 airplane 一列；
    hdir = 层级 Dirichlet，**α 按社区口径**（每个类的参数 = α；原文 Dir(α·p) 差 10 倍，F-056）；
    equal_random = 「edge 不对应机构」的对照（G0-random / G1-random / G2 / G5 用它）。
  - 随机性只来自 `default_rng([seed, 0x533])`，**不碰全局 np.random**；旧划分路径逐字节不变（AST 指纹守着）。
  - 自描述 `[Partition]` + 每 edge `[PartitionEdge]` → `run.data`（H_inter / H_intra、逐 edge airplane 占比、索引 sha），collect_metrics **schema 6**。
  - 离线预览（F0 数据，不需要 TF / GPU）：`python3 harness/partition_preview.py`（F-058）。
  - floor 验证 pilot **FLR**（G6(a) × ρ=0，预注册判定 `harness/flr_verdict.py`，D-061）决定 G0 要不要逐划分测 floor。
  - 守卫：`tests/test_designed_partition.py`（含集群上跑的 `build_clients` 集成测试）、`tests/test_flr_verdict.py`。
    > 那条集成测试直到 S9 才第一次在有 TF 的环境里跑，红在**夹具**少 `training.lr_decay`（F-063，已修；真实配置不受影响）。

- **S9 评估仪表 + 两个存盘开关**（2026-09-28，Exp3 改版 S9；DECISIONS D-071 … D-075）。
  取舍标准：一个量值得在线记，要能改变某个防御选择或预注册判定，且以后补比现在记贵
  （P2 确定，F-045 / F-050 → 评估侧的量多数可以带仪表重跑那一格补回来）。
  - **常开**（无配置键，不改 config_sha）：`attack/eval_detail.py`（纯 numpy）从**主列那一次前向**取
    逐客户端 ASR / 干净精度（含恶意端自身）、触发样本 margin（log p_t − max log p_k，与 logit 差相等）分位数、
    干净样本判为 y_t 的比例、按类 ASR、非目标翻转率 → `[EvalDetail]`（良性端池化）/ `[EvalDetailEdge]` /
    `[ClientEval]`（"/" 连接的列表里 None 写 `na`，`utils/kvline.fmt_list`）。collect_metrics **schema 7**
    （`rounds[]` 新列、`per_edge_detail_rounds` 紧凑行、`client_final` 按列、`dumps` manifest）。
    **硬约束**：不多做前向、不碰 RNG、每个探针只调一次触发器（AST 守卫）；客户端 ASR 由整数 argmax 计数，
    与主列逐位相同；细节打印放在 `t_asr` 计时之后。
  - **两个开关**（`alignment.EXTRA_SWITCHES`，默认关，**只在组的 `set:` 里开**）：
    `evaluation.dump_logits_every`（逐样本 fp16 对数概率）、`evaluation.snapshot_rounds: "30/70"`
    （**写成字符串**：列表在 `[设定4]` 里往返会变形）。落盘 `$ROOT/../tfdpfl-dumps/<run_id>.<job>/`，
    git 里只有 `[Dump]` manifest。快照 = 全局 + 每 edge 模型 + 每端 `private_state()` + 生成器（含 Adam）+
    探针顺序，**只供评估，不能续训**。何时开（D-073）：登记表写明了消费它的离线分析才开；logits ≤ 50 MB / run、
    快照 ≤ 3 次 / run，项目总预算 20 GB（组内共享 500 GB）。
  - 验收工具：`harness/instrumentation_check.py <ref> <new> --upto R`（只比 checksum 与改动前就有的数值字段；
    `check_reproducible.py` 比全部字段含计时，不适用）。G8 第 1–30 轮对 G6(a) 同 seed 就是它的 GPU 验证。
  - 守卫：`tests/test_eval_detail.py`（解析值 + AST）/ `test_eval_detail_tf.py`（同一次前向、开关不改数、快照复原 fresh-PM）/
    `test_collect_eval_detail.py`（打印 ↔ 解析同源、体积预算）/ `test_dump_switches.py` / `test_instrumentation_check.py`
    （真实数据的正反锚点）/ `test_decay_verdict.py`。

- **S4 投毒窗口起点 + 生成器语义**（2026-09-29，Exp3 改版 S4；DECISIONS D-076 … D-079）。
  - `backdoor.attack_start_round`（cloud round，含）+ 已有的 `attack_stop_round`（不含）= 投毒窗口 [start, stop)；
    `backdoor.generator_schedule ∈ {window, always}`（只对 badpfl）：window（缺省）= 生成器跟着窗口走、窗口外冻结（G8 的语义）；
    always = 窗口只管投毒、生成器每轮都训（窗口外 = ρ=0 影子攻击者）。两个键都不写 = 与 S4 之前逐字节一致。
  - 判定只在 `fedavg/attack/attack_window.py`（**不 import TF**）定义一次；`FLClientBase.attacking / generating` 调它，
    `on_round_start` 刷新 `_attack_active`（投毒闸门）与 `_gen_active`（生成器闸门，只有 Bad-PFL 的 `on_round_start` 读）。
    **窗口外不能消耗投毒随机数**（`on_batch` 在闸门关时直接返回）—— 否则 B 臂窗口前与 ρ=0 影子攻击者的轨迹不同。
  - `config_validate` 拒绝：起点 < 1、起点 ≥ 停止轮（空窗口）、起点 ≥ n_rounds、vanilla、`always` 配非 badpfl。
    自描述 `[设定7]`（独立 kv 行，**不扩 `[设定2]`**）→ `run.attack_start_round / generator_schedule`，collect_metrics **schema 8**；
    `status` 核对这两个键；`runs_table` 以它们为因素键（window 与没有 `[设定7]` 的老文件同格）。
  - 登记表：G0 固定 60 云轮；G5（3.3）5 格挂 `g5-schedule` 等 G5AB（A/B 对比，`harness/g5ab_verdict.py`）——
    **G5AB 判 insensitive → G5 用 A、已放行**（D-080）；G8F = G8 的 flat 对照（`decay_verdict.py --flat`）。
    GPU 上已证：B 臂窗口前与同 seed 的 G0-random 逐位相同（FINDINGS F-072）。
    G5 已回传（2026-09-30）：预注册判定 `harness/g5_verdict.py`（D-081）→ **`not_gated`**（F-076）；**3.3 收尾**（2026-10-01，D-082）。实验全貌见 `experiments/attack/hfl-mechanism/REPORT.md`，
    各组结果图 `harness/report_figures.py` → `hfl-mechanism/figures/final/`（REPORT §0；数取自各组判定脚本，守卫 `tests/test_report_figures.py`）。
  - 守卫：`tests/test_attack_window.py`（真值表 + 校验 + 闸门 AST）/ `test_attack_window_tf.py`（真 Bad-PFL 客户端：
    两种语义下生成器与投毒的开关、窗口外不耗投毒随机数、ρ=0 影子攻击者确实在训生成器且评估触发器用生成器）/
    `test_g5ab_verdict.py` / `test_decay_verdict.py`（flat 分支）/ `test_run_self_description.py`（`[设定7]` 往返）。

- **S5 统一评估网格**（2026-10-01，Exp3 改版 S5；DECISIONS D-055 预案 / D-084 实现取舍）。
  `evaluation.eval_grid: G`（有效轮；`alignment.EXTRA_SWITCHES`，缺省 None = 旧行为逐字节不变；**只在组的 `set:` 里开**）。
  规则只在 `fedavg/server/eval_grid.py`（**不 import TF**）定义一次：
  - 全量评估 = 网格上的云轮末，两个 eval_interval 必须 = lcm(G,R)/R（`config_validate` §4b' 核对）；
  - 其余网格点在 edge 轮之间做**轻评估**（`CloudServer._light_eval`，只有交错调度有这个时刻）：只算主列 ——
    fresh-PM 的 local / edge ASR（`evaluate_hierarchical_asr(light=True)`）+ fresh pm_acc + EM 精度；不算 global / 白盒 / 陈旧 / ASR4 / drift；
  - 停止判据横轴 = 网格序号 eff / G（`StoppingRule.update(..., x=)` + `observe()`；R = G 时就是云轮号 → 逐位不变），停止决定只在全量点；
  - GM 只在全量点算；非全量轮 `[Cloud] GM=n/a | EM=n/a`、`[Acc] em_acc=n/a`。
  **评估不得改变训练**：轻评估用专用草稿槽（`light_victim` / `light_attacker`）+ 独立 RNG 键，并整个包在
  `random.getstate()/setstate()` 里 —— `clone_model` 的未播种初始化器会消耗 **Python random**（F-078 实测；main.py 只调
  `tf.random.set_seed`），legacy 管线训练用它洗牌。网格因此**收窄到 P2 路径**：顺序调度 / 陈旧 PM / legacy 管线 /
  非共享生成器 / Bad-PFL 非固定攻击者 / 后处理防御，`config_validate` 一律拒绝。
  自描述 `[设定8]` → `run.eval_grid` / `run.grid`；轻评估点 `[Light]` / `[LightEdge]` → **单独成表** `light_rounds[]` /
  `per_edge_light_rounds`（不混进按云轮当键的 rounds[] / acc_rounds[]），collect_metrics **schema 9**；
  `runs_table.grid_series` 把全量 + 轻评估点按有效轮合并算 T_θ 与主列末 10 点；`instrumentation_check --grid` 比开 / 关网格。
  G2 写好网格但挂 **`g2-scale`**（规模未定）；G1 的网格留给 S6。GPU 探路组 **S5P** 已回传（2026-10-02）：
  开 / 关网格 checksum 与全量点数值逐位相同，轻评估点约 35 s ≈ 全量点（约 78 s）的 45%（F-079）。
  守卫：`tests/test_eval_grid.py`（规则真值表 + 校验 + 已有配置 sha 不变）/ `test_eval_grid_tf.py`（开 / 关网格 checksum 与全量点数值逐位相同、
  一次轻评估前后状态清单不变、去掉 random 复原就改变训练的反向锚点）/ `test_collect_eval_grid.py` / `test_stopping.py` §5
  （R5 逐位不变；pilot D029 flat 旧横轴停 150、网格横轴不停 —— F-052 的第一条数值证据）。
  > ⚠️ 测试夹具的 `_model()` 调了 `tf.keras.utils.set_random_seed`，之后 clone_model **不**消耗 Python random —— 与 main.py 不同。
  > 要测随机通道，先 `_prod_seeding`（把 Keras 的种子发生器复位成 None），见 `test_eval_grid_tf.py`。

- **S6a G1 的「便宜记录」**（2026-10-02，Exp3 改版 S6a；DECISIONS D-085 方案 / D-086 实现取舍；方案全文 `hfl-mechanism/S6a-PLAN.md`）。
  四个**只读记录**开关（`alignment.EXTRA_SWITCHES`，缺省关，只在组的 `set:` 里开；前提由 `config_validate` §4g 核对：
  eval_grid 已开 + 交错调度 + `hier_fedrep`；post_agg / frozen 另要 Bad-PFL 固定攻击者），**记录不得改变训练**：
  - ① `evaluation.update_geometry`（+`update_sketch_dim`）：edge 收齐上传后、`robust_mean` 之前，对每个上传的 body-only 更新
    Δ = w[base] − edge_w[base] 在线精确算 ‖Δ‖、对本 edge / 对全体 edge 其余更新的**留一余弦**（用「和」算，None = 无「其余」），
    另做 CountSketch 草图（常量种子的独立 Generator，fp16 进 `tfdpfl-dumps/<run>/sketch_rNNN.npz`）。纯算术在
    `server/update_geometry.py`（**不 import TF**）；挂点 `HierFedRepEdgeServer.update_observer`；打 `[UpdateGeo]`；
  - ② `evaluation.post_agg_eval`：云广播后、第 1 个 edge 轮前一次轻评估（`CloudServer._post_agg_eval`，第 2 个云轮起，有效轮记 (g−1)·R），
    与上一个全量点相减 = Δ_jump。**单独成表** `post_agg_rounds[]`，不进 grid_series、不喂停止判据、不动 `_eval_seq` / history；
  - ③ `evaluation.frozen_trigger`：每云轮广播后冻结固定攻击者的 fresh-PM 与生成器权重（`BackdoorCloudServer._freeze_trigger`），
    在 post / light / full 三类点上对良性端另算一列「冻结触发器」ASR（ξ 在冻结 PM 上求、δ 用冻结生成器；**换权重**实现，
    退出后原样换回含 BN 统计量）→ 把「受害 body 变了」和「触发器漂移」分开。`[FrozenASR]` / `[FrozenASREdge]`；
  - ④ 常开：轻评估点从同一次触发前向多取 margin 分位数 / 良性 ASR p90 / >0.5 比例 / flip_other（`[Light]` 加列）；
  - ⑤ 防呆：`stopping` 开着且 `eval_grid ∉ {None, 5}` → 拒绝启动（斜率容差按 5 有效轮标定，F-052）。
  自描述 `[设定9]` → `run.update_geometry / update_sketch_dim / post_agg_eval / frozen_trigger`（平铺标量，好让 `status` 核对）；
  collect_metrics **schema 10**（新表 `update_geometry` / `post_agg_rounds` / `per_edge_post_agg_rounds` / `frozen_rounds` /
  `frozen_edge_rounds`，`dumps.sketch`，`timing_summary.post_agg_eval_total_s / frozen_eval_total_s`）。
  三类新评估各用专用草稿槽 + 专用 PGD 噪声键（0x9057 / 0xF20E）并包在 `random.getstate()/setstate()` 里（F-078 的 Python random 通道）。
  **G8 快照上的 c_k 预检**：`fedavg/analysis/functional_score.py`（纯 numpy：NCM、c_k、AUROC）+ `ck_snapshot.py`（TF）+
  `hfl-mechanism/ck_precheck.sbatch`（回传 `analysis/ck_precheck.json`）；**判读阈值还没预注册**，回传前先写进 FINDINGS。
  **探路组 G1P**（3 run，C1 × R10 × s42 × 100 有效轮：`coll-on` / `coll-off` / `dist-on`，约 1.08 GPU-h，**已实测 F-081**）：
  开 / 关 checksum 与全量点、轻评估点数值逐位相同（`instrumentation_check coll-off coll-on`，现在也比 `light_rounds[]` 的参照列）+ 几何 AUROC。
  G1 的 `set:` 已补齐（布点 / 300 有效轮 / 网格），**不开记录开关**，仍挂 `S6`（= S6b：在线 c_k，等 G1P 回传再定）。
  守卫：`tests/test_update_geometry.py`（手算值 + 不碰全局 RNG + AST）/ `test_s6_tf.py`（开 / 关 checksum 与已有行逐字相同、状态清单不变、
  **去掉 random 围栏就改变训练的反向锚点**、冻结触发器 == 主触发器直到生成器被训动）/ `test_s6_switches.py` / `test_collect_s6.py` /
  `test_functional_score.py` / `test_ck_snapshot_tf.py`。
  > ⚠️ `python3 harness/registry.py … --materialize --group X` 会把 `INDEX.tsv` **整个重写成只含 X**（其余行丢失 → 这些格子失去 config_sha 核对：实测 G7 的 6 个 stale 被悄悄判成 done，done 87 → 93）。
  > 要补一个新组的配置，**不带 `--group`** 全量 materialize（已有配置字节不变，只多新行）。

- **S6b 在线 c_k + G1 登记**（2026-10-02，Exp3 改版 S6b；DECISIONS D-087；方案 `hfl-mechanism/S6b-PLAN.md`）。
  - 开关 `evaluation.update_ck`（+ `update_ck_every`=5 有效轮 / `update_ck_n`=64 张 / `update_ck_steps`=5 步；EXTRA_SWITCHES，缺省关；
    `config_validate` §4h：eval_grid + 交错调度 + hier_fedrep + S3 划分 + `clean_per_edge` ≥ n）。**只读记录，不得改变训练**。
  - `server/update_ck.py`（TF）`UpdateCk`：挂在 `HierFedRepEdgeServer.run_edge_round` 收齐上传后、`robust_mean` 前（`score_observer`）；
    评分点 = 有效轮 eff % every == 0 的 edge 轮。θ_before = 本轮下发的 edge_w，θ_i = edge_w 上换入上传的**可训练权重**，
    **BN moving 统计量保持 edge_w 的**；每个模型用 edge 干净集（`edge.clean_x / clean_y`，main.py 在开关开时由 `clean_indices` 取出）的类均值做 NCM head；
    c_k = 前 n 张的定向 PGD（`analysis/ck_snapshot.ck_reached_batched`：K 个类摞成一批；ε = `badpfl_epsilon`、无随机起点）失败率。
    专用评分模型第一次评分才创建（clone_model 推进 Python random）→ 整段包 `random.getstate()/setstate()`（F-078）。
    打 `[CkBefore]` / `[CkScore]`（每更新一行：cid / mal / ncm_acc / c = K 个值，无定义的类 `na`）/ `[TimingCk]`；
    **s_i（原文 §7 公式）不在线算**：`analysis/functional_score.update_score` 在 `harness/g1_scores.py` 里离线算。
  - **几何记录的混杂已拆开**：base 索引含 BN moving 统计量（A27 下私有），S6a 的 ‖Δ‖ / 余弦把「权重变化」和「客户端数据造成的统计量差」混在一起
    （F-081 补注：norm AUROC 可能部分来自统计量）。`UpdateGeometry(stat_idx=…)` 现在多记 `norm_w / norm_s / cos_edge_w / cos_global_w`（旧列逐位不变）+ 草图 `sketch_w`。
  - 自描述 `[设定10]` → `run.update_ck / update_ck_every / update_ck_n / update_ck_steps`（平铺标量，`status` 核对）；collect_metrics **schema 11**
    （`ck_scores` / `ck_before` 紧凑表、`update_geometry` 新列、`timing_summary.ck_eval_total_s`）。
  - **G1 登记（27 run）**：G1 24（{random, C1} × {集中 [10,0,0,0], 分散 [3,3,2,2]} × R{10, 20} × s42–44）+ **G1R5** 3（C1 × 集中 × R5）；
    固定 300 有效轮、网格 5、开 update_geometry / post_agg_eval / update_ck，**frozen 关**，不加 s45 / s46。G1R5 的精确对照是 G3-C1（不是 G8）。
    两组原挂 **`g1-prereg`**：判读规则 FINDINGS **N-007**（预注册），**用户已确认并放行（2026-10-03）**。
  - **开销（G1 探路包实测，F-083）**：在线 c_k 约 300–360 s / run（约 4% round_time，**比估的 +20% 低 5 倍**），显存不变（16.8–16.95 GiB / run）；
    云聚合后评估点约 39 s / 点；每 run 墙钟约 2.4–2.8 h；K=2 的 8 个包共 21.6 GPU-h（16 个 run）。K=3 安全。
  - **G1 / G1R5 全部回传（2026-10-03，27 个 run）**：GPU 上开 / 关逐位相同（含 c_k；G1R5 与 G3-C1 逐位相同，F-084）。

- **G1 判定 + Experiment 3 终版报告**（2026-10-03；DECISIONS D-088；FINDINGS F-084 … F-086）。
  - `harness/g1_verdict.py` 严格实现 N-007（**先冻结在 `1c3119e`、再打开 seed 44 与 G1R5**）：
    3-C 4 / 5 格 `self_cleaning`（random·R20 `user_decides`，G1R5 是 `3C_bridge`）；3-D `norm_w` `no_edge_gain`，余弦与在线 c_k `undetectable`。
    两份盲算的独立实现 + 三个视角的对抗式审查（F-085）：判定标签全部相同。
  - `harness/g1_scores.py` 拆出 `view_arrays` / `aurocs_from_arrays`（g1_scores.json 逐字节复现）；`analysis/functional_score.py` 加 `tpr_at_fpr` / `update_argmax_ties`
    （k* 并列均分：np.argmax 偏向类 0 = 目标类）。`harness/g1_explore.py` = 探索性读数（不进判定）。
  - ⚠ N-007 / F-083 里「集中布点 E0 全是恶意端」是错的：E0 = 15 良性 + 10 攻击者，edge 视角塌掉是参照池被污染（F-085 更正）。
  - 图 `F3_sawtooth_G1.png` / `F5_3D_views_G1.png`（`report_figures.py` 的 G1S / G1V）。`REPORT.md` 是**终版**（§1 结论一览、§5.12–§5.14、§9.4 防御含义、§10 综合讨论）。
  - ~~阶段三计划草案 `PLAN-draft.md`~~ → **阶段三计划已定稿**（2026-10-09，D-089 … D-095）：`experiments/defense/edge-native/PLAN.md`（D-008 已解除）；同目录 `DECISIONS.md` / `FINDINGS.md`（编号接续 D-089 / F-087）、`LITERATURE.md`（用户综述 + Bad-PFL / CCS 原文摘录）。**D0 已完成**（2026-10-09）：阶段三登记表 `experiments/defense/edge-native/registry.yaml`（SNAP 可交，`submit.sh` 同目录；工具与 hfl-mechanism 共用）、功效分析 `harness/d0_power.py`（F-091）、预注册草案 N-008（待用户确认）；**SNAP 已回传、有效**（2026-10-10，F-094：collocated 与 G1R5 逐位相同）；守卫 `tests/test_edge_native_registry.py`。交接 `experiments/defense/edge-native/current-focus.md`。
  - 守卫：`tests/test_g1_verdict.py`（合成数据覆盖每个分支；真实数据只钉 s42 / s43 的 F-083 数与已入库的 g1_scores.json；阈值反向锚点）/
    `test_g1_explore.py` / `test_report_figures.py`（图上的竖线 = 判定的 Δ_jump；ROC 在 FPR 5% 处 = 判定的 TPR）。
  - 守卫：`tests/test_update_ck_tf.py`（开 / 关 checksum 逐轮相同、**去掉 random 围栏就改变训练的反向锚点**、θ_i = 上传的权重 + edge 的统计量与 head、
    Δ = 0 时 c_i == c_before、一次评分不改任何权重 / RNG、批量 PGD == 逐类 PGD）/ `test_update_geometry.py`（拆分的手算值）/ `test_g1_scores.py` /
    `test_collect_s6.py`（schema 11）/ `test_s6_switches.py`（§4h + G1 27 run 的配置自洽）。

**留了接口但没有实现的**（不要以为它们能用）：
- 主动防御（需要客户端配合的防御）：接口齐了（`BaseDefense.layers` /
  `client_mixin` / `make_control` + 客户端侧 `set_control` / `get_aux`），无任何实现。
- cloud 层防御：`defense.layers` 缺省 `["edge"]`，写成 `[edge, cloud]` 才启用。
- cloud 层的方法专属聚合：`CloudServer.aggregate_edges` 是空壳，见陷阱 #8
  （唯一的非 FedAvg 行为是 S8 的「edge 段不聚合」，只在 `edge_shared_blocks > 0` 时生效）。

---

## 交互约定

- 在我明确说「**开始改**」之前，只做分析、只给方案，不要动代码。
- 每个方法开工前先输出**语义 diff 表**，我确认后再写代码：

  | 论文公式/步骤 | 官方实现 (reference/) | 本仓库实现 | 差异 | 怎么验证 |
  |---|---|---|---|---|

- 一个会话只处理**一个方法 / 一个模块**。做完 commit 后我会开新会话。
- **拿不出数值证据时不要说「应该没问题」**——那是危险信号，请明确指出
  「这一条我没有证据」。仓库里现存注释大量是这种自证式声明，不可信，
  一律以 `tests/` 里跑得出来的断言为准。
- 不 review、不跑测试，就不 commit。commit 是强制停顿点。

---

## 目录约定

```
fedavg/        唯一活跃代码（在集群上以 cwd=fedavg 运行，import 是 `from client.x import`）
               ⚠️ 绝大多数模块 import TF → 本地测不了。把纯算术抽成不依赖 TF 的
               小模块（如 `server/participation.py`），L1 就能在本地秒级覆盖它。
reference/     官方 torch/py 实现，gitignore，一次只 clone 1~2 个，用完即删
tests/         L1 算法不变量测试（本地秒级，见下）
experiments/<axis>/<method>/
               current-focus.md / smoke.yaml / expNNN.metrics.json / expNNN.notes.md
results/       集群回传的小产物（截断日志、误差数字）
scratch/       临时产物，全 gitignore
methods-registry.md   所有候选方法的台账 = 研究看板
```

`<axis>` ∈ `attack` / `defense` / `pfl`。三处目录用同一套 `<axis>/<method>` 命名。

---

## 验收标准：两级，缺一不可

**L1 — 算法不变量测试**（`tests/`，本地跑，秒级）
手工构造输入，正确输出可解析算出，断言必须**精确**而不是「看起来差不多」。
纯 numpy 的测试本地直接跑；需要 TF 的用 `pytest.importorskip("tensorflow")`
标记，在集群上跑。

**L2 — 端到端 smoke run**（`experiments/smoke-base.yaml`，集群，约 3 分钟）
10 client / 5 round 的最小配置，只回传一个 `metrics.json`。
由 `harness/collect_metrics.py` 从日志压出来，含：

| 字段 | 说明 |
|---|---|
| `run` | **自描述**：config 路径 / method / attack / defense / n_rounds / malicious_ids |
| `rounds[]` / `final` | 分层 ASR（global / edge / local，同 edge vs 异 edge） |
| `acc_rounds[]` / `final_acc` | GM / EM / PM 准确率 + 最终 PM 加权 C-Acc |
| `admitted[]` / `admitted_count_mean` / `rejected_ids` | 防御判决（见陷阱 #10） |
| `malicious_selected_rounds` / `n_malicious_participations` / `..._by_client` | 按 **client_id** 统计，vanilla 策略也算得出 |
| `client_failures[]` | 被 `_collect_updates_parallel` **吞掉**的客户端异常 |
| `errors[]` / `log_tail` | traceback 首行 / 末 40 行 |

> **`run` 段是硬要求**：`--config` 曾经被静默忽略（陷阱 #7），跑出来的 metrics.json
> 与「按预期跑」的那份长得一模一样。不写明自己跑了什么的 json 事后无法判读。

> **跑挂了先看 `client_failures[]`**：客户端异常会被 catch 掉、那个更新被踢出聚合，
> run 照常跑完、日志一切正常。若恰好是恶意客户端每轮都在这里，**ASR 必然是 0**，
> 而这与「攻击无效」看起来一模一样。

> **L1 过了不代表 ASR 会起来。** 本仓库已知的失败有一半在「接线」而不是算法本身
> （陷阱 #1 和 #7 都是这一类）。所以 L2 不可省略。

---

## 集群 ↔ Claude Code 协议

- **去程**：只推代码 / 配置 / 脚本 / 小夹具。
- **回程**：只回 `metrics.json`、误差数字、`tail` 后的日志、traceback。
- **checkpoint 留集群**，git 里只放 manifest（method / path / git_commit / step / metrics）。
- **大张量绝不回传**：让集群脚本当场算好统计量（max/mean diff、分位数、NaN 位置），
  只回这几个数字。
- 红线：`*.h5 / *.pt / *.ckpt / 大 npy / 大 log / venv` 永不进 git。
  守卫：`tests/test_repo_hygiene.py`（当前跟踪的文件；历史里已有约 280 MiB 这类垃圾，`python3 harness/git_size_report.py` 可复现，瘦身选项见 `hfl-mechanism/REPORT.md` 附录 A）。
  仓库 > 500 MB 或单文件 > 10 MB → 回去查 `.gitignore`。

---

## 已确认的陷阱（每确认一条就补一条，附证据）

1. ~~**攻击轴与 PFL 方法轴不正交**~~ ✅ **已修复**（commit `dd7fd5e`）
   旧：`use_cls = MalCls or ClientCls` 让恶意客户端类**替换掉** PFL 方法类，
   而三个攻击类都继承 `FedAvgClient` → `drift_correction=hierpfedme` 时良性客户端
   跑 pFedMe、恶意客户端跑朴素 FedAvg，上传语义不同，足以单独解释 ASR≈0。
   现：攻击/防御都是 **mixin**，与方法类**组合**（`client/compose.py`），
   MRO = `Attack → Defense → Method → FLClientBase`。
   证据：`tests/test_attack_method_orthogonality.py`（33 passed，改动前 18 failed）。
   **新增攻击/防御时不要再写成 Client 子类**，写 mixin + 钩子。

2. ~~**随机性未播种**~~ ⚠️ **大部分已修**（`d2717cd`），**Python `random` 已补**（`53a076d`），
   仍有 FLAME 一处漏网
   已修：`select_clients`、投毒选样、DnC 投影、ASR 子采样，都改用
   `np.random.default_rng([seed, edge_id/client_id])`。
   ~~**未修（第五处）**~~ ✅ **已修**（`53a076d`）：`main.set_seed` 只播了
   `np.random` 与 `tf.random`，而它自己的 docstring 写着「随机性来自**三处**」——
   漏掉的第三处是 Python 内置 `random`。五个方法客户端每个 epoch 都用它打乱 batch
   （`hier_fedrep.py:176` / `client_pfedme.py:137` / `hier_ditto.py:130` /
   `hier_ditto_rep.py:202` / `hier_pfedme_rep.py:189` 的 `random.shuffle(eb)`）
   → **Rep/Ditto/pFedMe 全家在固定 seed 下都不可复现**，即 Experiment 3 的每一个格子。
   这会伪装成「种子方差大」：3C 轴上 seed42≈0.80 / seed43≈0.58 的落差里有多少是
   真种子方差、多少是这个未播种的洗牌，修好之前无法分离。
   守卫：`tests/test_seeding_completeness.py`（并通用扫描 `client/`、`server/` 下
   任何 `random.*` 的使用，新增方法不会重蹈覆辙）。

   **仍未修（第六处）**：`defense/flame.py:76` 的高斯噪声用的是全局
   `np.random.normal`，且在**训练循环内**（每个 edge round 调一次）→
   `defense=flame` 的格子固定种子重跑对不上。修法一行：改用 seeded RNG。
   **新写的代码不要再碰全局 `np.random`**（setup 期的分区除外，那里顺序确定）。

3. **FLAME 未按文献实现**
   现实现用 "majority-cosine 近似" 代替 HDBSCAN，阈值取 off-diagonal 余弦相似度
   的**中位数** → 全良性、更新彼此接近时，接纳与否由数值噪声决定，会无故剔除良性
   客户端。另文献是**无权平均**，现实现按 `n_samples` 加权。
   `sklearn.cluster.HDBSCAN`（sklearn ≥ 1.3，集群已有 1.8）可直接照文献写。
   证据：`tests/test_flame.py`

4. **Neurotoxin mask 语义疑似反向（未修）**
   官方 `grad_mask_cv` 的 `ratio` 是「**保留**的坐标比例」，取 |grad| **最小**的那部分。
   现实现 `(np.abs(a) < thr)` + `mask_ratio=0.05` 只屏蔽 top-5%、放行 95%，
   ≈ 退化成普通 BadNet，持久性收益消失。另官方是**逐层**阈值，现实现是全局阈值。
   待 clone 官方实现确认。证据：`tests/test_neurotoxin_mask.py`（2 条红就是等它）。
   > 已顺带修好的两件（`dd7fd5e`，与 ratio 方向无关）：掩码基准从
   > 「`self.model` 训练前后之差」改为 `self.edge_weights`（非 FedAvg 方法下前者是
   > **个性化模型**，与上传物无关）；`on_upload` 强制纯函数，不再写回 `self.model`
   > 污染个性化模型。同文件的 `test_masked_coords_have_exactly_zero_update` 现在通过。

5. **Bad-PFL 生成器迁移性 / 归一化常数（未修）**
   生成器只在恶意客户端自己的模型上训练，评估时却要迁移到良性个性化模型
   （`build_eval_trigger` 取 `mal[0]` 的 generator）。另 `CIFAR10_STD` 被硬用在
   `dataset: cifar100` 的配置上。
   另：`build_autoencoder(img_size=8)` 输出 16×16（`test_badpfl_trigger` 里 2 条红），
   小 img_size 下 shape 不自洽；`client_cerp.py` 的 `_CERP_COLS` 最大到 14，
   **隐含要求 `img_size ≥ 15`**，越界会在 `__init__` 直接 IndexError。

6. **TF 框架陷阱**（移植 torch 实现时逐条确认）
   - BatchNorm：TF `momentum ≈ 1 − torch momentum`；`eps` 默认值不同
   - Conv padding：torch 显式 padding vs TF `"SAME"`，`stride > 1` 时不等价
   - 通道序：本项目统一 NHWC
   - 损失：交叉熵确认 `from_logits` 设置。**本仓库是 `from_logits=False`**，
     配 `models/cnn.py` 里末层的 `activation="softmax"`。写测试用的小模型时若忘了
     加 softmax，梯度会恒为 0、训练什么都不做，而断言在比较两个全零数组 → **假绿**。
   - eager 手写训练循环与良性客户端的 `@tf.function` 路径在线程池并发会互相污染
     （现有代码因此把 `n_workers` 强制设为 1，是接线不干净的代价，不是必然）
   - `tf.keras.optimizers.SGD(learning_rate=tf.Variable)` 在 **Keras 3 被拒**
     （只接受 float / LearningRateSchedule / callable）。`FLClientBase` 正是这么写的，
     所以本仓库**隐含要求 Keras 2.x**（TF ≲ 2.16），而 `requirements.txt` 只写了
     `tensorflow>=2.12` —— 上限没锁。

7. ~~**`--config` 被静默忽略**~~ ✅ **已修复**（commit `dd7fd5e`）
   `load_config` 旧实现**先 `open(path)` 再 `parse_known_args()`**，解析出的
   `args.config` 全仓库从未被使用 → `--config` 完全失效。
   `run_smoke.sh` 传的 `--config ../experiments/smoke-base.yaml` 无效，
   所谓「10 client / 5 round / cifar10 的 3 分钟 smoke」**每次实际跑的都是
   `fedavg/config/config.yaml`（cifar100 / 100 client / 40 round）**，日志无任何异常。
   > **此前所有标称「smoke」的历史结果都要重新解释** —— 它们跑的是全量配置。
   守卫：`tests/test_config_cli.py`（AST 断言 `args.config` 被使用、且 `open()` 在
   `parse_known_args()` 之后）。观察性确认：修好后 smoke 应 ~3 分钟跑完。

8. **cloud 层聚合分支全部被注释（未修，行为即「永远 FedAvg」）**
   `server/server.py` 的 `_aggregate_global` 里 feddyn / hierpfedme / scaffold 三个
   分支全被注释掉，函数体只剩一句朴素样本加权 FedAvg。
   → 不管 `drift_correction` 是什么，**cloud 层永远是朴素 FedAvg**；
   config 里的 `beta_hier`（Hier-pFedMe 式 9 的 β）、`alpha_feddyn_global`
   是**死配置，读都没读**。
   本会话只把入口收敛到 `CloudServer.aggregate_edges` 并留了防御通道，**未改行为**。
   要动它，先决定这是「有意的设计」还是「没做完」，并把结论写进 `config_validate`。
   > 2026-09-27（S8，D-057）：`aggregate_edges` 在聚合之后多了一步「edge 段换回聚合前的值」，
   > 只在 `federation.edge_shared_blocks > 0` 时生效；k=0 路径一字未改。feddyn / hierpfedme / scaffold 仍是注释。

9. **Rep 家族的私有 head 进了防御的距离计算（PFL 轴 × 防御轴的泄漏）**
   `server/hier_fedrep.py` / `hier_ditto_rep.py` 把**完整**权重列表传给 `robust_mean`，
   之后才只取 backbone 索引；而 `flame.py` / `multi_krum.py` / `dnc.py` 都是
   `flatten_weights(upd[0])` 展平全部坐标算余弦/欧氏距离。
   私有 head 逐客户端 warm-start、从不同步，是全部权重里**发散最快**的部分
   → 距离矩阵可能被 head 主导，而 head 恰恰是防御不该看的（连聚合结果都不用）。
   与陷阱 #1 同类但**独立**，正交化修好了它还在。
   > **这一条只有代码路径证据，没有数值证据。** 验法：构造 backbone 相同、
   > head 随机发散的一组更新，断言 FLAME 的接纳集合不变。
   修法方向：给 edge server 一个「送去防御的索引子集」的概念。

10. ~~**各防御的日志格式不统一 → `admitted_count` 4/5 瞎**~~ ✅ **已修复**（`4021e88`）
    旧：flame 打「admitted 7/10」、multi_krum 打「selected N clients」、
    dnc 打「keep N clients」、trimmed_mean/median 干脆不打。回程解析器只认 flame
    那句 → `admitted_count`（CLAUDE.md 要求回传的 4 个字段之一）**5 个防御里
    只有 1 个读得出来**，defense 轴 4/5 是瞎的。
    现：`RobustAggregationMixin.robust_mean` 这个**唯一收口处**发一条统一行：
    `[Decision] edge0 | flame | admitted 7/10 | rejected=[3, 5]`；坐标类防御发
    `coordinate-wise | n=10`，`admitted` 记 `None` 而**不是 0**（0 会被读成「全部剔除」）。
    **新增防御自动被覆盖，不要再各打各的。**
    守卫：`tests/test_cloud_aggregate_default.py`（5 个防御逐个断言真的发出了这行）
    + `tests/test_collect_metrics.py`（断言解析得出来）。这两件是分开测的 ——
    解析器认得格式 ≠ 代码会打印它。

11. ~~**ASR 探针用了官方 `x_test`，而它已被合并进客户端数据池**~~ ✅ **已修复**
    （`53a076d` 用了错的修法，`<本次>` 改正）

    **⚠️ 先说清楚什么不是 bug**：把 train+test **合并后再逐客户端分区**是
    PFLlib 的标准做法，本仓库照做，**这不造成任何泄漏**。因为
    `noniid_partition` 是一个**划分**（每个 index 恰好归一个客户端），
    `split_client_train_test` 再在**每个客户端分片内部**切 train/test ——
    于是「所有留出分片的并集」与「所有训练数据的并集」**全局不相交**。
    那才是真正的留出集，而且保住了全部 60k 数据。
    > 我们一度误判成「合并本身是泄漏」并把它删掉（改成只喂 x_train），
    > 那是错的，已回退。守卫 `tests/test_no_test_leakage.py` 里有一条
    > `test_merge_is_kept_because_it_is_not_the_bug` 专门挡这次误删重演。

    **真正错的地方在评估侧**：`BackdoorCloudServer` 曾把**原始 `x_test` 数组**
    又当成一份独立探针去算六个 ASR（`main.py` → `_backdoor_eval`）。
    合并之后官方 test split 里的图已经分给客户端、其中
    1−`per_client_test_ratio` 进了训练集，再拿它当探针就是**在训练过的图上
    测攻击成功率**。第二个连带问题：那样 ASR 测在类别均匀的官方测试集上，
    而 `pm_acc` 测在非 IID 的 per-client 分片上，两个数字不在同一个 population。

    **修法**：三层 ASR 一律测在留出分片上 ——
    global ← `merge_test_datasets(所有 edge)`｜edge ← `edge.get_test_dataset()`｜
    client ← `client.test_dataset`（与它自己的 `pm_acc` 同一个集合）。
    新增 `compute_asr_on_dataset`；`compute_asr` 无合格样本时返回 `None`
    而非 `0.0`（非 IID 下客户端留出分片可能一个非目标类样本都没有，
    换探针之后这从罕见变成常态）。

    **影响幅度未知，不要替它下结论**。我们曾在这里写过「抬高的是绝对值、
    组间相对趋势多半仍成立」——**那句没有证据，已撤回**。方向其实不单一：
    落在**恶意端**训练分片的那部分被真的投毒训过（模型是记忆 → 抬高 ASR），
    落在良性端的那部分按正确标签训过（更难被翻 → 压低 ASR）。
    净效果要同一 config 跑新旧两套探针各一次才能定。
    量化工具：`harness/evidence_data_split.py`（用真实分区函数，
    连「多少张落进恶意端训练分片」都数得出来），`bash run_evidence.sh` 即可跑，
    **不需要 GPU**。

    **此前所有 Experiment 3 的 ASR 都要重新解释**，旧结果已归档到
    `experiments/attack/hfl-propagation/results/archive-pre-fix/`。

12. ~~**每轮参与端数依赖 `n_edges`**~~ ✅ **已修复**（`a00a959`）
    `EdgeServerBase.select_clients` 逐 edge 各自向下取整
    `max(1, int(len(self.clients) * frac))`，但 `client_fraction` 的语义是**全局**参与率。
    N=100/frac=0.1 下：2 edge → 5×2=10；**4 edge → int(2.5)=2 ×4=8（少训 20%）**；
    10 edge → 1×10=10。于是 Experiment 3A 声称的「唯一自变量 = 恶意端布点」不成立。
    现改为整数配额（`server/participation.py`，**不 import TF** 以便本地秒级测试），
    余数**按轮轮转**——恶意端是按 edge 布点的，固定配额会让「布点」与「训练量」缠在一起。
    守卫：`tests/test_participation_quota.py`（含反向锚点：断言旧公式确实给出 8）。

13. ~~**无定义的分组指标被填 0.0**~~ ✅ **已修复**（`9d20c87`）
    `attack/backdoor_eval.py` 的 `_mean`/`_std` 空组返回 `0.0`，于是「该指标在本配置下
    无定义」与「后门完全没传过去」数值上完全一样。实际后果：所有 *distributed* 布点
    `diff_edge_asr=0.000`（没有干净 edge）；`10edge_collocated` 的 `same_edge_asr=0.000`
    且 `per_edge[0].client_benign=0.000`（E0 没有良性端）——后者还被画进逐 edge 图，
    把那条线拉到底。现改为 `None`，日志打 `n/a`，`collect_metrics` 解析成 JSON `null`
    （与陷阱 #10 的 `admitted=None` 同一约定）。守卫：`tests/test_undefined_metrics_are_null.py`。

14. **`experiments/` 下的文档会比数据旧（已发生，且骗过了一份报告）**
    `hfl-propagation/RESULTS.md` 写于 2026-08-26，称「3C seed43 因 GPU 分配失败」
    「3c_R40 两 seed 均失败」——这两批数据在**前一天**的 `0f7a716` 就已入库且
    `exit_code: 0`。报告里的 Figure 12 正是照着这份错误认知画的：R=2 是 2-seed 均值
    0.698、R≥4 是 seed42 单值，于是「从 0.70 跳到 0.81 然后走平」纯粹是**种子可得性
    的假象**，被当成了「ASR 对 R_edge 不敏感」的证据。
    **教训**：写结论前先 `ls results/` 数一遍文件，不要凭上一次会话的记忆。
    `run_exp3.sh --status` 是权威，手写的进度表不是。

16. ~~**两个 sbatch 绕过了容器（Alvis 时期的遗留）**~~ ✅ **已修复**（`<本次>`）
    `exp3_cell.sbatch` 在 commit `7ae2966`（"exp3 alvis part"）被改成裸
    `python3 main.py` + Alvis 的 SLURM 头（`--account=naiss2026-4-650` 不带
    `-gpu`、`--gpus-per-node=A40:1`）；换回 Arrhenius 之后没跟着改回来，
    新写的 `calib_cell.sbatch` 又照抄了它。**两个脚本都会在 GPU 排到之后才炸**，
    报的是 numpy/tensorflow 的 ImportError —— 看起来像「依赖没装」，
    真实原因是根本没进容器。仓库里其余 5 个脚本一直是对的
    （`run_full.sh` / `run_smoke.sh` / `experiment_tf.sh` / `run_l1.sh` /
    `run_evidence.sh` 都用 `$PY`）。
    现已改回 Arrhenius 头 + `$PY`。守卫：`tests/test_cluster_env_usage.py`
    （扫全部作业脚本：无裸 python 调用、用了 `$PY` 就必须 source
    `cluster_env.sh`、SLURM 头是 Arrhenius 式、容器路径不在 `cluster_env.sh`
    之外硬写；另有一条反向自检防止「零个文件全部通过」）。
    **新写作业脚本时从 `run_full.sh` 抄头，不要从 git 历史里翻。**

17. **自检的报错本身可能是错的（2026-09，三轮误判）** ⚠️ **教训条目**
    `cluster_env.sh` 的启动自检当时用 `$PY` 探测「容器能不能看见仓库」，
    而 `$PY` 带 `--nv`。给 `run_calibration.sh` 加了 `source cluster_env.sh`
    之后，自检**在登录节点**跑了 —— 那里没有 NVIDIA 驱动，容器根本起不来，
    自检却报「✗ 容器里看不到仓库目录」。
    那句话把人引向 `--bind`，于是连续三轮拆掉了 8 月验证过的正确设计
    （`--bind <仓库上一级>` + `cd $ROOT/fedavg` + 上一级 `tfdpfl-logs`），
    每轮换一个 `FileNotFoundError`：配置 → 日志 → keras 缓存，
    因为每轮只补上 `$PWD` 恰好没盖住的那一块。
    **证据**：`20515a4`（作者机 `arrhenius1`）提交的 3C 全批结果，
    当时的脚本正是 `--bind` + `cd fedavg`。
    现已全部回退；自检改用**不带 `--nv`** 的等价命令并回显 apptainer 原话。
    守卫：`test_bind_defaults_to_the_repo_parent` / `test_self_check_does_not_use_nv`
    / `test_training_runs_from_fedavg` / `test_log_dir_defaults_beside_the_repo`。
    **下次先确认容器起得来，再信「看不到仓库」这句话。**

15. **`run_exp3.sh` 按 `exit_code: 0` 跳过已完成格子**
    重跑前必须把旧的 `results/*.metrics.json` 移走，否则**一个 GPU 作业都不会提交**，
    而输出显示「已完成=N」，看起来一切正常。这类失败最难发现。

18. ~~**回退一段代码时删多了 → `build_clients` 成了没有调用者的死函数**~~
    ✅ **已修复**（`<本次>`）
    `a723a22`（方案 B 回退，撤销上一版对「合并 train+test」的误删）把

    ```python
    clients, baked_assignments, edge_fine_classes = build_clients(x_train, y_train, ...)
    ```

    连同它上面那段注释**整个 hunk** 换成了两行 `np.concatenate` —— 合并加回来了，
    `build_clients` 的调用没加回来。净效果：`build_clients` 全仓库无调用者，
    `x_all/y_all` 算完没人用，`clients` 从未绑定，`baked_assignments` /
    `edge_fine_classes` 恒为 `None`（superclass 分区那条路径连带失效）。
    集群上炸在 `main.py:695` 的 `NameError: name 'clients' is not defined`，
    **与 config / attack / framework 无关，任何格子都跑不起来**。
    现场特征：日志有 `[Setup] Building clients...`，但**没有**紧随其后的
    `[Setup] N clients built` —— 后者才是 `build_clients` 真的跑过的证据。
    顺带修掉：`[Setup] client pool = train split only (50000 samples)` 那行是
    方案 A 时期的文案，代码早已改回合并 60k，日志与行为**相反**（陷阱 #7/#14 同类）。

    **为什么 L1 全绿还是炸了**：`test_no_test_leakage.py::test_merge_is_kept_because_it_is_not_the_bug`
    只断言 `run_experiment` 里还有 `np.concatenate([x_train, x_test])` ——
    这句话在那份炸掉的代码上**照样成立**。而 `fedavg/` 绝大多数模块 import TF，
    本地跑不起来，于是这个 NameError 只能等 GPU 排到、跑到第 695 行才暴露。
    守卫：`tests/test_main_names_are_bound.py` —— 纯 stdlib AST 作用域分析，
    扫全部 `fedavg/**/*.py`，断言函数体里读取的每个名字都在「函数内某处被绑定 ∪
    模块级 ∪ builtins」里。反向锚点已实测：对**改动前**的 `main.py` 精确报出
    `('run_experiment', 'clients', 695)`，改动后为空。
    > 该扫描顺带查出三个**没有任何人 import 的死文件**（`client/Client_BadPFL.py`
    > 语法都不过、`data/partition_pfedme.py` 少 import `make_client_dataset`、
    > `data/clustering_pfedme.py` 少 import `_print_assignment`），记在测试的
    > `DEAD_FILES` 白名单里，并配 `test_dead_files_are_still_dead` 守着
    > 「它们必须仍然没人 import」。**要用其中任何一个，先把里面的未定义名修掉。**

    **教训**：改「一段」代码时，被删的和被加的**不是同一件事**也会落在同一个 hunk 里。
    回退前先问：这个 hunk 里除了我要撤的那句，还顺带带走了什么。

19. **CLI 参数静默盖掉 yaml → 防御轴实际没开**（2026-09-24 发现，**未修**，D-007 另开 S1b）
    `experiments/attack/hfl-propagation/exp3_cell.sbatch:46` 写死 `--defense none`，
    `main.py:225-230` 让它覆盖 yaml 的 `defense.name: median / multi_krum`。
    6 个 `def_*` 格子的 metrics.json 全是 `run.defense="none"`、`admitted=[]` ——
    文件名说有防御，实际没有，与陷阱 #7 同类。`calib_cell.sbatch` 照抄了同一行。
    现在的防线：`[Provenance]` 记录 `cli_overrides`；`harness/status.py` 把这类格子标成
    `mismatch`（`python3 harness/status.py experiments/attack/hfl-propagation/registry/v1.yaml`
    → 恰好 6 个）；`tests/test_registry.py::test_no_new_job_script_hardcodes_a_defense`
    禁止新脚本在命令行写死防御。**新写作业脚本只传 `--config`**（见 `hfl-mechanism/cell.sbatch`）。

20. ~~**同一配置、同一 seed，重跑结果不同**~~ ✅ **已修**（P2 口径；AUDIT A15 `done`，2026-09-27）
    第 19 条的 6 个格子等于 3 组同配置、同 seed 的重复，第 1 轮就分叉：flat 三次的第 1 轮
    gm_acc 为 0.1144 / 0.1116 / 0.1122，末 10 点 benign ASR 为 0.755 / 0.696 / 0.685。
    `3c_R5` 与 `2edge_distributed` 因素完全相同，实际是第 4 份重复（0.738 / 0.724 / 0.723 / 0.756）。
    **所以「固定 seed」只锁住了划分与布点，锁不住训练轨迹**；按 seed 配对的设计要按这个噪声算 seed 数。
    官方 Bad-PFL 设了 `cudnn.deterministic`（`utils.py:8-13`），但**只播了 torch**：划分与客户端顺序每次都不同（FINDINGS F-024）。
    A2 定为 D-028：打开 `enable_op_determinism()`，加 `[Checksum]` 行验收（A4 实现）。
    验收：pilot 第二轮 DET 两次 run（不同节点）前 5 轮 checksum 逐轮相同、指标到 4 位小数相同（F-045）。
    **只对开了 `training.deterministic_ops` 的配置成立**（P2 模板开了；P1 / 冻结配置仍不确定），
    **且每个 run 的 CPU 核数要相同**：CPU 上 2 核与 4 核的 checksum 不同（F-047）。别改 `cell.sbatch` 的 `-c 4`；
    一卡多跑的 `pack.sbatch` 让每个 run 恰好绑 4 核。

21. ~~**学习率按 cloud round 衰减 → flat 与 HFL 的 LR 日程不同**~~ ✅ **P2 已改按有效轮**（AUDIT A08 `deviate`，D-029 pilot 通过，2026-09-27）
    `client_base.py:117-126`：`lr0·0.992^cloud_round`。同样 200 有效轮，末端 lr：flat 0.020、
    R_edge=5 0.073、R_edge=40 0.096。flat vs HFL、R_edge 扫描都混进了这个差别。
    官方 Bad-PFL 是**常数** lr=0.1（`main.py:57`）。A2 定为 D-023：**保留**调过参的 0.992 与 head 0.005（来历见 F-030），
    改成**按有效轮衰减**，flat 逐字节不变；要先过 D-029 可行性实验。

22. **与官方 Bad-PFL 实现的差异**（2026-09-24 初查 16 条；A1 / A2 / A3 逐行拍板，**2026-09-27 AUDIT 全部关闭 → 口径 P2**）
    清单、双方行号与处理状态见 `experiments/attack/hfl-mechanism/AUDIT.md`。影响最大的几条：
    - 评估时的 ξ 算在**受害者**模型上（白盒），官方算在攻击者模型上（`fba.py:53,64`）；
    - 本地训练量：5 个 epoch vs 官方 15 步 —— **保留**（D-024：总本地训练量已与论文相当，F-029）；
    - ASR 只数非目标类，官方不过滤；
    - 聚合按样本加权，官方不加权 —— **保留**（D-027）；真正的问题是客户端不等大，S3 要求等大小（F-028）；
    - 生成器用干净数据训练，官方用已投毒的数据；
    - 数据增强只采一次就被缓存冻结，batch 组成也冻结，seed42 下 3.2% 的样本从未参与训练 —— **要修**（D-026，AUDIT A25）。
    - FedRep：本仓库 head 1 epoch（lr 0.005）+ body 5 epoch —— **维持**（D-024 取代了 D-012 的「各 15 步、lr 同为 0.1」）。

    A2 的原则（D-022）：攻击定义必须对齐；训练协议不为对齐而对齐，改调过参的值要先做可行性实验。

    A3（2026-09-26，D-030 … D-038）的要点：
    - **主 ASR 与主 pm_acc 必须在同一个个性化模型上测**，两者都改用 fresh-PM（当前 edge body + 自己的 head + 自己的 BN 统计量）；P1 的陈旧 PM 作副列（D-033）。
    - FedRep 下 BN 的 γ/β 共享、moving 统计量私有（D-032）。这是实现选择，不是 TF / torch 的框架差别（F-036）。
    - `build_resnet10` 的 stride-2 卷积用 `same`，主路与 shortcut 错位 1 像素，BN 与初始化也是 Keras 默认值（F-033）。A4 新增 `resnet10_torch` 对齐，冻结的 `resnet10` 不动（D-034）。
    - edge 改为按 edge 轮交错执行（共享生成器在顺序执行下更新次序偏斜，F-038），配额按有效轮轮转（D-036）。
    - FedRep 训练顺序：pilot 判 different（body_first 的 fresh pm_acc 低约 0.10），用户定**维持 head_first**（D-045，A26 `deviate`）。

    `experiments/METRICS.md` 里「ξ 用的是 mal[0] 的模型」这句与两边都不符；它与 Bad-PFL 库双份同步，改时两库一起改。
    **`AUDIT.md` 全部关闭之前，不跑任何 P2 run**（D-006）；`submit.sh` 与 `status.py` 会按这一条拦截。
    **2026-09-27 已全部关闭**（pilot `2853433`，F-045），`PROTOCOL_VERSION = "P2"`；各 G 组现在只等自己的功能会话（S3–S8）。

23. **GPU 确定性 × 在推理模式的 BN 上求梯度 → `UnimplementedError`（CPU 测不出）**
    ✅ **已修**（2026-09-26，D-043；**2026-09-27 GPU 上由 pilot 第二轮验证**，6 个 run 全部跑通，F-045）
    `enable_op_determinism()`（A15）下，TF 的 GPU `FusedBatchNormGradV3` 在
    `is_training=False` 时没有确定性实现，直接抛。Bad-PFL 的 PGD ξ、生成器训练（穿过冻结的 F
    回传）、评估侧 ξ 都在推理模式的 F 上求梯度 → exp3 改版的 pilot 第一轮 6 个 run 全崩（F-043）。
    **更糟的是崩之前**：训练侧的异常被 `_collect_updates_*` 吞掉，恶意端被踢出聚合，run 在
    **没有攻击者**的状态下照常往下跑 —— 只是评估侧恰好也崩了才暴露。
    这个检查只在 GPU kernel 里，**CPU 上的 L1 / 本地 smoke 永远是绿的**（F-042 就是 CPU 双跑）。
    现：`resnet10_torch` 的 BN 是 `models/cnn.py:TorchBatchNorm` —— 推理模式用
    `tf.nn.batch_normalization`，训练模式原样 fused。守卫 `tests/test_bn_inference_determinism.py`
    测**图里有没有 `is_training=False` 的 fused BN 算子**（CPU 上看得见），反向锚点是冻结的 `resnet10`；
    fixture `gpu_determinism_check` 把 GPU 的这条检查在 CPU 上模拟出来（改梯度注册表），
    直接跑 Bad-PFL 的三处调用点 —— 原生 BN 在 CPU 上就复现 pilot 的崩溃。
    pilot 判定另加有效性闸（D-044）：`client_failures` 非空或崩溃 → `invalid`，不再判 `pass`/`fail`。
    > **新模型要开确定性**：BN 一律用 `TorchBatchNorm`（或同样的推理路径），并把它加进那个测试。
    > **任何「GPU 上才会抛」的东西**：先用 `RUN_GROUPS=DET` 这类几分钟的短作业探路，再烧整 run。
