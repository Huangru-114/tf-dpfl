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

**留了接口但没有实现的**（不要以为它们能用）：
- 主动防御（需要客户端配合的防御）：接口齐了（`BaseDefense.layers` /
  `client_mixin` / `make_control` + 客户端侧 `set_control` / `get_aux`），无任何实现。
- cloud 层防御：`defense.layers` 缺省 `["edge"]`，写成 `[edge, cloud]` 才启用。
- cloud 层的方法专属聚合：`CloudServer.aggregate_edges` 是空壳，见陷阱 #8。

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
