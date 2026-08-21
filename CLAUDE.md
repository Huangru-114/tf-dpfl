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

**⚠️ `--bind` 不能省（踩过的坑）**：apptainer 默认只把 `$PWD` 和 `$HOME` 挂进容器。
本仓库的脚本会**跨目录**访问：

```
cd $ROOT/fedavg  →  读 $ROOT/experiments/smoke-base.yaml   （$PWD 的兄弟目录）
cd $ROOT         →  读 $ROOT/../tfdpfl-logs/*.log          （$PWD 的上一级）
```

这两处在容器里都不存在，报的是 `FileNotFoundError` —— **看起来像「文件丢了」，
其实文件好好的，只是容器看不见**。`cluster_env.sh` 默认绑定仓库的**上一级**目录
（同时覆盖仓库本身与 `tfdpfl-logs`），并在 source 时做一次自检：
容器里看不到仓库就直接报错退出，而不是等 GPU 作业跑到一半才炸。

**同一个坑的第二种长相**：`~/.keras/datasets` 往往是指向共享盘的**符号链接**。
容器里看不到链接目标时它就是**悬空**的，`os.path.isdir()` 为 False，
keras 的 `os.makedirs(..., exist_ok=True)` 去 mkdir 撞上链接本身，抛出

```
FileExistsError: [Errno 17] File exists: '/home/<user>/.keras/datasets'
```

这句话和真实原因毫无关系（既不是「已存在」也不是权限）。
`cluster_env.sh` 现在检测到 `<绑定根>/data/datasets` 就导出 `TFDPFL_KERAS_HOME`，
绕开软链；`data/dataset.py:resolve_keras_home` 另外会在悬空时**提前拦截**并打印
链接指向哪里、该设什么。
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

   | 环境 | 预期 | 说明 |
   |---|---|---|
   | **本地（无 TF）** | `172 passed / 4 skipped / 3 xfailed` → **PASS**（exit 0） | 4 skipped = 4 个需要 TF 的测试模块整体 skip |
   | **集群（有 TF）** | `213 passed / 3 skipped / 3 xfailed / 6 failed` → **FAIL** | 6 条红是**既有**的陷阱 #4/#5，见下 |

   > 若在容器里直接跑裸 `pytest`（不加 `tests/`），会多收 6 条
   > `fedavg/defense/test_defenses_offline.py` → **219 passed**。
   > `run_l1.sh` 只跑 `tests/`，所以是 213。两个数都对，别被吓到。
   > 已实测（2026-08-20，容器内 Python 3.10.12）：`6 failed, 219 passed, 3 skipped, 3 xfailed`。

   集群上 `run_l1.sh` 返回 FAIL 是**当前的预期状态**，不是回归：
   `test_neurotoxin_mask` ×2 是陷阱 #4（mask 语义方向未证实，测试按文献语义写、
   等实现被改过来）；`test_badpfl_trigger` ×4 是陷阱 #5 + `build_autoencoder(img_size=8)`
   的 shape bug + 一条测试自身的 float32 舍入写法。
   这 6 条修好之前，集群上的门禁请用 `bash run_l1.sh 2>&1 | tail -3` 人工核对数字，
   **多出来的红才是回归**。（3 xfailed 是 FLAME 的已知 bug，陷阱 #3。）
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

2. ~~**随机性未播种**~~ ⚠️ **大部分已修**（commit `d2717cd`），但**有一处漏网**
   已修：`select_clients`、投毒选样、DnC 投影、ASR 子采样，都改用
   `np.random.default_rng([seed, edge_id/client_id])`。
   **未修（第五处）**：`defense/flame.py:76` 的高斯噪声用的是全局
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
