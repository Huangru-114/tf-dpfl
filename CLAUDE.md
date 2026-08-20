# 项目：分层联邦学习 后门攻防 移植-实验 hub

本仓库是 hub：**唯一**与集群同步的 git 仓库。活跃代码只有 `fedavg/`。
移植对象是**训练循环 / 聚合算法**（不是网络结构），所以验收信号是
「算法不变量 + 端到端 smoke」，不是「逐层 activation 对拍」。

---

## 新会话开场：先做这三件事

1. **`git branch --show-current`** —— 确认在 `claude/repo-hub-refactor`（或我指定的分支）。
2. **`bash run_l1.sh`** —— 基线分两种环境，对不上就先停下来问我，别在坏掉的地基上改东西。

   | 环境 | 预期 | 说明 |
   |---|---|---|
   | **本地（无 TF）** | `172 passed / 4 skipped / 3 xfailed` → **PASS**（exit 0） | 4 skipped = 4 个需要 TF 的测试模块整体 skip |
   | **集群（有 TF）** | `213 passed / 3 skipped / 3 xfailed / 6 failed` → **FAIL** | 6 条红是**既有**的陷阱 #4/#5，见下 |

   集群上 `run_l1.sh` 返回 FAIL 是**当前的预期状态**，不是回归：
   `test_neurotoxin_mask` ×2 是陷阱 #4（mask 语义方向未证实，测试按文献语义写、
   等实现被改过来）；`test_badpfl_trigger` ×4 是陷阱 #5 + `build_autoencoder(img_size=8)`
   的 shape bug + 一条测试自身的 float32 舍入写法。
   这 6 条修好之前，集群上的门禁请用 `bash run_l1.sh 2>&1 | tail -3` 人工核对数字，
   **多出来的红才是回归**。（3 xfailed 是 FLAME 的已知 bug，陷阱 #3。）
3. **读 `experiments/<axis>/<method>/current-focus.md`** —— 本会话**唯一**要回答的问题
   和客观判据都在里面。没有这个文件就先和我一起写，不要直接开始改代码。

然后按下面「交互约定」走：先出语义 diff 表，我确认后再动代码。

## 当前地基（已完成，不要重复做）

- **聚合信息流已统一**：所有 PFL 方法的 edge 聚合都经 `EdgeServerBase.robust_mean`，
  defense 轴对每个方法都生效。守卫：`tests/test_defense_coverage.py`（AST 静态检查）。
- **上传带身份**：`aggregation/client_update.py` 的 `ClientUpdate` 继承 tuple，
  旧解包写法全部照旧，额外有 `.client_id`。收集顺序严格按 `selected` 顺序。
  防御的接纳/剔除经 `BaseDefense.record_decision` 翻译成 `last_admitted_ids`。
- **随机性已播种**（见陷阱 #2）。
- **启动时配置校验**：`config_validate.py`。新增 PFL 方法必须登记到
  `METHOD_SUPPORTS_DEFENSE`，否则拒绝启动。
- **Hier-PerFedAvg 已移除**（聚合元梯度，与防御接口语义不兼容），
  `_collect_updates_*` 的 `meta_grad` 模式随之删除。
- **矩阵提交器**：`matrix.conf` + `submit_matrix.sh` + `experiment_tf.sh`
  + `harness/collect_matrix.py`。

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

**L2 — 端到端 smoke run**（`experiments/<axis>/<method>/smoke.yaml`，集群，约 3 分钟）
10 client / 5 round 的最小配置，只回传一个 `metrics.json`：
`{asr, acc, admitted_count, malicious_selected_rounds}`。

> **L1 过了不代表 ASR 会起来。** 本仓库已知的失败有一半在「接线」而不是算法本身
> （见下方陷阱 #1）。所以 L2 不可省略。

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

1. **攻击轴与 PFL 方法轴不正交（架构级，未修复）**
   `main.py` 的 `use_cls = MalCls or ClientCls` 让恶意客户端类**替换掉** PFL 方法类。
   而 `NeurotoxinClient` / `BadPFLClient` 都继承 `FedAvgClient`。于是
   `drift_correction=hierpfedme` 时，良性客户端跑 pFedMe、恶意客户端跑朴素 FedAvg，
   上传语义都不同。这足以单独解释 ASR≈0。修法应是 mixin 组合而非替换基类。
   证据：`tests/test_attack_method_orthogonality.py`

2. ~~**随机性未播种**~~ ✅ **已修复**（commit `d2717cd`）
   训练循环内的随机（`select_clients`、投毒选样、DnC 投影、ASR 子采样）已全部
   改用独立 seeded RNG：`np.random.default_rng([seed, edge_id/client_id])`。
   **新写的代码不要再碰全局 `np.random`**（setup 期的分区除外，那里顺序确定）。

3. **FLAME 未按文献实现**
   现实现用 "majority-cosine 近似" 代替 HDBSCAN，阈值取 off-diagonal 余弦相似度
   的**中位数** → 全良性、更新彼此接近时，接纳与否由数值噪声决定，会无故剔除良性
   客户端。另文献是**无权平均**，现实现按 `n_samples` 加权。
   `sklearn.cluster.HDBSCAN`（sklearn ≥ 1.3，集群已有 1.8）可直接照文献写。
   证据：`tests/test_flame.py`

4. **Neurotoxin mask 语义疑似反向**
   官方 `grad_mask_cv` 的 `ratio` 是「**保留**的坐标比例」，取 |grad| **最小**的那部分。
   现实现 `(np.abs(a) < thr)` + `mask_ratio=0.05` 只屏蔽 top-5%、放行 95%，
   ≈ 退化成普通 BadNet，持久性收益消失。另官方是**逐层**阈值，现实现是全局阈值。
   待 clone 官方实现确认。证据：`tests/test_neurotoxin_mask.py`

5. **Bad-PFL 生成器迁移性 / 归一化常数**
   生成器只在恶意客户端自己的模型上训练，评估时却要迁移到良性个性化模型
   （`build_eval_trigger` 取 `mal[0]` 的 generator）。另 `CIFAR10_STD` 被硬用在
   `dataset: cifar100` 的配置上。

6. **TF 框架陷阱**（移植 torch 实现时逐条确认）
   - BatchNorm：TF `momentum ≈ 1 − torch momentum`；`eps` 默认值不同
   - Conv padding：torch 显式 padding vs TF `"SAME"`，`stride > 1` 时不等价
   - 通道序：本项目统一 NHWC
   - 损失：交叉熵确认 `from_logits` 设置
   - eager 手写训练循环与良性客户端的 `@tf.function` 路径在线程池并发会互相污染
     （现有代码因此把 `n_workers` 强制设为 1，是接线不干净的代价，不是必然）
