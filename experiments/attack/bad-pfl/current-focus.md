# current-focus：Bad-PFL

> **前置**：本方法依赖「陷阱 #1 正交化」先完成。若 `tests/test_attack_method_orthogonality.py`
> 还是失败/skip 状态，先停下来告诉我。

## 现在要回答的唯一问题

**攻击在恶意客户端**自己的**模型上到底成不成立？**

不是「ASR 为什么是 0」。现在一直在盯良性客户端的 ASR，但那是**第二级**问题。
生成器只在恶意客户端自己的模型上训练，如果连它自己都骗不动，讨论迁移到良性
个性化模型毫无意义。**先把第一级钉死。**

## 判定「完成」的客观标准（分两级，必须按顺序）

| 级别 | 判据 | 含义 | 现状 |
|---|---|---|---|
| L1 | `pytest tests/test_badpfl_trigger.py -v` → 5 passed（需 TF，集群跑） | 扰动预算 / 投毒比例 / 归一化 / 可复现 | ✅ 集群 5 passed（commit `e34da4b`，2026-08-21） |
| **L2-一级** | `final.local_malicious_asr > 0.9` | **攻击本身学会了** | smoke(exp001) 0.889 → 机制成立✅；接线全通 |
| L2-二级 | `final.local_benign_asr` | 迁移性是否成立 | **待全长 run**：smoke 仅 10 有效轮不足以判；需 round×edge_round≈300~500 |

L2 跑法（本会话定：PFL 方法轴用 **fedrep**，故第 6 参数传 framework 键 `hier_fedavg_fedrep`）：
```bash
# framework 键是 hier_fedavg_fedrep（→ drift_correction=hier_fedrep），不是 hier_fedrep
sbatch run_smoke.sh attack bad-pfl badpfl none exp001 hier_fedavg_fedrep
```

### 关于「修好了」的定义 —— 开工前必须先认这一条

- **一级不过** → 是 bug，继续修。
- **一级过、二级 ≈ 0** → 这**不是 bug，是科学结论**：在本仓库的设定下生成器不可迁移。
  这时候要么按论文的做法改建模（官方靠触发器绑定**目标类的自然特征**，而不是靠
  一个学出来的生成器硬迁移），要么把它记录成 negative result 写进 notes。
  **不要通过调超参去把二级"调"上来** —— 那是在拟合噪声。

> 没有这条约定，这个会话会无限期地追 ASR。

## 参考实现

- 文献：*Bad-PFL: Exploiting Backdoor Attacks against Personalized Federated Learning*, ICLR 2025
- 源仓库：`github.com/fmy266/Bad-PFL`（**clone 前先核实 URL**）
- 关键文件：`fba.py` / `generator.py`
- clone 的 commit：**待填**

## 语义 diff 表（逐格用官方代码证实/证伪）

| 步骤 | 官方 | 本仓库现实现 `client/client_badpfl.py` | 差异 | 怎么验证 |
|---|---|---|---|---|
| 触发器 | `T(x) = x + ξ + δ` | 同左 | 一致 | — |
| ξ（破坏噪声） | FGSM，σ=4/255（[0,1] 空间） | `sigma_norm·sign(∇_x CE)` | 一致 | `test_fgsm_noise_respects_sigma_budget` |
| δ（生成器） | Autoencoder，ε=4/255，输出有界 | `generator(x)·eps_norm` | 待核实输出层是否 tanh | `test_generator_delta_respects_eps_budget` |
| 归一化换算 | — | 硬用 `attack/triggers.CIFAR10_STD` | **不同**：`data/dataset.py:121` 对 cifar100 用 `CIFAR100_STD`，差 ~8% | `test_eps_conversion_uses_dataset_matching_std` |
| 生成器训练目标 | 让加噪样本被判为 target | 同左（model 冻结，30 步） | 一致 | **L2-一级** |
| 生成器是否上传 | 本地训练、不上传聚合 | 同左 | 一致 | — |
| 触发器与目标类的关系 | 绑定目标类的**自然特征**（可迁移性的来源） | 无此机制 | **可能是关键差异** | L2-二级 |
| 训练路径 | — | 手写 eager 循环，绕开父类 `_train_step` | 丢 lr 衰减/近端项；逼得 `n_workers=1` | 正交化的 `on_batch` 钩子应替掉它 |
| 评估侧触发器 | 每个恶意端用自己的 generator | `build_eval_trigger` 统一取 `mal[0]` | 多恶意端时不对 | L2-二级 |

## 已排除的可能

- ~~随机性未播种导致不可复现~~ —— 已在 commit `d2717cd` 修掉（`_poison_batch` 改用 `self.rng`）。

## 进展日志

| 日期 | 做了什么 | 证据 | 结论 |
|---|---|---|---|
| 2026-08-19 | 写 5 条 L1 不变量；确认 cifar100 归一化常数不一致 | `data/dataset.py:121` 用 CIFAR100_STD，`client_badpfl.py:24` 导入 CIFAR10_STD | 归一化偏差属实但幅度小（~8%），**不足以解释 ASR≈0**；主嫌疑是迁移性 + 接线 |
| 2026-08-21 | 阶段0/1：clone 官方（`fmy266/Bad-PFL`@`ad845a5`）对拍；修 L1 三处（P1 生成器 img_size 断言+测试 IMG 8→16；P2 ε/σ 按 dataset 选 STD；P3 FGSM 断言改 np.isclose）+ 修复复现性测试方法论 | 集群 L1：`test_badpfl_trigger.py` 5 passed（总数 6→2 failed，剩 2 条为陷阱#4 Neurotoxin）；commits `f76ff71`/`e34da4b` | L1 全绿。**新发现**：官方用**单个共享生成器**跨全恶意端 co-train（`fba.py:27` closure），本仓库每端独立→ 迁移性(L2-二级)关键差异，记为 P4 缓做。官方数据空间是 [0,1]（仅 ToTensor），本仓库标准化→ `ε/STD` 换算方向正确 |

---

# 交接（2026-08-22，本会话结束，供新会话接手）

## 当前定论（按证据，已纠正此前误判）

- **BN 不是问题**（此前误判，已撤回）。无攻击基线 `fedrep+resnet10(BN)`：PM 0.517→0.720(r5→r35)仍升 → 干净时健康。
- **exp004（badpfl, resnet10, 100client/10mal, lr?）**：GM_ASR=0.988、malicious=0.953，但 **PM acc 崩到 0.41**、benign ASR 卡 0.49。论文同 FedRep 是 **ACC 80.29 / ASR 97.95**。
- 结论：**攻击不隐蔽、过度投毒把主任务打崩**（不是 BN、不是迁移性壁垒）。

## 两个待解问题

### 问题1：让 badpfl+fedrep+resnet 同时拿到正常 acc 和高 asr（对齐论文 80/98）
主嫌疑 = **过度参与/过度投毒**：
- `client_fraction=1.0` → 每轮**全部**恶意端参与，再 ×edge_rounds=5；论文每轮从 100 选 10 → 平均~1 恶意端/轮。实际投毒强度 ≈ 论文 10×。
- `poison_ratio=0.5`（论文 0.2）。
- **下一步（设定对齐，非调攻击超参）**：把 `client_fraction` 降到论文的部分参与（每轮选 ~10%），`poison_ratio`→0.2，看 acc 是否回到 ~0.72 且 asr 仍高。这是最高优先级实验。

### 问题2：badpfl 并行提速
- 现象：`badpfl_allow_parallel=true` + P4 共享 generator → 多线程并发读写**同一** generator → `conv2d_transpose` shape 崩 / `input depth ...` 报错。
- 官方 `reference/Bad-PFL/fl_process.py` **也是顺序遍历客户端**，快是靠 PyTorch+扁平结构，**不是客户端并行** → 线程并行不是论文提速来源。
- 方向：要么（a）串行但降单客户端开销（eager FGSM/retrace）；（b）generator 训练串行、其余并行；（c）每线程独立 generator（与 P4 共享冲突，需权衡）。**当前 `badpfl_allow_parallel=true` 不安全，默认应保持串行。**

## ⚠️ 配置漂移（必须先修）
仓库 `full_p4_resnet.yaml` = 20client/2mal/lr0.02；但 exp004 实际跑的是 100client/10mal（`malicious_ids` 0..99）。**用户改了集群 config 但没 push。** 新会话第一步：让用户 push 真实 config，或据 exp004.metrics.json 的 run 段重建，否则不可复现（陷阱#7）。

## 关键代码位置
- P4 共享 generator：`client/client_badpfl.py:set_shared_generator`；`main.py:449-486`（创建+注入，`badpfl_allow_parallel` 开关也在此附近 ~441）。
- resnet10（BN）：`models/cnn.py:build_resnet10`；fedrep 切分 `get_base_head_indices` 通用（已验证兼容）。
- 并行/串行：`server/edge_server_base.py:_collect_updates_parallel`（ThreadPoolExecutor, n_workers）。
- 参考实现（gitignore，可能需重新 clone）：`reference/Bad-PFL`@`ad845a5`（`fba.py`/`fl_process.py`/`main.py` argparse 默认值 = 论文设定）。

## 分支
`claude/bad-pfl-attack-roadmap-7dt5ng`。新会话按 CLAUDE.md：从此分支继续或按需从 main 重开。**推送铁律：commit → git pull --rebase → push，永不 force。**

---

# 交接（2026-08-23，本会话结束，供新会话接手）

## 本会话定论：问题1「对齐 fraction/poison 也没用」**前提不成立**

exp006（本应是 frac=0.1 / poison=0.2 的 paper-aligned run）**实际没跑在 frac=0.1 上**。
数值反证：`malicious_participation_by_client` 显示 10 个恶意端 × 全部 80 轮参与，而 frac=0.1
下单恶意端每轮命中率仅 ≈0.41（100client/2edge/5edge_round），全部 80 轮的概率 ≈10⁻³¹ → 不可能。
只可能是**有效全参与（fraction≈1.0）**，投毒强度约标称 10×，与 exp004 同类 →
pm_acc 崩到 0.42、benign ASR 卡 0.44 都由此解释。**完整证据链见 `exp006.notes.md`。**

根因是陷阱 #7 的同类：`metrics.json` 的 `run` 块此前不记录 client_fraction / poison_ratio，
「改成论文值」无法从 artifact 证实。

## 本会话已落地（代码，L1 全绿 174 passed）

- `config_validate.py`：打 `[设定] client_fraction=… | poison_ratio=… | n_clients=… |
  n_edges=… | n_malicious=… | arch=…` 自描述行。
- `harness/collect_metrics.py`：解析进 `run` 块；并对「frac<1 却恶意端全轮参与」打红警告
  （exp006 失败模式，从此不静默）。
- `tests/test_collect_metrics.py`：+2 条 L1。

## 下一步（最高优先级，turnkey）

**可验证重跑 exp007**：集群 `git pull --rebase`，跑仓库内 `full_p4_resnet.yaml`
（已是 frac=0.1，别用集群本地改过的），命令见 `exp006.notes.md`。
**验收前置**：新 metrics 的 `run.client_fraction==0.1` 且 participation ≈40% 轮次，否则重来。
过前置后再判 pm_acc / benign ASR。

## 未解澄清（非 bug，记录以免重复踩）

- diff_edge=0.0 是布点假象（2 edge + 恶意 spread → 无 diff_edge 样本），跨 edge 迁移性
  在当前 config 下**未被测到**。想测需 >2 edge 或留干净 edge。
