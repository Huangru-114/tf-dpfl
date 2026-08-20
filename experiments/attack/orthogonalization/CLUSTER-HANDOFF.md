# 集群交付清单 —— 正交化会话

分支：`claude/attack-pfl-orthogonalization-e3c3dd`
提交：`179824a`（方案 v1）→ `87e10ff`（方案 v2）→ `dd7fd5e`（实现）→ `4021e88`（harness）

```bash
git fetch origin claude/attack-pfl-orthogonalization-e3c3dd
git checkout claude/attack-pfl-orthogonalization-e3c3dd
```

---

## 零、运行环境（跑之前必读）

集群上**所有 python 都必须在 apptainer 容器里跑**，裸 `python3` 一个库都找不到；
`.sh` 要用 `sbatch` 提交。仓库里这件事已收口到 `cluster_env.sh`，
`run_l1.sh` / `run_smoke.sh` / `experiment_tf.sh` 都已改为经它解析出 `$PY`，
所以下面的命令直接可用。

**每次跑之前扫一眼第一行输出**：

```
[env] python = apptainer exec --nv /nobackup/.../torch_fl.sif python3   (mode=apptainer)
```

若显示 `mode=local`，说明没检测到容器 → 后面必然一串 ImportError，先查
`TFDPFL_SIF` 指的路径存不存在。

> 顺带修了一处：`experiment_tf.sh` 原先硬写 `$ROOT/tensorflow.sif`，
> 与实际容器路径 `/nobackup/proj/disk/naiss2025-22-1095/personal/ziangg/torch_fl.sif`
> 不一致 —— 矩阵提交器此前很可能根本跑不起来。现在统一走 `cluster_env.sh`。

---

## 零点五、跑之前先知道三件事

**1. `run_l1.sh` 在集群上会返回 FAIL，这是预期的。**
6 条红全部是**既有的**陷阱 #4 / #5，不是本次改动引入的。
本次改动前在同样环境下是 **24 条红**，现在是 **6 条**。

**2. `--config` 此前被静默忽略，所以你以前跑的所有「smoke」都不是 smoke。**
旧 `load_config` 先 `open(path)` 再 `parse_known_args()`，`args.config` 全仓库从未使用
→ `run_smoke.sh` 传的 `--config ../experiments/smoke-base.yaml` 无效，
每次实际跑的是 `fedavg/config/config.yaml`（**cifar100 / 100 client / 40 round**）。
**观察性确认**：修好之后 smoke 应该在 ~3 分钟内跑完；如果又跑了几十分钟，说明没生效。

**3. 本会话的 L2 是在合成数据上跑的**（容器代理封了 `cs.toronto.edu`）。
接线验通了，但**所有 ASR / acc 的绝对值都不可引用**。下面第二步就是补真实数据。

---

## 一、Step 0：L1（秒级）

```bash
bash run_l1.sh
```

**预期（精确）**：`213 passed, 3 skipped, 3 xfailed, 6 failed`
（本地无 TF 时是 `172 passed / 4 skipped / 3 xfailed` → PASS；这两个数字都实测过）

| 类别 | 数量 | 说明 |
|---|---|---|
| passed | 213 | |
| skipped | 3 | `test_malicious_client_does_not_silently_downgrade_to_fedavg` 的 `hierfedavg` 格，按设计 skip |
| xfailed | 3 | FLAME 的已知 bug（陷阱 #3），本来就是 xfail |
| **failed** | **6** | 全部既有：`test_neurotoxin_mask` ×2（陷阱 #4）、`test_badpfl_trigger` ×4（陷阱 #5 + 两个 autoencoder shape + 一个测试自身的 float32 舍入写法） |

**对不上就停下来告诉我**，尤其是：
- 6 之外多出来的红 → 我引入了回归
- `test_attack_method_orthogonality` 出现任何红 → 正交化没生效（本地是 33 passed）

回传方式：直接把最后那行 summary 贴给我；有多余的红就再贴 `--tb=short` 的首个断言。

---

## 二、Step 1：正交化的 2×2（这是本次改动的核心验收）

四格，每格约 3 分钟。**这是 `--config` 修好之后第一次真正的 smoke。**

```bash
sbatch run_smoke.sh attack orthogonalization badnet     none exp101 hier_fedavg
sbatch run_smoke.sh attack orthogonalization badnet     none exp102 hier_pfedme
sbatch run_smoke.sh attack orthogonalization neurotoxin none exp103 hier_fedavg
sbatch run_smoke.sh attack orthogonalization neurotoxin none exp104 hier_pfedme

squeue -u "$USER"        # 看排队/运行状态
```

四个作业互相独立，可以同时提交。`run_smoke.sh` 自带 SLURM 头
（`-c 4 --gpus 1 --mem=24G -t 04:00:00 -A naiss2026-4-650-gpu -p gpu`），
`sbatch` 会把位置参数原样传进去。

| exp | 攻击 | 方法 | 这一格在回答什么 |
|---|---|---|---|
| 101 | badnet | hier_fedavg | 基线。vanilla 走 `MalMixin is None` 分支，与改动前同路径 |
| 102 | badnet | hier_pfedme | **「没跑坏」**：良性路径数值应零变化 |
| 103 | neurotoxin | hier_fedavg | 改动前这一格**碰巧是对的**（攻击类本来就继承 FedAvgClient） |
| 104 | neurotoxin | hier_pfedme | **改动前这一格是坏的** —— 恶意客户端跑朴素 FedAvg |

### 判据（从 metrics.json 直接读，不靠人肉判读）

1. **exp104 的恶意客户端必须真的跑 pFedMe。** 在日志里 grep：
   ```bash
   LOGDIR="${TFDPFL_LOGDIR:-$(cd .. && pwd)/tfdpfl-logs}"
   grep -E "client classes|\[Client  0\] Round 1" \
        "$LOGDIR/smoke_attack_orthogonalization_neurotoxin_none_hier_pfedme.log" | head -4
   ```
   必须看到：
   ```
   [Setup] client classes | benign=PFedMeClient | malicious=NeurotoxinPFedMeClient
     [Client  0] Round 1 | Hier-pFedMe(λ2=..., K=...) | loss=...
     [Client  0] Round 1 | Neurotoxin(mask=top-5%, ref=edge_weights)
   ```
   若 `malicious=` 不是 `Neurotoxin*PFedMeClient`，或中间那行缺失 → 正交化没生效。

2. **exp101 vs exp103 的 `final_acc.gm_acc` 应同量级**（都是 hier_fedavg，攻击强度不同但不该崩）。

3. **exp102 的 `final_acc` 不应显著低于 exp101 之外的 pFedMe 基线**
   —— 若你手上有改动前 hier_pfedme 的历史数字可以对，但注意历史「smoke」跑的其实是全量配置，
   **不可直接比**。这也是我说「改动前的对照数字不存在」的原因。

4. **四格的 `run.n_rounds == n_malicious_participations == 5`**（neurotoxin/badpfl/cerp 强制 Q=1；
   badnet 用 `attack_freq_Q`，smoke 配置里若 Q>1 则这个数会小于 5，属正常）。

---

## 三、Step 2：防御轴一格（验新的统一判决行）

```bash
sbatch run_smoke.sh defense flame badnet flame exp201 hier_pfedme
```

判据：`metrics.json` 里
- `admitted[]` **非空**，每条带 `layer` / `defense` / `admitted` / `total` / `rejected`
- `admitted_count_mean` 不是 `null`
- `rejected_ids` 非空

改动前这三个字段只有 flame 才有值（其余 4 个防御全瞎）。若想一次把 5 个防御都验了：

```bash
for d in flame multi_krum dnc median trimmed_mean; do
  sbatch run_smoke.sh defense $d badnet $d exp2_$d hier_pfedme
done
```
`median` / `trimmed_mean` 是坐标类防御，预期 `admitted` 为 `null` + 日志里是
`[Decision] edge0 | median | coordinate-wise | n=...`（**不是** `admitted=0`）。

---

## 四、怎么把结果交给我

按 CLAUDE.md 的回程协议：**只回小产物，大日志留集群。**

**首选（最省事）**：把 metrics.json 提交推上来，我直接读。

```bash
git add experiments/attack/orthogonalization/*.metrics.json experiments/defense/*/*.metrics.json
git commit -m "results: 正交化 2x2 + 防御轴 smoke（真实 CIFAR-10）"
git push -u origin claude/attack-pfl-orthogonalization-e3c3dd
```
然后跟我说一句「结果推上去了」。每个 json 约 4 KB。

**次选（不想推 git）**：贴这几样，够我判读：

1. `bash run_l1.sh` 的最后一行 summary
2. 每格的 `slurm-<jobid>.out` 里结尾那几行 `[collect]`（它已经把关键数打出来了）
3. Step 1 判据 1 的那三行 grep 输出
4. 任何 traceback 的**首行 + 最后 10 行**

**绝对不要回传**：完整日志、`report_*.txt`、checkpoint、大 npy。
（`report_*.txt` 和 `slurm-*.out` 我已加进 `.gitignore`。）

### 产物都在哪

| 东西 | 路径 |
|---|---|
| 小 metrics.json（**回传这个**） | `experiments/<axis>/<method>/<exp_id>.metrics.json` |
| 大日志（留集群） | `${TFDPFL_LOGDIR:-<仓库同级>/tfdpfl-logs}/smoke_*.log` |
| sbatch 的 stdout | 提交目录下的 `slurm-<jobid>.out` |

---

## 五、跑挂了怎么办

先看 `metrics.json` 里这两个字段，它们就是为此加的：

- **`errors[]`** —— traceback 首行
- **`client_failures[]`** —— 被 `_collect_updates_parallel` 吞掉的客户端异常。
  run 会照常跑完、日志一切正常，但那个客户端的更新根本没进聚合。
  **如果恰好是恶意客户端每轮都在这里，ASR 必然是 0**，而这与「攻击无效」看起来一模一样。

若 `client_failures` 非空，把那几条贴给我，别急着调超参。

---

## 六、这次会话之后还欠什么

| # | 事项 | 状态 |
|---|---|---|
| 1 | 真实数据的 L2（本文件 Step 1/2） | **待你跑** |
| 2 | 陷阱 #4：Neurotoxin `ratio` 是保留比例还是屏蔽比例 | 未动。需 clone 官方实现核对，`tests/test_neurotoxin_mask.py` 那 2 条红就是等它 |
| 3 | 陷阱 #5：`CIFAR10_STD` 硬用在 cifar100 配置上 | 未动 |
| 4 | `build_autoencoder(img_size=8)` 输出 16×16 | 未动（badpfl 那 2 条红） |
| 5 | Rep 家族的**私有 head 进了 FLAME/multi-Krum 的距离计算** | 只有代码路径证据，**无数值证据**。验法：构造 backbone 相同、head 随机发散的一组更新，断言 FLAME 接纳集合不变 |
| 6 | `defense/flame.py:76` 用全局 `np.random.normal` | 陷阱 #2 的第五处漏网，在训练循环内 → `defense=flame` 固定种子重跑对不上 |
| 7 | cloud 层聚合分支全被注释（`beta_hier` 是死配置） | 本会话只留了 `aggregate_edges` 接口，行为未改 |
| 8 | 主动防御的第一个实现 | 接口已就位（`layers` / `client_mixin` / `make_control` / `set_control` / `get_aux`），无实现 |

第 5、6 条我建议下一个会话优先做 —— 它们都在**防御轴**上，而你 Step 2 一旦跑出真实数字，
这两条会直接影响怎么解释那些数字。
