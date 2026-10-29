# current-focus —— 阶段三（edge 原生防御）· 交接给 SA-C（CCS）

> **写于 2026-10-28**（SA0 / P0 / SA1 会话收尾；同日 SA-C 会话改写第一节）。这一版取代之前叠在一起的各段状态；过程与数字都在本目录 `FINDINGS.md`（F-090 … F-100）与 `DECISIONS.md`（D-096 … D-108）。
> ⚠️ 阶段三的实验全部做完之前，不要把分支合并到 main（CLAUDE.md 分支纪律，用户 2026-10-10）。分支 `claude/eloquent-faraday-b9kmop`。

## 一、SA-C（CCS）：代码已完成（2026-10-28，D-109 / F-101）；下一步 = 集群上跑对抗空间探针 → 用户定空间 → 交 CCSF

**做了什么**（语义对照表与取舍：DECISIONS D-109；官方代码的读法：FINDINGS F-101 —— 官方代码本身跑不起来，照的是意图）：
- 开关：`defense.name: ccs` + `defense.layers: [client, edge]` + `defense.ccs.{at, cluster, space, pool, kl, mmd, steps, …}`（`fedavg/defense/ccs_spec.py`，纯算术、不 import TF）。
  CCSP 的三臂 = `at` / `cluster` 两个布尔；只聚类的臂可以用 `resnet10_torch`、不要 client 层。
- 客户端：`client/ccs_mixin.py`（`CCSClientMixin` + 与离线探针共用的 `CCSAttack`）；`FLClientBase.on_body_step` 新钩子（缺省 None = 原样走方法类的 body 步）。
- 服务端：`defense/ccs.py`（复用 `analysis/bn_cluster.detect`；`[CCS]` 行 + 统一的 `[Decision]` 行）；`pool: edge_cache` 给 CCSP 用。
- 模型：`resnet10_torch_proj`（私有投影网络，P2 核对视为与模板等价）。自描述 `[设定12]`、`[MalShare]`；collect_metrics **schema 13**（`run.ccs*`、`run.malicious_data_share`、`ccs_decisions[]`）。
- 判定：`harness/ccsf_verdict.py`（N-008 CCSF 段，**已 commit**）。登记：CCSF 两格已补 `set:`，挂 `ccs-space`（`SA-C` 已加进 available）。
- 守卫：`tests/test_ccs_spec.py`（本地）/ `test_ccs_config.py`（本地）/ `test_ccsf_verdict.py`（本地）/ `test_ccs_tf.py`（TF）。

**对抗样本空间已定**（2026-10-29，F-102 / D-110）：探针显示标准化空间（照官方代码）与像素空间都让模型全错、像素约强 1.9 倍（同一量级）→ CCSF 只跑 input；
`ccs-space` 已放行，CCSF 两格已 materialize（`configs/CCSF__{off,ccs-full}__s42.yaml`）。机时更正：每个 ccs run 约 6–7 GPU-h（不是之前写的 15–25）。

**集群上要做的**（用户）：
1. `git pull`，看 `git log --oneline -1`。
2. 先探路（新模型 + 新代码路径第一次上 GPU；陷阱 #23：对抗训练在推理模式的 BN 上求梯度）：只交 ccs-full、短时限，回来看日志里没有被吞的客户端异常、`[CCS]` 每轮一行、显存：

   `RUN_GROUPS=CCSF bash experiments/defense/edge-native/submit.sh --dry-run`（先看会交什么）

   探路可以直接 `sbatch -t 0:30:00 experiments/attack/hfl-mechanism/cell.sbatch experiments/defense/edge-native/configs/CCSF__ccs-full__s42.yaml CCSF__ccs-full__s42 <临时 metrics 路径>`
   （metrics 写到 results/ 以外，免得被当成完成的 run）。
3. 探路没问题 → `RUN_GROUPS=CCSF bash experiments/defense/edge-native/submit.sh`（2 run；off 约 3–4 GPU-h，ccs-full 约 6–7 GPU-h）。
4. 回传后：

   `python3 harness/ccsf_verdict.py --off experiments/defense/edge-native/results/P2/CCSF/CCSF__off__s42.metrics.json --ccs experiments/defense/edge-native/results/P2/CCSF/CCSF__ccs-full__s42.metrics.json --out experiments/defense/edge-native/analysis/ccsf_verdict.json`

   读 `rejection.fpr` 时记住 F-101 ⑧：官方聚类参数在彼此很近的良性端上也会剔掉一部分。判定不是 `ccs_reproduces` → pixel 版是第一个排查项（D-110）。
   ccs-full 的日志会有一条「backdoor.enabled=false 但 defense.name=…」的假警告，是已有代码的问题（F-102 末条），不用管。

**还开着的**：D-096（「CCS 消融里对抗训练约占 85%」的出处）—— 找到要在 **CCSP 交作业之前**补成预注册预测。CCSP 登记时每布点补一个 off 臂（D-109 ②）。

## 二、P1（在线上传前加固，D-108）：已回传、判 `go_main_col`（2026-10-29，F-103）

- `P1__col-pre__s42`（SAU 8/255、BN 更新、5 遍，上传前加固）vs SNAP__collocated__s42。闸通过（第 1 云轮 `[HardenAgg] unhardened` = SNAP `[Checksum] Round 1`；`[Harden]` 240 行、0 发散）。
- 受害 edge V 0.995 → 0.519（ΔV 0.476）、ΔMTA 0.0086 → `go_main_col`；但 jump 只降 21%（比值 0.79，jump 那一项不过）、攻击者 edge E0 0.998 → 0.950 基本没动；
  臂的 V 不是平台值（第 46–60 云轮 0.24–0.65 振荡）→ **单 seed，幅度没有可外推的证据**。加固墙钟 +116% round_time。
- 判定 `analysis/p1_verdict.json`。**等用户定**：按 PLAN 进 s43–44 复验（`P1__col-pre__s4{3,4}`，各约 7.5 GPU-h），还是先做分散布点 / A-every。

## 三、到现在为止的结论（细节见 FINDINGS）

| 步骤 | 结果 | 出处 |
|---|---|---|
| SNAP（无防御对照 + 快照） | 有效；集中布点与 G1R5 逐位相同 | F-094 |
| SA0（加固核心 + 离线回放工具） | 三轮对抗式审查后完成；离线回放在 GPU 上与原训练一致（V0） | F-096、F-097 |
| 加固 lr | 客户端 lr 在 BN 固定模式下发散；按预注册规则选 train ×0.1、frozen ×0.03 | F-098、F-099、D-106、D-107 |
| P0（离线回放，s42 全 36 个配置） | `accuracy_bound`：砍得动的都伤精度（PGD 4/255 砍 83–88%、精度降约 0.10；SAU 8/255 BN 更新 5 遍砍 44–58%、降 0.02–0.03）；只做干净微调砍不动；BN 固定模式多数更糟 | F-100、D-108；对比表 `analysis/p0_s42_screen_tables.md` |
| 「只吸收部分中心更新」（阻尼） | 一次性回放里看起来好，但只是延缓灌入，对持久 / 强攻击（如 IBA）无用 → 只作参照，不当候选 | 用户 2026-10-28 |
| P1（在线 SAU 一个 run，s42） | `go_main_col`：受害 V 0.995 → 0.519、ΔMTA 0.009；jump 只降 21%、E0 不动、末段振荡 | F-103 |

**用户 2026-10-28 提出的另一条路线**（还没讨论完）：用本区域数据产生的**合成数据**代替区域服务器手里的真实干净集。
看法（会话里已说明）：可以去掉「区域服务器持有真实数据」这个强假设，但合成数据可能把后门特征一起反演出来，且依赖于区域侧加固本身先被证明有用 → 排在 P1 结果之后再议（P1 已回传，F-103）。

## 四、还等用户定的旧事项

- F-095：G1 / G1R5 的在线 c_k 只在飞机 / 汽车两类上测 —— `update_ck.py` 的取样要不要改、REPORT §5.13 要不要加注。
- PLAN §8 仍待定：G8 存盘删不删、F-089 的测试要不要改成与 Python 版本无关。

## 五、环境 / 基线

- 本地 L1：Python 3.11 venv（F-089）；本地 TF 测试用 tensorflow-cpu 2.15（Keras 2）+ scikit-learn 的 venv，`TFDPFL_PY=<venv>/bin/python bash run_l1.sh`。
  2026-10-28 基线（`1b22d10`）：无 TF 1719 passed / 53 skipped / 3 xfailed；有 TF 只有陷阱 #4 的 2 条红。
  SA-C 之后（2026-10-28 实测）：无 TF **1781 passed / 52 skipped / 3 xfailed**；有 TF（tensorflow-cpu 2.15）**2043 passed / 26 skipped / 3 xfailed / 2 failed**（陷阱 #4；改动前同环境 1972 / 26 / 3 / 2）。
- 集群：先 `git pull`，看 `git log --oneline -1` 再交；作业脚本只传参数（陷阱 #19）；排队靠 fairshare（10-27 实测 `sprio` 全部来自 FAIRSHARE），短作业用 `sbatch -t …` 改短时限便于回填。
- 机时：上限约 100 GPU-h / 周（D-047）。
