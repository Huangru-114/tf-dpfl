# Stage B —— 收敛标定（回应导师意见 #5）

> **必须先于主力重跑。** 用数据定预算，而不是凭感觉把 local epoch 从 1 改成 3。
> 这一步的结果直接决定 Experiment 3 每一格的成本。

导师原话是「200 rounds + 1 local epoch 效率低，把 local epoch 提到 3–5、把轮数降下来」。
这批实验回答的是：**等算力下哪个 `(local_epochs, n_rounds)` 组合最快到平台。**

顺带更正报告 §2 里一句不成立的陈述：Exp 1 写着「1 local epoch」，实际是
**15 local *steps* ≈ 0.38 epoch**；Exp 3 才是真的 1 epoch。

## 实验格（6 个）

| 自变量 | 取值 | 说明 |
|---|---|---|
| `training.local_epochs` | 1 / 3 / 5 | **唯一自变量** |
| `seed` | 42 / 43 | 2 seed，看落差有没有大到淹掉自变量 |

其余逐字继承 `experiments/attack/hfl-propagation/2edge_distributed.yaml`（3A 的锚点拓扑），
只另改两处，都是为标定本身服务：

| 键 | 锚点 | 标定 | 为什么 |
|---|---|---|---|
| `n_rounds` | 80（=400 有效轮） | 30（=150 有效轮） | 标定只要「什么时候到平台」，而 epoch 越大平台来得越早。若 epochs=1 到 150 有效轮还没平，**那本身就是结论** |
| `eval_interval` | 2 | 1 | 每 5 个有效轮一个点（30 个点）。轨迹要够密才看得出拐点，同时给 30 个 `[Timing]` 样本 |

守卫：`tests/test_calibration.py` 逐格断言「与锚点只差这几个键」。多漂一个键，
得出的 (epoch, rounds) 建议就是错的，而事后从 `metrics.json` 分辨不出来。

## 怎么跑

```bash
bash experiments/calibration/run_calibration.sh --dry-run   # 先看清单
bash experiments/calibration/run_calibration.sh            # 提交 6 个 GPU job
bash experiments/calibration/run_calibration.sh --status   # 看进度
```

⚠️ 与 `run_exp3.sh` 同一个坑（陷阱 #15）：重跑前把旧的 `results/*.metrics.json`
移走，否则**一个作业都不会提交**，而输出显示「已完成=6」，看起来一切正常。

## 怎么读

```bash
python3 experiments/calibration/read_calibration.py          # 不需要 GPU
python3 experiments/calibration/read_calibration.py --tol 0.005 --json
```

出一张表：

| 列 | 含义 |
|---|---|
| `plat_MTA` / `plat_ASR` | 各自到平台的 cloud round；**判不出来报 `n/a`，不猜** |
| `plat_eff` | 到平台的**有效轮** = cloud × `edge_rounds`，跨 local_epochs 才可比 |
| `GPU-s→plat` | 到平台的**实测**墙钟（train + 后门评估求和，不是均值×轮数）。**选预算就看这一列** |
| `eval%` | 后门评估占全程墙钟的比例。它高 → 降轮数比提 epoch 更划算；它低 → 反之 |

**平台的定义**（可计算，不看图）：令 `final` = 最后 3 个评估点的均值，
平台轮 = 最小的评估轮 r，使得从 r 起**每一个**后续点都落在 `final ± tol` 内。
找不到就报 `None` —— 「150 有效轮还没收敛」是结论，不能偷换成一个数字。

时间数据来自本轮新加的仪表（`[Timing]` 行 + `[Cloud]` 行的 `time=`）。
注意两段口径不同、必须分开累加：`round_time` 每轮都有且**不含**后门评估，
`_backdoor_eval` 是在 `super().run_round()` 返回之后才跑的。见 CLAUDE.md「墙钟已可拆分」。

## 成本估计（待实测校正）

`exp007.notes.md` 记录锚点是 **36.8 s/round × 400 轮 ≈ 4 GPU-h**（`local_epochs=1`）。
若每轮成本对 `local_epochs` 近似线性，则本批 ≈ `(1+3+5)/1 × 150/400 × 4 × 2 seed ≈ 27 GPU-h`。
**「近似线性」这一条我没有证据** —— 评估开销不随 epoch 变，而它占多少正是本批要测的。
真实数字以 `read_calibration.py` 的 `GPU-s→plat` 为准。
