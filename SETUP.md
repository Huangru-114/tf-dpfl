# 仓库组织与集群交互标准（本仓库特化版）

本文件是通用「整合-实验任务标准」针对 **tf-dpfl** 的落地版本。
与通用版的差别集中在一点：

> 通用版假设你在移植**网络结构**（`model.py` + `convert_weights.py` + 逐层 activation
> 对拍 < 1e-5）。本仓库移植的是**训练循环 / 聚合算法**（Neurotoxin、Bad-PFL、FLAME…）
> ——没有层、没有权重要转换。所以对拍循环换成**算法不变量 + 端到端 smoke** 两级验收。

---

## 0. 一条贯穿始终的原则

**大的东西永远不进 git、永远不给 Claude Code 看；只有小的、决策相关的东西跨回来。**

| 类别 | 存哪里 | Claude Code 能看到吗 |
|---|---|---|
| 代码 / 配置 / 脚本 | git | 是 |
| 指标摘要 `metrics.json` | git（`experiments/`） | 是 |
| 截断日志 / traceback | git（`results/`）或直接粘贴 | 是 |
| 模型 checkpoint `*.h5` | **仅集群** | 否，只看 manifest |
| 全量日志 / 大张量 | **仅集群** | 否，只看统计量 |

需要看某个大张量时，让集群脚本先算好统计量（与参照的 max/mean diff、分位数、NaN 位置），
只把这几个数字传回来。

---

## 1. 目录结构

```
tf-dpfl/
├── SETUP.md                本文件
├── CLAUDE.md               每次会话自动带上的约定 + 已确认陷阱清单
├── methods-registry.md     所有方法的台账 = 研究看板
├── run_l1.sh               本地：一条命令跑完 L1
├── run_smoke.sh            集群：一条命令跑完 L2 并吐出 metrics.json
│
├── fedavg/                 唯一活跃代码（cwd=fedavg 运行）
├── reference/              官方实现，gitignore，一次只 clone 1~2 个，用完即删
├── harness/                跨方法复用的测试/回收基础设施
│   └── collect_metrics.py  全量日志 → 小 metrics.json
├── tests/                  L1 算法不变量
├── experiments/<axis>/<method>/
│   ├── current-focus.md    本轮唯一要回答的问题 + 客观判据（模板见 TEMPLATE-）
│   ├── expNNN.metrics.json 集群回传的小结果
│   └── expNNN.notes.md     你的分析和结论
├── results/                截断日志、误差数字的落脚点
└── scratch/                临时产物，全 gitignore
```

`<axis>` ∈ `attack` / `defense` / `pfl`。三处目录同一套命名，便于模式匹配和 prompt 复用。

---

## 2. 两级验收（替代「逐层对拍 < 1e-5」）

### L1 — 算法不变量（`tests/`，本地秒级）

手工构造输入，正确输出可**解析算出**，断言必须精确。

```bash
bash run_l1.sh          # 或 pytest tests/ -v
```

纯 numpy 的直接跑；需要 TF 的用 `pytest.importorskip("tensorflow")` 标记，本地自动 skip、
在集群上跑。

### L2 — 端到端 smoke（集群，约 3 分钟）

```bash
bash run_smoke.sh attack neurotoxin neurotoxin none exp001
```

10 client / 5 round 的最小配置，大日志留 `/tmp`，只产出一个 KB 级
`experiments/<axis>/<method>/exp001.metrics.json`。

> **L1 过了不代表 ASR 会起来。** 本仓库已知失败有一半在「接线」而不是算法本身
> （CLAUDE.md 陷阱 #1）。L2 不可省略。

---

## 3. 单个方法的整合循环

1. **立问题**：填 `experiments/<axis>/<method>/current-focus.md`——本轮**唯一**要回答的
   问题 + 能从 metrics.json 直接读出的客观判据。
2. **引入源库**：`git clone <url> reference/<method>-src`，把 URL 和 commit hash 记进台账。
   此刻只有这一个库在场。
3. **填语义 diff 表**：论文步骤 → 官方实现 → 本仓库实现 → 差异 → 怎么验证。**逐格用官方
   代码证实/证伪，不靠印象。**
4. **先计划后代码**：确认 diff 表后再动 `fedavg/`，同时补 `tests/` 里的不变量。
5. **本地 L1** → 绿了才 push。
6. **集群 L2**：`git pull && bash run_smoke.sh ...` → 回传那一个 metrics.json。
7. **收敛或调试**：过了就 commit + 更新台账状态；不过就回到语义 diff 表，找**第一格**
   对不上的地方，问题从「整个方法」缩小到「一个步骤」。
8. **清场**：`rm -rf reference/<method>-src`，换下一个。

> 不 review、不跑测试，就不 commit。commit 是强制停顿点。

---

## 4. 规模纪律

- **台账驱动**：所有候选方法只在 `methods-registry.md` 里以一行存在，不预先 clone。
- **一次一个**：串行推进。别并行开五个——那是上下文稀释和目录混乱的根源。
- **harness 复用**：`collect_metrics.py`、`tests/conftest.py`、`run_smoke.sh` 写一次，
  之后每个方法纯复用。
- **一个类别一个 hub**：本仓库 = 「分层 FL 后门攻防」这一类。别往里塞别的类别。

---

## 5. 仓库规模红线

| 指标 | 现状（2026-08-19 refactor 后） | 红线 |
|---|---|---|
| 追踪文件数 | 77 | — |
| 追踪内容 | ~740 KB | — |
| 单文件 | 最大 116 KB（`hello_kitty.png`） | > 10 MB 就查 `.gitignore` |
| `.git` | 288 MB（**历史里仍有旧的 venv/wandb/h5**） | > 500 MB 考虑重写历史 |

`.git` 偏大是因为大文件还留在**历史**里。要真正瘦下来需要
`git filter-repo` 重写历史（代价：commit hash 全变，集群侧要重新 clone）。
当前选择是**不重写**，refactor 前的完整状态保留在分支
`claude/repo-refactor-review-xgyi0k`（commit `8ae9b48`）。

**永不进 git**：`*.h5 / *.pt / *.ckpt / 大 npy / *.log / wandb/ / runs/ / venv(lib,bin,pyvenv.cfg)`
