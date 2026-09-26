# Experiment 3（改版）—— HFL 中个性化后门的机制研究

旧方案（`../hfl-propagation/`）已冻结，只作参照。本目录是改版方案的唯一入口。

## 先读哪个

| 文件 | 内容 |
|---|---|
| `current-focus.md` | 下一会话唯一要回答的问题 |
| `PLAN.md` | 评审、子实验→代码映射、预注册判定、运行组、执行路线、图目录 |
| `AUDIT.md` | 与官方 Bad-PFL / FedRep / 原文献的对齐审计。**全部关闭前不开跑** |
| `DECISIONS.md` | 决策日志（只追加） |
| `FINDINGS.md` | 结论台账（带证据和状态） |
| `PLAN-original-2026-09-24.md` | 原始规划，逐字保存 |

## 目录

```
registry.yaml           运行组声明（因素网格、seed、requires）
configs/                materialize 生成的完整配置（GENERATED，不要手改）+ INDEX.tsv
results/P<N>/<group>/   <run_id>.metrics.json（P = 口径版本，见 PLAN §0）
analysis/               runs.csv / series.csv / verdicts.json（脚本生成）
figures/                生成物（gitignore）；定稿图放 figures/final/
```

## 工作流

所有 harness 脚本都是纯标准库 + PyYAML（只有画图要 matplotlib），本地和登录节点都能跑。

```bash
# 0. 看每组有多少 run、还缺什么（审计 / 功能会话）
python3 harness/registry.py experiments/attack/hfl-mechanism/registry.yaml

# 1. 生成配置（base 在 A4 之后才有；在那之前这一步会明确报错）
python3 harness/registry.py experiments/attack/hfl-mechanism/registry.yaml --materialize

# 2. 提交（登录节点，纯 bash）。AUDIT.md 未全部关闭时只列清单、不提交
bash experiments/attack/hfl-mechanism/submit.sh --status
bash experiments/attack/hfl-mechanism/submit.sh --dry-run
RUN_GROUPS="G0 G2" bash experiments/attack/hfl-mechanism/submit.sh

# 3. 回传后对账：todo / blocked / failed / mismatch / stale / done + orphan
python3 harness/status.py experiments/attack/hfl-mechanism/registry.yaml

# 4. 整洁表（按实际因素分组，末 10 点均值，拒绝混合口径版本）
python3 harness/runs_table.py experiments/attack/hfl-mechanism/results/P2 \
    --out experiments/attack/hfl-mechanism/analysis

# 5. 预注册判定（PLAN §3）
python3 harness/verdicts.py experiments/attack/hfl-mechanism/analysis/runs.csv \
    --out experiments/attack/hfl-mechanism/analysis/verdicts.json

# 6. 出图（组内因素不唯一会被拒绝；--floor 画下限虚线）
python3 harness/figures.py trajectory --tables experiments/attack/hfl-mechanism/analysis \
    --metric local_benign_asr --group-by n_edges,edge_rounds \
    --floor poison_ratio=0 --out experiments/attack/hfl-mechanism/figures/F2.png
python3 harness/figures.py per-edge --tables experiments/attack/hfl-mechanism/analysis \
    --run <run> --metric edge.client_benign --out experiments/attack/hfl-mechanism/figures/F3.png
```

## A4 的可行性 pilot（P1 口径，不进 P2）

`pilot/registry.yaml` 登记了 6 个 run（D-029 / D-031 / A15 的确定性对，D-042）。它不受 D-006 的
审计门槛约束，所以有自己的提交脚本：

```bash
python3 harness/registry.py experiments/attack/hfl-mechanism/pilot/registry.yaml --materialize   # 已生成、已入库
bash experiments/attack/hfl-mechanism/pilot/submit_pilot.sh --dry-run
bash experiments/attack/hfl-mechanism/pilot/submit_pilot.sh
python3 harness/pilot_a4.py experiments/attack/hfl-mechanism/pilot/registry.yaml --json <out>   # 预注册判定
```

P2 = 基配置 `base.yaml` + overlays 里的「P2 对齐」模板 `fedavg/config/alignment_p2.yaml`（每个
AUDIT 对齐项一个开关，默认旧行为；D-039）。`meta.protocol: P2` 的配置少开一项会被拒绝启动。

## 旧方案数据（P1）怎么看

旧方案的 26 个 metrics.json 没有 `[Provenance]` 行，口径版本要显式指定：

```bash
python3 harness/status.py experiments/attack/hfl-propagation/registry/v1.yaml
python3 harness/runs_table.py experiments/attack/hfl-propagation/results --out <dir> --legacy-protocol P1
```

P1 只作试点（DECISIONS D-009）：用来估噪声和效应量、提出假设、测试工具，不进结论。

## 规矩

- **数字只从脚本产物来**：FINDINGS / 报告里的数，要能用上面的命令重算出来（陷阱 #14）。
- **改判定阈值**要在 DECISIONS.md 留一条，不能悄悄改 `harness/verdicts.py`。
- **结论的状态**：`provisional`（P1 或 seed 不够）→ `confirmed`（P2 + 判定通过）→ 被推翻时改成 `retracted` 并写原因，不删除。
- **出图**：横轴是有效轮；ASR 图要画下限；图例写 n；图上文字用英文（容器里没有中文字体）。
