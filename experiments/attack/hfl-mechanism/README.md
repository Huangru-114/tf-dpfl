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

工作流命令在 S1 完成后补在下面。
