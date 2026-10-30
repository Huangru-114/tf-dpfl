# CCSF 的 FedRep 版结果（旧口径，D-113 / D-114）

CCSF 在 2026-10-30 改到 FedPer + 官方超参数之前，用 **hier_fedrep** 跑出来的两份结果，移出 `results/` 以免被新 run 覆盖：

| 文件 | 内容 |
|---|---|
| `CCSF__off__s42.metrics.json` | FedRep、无防御，300 轮跑完（exit 0；git `769518f`） |
| `CCSF__ccs-full__s42.metrics.json` | FedRep、CCS，第 54 轮良性端 NaN 崩溃（exit 1；git `71b1255`，48G）；主任务从第 10 轮起垮（F-109） |
| `CCSF__ccs-full__pack-k1-s42.gpu.json` | 上一行那个作业的显存记录 |

用途：D-114 ⑥ 消融里的 FedRep 两臂（前 30 轮；P2 下同配置逐位可复现，不重跑），以及「CCS 能否迁移到 FedRep」的参照。
配置 sha 与现在登记表里的 CCSF 不同 —— 不要放回 `results/`。
