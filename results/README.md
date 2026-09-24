# results/ —— 集群回传的小产物落脚点

只放**小的、决策相关的**东西：

- 截断日志（集群侧先 `tail`/`grep` 抽出相关几十行）
- 误差数字、traceback
- checkpoint **manifest**（不是 checkpoint 本身）：

```json
{
  "method": "neurotoxin",
  "checkpoint_path": "/mimer/NOBACKUP/.../neurotoxin/round_40.h5",
  "git_commit": "a1b2c3d",
  "round": 40,
  "metrics": {"global_asr": 0.213, "local_acc_mean": 0.887}
}
```

指标摘要请放 `experiments/<axis>/<method>/expNNN.metrics.json`，不要放这里。

**大文件在这个目录里同样被 gitignore 挡住**（`*.log` / `*.npy` / `*.h5` …）。

## 图放哪里（2026-09-24 起）

- 图是**生成物**：由 `harness/figures.py` 从 `runs.csv` / `series.csv` 重新画出来，默认不进 git。
- 改版实验 3 的图放 `experiments/attack/hfl-mechanism/figures/`（gitignore）；要进报告的定稿图
  放 `experiments/attack/hfl-mechanism/figures/final/` 并提交。
- 本目录下的 `figures/*.png`（`df015b2`，2026-09-21）是旧方案 P1 试点批次的图，只作参照，
  **不进结论**（DECISIONS D-009）。
