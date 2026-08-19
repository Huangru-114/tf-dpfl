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
