# current-focus —— 阶段三（edge 原生防御）· D0 完成，交接

> **写于 2026-10-09**（D0 会话）。上一版（同日，「交接给 D0」）的五项已全部处理，结果在本目录 `FINDINGS.md` F-090 … F-092 与 N-008（草案）。

## D0 做了什么（没有改 `fedavg/`）

1. **登记表** `registry.yaml`（base / overlay 与 hfl-mechanism 同一条链；审计门槛沿用 hfl-mechanism/AUDIT.md）：
   - **SNAP**（6 run，已 materialize，**可交**）：`D-base` + `snapshot_rounds: "6/15/60"` × {collocated, distributed} × s42–44。
     SNAP-col 与 G1R5 同 seed 的配置只差记录 / 快照开关（守卫 `tests/test_edge_native_registry.py`），6 个配置过 `config_validate`。
   - **CAL**（6 run）挂 `SA1`、**CCSF**（2 run）挂 `SA-C` + `ccsf-partition`；**P0** 是离线组，写在 `offline:`，挂 `SA0` + `d0-prereg`。
   - 快照的 edge 模型 = 本轮上传物，`global` = 本轮云聚合结果（F-091，代码核对）。
2. **提交脚本** `submit.sh`（照 `pilot/submit_pilot.sh`，只读本目录 INDEX，复用 hfl-mechanism 的 `cell.sbatch` / `pack.sbatch`，保留审计门槛）。
3. **功效分析** `harness/d0_power.py` → `analysis/d0_power.json`（F-090）：对照臂饱和（G1R5 V 0.996）；假阳性很低；配对 SD 0.035 时 μ = 0.20 几乎必过；
   **`no_effect` 对真零效应也常判不出**（62% / 32%）；margin 2 logit 有余量；P0 的 6 个 col 快照分母都 ≥ 0.146。
4. **文献核对**（F-092）：PLAN 凭记忆引用的 13 条出处都对；两处要带进 SA0 的语义 diff：SAU 官方 ε = 0.2（PLAN 写 8/255）、TRADES 官方 10 步 / β 6。EOT 引文已改写。
5. **预注册草案 N-008**：SNAP 有效性闸、P0（V0 / 精度过滤 / 选择 / `go_online` / `kill_pre` / `accuracy_bound`）、P1（`go_main` / `stop`，新增分散布点的 `stop`）、CCSF（加了「off 臂必须植入」的闸）。

L1（Python 3.11 venv）：基线 1599 passed / 45 skipped / 3 xfailed → 本会话后 **1617 passed / 47 skipped / 3 xfailed → PASS**（+16 = 新测试文件；`test_cluster_env_usage` 自动扫到新的 `submit.sh`：+2 passed、+2 skipped「不调 python / 不写日志」）。

## 等用户的事

1. **交 SNAP**（先把本分支合进 main —— 集群只 pull main；然后在集群登录节点）：
   ```bash
   git pull && PACK=3 RUN_GROUPS=SNAP bash experiments/defense/edge-native/submit.sh --dry-run   # 两个 k=3 包
   PACK=3 RUN_GROUPS=SNAP bash experiments/defense/edge-native/submit.sh
   ```
   约 6 GPU-h，盘约 1.8 GB。回传 6 个 `results/P2/SNAP/*.metrics.json`（+ `*.gpu.json`）。快照留集群（`tfdpfl-dumps/SNAP__*`）。
2. **确认 N-008**（或改写）→ 把 `d0-prereg` 加进 `registry.yaml` 的 `available`。特别是：
   - `no_effect` 规则要不要改（F-090 第 3 点）；
   - P1 新增的分散布点 `stop`（1.0 logit）；CCSF 的「off 臂必须植入」闸；
   - P0 里 R_H 0.5 / 0.2、每 edge 精度 0.04 —— 这些没有证据，只是推理。
3. `PLAN.md` §8 其余各条仍待定（CCS 消融出处、头条结论组、DPR 初始化、CCSF 划分、H5 筛选、G8 存盘、F-089 测试）。

## SNAP 回传后（下一个会话的第一件事）

- `python3 harness/status.py experiments/defense/edge-native/registry.yaml` → 6 个 `done`；
- 有效性闸（N-008 SNAP）：`python3 harness/instrumentation_check.py <G1R5 s4x> <SNAP col s4x> --upto 60`，三个 seed 逐轮相同；
  `dumps.snapshots` 轮号 = 6 / 15 / 60。不过 → 停下查，不读 P0。
- 然后 = **SA0**（edge 侧 AT 核心 + 快照离线探针）：先出语义 diff 表（Madry PGD-AT / TRADES / SAU 官方实现 vs 本仓库；F-092 的两处），用户说「开始改」后再写代码。
  SA0 在打开 s42 快照之前要把 P0 的完整配置清单与预算写进 git（N-008）。

## 环境 / 基线

- 分支 `claude/eloquent-faraday-b9kmop`（起点 `de0e8b2`）。
- 本地 L1 用 Python 3.11 venv（F-089）：`uv venv -p python3.11 <dir> && uv pip install -p <dir>/bin/python pytest numpy pyyaml matplotlib && TFDPFL_PY=<dir>/bin/python bash run_l1.sh`。
- 机时：上限约 100 GPU-h / 周（D-047）。
