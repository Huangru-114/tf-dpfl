# current-focus —— 阶段三（edge 原生防御）· D0 完成、SNAP 已回传，交接给 SA0

> **写于 2026-10-09**（D0 会话）。上一版（同日，「交接给 D0」）的五项已全部处理，结果在本目录 `FINDINGS.md` F-091 … F-093 与 N-008（草案）。

## D0 做了什么（没有改 `fedavg/`）

1. **登记表** `registry.yaml`（base / overlay 与 hfl-mechanism 同一条链；审计门槛沿用 hfl-mechanism/AUDIT.md）：
   - **SNAP**（6 run，已 materialize，**可交**）：`D-base` + `snapshot_rounds: "6/15/60"` × {collocated, distributed} × s42–44。
     SNAP-col 与 G1R5 同 seed 的配置只差记录 / 快照开关（守卫 `tests/test_edge_native_registry.py`），6 个配置过 `config_validate`。
   - **SNAP5**（4 run，s45 / s46，D-099 的 5 seed 对照）挂 `h-main`；**CAL**（6 run）挂 `SA1`、**CCSF**（2 run，划分 = 旧 `noniid`，D-098）挂 `SA-C`；**P0** 是离线组，写在 `offline:`，挂 `SA0` + `d0-prereg`。
   - 快照的 edge 模型 = 本轮上传物，`global` = 本轮云聚合结果（F-092，代码核对）。
2. **提交脚本** `submit.sh`（照 `pilot/submit_pilot.sh`，只读本目录 INDEX，复用 hfl-mechanism 的 `cell.sbatch` / `pack.sbatch`，保留审计门槛）。
3. **功效分析** `harness/d0_power.py` → `analysis/d0_power.json`（F-091）：对照臂饱和（G1R5 V 0.996）；假阳性很低；配对 SD 0.035 时 μ = 0.20 几乎必过；
   **`no_effect` 对真零效应也常判不出**（62% / 32%）；margin 2 logit 有余量；P0 的 6 个 col 快照分母都 ≥ 0.146。
4. **文献核对**（F-093）：PLAN 凭记忆引用的 13 条出处都对；两处要带进 SA0 的语义 diff：SAU 官方 ε = 0.2（PLAN 写 8/255）、TRADES 官方 10 步 / β 6。EOT 引文已改写。
5. **预注册草案 N-008**：SNAP 有效性闸、P0（V0 / 精度过滤 / 选择 / `go_online` / `kill_pre` / `accuracy_bound`）、P1（`go_main` / `stop`，新增分散布点的 `stop`）、CCSF（加了「off 臂必须植入」的闸）。

L1（Python 3.11 venv）：基线 1599 passed / 45 skipped / 3 xfailed → 本会话后 **1619 passed / 47 skipped / 3 xfailed → PASS**（+18 = 新测试文件；`test_cluster_env_usage` 自动扫到新的 `submit.sh`：+2 passed、+2 skipped「不调 python / 不写日志」）。

## 状态（2026-10-10）：SNAP 已回传、有效（F-094）

- 6 个 run 全部 `done`；collocated 三个与 G1R5 同 seed 前 60 轮逐位相同（含云聚合后评估点）；快照 6 / 15 / 60 都在（集群 `tfdpfl-dumps/SNAP__*`，约 1.93 GB）。
- P0 的 6 个 col 快照分母都计入（与 F-091 的预测逐位相同）。对照臂两种布点都饱和。

## N-008 已生效（2026-10-10，D-101）

- 攻击者参与的闸 → `n_malicious_participations > 0`；H1 / H3 的 `no_effect` → 「|均值| < 0.05 且最大 |Δ| < 0.10」（`PLAN.md` §4 H1）；
  P1 分散布点的 `stop`、CCSF 的植入闸、P0 的阈值按草案原值确认。`registry.yaml` 的 `available` 已加 `d0-prereg` → P0 只等 SA0。
- `PLAN.md` §8 仍待定：G8 存盘删不删、F-089 的测试要不要改成与 Python 版本无关。

## 下一个会话 = SA0

- edge 侧 AT 核心 + 快照离线探针。先出语义 diff 表（Madry PGD-AT / TRADES / SAU 官方实现 vs 本仓库；F-093 的两处），用户说「开始改」后再写代码。
- 读快照的第一步：核对 npz 里 `meta_json.edge_matches_eval == true`（本地没核对，F-094）。
- 在打开 s42 快照**之前**把 P0 的完整配置清单与预算写进 git（N-008）；`harness/p0_verdict.py` 也要先 commit 再读数据（D-088 的做法）。

## 环境 / 基线

- 分支 `claude/eloquent-faraday-b9kmop`（起点 `de0e8b2`）。
- 本地 L1 用 Python 3.11 venv（F-089）：`uv venv -p python3.11 <dir> && uv pip install -p <dir>/bin/python pytest numpy pyyaml matplotlib && TFDPFL_PY=<dir>/bin/python bash run_l1.sh`。
- 机时：上限约 100 GPU-h / 周（D-047）。
