"""
utils/dumps.py  —  logits 存盘与分析快照的写盘小工具（S9 / D-072 / D-073）

numpy + 标准库，不 import TF（本地 L1 与 config_validate 都能直接用）。

两个开关都在 `alignment.EXTRA_SWITCHES`，默认关，只在登记表组的 `set:` 里开：
  evaluation.dump_logits_every  每第 k 个后门评估点存一次逐样本对数概率（0 = 关）
  evaluation.snapshot_rounds    在这些 cloud 轮末存一次分析快照，写成 "30/70"（None = 关）

**何时开（D-073）**：登记表里写明了消费它的离线分析才开；logits ≤ 50 MB / run；
快照每 run ≤ 3 次、只在有实验意义的时刻（攻击者退出、run 结束、窗口结束）。
快照**只供评估**：不含客户端 / edge / 数据的随机状态，也不含陈旧 client.model，不能续训。

落盘位置：`$ROOT/../tfdpfl-dumps/<run_id>.<SLURM_JOB_ID>/`（与 tfdpfl-logs 同级，在
cluster_env.sh 的 `--bind <仓库上一级>` 之内；`TFDPFL_DUMPDIR` 可覆盖）。
带作业号：同一个 run 被重复提交时两个作业不互相覆盖（D-070 的同一教训）。
每写一个文件打一行 `[Dump]`（manifest：相对路径、字节、sha），collect_metrics 收进
metrics.json 的 `dumps` —— git 里只放 manifest，大文件留集群（CLAUDE.md 协议）。
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

from utils.kvline import format_kv

MAX_SNAPSHOTS = 3


def parse_rounds(value) -> tuple:
    """`evaluation.snapshot_rounds` → 升序、不重复的正整数元组。None → ()。

    写成 "30/70"（不是 YAML 列表）：`[设定4]` 按 kvline 打印与解析，列表会被打成
    "[30, 70]" 再解析成字符串，往返不再逐字相同。YAML 里的单个数（`70`）也接受。
    """
    if value is None:
        return ()
    if isinstance(value, bool):
        raise ValueError(f"snapshot_rounds 不能是 bool：{value!r}")
    if isinstance(value, int):
        parts = [value]
    elif isinstance(value, str):
        try:
            parts = [int(p) for p in value.split("/")]
        except ValueError:
            raise ValueError(f"snapshot_rounds 要写成 \"30/70\" 这样的整数串，收到 {value!r}")
    else:
        raise ValueError(f"snapshot_rounds 要写成 \"30/70\" 这样的字符串，收到 {value!r}")
    if any(p < 1 for p in parts):
        raise ValueError(f"snapshot_rounds 的轮号从 1 起，收到 {value!r}")
    if list(parts) != sorted(set(parts)):
        raise ValueError(f"snapshot_rounds 必须升序且不重复，收到 {value!r}")
    return tuple(parts)


def dump_root() -> Path:
    env = os.environ.get("TFDPFL_DUMPDIR")
    if env:
        return Path(env)
    # fedavg/utils/dumps.py → parents[2] = 仓库根 → parents[3] = 仓库上一级
    return Path(__file__).resolve().parents[3] / "tfdpfl-dumps"


def run_dir_name(config) -> str:
    run_id = ((config or {}).get("meta") or {}).get("run_id") or "norun"
    job = os.environ.get("SLURM_JOB_ID") or "local"
    return f"{run_id}.{job}"


def write_npz(root: Path, rel: str, arrays: dict) -> dict:
    """写 `root/rel`（未压缩 npz），返回 {path, bytes, sha, write_s}。path 相对 root。"""
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": rel, "bytes": int(path.stat().st_size), "sha": h.hexdigest()[:12],
            "write_s": round(time.perf_counter() - t0, 2)}


def dump_line(round_idx: int, kind: str, info: dict) -> str:
    """`[Dump] Round N | kind=… | path=… | bytes=… | sha=… | write_s=…`"""
    return format_kv("[Dump]", {"kind": kind, "path": info["path"], "bytes": info["bytes"],
                                "sha": info["sha"], "write_s": float(info["write_s"])},
                     round_idx=round_idx, digits=2)


def estimate_snapshot_bytes(n_snapshots: int, n_models: int, n_params: int,
                            n_clients: int, private_per_client: int) -> int:
    """fp32 快照的粗估：(edge 模型 + 全局模型) × 参数 + 每端私有部分。只给 config_validate 报警用。"""
    return int(n_snapshots * 4 * (n_models * n_params + n_clients * private_per_client))
