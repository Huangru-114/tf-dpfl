"""
harness/runs_table.py  —  把一堆 metrics.json 压成两张「整洁表」，所有分析与画图都从这两张表出发

    python3 harness/runs_table.py <结果目录或文件>... --out <目录>
        → <目录>/runs.csv    每个 run 一行：实际因素 / 溯源 / 末 10 点均值 / T_θ / 机时 / replicate 编号
        → <目录>/series.csv  长表：run, r_eff, cloud_round, edge_id, metric, value

为什么是这两张表（FINDINGS F-010）：
  · 旧的 plot_exp3.py 按**文件名**认格子（ORDER_3AB、^3c_R、FLAT_CELL 硬编码），
    新运行组一来就得改代码；analyze_exp3.py 的终值取**单个末轮**，而标定规定用
    末 10 点均值（逐点 σ≈0.09 与效应同量级）。
  · 这里按 run 块里的**实际**因素分组（不看文件名），终值一律用末 10 点均值。
    同一因素键 + 同一 seed 出现多份时编 replicate 号 —— 例：def_median_flat_baseline
    实际没开防御，它就作为 flat_baseline 的第 2 份重复出现（F-002 的噪声估计就是这么来的）。

口径版本（PROTOCOL，不是训练 epoch）：取自 run.provenance.protocol；老文件没有这一行，
用 --legacy-protocol 指定（旧方案的 seed42 批次是 P1）。表里出现两种版本就拒绝输出，
除非显式 --allow-mixed —— P1 和 P2 的数字不能放进同一张结论表（AUDIT.md）。

纯标准库，本地与容器里都能跑，不 import TF / numpy。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp3 import THETAS, effective_round_series, first_crossing, hhi   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

LAST_K = 10                       # 终值 = 末 LAST_K 个评估点的均值（标定 RESULTS.md §4.2）

ASR_METRICS = ("global_asr", "edge_asr", "local_benign_asr", "local_malicious_asr",
               "same_edge_asr", "diff_edge_asr")
ACC_METRICS = ("gm_acc", "em_acc", "pm_acc")
T_THETA_METRICS = ("global_asr", "edge_asr", "local_benign_asr")

# 实际因素 = 决定「这是哪一格」的 run 块字段（seed 单独一列，不进因素键）。
FACTOR_KEYS = ("method", "attack", "defense", "n_clients", "n_edges", "edge_rounds",
               "client_fraction", "poison_ratio", "malicious_per_edge",
               "malicious_placement", "edge_assignment", "local_epochs", "plocal_epochs",
               "attack_stop_round")
PROV_KEYS = ("protocol", "git", "config_sha", "study", "group", "run_id")

UNKNOWN_PROTOCOL = "unknown"


class MixedProtocolError(ValueError):
    pass


# ══════════════════════════════════════════════════════════════════════════
# 纯计算
# ══════════════════════════════════════════════════════════════════════════

def last_k_mean(values, k: int = LAST_K):
    """末 k 个**有定义**的点的均值与实际用到的点数；一个都没有 → (None, 0)。"""
    vals = [v for v in values if v is not None][-k:]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def factor_key(run_block: dict) -> tuple:
    out = []
    for k in FACTOR_KEYS:
        v = run_block.get(k)
        out.append(json.dumps(v) if isinstance(v, list) else v)
    return tuple(out)


def _fmt_key(key: tuple) -> str:
    return " | ".join(f"{k}={v}" for k, v in zip(FACTOR_KEYS, key) if v is not None)


def summarize_run(m: dict, *, name: str, source: str, legacy_protocol=None) -> dict:
    """一个 metrics.json → runs.csv 的一行。"""
    run = m.get("run") or {}
    prov = run.get("provenance") or {}
    er = run.get("edge_rounds") or 1
    row = {"run": prov.get("run_id") or name, "source": source}
    for k in PROV_KEYS:
        row[k] = prov.get(k)
    row["protocol"] = prov.get("protocol") or legacy_protocol or UNKNOWN_PROTOCOL
    row["seed"] = run.get("seed")
    for k in FACTOR_KEYS:
        v = run.get(k)
        row[k] = json.dumps(v) if isinstance(v, list) else v
    row["hhi"] = hhi(run.get("malicious_per_edge") or [])
    row["factor_key"] = _fmt_key(factor_key(run))
    row["exit_code"] = m.get("exit_code")
    row["cli_overrides"] = (None if run.get("cli_overrides") is None
                            else json.dumps(run["cli_overrides"], ensure_ascii=False))
    row["stop_reason"] = run.get("stop_reason")
    row["stopped_at_effective"] = run.get("stopped_at_effective")

    rounds = m.get("rounds") or []
    for mk in ASR_METRICS:
        mean, n = last_k_mean([r.get(mk) for r in rounds])
        row[f"{mk}_last{LAST_K}"] = mean
        row[f"{mk}_n"] = n
    acc = m.get("acc_rounds") or []
    for mk in ACC_METRICS:
        mean, n = last_k_mean([r.get(mk) for r in acc])
        row[f"{mk}_last{LAST_K}"] = mean
        row[f"{mk}_n"] = n

    for mk in T_THETA_METRICS:
        series = effective_round_series(rounds, er, mk)
        for th in THETAS:
            c = first_crossing(series, th)
            row[f"t{th}_{mk}"] = c.t_theta
            row[f"t{th}_{mk}_state"] = ("crossed" if c.crossed else
                                        "left_censored" if c.left_censored else
                                        c.reason or "censored")

    ts = m.get("timing_summary") or {}
    wall = ts.get("wall_total_s")
    row["gpu_h"] = (wall / 3600.0) if wall else None
    return row


def series_rows(m: dict, run_name: str) -> list:
    """一个 metrics.json → series.csv 的若干行（edge_id=-1 表示全局 / 跨 edge 聚合量）。"""
    run = m.get("run") or {}
    er = run.get("edge_rounds") or 1
    out = []
    for r in m.get("rounds") or []:
        for mk in ASR_METRICS:
            if r.get(mk) is not None:
                out.append((run_name, r["round"] * er, r["round"], -1, mk, r[mk]))
    for r in m.get("acc_rounds") or []:
        for mk in ACC_METRICS:
            if r.get(mk) is not None:
                out.append((run_name, r["round"] * er, r["round"], -1, mk, r[mk]))
    for rnd, edges in (m.get("per_edge_rounds") or {}).items():
        for e in edges:
            for mk in ("edge_asr", "client_benign", "client_malicious"):
                if e.get(mk) is not None:
                    out.append((run_name, int(rnd) * er, int(rnd), e["edge_id"],
                                f"edge.{mk}", e[mk]))
    for rnd, edges in (m.get("per_edge_acc_rounds") or {}).items():
        for e in edges:
            for mk in ("em_acc", "pm_acc"):
                if e.get(mk) is not None:
                    out.append((run_name, int(rnd) * er, int(rnd), e["edge_id"],
                                f"edge.{mk}", e[mk]))
    out.sort(key=lambda t: (t[4], t[3], t[1]))
    return out


def assign_replicates(rows: list) -> None:
    """同一 (因素键, seed) 的多份结果按 source 排序编号 1, 2, …；并写 n_replicates。"""
    groups = {}
    for r in rows:
        groups.setdefault((r["factor_key"], r["seed"]), []).append(r)
    for rs in groups.values():
        rs.sort(key=lambda r: r["source"])
        for i, r in enumerate(rs, 1):
            r["replicate"] = i
            r["n_replicates"] = len(rs)


# ══════════════════════════════════════════════════════════════════════════
# IO
# ══════════════════════════════════════════════════════════════════════════

def find_metrics(paths) -> list:
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.metrics.json")))
        elif p.name.endswith(".metrics.json"):
            files.append(p)
    return sorted(set(files))


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def build(paths, *, legacy_protocol=None, allow_mixed=False, exclude_archive=True):
    """→ (runs_rows, series_rows)。默认跳过 archive* 目录（P0 归档）。"""
    runs, series = [], []
    for f in find_metrics(paths):
        if exclude_archive and any(part.startswith("archive") for part in f.parts):
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        name = f.name[:-len(".metrics.json")]
        row = summarize_run(m, name=name, source=_rel(f), legacy_protocol=legacy_protocol)
        runs.append(row)
        series.extend(series_rows(m, row["run"]))
    protocols = sorted({r["protocol"] for r in runs})
    if len(protocols) > 1 and not allow_mixed:
        raise MixedProtocolError(
            f"表里混着多个口径版本 {protocols}：P1 与 P2 的数字不能放进同一张表。"
            "分开跑，或确实要比较时加 --allow-mixed")
    assign_replicates(runs)
    return runs, series


RUN_COLUMNS_HEAD = ("run", "source", "protocol", "study", "group", "run_id", "git",
                    "config_sha", "seed", "replicate", "n_replicates", "factor_key",
                    *FACTOR_KEYS, "hhi", "exit_code", "cli_overrides",
                    "stop_reason", "stopped_at_effective", "gpu_h")
SERIES_COLUMNS = ("run", "r_eff", "cloud_round", "edge_id", "metric", "value")


def run_columns(rows) -> list:
    rest = []
    for r in rows:
        for k in r:
            if k not in RUN_COLUMNS_HEAD and k not in rest:
                rest.append(k)
    return [*RUN_COLUMNS_HEAD, *rest]


def write_tables(runs, series, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = run_columns(runs)
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(runs, key=lambda r: (r["factor_key"], str(r["seed"]), r["replicate"])):
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    with (out_dir / "series.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SERIES_COLUMNS)
        w.writerows(series)


def main(argv=None):
    ap = argparse.ArgumentParser(description="metrics.json → runs.csv + series.csv")
    ap.add_argument("paths", nargs="+", help="结果目录（递归）或单个 metrics.json")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--legacy-protocol", help="没有 [Provenance] 行的老文件算哪个口径版本（如 P1）")
    ap.add_argument("--allow-mixed", action="store_true", help="允许混合口径版本（只用于对照）")
    ap.add_argument("--include-archive", action="store_true", help="连 archive* 目录一起读")
    args = ap.parse_args(argv)
    runs, series = build(args.paths, legacy_protocol=args.legacy_protocol,
                         allow_mixed=args.allow_mixed,
                         exclude_archive=not args.include_archive)
    write_tables(runs, series, Path(args.out))
    reps = sum(1 for r in runs if r["n_replicates"] > 1 and r["replicate"] == 1)
    print(f"[runs_table] {len(runs)} runs / {len(series)} series 行 → {args.out}")
    print(f"[runs_table] 口径版本: {sorted({r['protocol'] for r in runs})} · "
          f"有重复的 (因素键, seed): {reps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
