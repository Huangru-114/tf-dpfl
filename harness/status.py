"""
harness/status.py  —  声明（登记表）对实际（metrics.json）逐格对账

    python3 harness/status.py experiments/attack/hfl-mechanism/registry.yaml
    python3 harness/status.py experiments/attack/hfl-propagation/registry.yaml --json out.json

每个登记的 run 落进且只落进一个状态：

    todo      还没有 metrics.json，而且门槛已满足
    blocked   还没有 metrics.json，门槛没满足（审计没关 / 功能会话没做）
    failed    有文件，但 exit_code 缺失或 ≠ 0
    mismatch  跑完了，但 run 块与声明不符 —— 例：声明 defense=median，实际 none（F-001）
    stale     跑完了、声明相符，但口径版本不对，或 config_sha 与当前声明不同（yaml 被改过）
    done      跑完了、相符、不过期

另外报 orphan：结果目录里有、登记表里没有的 metrics.json。

`done` 再细分 verified（有 [Provenance] 行，config_sha 核对过）与 unverified（老文件没有
溯源行，只核对了 run 块）。有 mismatch 时退出码为 1。

纯标准库 + PyYAML，本地与登录节点都能跑，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import Registry, read_index   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))
from utils.provenance import config_sha      # noqa: E402

STATUSES = ("todo", "blocked", "failed", "mismatch", "stale", "done")

ROOT = Path(__file__).resolve().parent.parent


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _same(a, b) -> bool:
    """数值宽松比较（0.1 与 0.10、5 与 5.0 视为相同），其余严格相等。"""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def run_block_mismatches(expected: dict, run_block: dict) -> list:
    """[[key, 声明, 实际], ...]。声明为 None 的键不核对（登记表没说的不算）。"""
    out = []
    for k, v in expected.items():
        if v is None:
            continue
        got = run_block.get(k)
        if not _same(v, got):
            out.append([k, v, got])
    return out


def classify(reg: Registry) -> dict:
    index = read_index(reg)
    rows = []
    seen_paths = set()
    for run in reg.runs():
        path = reg.metrics_path(run)
        seen_paths.add(path.resolve())
        row = {"run_id": run["run_id"], "group": run["group"], "cell": run["cell"],
               "seed": run["seed"], "path": _rel(path), "status": None,
               "verified": None, "detail": ""}
        if not path.exists():
            unmet = reg.unmet_requires(run["group"])
            row["status"] = "blocked" if unmet else "todo"
            row["detail"] = ("缺: " + ", ".join(unmet)) if unmet else ""
            rows.append(row)
            continue

        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as e:
            row["status"], row["detail"] = "failed", f"json 读不了: {e}"
            rows.append(row)
            continue

        rc = m.get("exit_code", "MISSING")
        if rc != 0:
            row["status"] = "failed"
            row["detail"] = "没有 exit_code" if rc == "MISSING" else f"exit_code={rc}"
            rows.append(row)
            continue

        run_block = m.get("run") or {}
        mism = run_block_mismatches(reg.expected_run_block(run), run_block)
        if mism:
            row["status"] = "mismatch"
            row["detail"] = "; ".join(f"{k}: 声明 {v!r} / 实际 {g!r}" for k, v, g in mism)
            rows.append(row)
            continue

        prov = run_block.get("provenance")
        if prov:
            stale = []
            if reg.protocol and prov.get("protocol") != reg.protocol:
                stale.append(f"口径版本 {prov.get('protocol')} ≠ 登记表 {reg.protocol}")
            want_sha = (index.get(run["run_id"]) or {}).get("config_sha")
            if want_sha is None and run.get("config"):
                want_sha = config_sha(reg.declared_config(run))
            if want_sha and prov.get("config_sha") != want_sha:
                stale.append(f"config_sha {prov.get('config_sha')} ≠ 当前声明 {want_sha}")
            if stale:
                row["status"], row["detail"] = "stale", "; ".join(stale)
                rows.append(row)
                continue
            row["verified"] = True
        else:
            row["verified"] = False
            row["detail"] = "无 [Provenance]（老文件）：只核对了 run 块"
        row["status"] = "done"
        rows.append(row)

    orphans = []
    if reg.results_dir.exists():
        pattern = "*.metrics.json" if reg.layout == "flat" else "**/*.metrics.json"
        for p in sorted(reg.results_dir.glob(pattern)):
            if p.resolve() not in seen_paths:
                orphans.append(_rel(p))

    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    counts["orphan"] = len(orphans)
    return {"registry": str(reg.path), "protocol": reg.protocol, "counts": counts,
            "runs": rows, "orphans": orphans}


def _print(report: dict, quiet: bool):
    if not quiet:
        for r in report["runs"]:
            v = {True: "✓", False: "·", None: " "}[r["verified"]]
            print(f"  {r['status']:<9}{v} {r['run_id']:<48} {r['detail']}")
        for o in report["orphans"]:
            print(f"  orphan     {o}")
    c = report["counts"]
    print("──────────────────────────────────────────────")
    print("  ".join(f"{k}={c[k]}" for k in (*STATUSES, "orphan")))
    print("（✓ = 有溯源、config_sha 核对过；· = 老文件，只核对了 run 块）")


def main(argv=None):
    ap = argparse.ArgumentParser(description="登记表 vs metrics.json 对账")
    ap.add_argument("registry")
    ap.add_argument("--json", help="把完整报告写到这个文件")
    ap.add_argument("--quiet", action="store_true", help="只打汇总")
    args = ap.parse_args(argv)
    report = classify(Registry(args.registry))
    _print(report, args.quiet)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return 1 if report["counts"]["mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
