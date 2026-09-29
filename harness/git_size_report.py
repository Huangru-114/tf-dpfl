"""
harness/git_size_report.py  —  git 仓库「谁占了空间」的只读统计（2026-09-29）

    python3 harness/git_size_report.py [--top 20] [--repo .]

按类别汇总**全部历史**里 blob 的磁盘占用（压缩后，`objectsize:disk`），标出路径现在是否还被跟踪。
只读：只调 `git rev-list --objects --all` / `git cat-file --batch-check` / `git ls-files` / `git count-objects`，
不改任何东西。结论与瘦身选项见 experiments/attack/hfl-mechanism/REPORT.md 的附录。

「垃圾」= CLAUDE.md 红线里永不该进 git 的东西（*.h5 / *.pt / *.ckpt / venv / wandb 运行记录）。
它们都已在 `.gitignore` 里、也早已从工作树删掉，但**仍留在历史里** —— 只能靠改写历史去掉。
纯标准库。
"""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict

JUNK = ("*.h5 checkpoints", "venv (lib/ bin/ include/)", "wandb runs", "arrays / pickles")


def category(path: str) -> str:
    if path.endswith((".h5", ".pt", ".ckpt", ".pth")):
        return "*.h5 checkpoints"
    top = path.split("/", 1)[0]
    if top in ("lib", "lib64", "bin", "include", "share") or "site-packages/" in path \
            or path.endswith("pyvenv.cfg"):
        return "venv (lib/ bin/ include/)"
    if "wandb/" in path:
        return "wandb runs"
    if path.endswith((".npy", ".npz", ".pkl")):
        return "arrays / pickles"
    if path.endswith(".metrics.json"):
        return "metrics.json"
    if path.endswith((".log", ".out", ".err", ".txt")):
        return "logs / txt"
    if path.endswith((".png", ".pdf", ".jpg", ".svg")):
        return "figures"
    return "code / config / docs / other"


def _git(repo, *args, stdin=None) -> str:
    return subprocess.run(["git", "-C", repo, *args], input=stdin, capture_output=True,
                          text=True, check=True).stdout


def blobs(repo: str) -> list:
    """[(path, raw_bytes, disk_bytes)]，全部历史里可达的 blob（同一 blob 多路径时取 rev-list 给的第一个）。"""
    objs = _git(repo, "rev-list", "--objects", "--all")
    out = _git(repo, "cat-file",
               "--batch-check=%(objecttype) %(objectname) %(objectsize) %(objectsize:disk) %(rest)",
               stdin=objs)
    rows = []
    for ln in out.splitlines():
        parts = ln.split(" ", 4)
        if len(parts) >= 4 and parts[0] == "blob":
            rows.append((parts[4] if len(parts) > 4 else "", int(parts[2]), int(parts[3])))
    return rows


def summarize(rows, tracked: set) -> dict:
    agg = defaultdict(lambda: {"disk": 0, "raw": 0, "blobs": 0, "tracked_blobs": 0, "tracked_disk": 0})
    for path, raw, disk in rows:
        a = agg[category(path)]
        a["disk"] += disk
        a["raw"] += raw
        a["blobs"] += 1
        if path in tracked:
            a["tracked_blobs"] += 1
            a["tracked_disk"] += disk
    total = sum(a["disk"] for a in agg.values())
    junk = sum(a["disk"] for c, a in agg.items() if c in JUNK)
    return {"categories": dict(agg), "total_disk": total, "junk_disk": junk}


def main(argv=None):
    ap = argparse.ArgumentParser(description="git 历史的空间占用（只读）")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    print(_git(a.repo, "count-objects", "-vH").strip())
    rows = blobs(a.repo)
    tracked = set(_git(a.repo, "ls-files").splitlines())
    s = summarize(rows, tracked)
    mib = 2 ** 20
    print(f"\n历史 blob 共 {len(rows)} 个，磁盘 {s['total_disk'] / mib:.1f} MiB；"
          f"其中垃圾类 {s['junk_disk'] / mib:.1f} MiB（{100 * s['junk_disk'] / max(s['total_disk'], 1):.1f}%）\n")
    print(f"{'类别':28s} {'磁盘 MiB':>9s} {'占比':>6s} {'blob 数':>7s}  {'路径仍被跟踪':>12s}")
    for c, v in sorted(s["categories"].items(), key=lambda kv: -kv[1]["disk"]):
        print(f"{c:28s} {v['disk'] / mib:9.1f} {100 * v['disk'] / max(s['total_disk'], 1):5.1f}% "
              f"{v['blobs']:7d}  {v['tracked_blobs']:5d}（{v['tracked_disk'] / mib:.1f} MiB）")
    print(f"\n最大的 {a.top} 个 blob（磁盘）：")
    for path, raw, disk in sorted(rows, key=lambda r: -r[2])[:a.top]:
        flag = "跟踪中" if path in tracked else "已删除"
        print(f"  {disk / mib:7.1f} MiB（原始 {raw / mib:6.1f}） {flag}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
