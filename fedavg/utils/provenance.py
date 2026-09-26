"""
utils/provenance.py  —  每个 run 打一行 `[Provenance]`，说清楚「这份数字是哪份代码、哪份配置跑的」

为什么要有这一行（experiments/attack/hfl-mechanism/FINDINGS.md F-001 / F-011）：

  · 旧 metrics.json 里没有 git commit、没有 config hash、没有 run_id，一个格子的身份
    只靠文件名。6 个 `def_*` 格子文件名写着 median / multi_krum，实际跑的是 none
    （`exp3_cell.sbatch` 写死 `--defense none` 覆盖了 yaml），json 里看不出来。
  · 所以这里把「声明」（yaml 原文的 hash）和「实际」（CLI 改掉了哪些值）**分开**记：
      config_sha    = yaml 原文解析出的 dict 的 hash —— 登记表 materialize 时算的是同一个数，
                      `harness/status.py` 拿它判断一格是不是按当前配置跑的
      cli_overrides = CLI（--seed/--defense/--attack_method/--framework/--override …）
                      相对 yaml 改动过的叶子值 [key, old, new]。**只记录，不拦**。

口径版本 PROTOCOL_VERSION（**不是训练 epoch**）：
  P0  探针修正之前的归档（hfl-propagation/results/archive-pre-fix/）
  P1  统一标准后的 seed42 批次（d8c8d0b）以及本常量生效后、对齐审计完成前的全部 run
  P2  AUDIT.md 全部关闭、代码按审计改完之后（2026-09-27 A4 收口：pilot `2853433` 关掉最后 4 行后改成 "P2"）
分析工具（harness/runs_table.py）拒绝把不同版本的 run 混在一起。

纯标准库，**不 import TF** —— tests/test_provenance.py 本地秒级。
`git_commit` 直接读 .git 下的文件，不依赖 git 可执行文件（容器里可能没有）。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

PROTOCOL_VERSION = "P2"

PROVENANCE_TAG = "[Provenance]"

# `[Provenance]` 行里 cli_overrides 必须是最后一个字段：它的 JSON 值里可能出现 " | "。
_FIELD_ORDER = ("protocol", "git", "branch", "dirty", "config_sha",
                "study", "group", "run_id", "host", "job", "start")


# ══════════════════════════════════════════════════════════════════════════
# 配置 hash 与 CLI 覆盖
# ══════════════════════════════════════════════════════════════════════════

def config_sha(config: dict, n: int = 12) -> str:
    """dict 的稳定 hash：键排序 + 紧凑 JSON + sha256，取前 n 位。

    登记表（harness/registry.py）在 materialize 时对**同一份** dict 调这个函数，
    所以「yaml 文件 → safe_load → config_sha」两边必然一致。
    """
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v:
                out.update(_flatten(v, key))
            else:
                out[key] = v
    else:
        out[prefix] = d
    return out


_MISSING = "<missing>"


def flat_diff(before: dict, after: dict) -> list:
    """两份嵌套 dict 的叶子差异 → [[key, old, new], ...]，按 key 排序。

    只在一侧存在的叶子，另一侧记成 "<missing>"。值原样保留（不转字符串），
    写进日志时再统一 JSON 编码。
    """
    fb, fa = _flatten(before), _flatten(after)
    diffs = []
    for key in sorted(set(fb) | set(fa)):
        old = fb.get(key, _MISSING)
        new = fa.get(key, _MISSING)
        if old != new:
            diffs.append([key, old, new])
    return diffs


# ══════════════════════════════════════════════════════════════════════════
# git（只读 .git 文件，不调 git 命令也能拿到 commit）
# ══════════════════════════════════════════════════════════════════════════

def _resolve_git_dir(root: Path):
    dot = root / ".git"
    if dot.is_dir():
        return dot
    if dot.is_file():                              # worktree / submodule: "gitdir: <path>"
        text = dot.read_text().strip()
        if text.startswith("gitdir:"):
            p = Path(text.split(":", 1)[1].strip())
            return p if p.is_absolute() else (root / p)
    return None


def git_commit(root) -> tuple:
    """返回 (sha 或 None, branch 或 None)。detached HEAD 时 branch=None。

    查找顺序：HEAD → 松散 ref 文件 → packed-refs。worktree 的 HEAD 在自己的 gitdir 里，
    ref 可能在 commondir 里，两处都找。
    """
    root = Path(root)
    gdir = _resolve_git_dir(root)
    if gdir is None or not (gdir / "HEAD").exists():
        return None, None
    head = (gdir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return (head or None), None
    ref = head.split(":", 1)[1].strip()
    branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref

    search = [gdir]
    common = gdir / "commondir"
    if common.exists():
        c = Path(common.read_text().strip())
        search.append(c if c.is_absolute() else (gdir / c))
    for g in search:
        loose = g / ref
        if loose.exists():
            sha = loose.read_text().strip()
            if sha:
                return sha, branch
    for g in search:
        packed = g / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                parts = line.strip().split(" ", 1)
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0], branch
    return None, branch


def git_dirty(root) -> str:
    """"0" / "1" / "unknown"。只看已跟踪文件（未跟踪的 scratch 不算）。
    需要 git 可执行文件；容器里没有就是 unknown，不猜。"""
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if r.returncode != 0:
        return "unknown"
    return "1" if r.stdout.strip() else "0"


# ══════════════════════════════════════════════════════════════════════════
# 打印与解析（两侧分开测：tests/test_provenance.py）
# ══════════════════════════════════════════════════════════════════════════

def _clean(v) -> str:
    """字段值里不允许出现分隔符与换行。None → "n/a"。"""
    if v is None or v == "":
        return "n/a"
    s = str(v).replace("|", "/").replace("\n", " ").strip()
    return s or "n/a"


def provenance_fields(config: dict, *, declared_sha: str, cli_overrides: list,
                      repo_root=None, now=None) -> dict:
    """收集 `[Provenance]` 行的全部字段（不打印）。"""
    meta = (config or {}).get("meta") or {}
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    sha, branch = git_commit(root)
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return {
        "protocol":   PROTOCOL_VERSION,
        "git":        sha[:12] if sha else None,
        "branch":     branch,
        "dirty":      git_dirty(root),
        "config_sha": declared_sha,
        "study":      meta.get("study"),
        "group":      meta.get("group"),
        "run_id":     meta.get("run_id"),
        "host":       socket.gethostname(),
        "job":        os.environ.get("SLURM_JOB_ID"),
        "start":      now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cli_overrides": cli_overrides or [],
    }


def format_provenance(fields: dict) -> str:
    parts = [f"{k}={_clean(fields.get(k))}" for k in _FIELD_ORDER]
    ov = json.dumps(fields.get("cli_overrides") or [], separators=(",", ":"),
                    ensure_ascii=False, default=str)
    parts.append(f"cli_overrides={ov}")
    return f"{PROVENANCE_TAG} " + " | ".join(parts)


def print_provenance(config: dict, *, declared_sha: str, cli_overrides: list,
                     repo_root=None) -> dict:
    """main.load_config 调这一个函数：先逐条打印 CLI 覆盖，再打印 `[Provenance]` 行。"""
    for key, old, new in cli_overrides or []:
        print(f"[Config] CLI 覆盖 yaml: {key} {old!r} → {new!r}")
    fields = provenance_fields(config, declared_sha=declared_sha,
                               cli_overrides=cli_overrides, repo_root=repo_root)
    print(format_provenance(fields))
    return fields


def parse_provenance(line: str):
    """`[Provenance]` 行 → dict；不是这一行 → None。

    与 `[设定]` 的全或无正则不同，这里按 `key=value` 逐字段解析：
    以后加字段不会让旧字段一起变 None；缺的字段就是 None。
    """
    line = line.strip()
    if not line.startswith(PROVENANCE_TAG):
        return None
    body = line[len(PROVENANCE_TAG):].strip()
    overrides = []
    marker = "cli_overrides="
    idx = body.find(marker)
    if idx >= 0:
        raw = body[idx + len(marker):].strip()
        body = body[:idx].rstrip(" |")
        try:
            overrides = json.loads(raw)
        except ValueError:
            overrides = None                     # 格式坏了：None ≠ []（[] 是「没有覆盖」）
    out = {k: None for k in _FIELD_ORDER}
    for part in body.split(" | "):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        out[k] = None if v == "n/a" else v
    out["cli_overrides"] = overrides
    return out
