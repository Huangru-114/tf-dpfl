"""
utils/kvline.py  —  `[标签] k=v | k=v | …` 自描述行的打印与解析（同源）

A4 之后新加的自描述行（`[设定4]` `[设定5]` `[ASR4]` `[ASRwb]` `[Stale]` `[Checksum]`）
一律走这里。理由同 `[Provenance]`（utils/provenance.py:parse_provenance）：

  · 旧的 `[设定]` / `[Backdoor]` 行是**全或无**的正则 —— 加一个字段、格式一变，
    原有字段会一起变 None，而日志毫无异常（CLAUDE.md `[设定2]` 那一段）。
  · 按 `k=v` 逐字段解析：以后加字段不会连累旧字段；缺的字段就是 None。

值的约定（打印与解析互逆）：
  None ↔ `n/a`（**绝不打成 0**：无定义与「值为零」不能长得一样，陷阱 #13）
  bool ↔ `true` / `false`
  float → 4 位小数（`digits` 可改）；int / str 原样
形如 `Round 12` 的段解析成 `round=12`；形如 `edge3` 的段解析成 `edge_id=3`。

纯标准库，不 import TF —— collect_metrics 与本地 L1 都能直接用。
"""

from __future__ import annotations

import re

_ROUND = re.compile(r"^Round\s+(\d+)$")
_EDGE = re.compile(r"^edge(\d+)$")
_INT = re.compile(r"^-?\d+$")
_FLOAT = re.compile(r"^-?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$")


def fmt_value(v, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def format_kv(tag: str, fields: dict, *, round_idx=None, edge_id=None,
              digits: int = 4) -> str:
    """`[tag] Round N | edgeK | k=v | …`。round / edge 段可选，按此顺序在最前。"""
    parts = []
    if round_idx is not None:
        parts.append(f"Round {int(round_idx)}")
    if edge_id is not None:
        parts.append(f"edge{int(edge_id)}")
    for k, v in fields.items():
        s = fmt_value(v, digits)
        if "|" in s or "=" in str(k):
            raise ValueError(f"{tag}: 字段 {k!r} 的值 {s!r} 含分隔符，解析会错位")
        parts.append(f"{k}={s}")
    return f"{tag} " + " | ".join(parts)


def parse_value(s: str):
    s = s.strip()
    if s == "n/a":
        return None
    if s == "true":
        return True
    if s == "false":
        return False
    if _INT.match(s):
        return int(s)
    if _FLOAT.match(s):
        return float(s)
    return s


def parse_kv(line: str, tag: str):
    """这一行是 `tag` 行 → dict；否则 None。行里 tag 之前可以有别的前缀（如时间戳）。"""
    idx = line.find(tag + " ")
    if idx < 0:
        return None
    body = line[idx + len(tag):].strip()
    out = {}
    for part in body.split(" | "):
        part = part.strip()
        if not part:
            continue
        m = _ROUND.match(part)
        if m:
            out["round"] = int(m.group(1))
            continue
        m = _EDGE.match(part)
        if m:
            out["edge_id"] = int(m.group(1))
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = parse_value(v)
    return out


def collect_kv(lines, tag: str) -> list:
    """全部 `tag` 行，按出现顺序。"""
    out = []
    for ln in lines:
        d = parse_kv(ln, tag)
        if d is not None:
            out.append(d)
    return out
