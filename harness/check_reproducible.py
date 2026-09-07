"""比对两个（或多个）metrics.json，判定同 seed 的重跑是否逐位一致。

这是 `main.set_seed` 补播 Python `random` 之后的**验收标准**，
没法写成单元测试 —— 它要的是两次真实训练的产物。

背景：`set_seed` 曾只播 `np.random` 与 `tf.random`，而五个方法客户端
（hier_fedrep / client_pfedme / hier_ditto / hier_ditto_rep / hier_pfedme_rep）
每个 epoch 都用内置 `random.shuffle(eb)` 打乱 batch。于是 Rep/Ditto/pFedMe 全家
**在固定 seed 下都不可复现**，也就是 Experiment 3 的每一个格子。
这会伪装成「种子方差大」：3C 轴上 seed42≈0.80 / seed43≈0.58 的落差里，
有多少是真种子方差、多少是这个未播种的洗牌，修好之前无法分离。

用法::

    # 同一个 config、同一个 seed，跑两次，落到不同的 exp_id
    sbatch run_full.sh attack hfl-propagation badpfl none repro_a hier_fedavg_fedrep \\
           experiments/attack/hfl-propagation/3c_R5.yaml
    sbatch run_full.sh attack hfl-propagation badpfl none repro_b hier_fedavg_fedrep \\
           experiments/attack/hfl-propagation/3c_R5.yaml

    python3 harness/check_reproducible.py .../repro_a.metrics.json .../repro_b.metrics.json

退出码 0 = 一致（可复现），1 = 有分歧（列出第一处）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List, Tuple

# 这些字段本来就该逐次不同，不参与比对
IGNORED_TOP = {"log_tail", "errors", "exit_code"}
IGNORED_RUN = {"config_path", "run_name"}


def _flatten(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    """把嵌套结构压成 [(路径, 标量)]，只保留数值与 None。"""
    out: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k in sorted(obj):
            if prefix == "" and k in IGNORED_TOP:
                continue
            if prefix == "run" and k in IGNORED_RUN:
                continue
            out += _flatten(obj[k], f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _flatten(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float, bool)) or obj is None:
        out.append((prefix, obj))
    return out


def compare(paths: List[Path]) -> int:
    if len(paths) < 2:
        print("需要至少两个 metrics.json")
        return 2
    flats = []
    for p in paths:
        if not p.is_file():
            print(f"❌ 找不到 {p}")
            return 2
        flats.append(dict(_flatten(json.loads(p.read_text(encoding="utf-8")))))

    ref_path, ref = paths[0], flats[0]
    worst = 0
    for path, cur in zip(paths[1:], flats[1:]):
        only_ref = sorted(set(ref) - set(cur))
        only_cur = sorted(set(cur) - set(ref))
        diffs = [(k, ref[k], cur[k]) for k in sorted(set(ref) & set(cur))
                 if ref[k] != cur[k]]

        if not (only_ref or only_cur or diffs):
            print(f"✅ {path.name} 与 {ref_path.name} 逐位一致"
                  f"（比对了 {len(ref)} 个数值字段）")
            continue

        worst = 1
        print(f"❌ {path.name} 与 {ref_path.name} 不一致")
        if only_ref or only_cur:
            print(f"   字段集合不同：仅在 {ref_path.name} 的 {len(only_ref)} 个"
                  f"、仅在 {path.name} 的 {len(only_cur)} 个")
            for k in (only_ref[:3] + only_cur[:3]):
                print(f"     - {k}")
        if diffs:
            print(f"   {len(diffs)} 个数值不同，前几处：")
            for k, a, b in diffs[:8]:
                print(f"     {k}: {a!r}  vs  {b!r}")
            # 最早出现分歧的轮次最能说明问题
            rounds = [k for k, _, _ in diffs if k.startswith("rounds[")]
            if rounds:
                print(f"   最早分歧出现在：{rounds[0]}")
                print("   → 若第一轮就分歧，随机性从 setup 期就没被控住；"
                      "若靠后才分歧，多半是训练循环里的某个未播种 RNG。")
    return worst


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    return compare([Path(a) for a in argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
