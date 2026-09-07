#!/usr/bin/env python3
"""Stage B 标定读数 —— 把 6 个 metrics.json 压成一张「等算力下谁最快到平台」的表。

回应导师意见 #5。**判据是可计算的，不是看图**：

  平台（plateau）
      令 `final` = 最后 3 个评估点的均值。平台轮 = 最小的评估轮 r，使得从 r 起
      **每一个**后续点都落在 `final ± tol` 内。找不到这样的 r（含只有 ≤3 个点）
      → 报 `None`，**不猜**。「到 150 有效轮还没平」本身就是结论。

  到平台的 GPU-秒
      = sum(round_time)（≤ 平台轮的所有轮）+ sum(bd_eval total_s)（≤ 平台轮的评估轮）。
      两段口径不同、必须分开累加：`round_time` 每轮都有且**不含**后门评估，
      `[Timing]` 只在评估轮有。见 harness/collect_metrics.py 的 RE_ROUND_TIME。
      这是**实测求和**，不是「均值 × 轮数」的估算。

纯 stdlib，本地和集群都能跑，不需要 GPU / numpy / matplotlib。

    python3 experiments/calibration/read_calibration.py
    python3 experiments/calibration/read_calibration.py --tol 0.005 --results-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

CELL_RE = re.compile(r"local_epochs(\d+)_seed(\d+)\.metrics\.json$")


def plateau_round(points: List[Dict[str, Any]], key: str, tol: float,
                  tail: int = 3) -> Optional[int]:
    """最早的「此后再没离开 final ± tol」的评估轮；判不出来就 None。

    `points` 是 [{round, <key>}, ...]，`None` 值直接跳过（无定义的指标留空，
    绝不当 0 参与统计）。
    """
    usable = [(p["round"], p[key]) for p in points
              if p.get("round") is not None and p.get(key) is not None]
    if len(usable) <= tail:
        return None
    usable.sort()
    final = sum(v for _, v in usable[-tail:]) / tail
    for i, (r, _) in enumerate(usable):
        if all(abs(v - final) <= tol for _, v in usable[i:]):
            return r
    return None


def cost_up_to(metrics: Dict[str, Any], upto_round: Optional[int]) -> Dict[str, Optional[float]]:
    """≤ upto_round 的实测墙钟拆分。upto_round 为 None → 全程（并注明未到平台）。"""
    acc = metrics.get("acc_rounds") or []
    tim = metrics.get("timing_rounds") or []
    def _in(r):
        return r is not None and (upto_round is None or r <= upto_round)
    train = [a["round_time"] for a in acc
             if _in(a.get("round")) and a.get("round_time") is not None]
    ev = [t["total_s"] for t in tim
          if _in(t.get("round")) and t.get("total_s") is not None]
    return {
        "train_s": round(sum(train), 1) if train else None,
        "eval_s": round(sum(ev), 1) if ev else None,
        "total_s": round(sum(train) + sum(ev), 1) if (train and ev) else None,
        "n_rounds": len(train) or None,
    }


def load_cells(results_dir: Path) -> List[Dict[str, Any]]:
    cells = []
    for path in sorted(results_dir.glob("local_epochs*_seed*.metrics.json")):
        m = CELL_RE.search(path.name)
        if not m:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        cells.append({"path": path, "local_epochs": int(m.group(1)),
                      "seed": int(m.group(2)), "metrics": data})
    return cells


def _fmt(v, spec="{:.1f}"):
    return "n/a" if v is None else spec.format(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--tol", type=float, default=0.01,
                    help="平台带宽（绝对值）。默认 0.01 = 1 个百分点")
    ap.add_argument("--json", action="store_true", help="额外输出机读 JSON")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    cells = load_cells(results_dir)
    if not cells:
        print(f"[calib] {results_dir} 下没有 local_epochs*_seed*.metrics.json —— "
              f"先跑 bash experiments/calibration/run_calibration.sh")
        return 1

    print(f"[calib] tol = ±{args.tol}（平台带宽，绝对值）；"
          f"plateau 定义见本文件 docstring")
    print(f"[calib] {len(cells)} 个格子，来自 {results_dir}\n")

    rows = []
    for c in cells:
        m = c["metrics"]
        ep, seed = c["local_epochs"], c["seed"]
        rc = m.get("exit_code")
        acc = m.get("acc_rounds") or []
        asr = m.get("rounds") or []
        ts = m.get("timing_summary") or {}

        p_mta = plateau_round(acc, "pm_acc", args.tol)
        p_asr = plateau_round(asr, "local_benign_asr", args.tol)
        # 「跑完这个 config 要多久」按两个平台里**晚**的那个算 —— 两个指标都稳了
        # 才算这个预算够用。任一判不出来 → 整格判为「未到平台」，代价按全程报。
        both = None if (p_mta is None or p_asr is None) else max(p_mta, p_asr)
        cost = cost_up_to(m, both)
        full = cost_up_to(m, None)

        rows.append({
            "local_epochs": ep, "seed": seed, "exit_code": rc,
            "n_eval_points": len(asr),
            "plateau_round_mta": p_mta, "plateau_round_asr": p_asr,
            "plateau_round": both,
            "plateau_effective_round": (
                None if both is None
                else both * int((m.get("run") or {}).get("edge_rounds") or 1)),
            "cost_to_plateau": cost, "cost_full_run": full,
            "round_time_mean_s": ts.get("round_time_mean_s"),
            "bd_eval_mean_s": ts.get("bd_eval_mean_s"),
            "bd_eval_fraction": ts.get("bd_eval_fraction"),
            "final_pm_acc": (m.get("final_acc") or {}).get("pm_acc"),
            "final_benign_asr": (m.get("final") or {}).get("local_benign_asr"),
        })

    hdr = (f"{'ep':>3} {'seed':>5} {'rc':>3} {'plat_MTA':>9} {'plat_ASR':>9} "
           f"{'plat_eff':>9} {'GPU-s→plat':>11} {'train_s':>9} {'eval_s':>8} "
           f"{'eval%':>6} {'final_MTA':>10} {'final_ASR':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda d: (d["local_epochs"], d["seed"])):
        frac = r["bd_eval_fraction"]
        print(f"{r['local_epochs']:>3} {r['seed']:>5} "
              f"{'n/a' if r['exit_code'] is None else r['exit_code']:>3} "
              f"{_fmt(r['plateau_round_mta'], '{:.0f}'):>9} "
              f"{_fmt(r['plateau_round_asr'], '{:.0f}'):>9} "
              f"{_fmt(r['plateau_effective_round'], '{:.0f}'):>9} "
              f"{_fmt(r['cost_to_plateau']['total_s']):>11} "
              f"{_fmt(r['cost_to_plateau']['train_s']):>9} "
              f"{_fmt(r['cost_to_plateau']['eval_s']):>8} "
              f"{('n/a' if frac is None else f'{100 * frac:.0f}%'):>6} "
              f"{_fmt(r['final_pm_acc'], '{:.4f}'):>10} "
              f"{_fmt(r['final_benign_asr'], '{:.4f}'):>10}")

    print("\n读法：")
    print("  plat_eff   = 到平台的**有效轮**（cloud round × edge_rounds），跨 local_epochs 可比")
    print("  GPU-s→plat = 到平台的实测墙钟（train + 后门评估），**这一列才是选预算的依据**")
    print("  eval%      = 后门评估占全程墙钟的比例；它高就说明降轮数比提 epoch 更划算")
    print("  n/a 表示该量在本格**无定义或判不出来**（如到 150 有效轮还没平），不是 0")

    if args.json:
        print("\n" + json.dumps(rows, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
