"""
harness/g7_posthoc.py  —  G7（预处理对比，D-025）的**事后**判定（D-049）

    python3 harness/g7_posthoc.py [--json out.json]

⚠ **这不是预注册判定**：D-025 只写了「检验官方设定降低了攻击难度」，没有量化规则；
规则是 2026-09-27 用户**看过 G7 数据之后**定的（D-049），结论只能按事后判据引用。

规则：按 seed 配对（std vs official，seed 42 / 43 / 44），每个 seed 同时满足
  ① 官方臂的良性 T50（主列 local_benign_asr，插值）**更早**：
       官方左删失（第一个评估点就 ≥ 0.5）而标准臂不是 → 更早；两臂都越阈 → 比数值；
       其余情况（官方从未越阈、两臂都左删失、标准臂左删失）→ 不算更早；
  ② 官方臂末 10 点良性 ASR ≥ 标准臂；
3 个 seed 都满足 → 「是（事后判据）」；否则「否（事后判据）」；有 run 缺 / 无效 → missing / invalid。
另报 fresh pm_acc 与陈旧 pm_acc 的两臂之差 —— **混杂**：官方预处理下模型本身也更弱（F-050）。

纯标准库，不 import TF。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_exp3 import first_crossing   # noqa: E402
from pilot_a4 import invalid_reasons       # noqa: E402
from runs_table import last_k_mean, window_mean, SIDE_COLUMN_ANCHOR   # noqa: E402

RESULTS = HERE.parent / "experiments/attack/hfl-mechanism/results/P2/G7"
SEEDS = (42, 43, 44)
THETA = 0.5
KEY = "local_benign_asr"


def crossing(m: dict, key: str = KEY, theta: float = THETA):
    eff = (m.get("run") or {}).get("edge_rounds") or 1
    series = [(r["round"] * eff, r[key]) for r in m.get("rounds") or [] if r.get(key) is not None]
    return first_crossing(series, theta)


def earlier(off, std) -> bool:
    """官方臂的越阈是否严格早于标准臂（左删失 = 早于第一个评估点）。"""
    if off.left_censored:
        return not std.left_censored
    if off.crossed and std.crossed:
        return off.t_theta < std.t_theta
    if off.crossed and not std.crossed and not std.left_censored:
        return True                      # 标准臂从未越阈（右删失）
    return False


def _last10(rows, key):
    if key in SIDE_COLUMN_ANCHOR:                  # 陈旧 pm_acc 隔点算之后按窗口取（D-054）
        return window_mean(rows, key, anchor=SIDE_COLUMN_ANCHOR[key])[0]
    return last_k_mean([r.get(key) for r in rows or []])[0]


def judge_seed(std: dict | None, off: dict | None, seed: int) -> dict:
    if std is None or off is None:
        return {"seed": seed, "verdict": "missing"}
    bad = [f"{arm}: {r}" for arm, m in (("std", std), ("official", off)) for r in invalid_reasons(m)]
    if bad:
        return {"seed": seed, "verdict": "invalid", "reasons": bad}
    cs, co = crossing(std), crossing(off)
    a_s, a_o = _last10(std["rounds"], KEY), _last10(off["rounds"], KEY)
    ok_t, ok_a = earlier(co, cs), (a_o is not None and a_s is not None and a_o >= a_s)

    def _t(c):
        return "<首点" if c.left_censored else (None if c.t_theta is None else round(c.t_theta, 2))
    return {"seed": seed, "verdict": "yes" if (ok_t and ok_a) else "no",
            "t50_std": _t(cs), "t50_official": _t(co), "earlier": ok_t,
            "asr10_std": a_s, "asr10_official": a_o, "asr_not_lower": ok_a,
            "d_pm_acc": _d(off, std, "pm_acc"), "d_pm_acc_stale": _d(off, std, "pm_acc_stale")}


def _d(off, std, key):
    a, b = _last10(off["acc_rounds"], key), _last10(std["acc_rounds"], key)
    return None if a is None or b is None else round(a - b, 4)


def judge(pairs: dict) -> dict:
    """pairs：{seed: (std_metrics, official_metrics)}。"""
    per = [judge_seed(*pairs.get(s, (None, None)), s) for s in SEEDS]
    vs = {p["verdict"] for p in per}
    if "invalid" in vs:
        overall = "invalid"
    elif "missing" in vs:
        overall = "missing"
    else:
        overall = "yes" if vs == {"yes"} else "no"
    text = {"yes": "官方预处理降低攻击难度：是（事后判据，D-049；干净精度同时下降 → 混杂，F-050）",
            "no": "官方预处理降低攻击难度：否（事后判据，D-049）",
            "missing": "结果不全", "invalid": "有 run 无效（D-044 的闸）"}[overall]
    return {"overall": overall, "summary": text, "per_seed": per, "posthoc": True}


def load(results_dir: Path = RESULTS) -> dict:
    def _l(p):
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    return {s: (_l(results_dir / f"G7__std__s{s}.metrics.json"),
                _l(results_dir / f"G7__official__s{s}.metrics.json")) for s in SEEDS}


def main(argv=None):
    ap = argparse.ArgumentParser(description="G7 的事后判定（D-049，非预注册）")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = judge(load())
    for p in res["per_seed"]:
        print(f"  seed {p['seed']}: {p['verdict']:<7} "
              f"{json.dumps({k: v for k, v in p.items() if k not in ('seed', 'verdict')}, ensure_ascii=False)}")
    print(res["summary"])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
