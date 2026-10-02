"""
fedavg/server/eval_grid.py  —  统一评估网格（S5；DECISIONS D-055 预案 / D-084 实现取舍）

配置键 `evaluation.eval_grid: G`（有效轮；缺省 None = 旧行为，逐字节不变）。
网格规则**只在这里定义一次**：server（全量 / 轻评估的位置）、config_validate（两个
eval_interval 的核对）、停止判据的横轴、harness 都调它。纯算术，**不 import TF**
（tests/test_eval_grid.py 本地秒级）。

为什么要它（FINDINGS F-046 / F-052）：
  · 旧行为每个云轮末评一次 → 一个点 = R 个有效轮。R20 每 20 有效轮才一点（T50 在 20–45
    有效轮的格子上没有分辨率），R2 反过来每 2 有效轮一次全量评估（G2 的 R2 格 150 次）。
  · 停止判据的斜率横轴是云轮号 → 容差折成「每有效轮」随 R 变（flat 比 R5 宽松 5 倍）。

规则（有效轮 eff = (g−1)·R + er，g 为云轮、er 为 edge 轮，都从 1 起）：
  · 网格点 = eff % G == 0 的有效轮。
  · **全量评估**：网格点恰好是云轮末（er = R，即 g·R % G == 0）。两个 eval_interval
    （`backdoor.` / `evaluation.`）必须 = lcm(G, R) / R —— 代码里全量评估仍由它们触发，
    config_validate 核对它们与网格一致。
  · **轻评估**：其余网格点（er < R），落在「所有 edge 都跑完第 er 个 edge 轮」之后
    （只有交错调度有这个时刻）。只算主列：fresh-PM 的 local / edge ASR + fresh pm_acc
    + EM 精度；不算 global、白盒、陈旧、ASR4、drift。
  · 停止判据横轴 = 网格序号 eff / G（R = G 时就是云轮号 → 逐位不变）。
  · GM 只在全量点算（云轮中间全局模型不变，算了没有信息）。

R = 1（flat）：没有轻评估点；全量每 G 个云轮一次，与旧的 eval_interval = G 相同，
只是停止判据的横轴变了（容差严 G 倍，正是 F-052 要修的）。
"""

from __future__ import annotations

from math import gcd

from alignment import get_switch

KEY = "evaluation.eval_grid"


def lcm(a: int, b: int) -> int:
    a, b = int(a), int(b)
    return a * b // gcd(a, b)


def grid_size(config: dict):
    """配置的网格 G（有效轮）；没写 → None（旧行为）。类型由 config_validate 检查。"""
    g = get_switch(config, "evaluation.eval_grid")      # 字面键：test_alignment_switches 按它认「真的被读到」
    return None if g is None else int(g)


def edge_rounds(config: dict) -> int:
    return int(((config or {}).get("federation") or {}).get("edge_rounds", 1) or 1)


def full_interval(G: int, R: int) -> int:
    """全量评估之间隔几个云轮 = 两个 eval_interval 必须取的值。"""
    return lcm(G, R) // int(R)


def period_eff(G: int, R: int) -> int:
    """网格与云轮对齐的周期（有效轮）：每 lcm(G, R) 个有效轮，全量点重复一次。"""
    return lcm(G, R)


def light_per_period(G: int, R: int) -> int:
    """一个周期里有几个轻评估点。"""
    return lcm(G, R) // int(G) - 1


def is_full_round(G: int, R: int, g: int) -> bool:
    """第 g 个云轮末是不是网格点（= 全量评估轮）。"""
    return (int(g) * int(R)) % int(G) == 0


def light_edge_rounds(G: int, R: int, g: int) -> list:
    """第 g 个云轮里要做轻评估的 edge 轮 er（1 ≤ er < R）。"""
    G, R, g = int(G), int(R), int(g)
    return [er for er in range(1, R) if ((g - 1) * R + er) % G == 0]


def grid_x(eff: int, G: int) -> int:
    """停止判据的横轴：网格序号。只对网格点有定义。"""
    eff, G = int(eff), int(G)
    if eff % G:
        raise ValueError(f"有效轮 {eff} 不在网格 G={G} 上")
    return eff // G


def eval_points(G: int, R: int, n_rounds: int) -> list:
    """全部评估点 [(g, er, eff, kind)]，按时间顺序；kind ∈ {"light", "full"}。"""
    out = []
    for g in range(1, int(n_rounds) + 1):
        for er in light_edge_rounds(G, R, g):
            out.append((g, er, (g - 1) * int(R) + er, "light"))
        if is_full_round(G, R, g):
            out.append((g, int(R), g * int(R), "full"))
    return out


def legacy_eval_points(R: int, eval_interval: int, n_rounds: int) -> list:
    """旧行为的评估点（云轮末，每 eval_interval 个云轮 + 末轮）—— 反向锚点用。"""
    out = []
    for g in range(1, int(n_rounds) + 1):
        if g % int(eval_interval) == 0 or g == int(n_rounds):
            out.append((g, int(R), g * int(R), "full"))
    return out


def describe(config: dict) -> dict:
    """`[设定8]` 的字段。没开网格：横轴是云轮号、GM / EM 每个云轮都算（旧行为）。"""
    G = grid_size(config)
    if G is None:
        return {"eval_grid": None, "period_eff": None, "full_every": None,
                "light_per_period": None, "slope_axis": "cloud", "gm_em": "every_round"}
    R = edge_rounds(config)
    return {"eval_grid": G, "period_eff": period_eff(G, R), "full_every": full_interval(G, R),
            "light_per_period": light_per_period(G, R), "slope_axis": "grid", "gm_em": "grid"}
