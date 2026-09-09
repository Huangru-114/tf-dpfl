"""
server/stopping.py  —  自适应轮数：地板 + 按需延长（只延长，不早停）

## 为什么需要它

固定轮数下，跨拓扑比较其实是在比「谁离收敛更近」——
**`n_edges` 影响收敛速度，而 `n_edges` 正是 Experiment 3 的自变量**。
实测（`experiments/calibration/RESULTS.md` §8，两格只差拓扑）：

    10edge 在 round 30 的样子 ≈ 2edge 在 round 12–18 的样子

于是「稳态拓扑效应」与「收敛滞后」在等预算下分不开。
改成「跑到各自判据满足」，每格就在**相同的完成状态**下被读取。

## 两条判据（都从实测噪声标定，不是拍的）

**A. `thresholds_crossed`** —— 三层 ASR × θ∈{0.25,0.5,0.75} 全部越过。
   越过的定义是**连续 2 个评估点 ≥ θ**（去抖；ASR 逐点 σ≈0.09，单点跨越是噪声）。
   实测满足于：2edge round 20 / 10edge round 28 —— 都 ≤ 地板 30。

**B. `pm_acc_plateau`** —— 末 W 个评估点的 OLS 斜率 < tol。
   W=10、tol=0.0010/轮 的依据：残差 σ≈0.0023 → 斜率 SE≈0.00024；
   2edge 实测 +0.00036（判平，1.5×SE），10edge +0.00229（判未平，8.8×SE），
   两边各有约 2 倍余量。

   > ⚠️ **不能复用 `read_calibration.plateau_round`** —— 它是**回溯式**的
   > （「此后再没离开 **final** ± tol」），需要知道最终值。跑的时候没有 final。
   > 在线判据必须是前瞻式的：末 W 点斜率。

**两条都满足才算完成。** 只用 A 更省机时，但会让 `pm_acc` 还在爬的格子停在
欠训状态 —— 那正是这个模块要修的 confound。criteria 是 config 列表，可单独关掉。

## 三种结束，各报各的（绝不混成一个数）

| 情况 | `reason` |
|---|---|
| 所有启用的判据都满足 | `converged` |
| 到 cap 仍未满足 | `cap_reached` ← **是 censored，不是「收敛在 cap」** |
| 评估点数不可能凑够 W（`3c_R40` 只有 5–7 个点） | `grid_too_coarse`，跑满 cap |

## 只延长，不早停

`floor_effective` 之前**永不**停，哪怕判据已满足。因此每格都有 ≥ floor 的轨迹，
**同轮比较点（round 30）人人都有** —— 跑到 45 的 run 里含着第 30 轮快照，
延长不破坏任何同轮分析。

纯 numpy，**不 import TF** —— tests/test_stopping.py 本地秒级跑得动。
"""

from __future__ import annotations

# 三层 ASR 在 _backdoor_eval 的 metrics 字典里的键名。
# 顺序即报告顺序；改名要同时改 backdoor_server 的打印与本表。
ASR_KEYS = ("global_asr", "edge_asr_mean", "local_asr_benign_mean")

VALID_CRITERIA = ("thresholds_crossed", "pm_acc_plateau")

REASON_CONVERGED = "converged"
REASON_CAP = "cap_reached"
REASON_COARSE = "grid_too_coarse"


def _ols_slope(xs, ys):
    """最小二乘斜率。少于 2 点或 x 全同 → None（不猜）。"""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


class StoppingRule:
    """
    每个后门评估轮喂一次观测，回一个是否停止的判定。

    调用方（`CloudServer.run`）只需要：

        dec = rule.update(round_idx, signals)      # signals: {键: 值 or None}
        if dec.stop:
            print(rule.log_line(round_idx, dec));  break
    """

    def __init__(self, cfg: dict | None, edge_rounds: int, n_rounds: int):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True)) and bool(cfg)
        self.edge_rounds = max(1, int(edge_rounds))
        self.n_rounds = int(n_rounds)

        self.criteria = tuple(cfg.get("criteria", VALID_CRITERIA))
        self.floor_effective = int(cfg.get("floor_effective", 0))
        self.cap_effective = int(cfg.get("cap_effective",
                                         self.n_rounds * self.edge_rounds))
        self.thetas = tuple(float(t) for t in cfg.get("thetas", (0.25, 0.5, 0.75)))
        self.debounce = int(cfg.get("debounce", 2))
        self.pm_window = int(cfg.get("pm_window", 10))
        self.pm_slope_tol = float(cfg.get("pm_slope_tol", 0.0010))

        # 观测序列：键 → [(round_idx, value), ...]。**None 一律不记**
        # （陷阱 #13：无定义的分组指标返回 None 而不是 0，混进来会污染斜率与越阈判定）。
        self._series: dict[str, list] = {}
        # 越阈状态：(键, θ) → 已确认越过。一旦确认不再撤销 ——
        # 判据问的是「有没有达到过」，不是「此刻在不在上面」。
        self._crossed: dict[tuple, bool] = {}
        self._run_len: dict[tuple, int] = {}     # 连续 ≥ θ 的点数（去抖计数）

    # ── 网格够不够 ────────────────────────────────────────────────────────
    @property
    def max_eval_points(self) -> int:
        """本 run 最多能拿到多少个评估点（由 cap 与 eval_interval 决定）。"""
        return len(self._series.get("pm_acc", ())) or 0

    def _grid_too_coarse(self, cap_rounds: int, eval_interval: int) -> bool:
        return (cap_rounds // max(1, eval_interval)) < self.pm_window

    # ── 主入口 ────────────────────────────────────────────────────────────
    def update(self, round_idx: int, signals: dict) -> "Decision":
        for key, val in (signals or {}).items():
            if val is None:                     # 无定义 → 不记，不当 0
                continue
            self._series.setdefault(key, []).append((int(round_idx), float(val)))
            self._bump_crossings(key, float(val))

        effective = int(round_idx) * self.edge_rounds
        detail = {
            "effective": effective,
            "crossed": self.n_crossed,
            "crossed_total": self.n_thresholds_total,
            "pm_slope": self.pm_slope(),
        }

        if not self.enabled:
            return Decision(False, None, detail)

        # 地板之前永不停 —— 哪怕判据已满足。同轮比较点靠这条保住。
        if effective < self.floor_effective:
            return Decision(False, None, detail)

        if self._all_criteria_met():
            return Decision(True, REASON_CONVERGED, detail)

        if effective >= self.cap_effective:
            reason = REASON_CAP
            if ("pm_acc_plateau" in self.criteria
                    and len(self._series.get("pm_acc", ())) < self.pm_window):
                # 点数根本凑不够 W —— 那不是「没收敛」，是网格太粗判不出来
                reason = REASON_COARSE
            return Decision(True, reason, detail)

        return Decision(False, None, detail)

    # ── 判据 A：越阈 ──────────────────────────────────────────────────────
    def _bump_crossings(self, key: str, val: float):
        if key not in ASR_KEYS:
            return
        for th in self.thetas:
            k = (key, th)
            if self._crossed.get(k):
                continue
            run = self._run_len.get(k, 0) + 1 if val >= th else 0
            self._run_len[k] = run
            if run >= self.debounce:
                self._crossed[k] = True

    @property
    def n_thresholds_total(self) -> int:
        return len(ASR_KEYS) * len(self.thetas)

    @property
    def n_crossed(self) -> int:
        return sum(1 for v in self._crossed.values() if v)

    def _thresholds_crossed(self) -> bool:
        return self.n_crossed >= self.n_thresholds_total

    # ── 判据 B：pm_acc 平台 ───────────────────────────────────────────────
    def pm_slope(self):
        pts = self._series.get("pm_acc", [])
        if len(pts) < self.pm_window:
            return None                      # 点数不够 → None，不猜
        tail = pts[-self.pm_window:]
        return _ols_slope([r for r, _ in tail], [v for _, v in tail])

    def _pm_plateau(self) -> bool:
        s = self.pm_slope()
        return s is not None and abs(s) < self.pm_slope_tol

    def _all_criteria_met(self) -> bool:
        checks = {"thresholds_crossed": self._thresholds_crossed,
                  "pm_acc_plateau": self._pm_plateau}
        return all(checks[c]() for c in self.criteria if c in checks)

    # ── 自描述 ────────────────────────────────────────────────────────────
    def log_line(self, round_idx: int, dec: "Decision") -> str:
        """
        **硬要求**：停在第 32 轮的 run 与跑满的 run 长得一模一样（陷阱 #7 / #14 同类）。
        这一行必须可解析，进 metrics.json 的 run 块。
        """
        s = dec.detail.get("pm_slope")
        return (f"[Stop] round={int(round_idx)} | effective={dec.detail['effective']} | "
                f"reason={dec.reason} | "
                f"crossed={dec.detail['crossed']}/{dec.detail['crossed_total']} | "
                f"pm_slope={'n/a' if s is None else f'{s:.5f}'} | "
                f"window={self.pm_window}")


class Decision:
    __slots__ = ("stop", "reason", "detail")

    def __init__(self, stop: bool, reason, detail: dict):
        self.stop = bool(stop)
        self.reason = reason
        self.detail = dict(detail)

    def __repr__(self):
        return f"Decision(stop={self.stop}, reason={self.reason!r})"
