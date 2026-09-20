"""
harness/analyze_exp3.py  —  把 metrics.json 算成 README 的四条判据 + T_theta + HHI 回归

## 为什么

experiments/attack/hfl-propagation/current-focus.md 记录的导师意见 #7「Exp 3 需要
更多实验和分析」拆成两件事，这个文件解决其中「分析不够」的一半：29 个
metrics.json 早就在盘上，但没有代码把 README.md 定义的四条判据从原始数据里算出来，
RESULTS.md 至今报不出这四条、也报不出「到 ASR=0.5 的有效轮数」和「干净 edge 的
爬升延迟」。

不 import TF，纯 stdlib + numpy 都不需要（连 numpy 都没用到，算术全是 Python 内建），
本地秒级跑得动，L1 测试见 tests/test_analyze_exp3.py。

## README 的四条判据（用 per_edge_final，不用被均值抹平的聚合值）

- Local amplification：edge_asr[e] > client_benign[e]
- Hierarchical amplification：final.global_asr > mean_e(edge_asr)
- Dilution：edge_asr[e] < client_malicious[e]
- Cross-edge cancellation：各 edge_asr 高但 final.global_asr 低

前两条是同一个量的正负两面，本模块报成一个带符号数 Δ_hier
（>0 = amplification，<0 = cancellation）；Local amplification / Dilution
报成逐 edge 的带符号差值。mean_e 是**跨 edge 不加权平均**（问的是 edge 之间，
不是样本之间，不按 n_samples 加权）。

## 插值（T_theta）是为了什么

评估是间隔采样的（比如每 5 个有效轮测一次），真正越过阈值 θ 的瞬间几乎不会
刚好落在采样点上。线性插值只假设**相邻两个采样点之间**大致线性（比"整条曲线
线性"弱得多的局部假设），去估计理论上越阈的位置——这是生存分析/事件时间统计
的标准做法。但点特别稀疏时插值会失真："插值出来的数看着精确其实是编的"
（current-focus.md 原话，指 3c_R20/R40 这类粗网格）。`MIN_POINTS_FOR_INTERPOLATION`
就是这道门槛。

## 退化情况一律返回 None，不编造 0 或者拿别的数顶替

这个仓库已经因为「无定义指标被填 0.0」出过真的研究性错误（陷阱 #13：
diff_edge_asr/same_edge_asr 曾经在无对应分组时被填 0，与「后门真的没有效果」
在数值上分不出来）。本模块延续这个约定：标量的退化结果一律 None，从不返回 0
或者一个看似合理但其实是编的数字；不同的退化原因（从未越阈 / 一开始就越阈 /
网格太粗 / 没有干净 edge / ...）必须能互相区分，不能塌成同一个哨兵值。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "attack" / "hfl-propagation" / "results"

# 这是 metrics.json 里的字面字段名。**不要**跟 fedavg/server/stopping.py 的
# ASR_KEYS = ("global_asr", "edge_asr_mean", "local_asr_benign_mean") 搞混——
# 那是训练循环内存态 dict 的键名（StoppingRule 直接吃它），本质是同一份数字
# 在流水线三个阶段的三套键名：内存态 edge_asr_mean/local_asr_benign_mean
# → 日志标签 EM_ASR=/local_benign= → collect_metrics.py 解析后写进 JSON 时
# 变成 edge_asr/local_benign_asr（fedavg/server/backdoor_server.py:250-251，
# harness/collect_metrics.py:452-454）。本模块只对接 metrics.json 这一套。
METRIC_KEYS = ("global_asr", "edge_asr", "local_benign_asr")
THETAS = (0.25, 0.5, 0.75)

# 插值只需要 2 个点（卡住 θ 的那一对），但只有 2 个点时那对点就是全部数据——
# 「插值出来的数看着精确其实是编的」。取 4 留出「越阈前至少一点 + 越阈后至少
# 一点」之外的余量，确认爬升不是单点噪声。比 stopping.pm_window=10 宽松：
# 后者要判断稳定平台需要更多点撑住斜率估计，这里只要局部卡住 θ 的行为，
# 10 会把真实存在的 3c_R40（当前 8 点）也误伤。
MIN_POINTS_FOR_INTERPOLATION = 4

# 字面抄 experiments/attack/hfl-propagation/plot_exp3.py 的 CELL_RE，
# 不 import 那个文件（它模块级 import matplotlib，会给纯分析脚本强加绘图依赖）。
CELL_RE = re.compile(r"^(?P<cell>.+)_seed(?P<seed>\d+)$")

REASON_COARSE = "grid_too_coarse"
REASON_NO_CLEAN_EDGE = "no_clean_edge"
REASON_NO_CONTAMINATED_EDGE = "no_contaminated_edge"
REASON_CLEAN_ALL_CENSORED = "clean_all_censored"
REASON_CONTAMINATED_ALL_CENSORED = "contaminated_all_censored"


# ── 结果类（__slots__，不用 @dataclass，照 stopping.py 的 Decision 风格）──────

class Crossing:
    """first_crossing() 的返回值。四态互斥，见模块 docstring。"""
    __slots__ = ("crossed", "t_theta", "left_censored", "censored_at", "reason", "n_points")

    def __init__(self, *, crossed, t_theta=None, left_censored=False,
                 censored_at=None, reason=None, n_points=0):
        self.crossed = crossed
        self.t_theta = t_theta
        self.left_censored = left_censored
        self.censored_at = censored_at
        self.reason = reason
        self.n_points = n_points

    def as_dict(self):
        return {
            "crossed": self.crossed, "t_theta": self.t_theta,
            "left_censored": self.left_censored, "censored_at": self.censored_at,
            "reason": self.reason, "n_points": self.n_points,
        }

    def __repr__(self):
        return f"Crossing({self.as_dict()!r})"


class CleanEdgeDelay:
    """clean_edge_delay() 的返回值。"""
    __slots__ = ("delay", "reason", "t_clean", "t_contaminated",
                 "n_clean_edges", "n_contaminated_edges")

    def __init__(self, *, delay, reason, t_clean, t_contaminated,
                 n_clean_edges, n_contaminated_edges):
        self.delay = delay
        self.reason = reason
        self.t_clean = t_clean
        self.t_contaminated = t_contaminated
        self.n_clean_edges = n_clean_edges
        self.n_contaminated_edges = n_contaminated_edges

    def as_dict(self):
        return {
            "delay": self.delay, "reason": self.reason,
            "t_clean": self.t_clean, "t_contaminated": self.t_contaminated,
            "n_clean_edges": self.n_clean_edges,
            "n_contaminated_edges": self.n_contaminated_edges,
        }

    def __repr__(self):
        return f"CleanEdgeDelay({self.as_dict()!r})"


class SeedBand:
    """跨 seed 聚合的标准形状：{mean, min, max, n_seeds, seeds, single_seed}。"""
    __slots__ = ("mean", "min", "max", "n_seeds", "seeds", "single_seed")

    def __init__(self, *, mean, min, max, n_seeds, seeds, single_seed):
        self.mean = mean
        self.min = min
        self.max = max
        self.n_seeds = n_seeds
        self.seeds = seeds
        self.single_seed = single_seed

    def as_dict(self):
        return {
            "mean": self.mean, "min": self.min, "max": self.max,
            "n_seeds": self.n_seeds, "seeds": self.seeds,
            "single_seed": self.single_seed,
        }

    def __repr__(self):
        return f"SeedBand({self.as_dict()!r})"


class OLSFit:
    """_ols_fit() 的返回值。"""
    __slots__ = ("slope", "intercept", "r2", "n")

    def __init__(self, *, slope, intercept, r2, n):
        self.slope = slope
        self.intercept = intercept
        self.r2 = r2
        self.n = n

    def as_dict(self):
        return {"slope": self.slope, "intercept": self.intercept,
                "r2": self.r2, "n": self.n}

    def __repr__(self):
        return f"OLSFit({self.as_dict()!r})"


# ── 纯计算函数（无 IO，可直接单测）────────────────────────────────────────────

def edge_of(cid: int, n_clients: int, n_edges: int) -> int:
    """cid -> edge_id，edge_assignment == "block" 时的公式（本实验唯一用到的分配方式）。"""
    block = n_clients // n_edges
    return cid // block


def _mean_min_max(values):
    """values：已经过滤掉 None 的 float 序列。空 -> 全 None。"""
    values = list(values)
    if not values:
        return None, None, None
    return sum(values) / len(values), min(values), max(values)


def unweighted_mean_over_edges(values):
    """
    values: list[float|None]，跨 edge 的某个量。返回 (mean, n_used, n_dropped)。

    **不加权**——这是 Δ_hier / 局部放大 / 稀释 / 干净-edge 分组共用的唯一
    「跨 edge 平均」实现点，问的是 edge 之间的差异，不是样本之间的差异，
    不按 n_samples 加权。全部 None -> (None, 0, n_dropped)，不是 0。
    """
    used = [v for v in values if v is not None]
    n_dropped = len(values) - len(used)
    if not used:
        return None, 0, n_dropped
    return sum(used) / len(used), len(used), n_dropped


def hhi(malicious_per_edge):
    """Herfindahl-Hirschman 集中度指数：Σ(m_e/M)²。空/全零 -> None。"""
    if not malicious_per_edge:
        return None
    total = sum(malicious_per_edge)
    if total <= 0:
        return None
    return sum((m / total) ** 2 for m in malicious_per_edge)


def _safe_sub(a, b):
    if a is None or b is None:
        return None
    return a - b


def delta_hier(final_global_asr, per_edge_final):
    """
    Δ_hier = final_global_asr − mean_e(edge_asr)。返回 (delta, n_used, n_dropped)。

    >0 = Hierarchical amplification（cloud 层比各 edge 平均水平更容易触发后门）；
    <0 = Cross-edge cancellation（各 edge 单独看不低，合到 cloud 层反而下降）。
    这是 README 判据 2/4 的带符号合并版——同一个量的正负两面。
    global_asr 或 mean_e 任一 None -> delta 为 None。
    """
    edge_asrs = [pe.get("edge_asr") for pe in per_edge_final]
    mean_e, n_used, n_dropped = unweighted_mean_over_edges(edge_asrs)
    if final_global_asr is None or mean_e is None:
        return None, n_used, n_dropped
    return final_global_asr - mean_e, n_used, n_dropped


def local_amplification_per_edge(per_edge_final):
    """
    {edge_id: edge_asr - client_benign}。= README「Local amplification」的
    带符号版，也是「私有头挡住多少」：>0 说明良性客户端自己的个性化模型比
    这个 edge 的共享模型更难触发后门——私有头过滤掉了一部分，但共享部分仍带着它。
    """
    return {pe["edge_id"]: _safe_sub(pe.get("edge_asr"), pe.get("client_benign"))
            for pe in per_edge_final}


def dilution_per_edge(per_edge_final):
    """
    {edge_id: client_malicious - edge_asr}。= README「Dilution」的带符号版：
    >0 说明恶意客户端自己训的模型 ASR 很高，但跟同 edge 良性客户端的更新平均
    到一起后被稀释了——这正是 FedAvg 聚合本该起的作用。
    """
    return {pe["edge_id"]: _safe_sub(pe.get("client_malicious"), pe.get("edge_asr"))
            for pe in per_edge_final}


def clean_edge_propagation_strength(per_edge_final):
    """
    {edge_id: client_benign}，仅 has_malicious=False 的 edge。

    这种 edge 从没见过投毒样本，它的 client_benign ASR 若不是 0，唯一解释是
    后门从别的（有恶意客户端的）edge 经 cloud 层聚合"传染"过来的——直接测跨
    edge 传播强度。只在 collocated/mixed 拓扑里有意义；distributed 每个 edge
    都有恶意端，没有干净 edge 可测，返回 {}（调用方不得把 {} 当成 0）。
    """
    return {pe["edge_id"]: pe["client_benign"] for pe in per_edge_final
            if pe.get("has_malicious") is False and pe.get("client_benign") is not None}


def effective_round_series(rounds, edge_rounds, metric_key):
    """[(round*edge_rounds, value), ...]，过滤 None，按 r_eff 排序。"""
    out = []
    for r in rounds:
        v = r.get(metric_key)
        if v is None:
            continue
        out.append((r["round"] * edge_rounds, v))
    out.sort(key=lambda t: t[0])
    return out


def per_edge_round_series(per_edge_rounds, edge_rounds, edge_id, metric_key):
    """同 effective_round_series，但读 per_edge_rounds 形状（{"<round>": [...]}）。"""
    out = []
    for round_str, edges in per_edge_rounds.items():
        r_eff = int(round_str) * edge_rounds
        for e in edges:
            if e["edge_id"] == edge_id:
                v = e.get(metric_key)
                if v is not None:
                    out.append((r_eff, v))
                break
    out.sort(key=lambda t: t[0])
    return out


def first_crossing(series, theta, *, min_points=MIN_POINTS_FOR_INTERPOLATION):
    """
    series: [(r_eff, value), ...]（调用方已过滤 None）。找第一次越过 theta 的
    有效轮，线性插值。四态互斥，见模块 docstring 与类 Crossing：

      正常越阈       crossed=True  t_theta=插值值
      从未越阈(右删失) crossed=False t_theta=None censored_at=最后一个 r_eff
      一开始就≥θ(左删失) crossed=False t_theta=None left_censored=True
      网格太粗       crossed=False t_theta=None reason="grid_too_coarse"

    故意不做 stopping.py 式的 debounce——这里要的是字面的首次越阈插值，
    不是稳定性触发，这是与 stopping.StoppingRule 语义的有意分歧。
    """
    n = len(series)
    if n < min_points:
        return Crossing(crossed=False, reason=REASON_COARSE, n_points=n)

    series = sorted(series, key=lambda t: t[0])
    r0, v0 = series[0]
    if v0 >= theta:
        return Crossing(crossed=False, left_censored=True, n_points=n)

    for i in range(len(series) - 1):
        r_a, v_a = series[i]
        r_b, v_b = series[i + 1]
        if v_a < theta <= v_b:
            if v_b == v_a:
                continue  # 防御性：v_a<theta<=v_b 已保证 v_b>v_a，理论上到不了这里
            t = r_a + (theta - v_a) * (r_b - r_a) / (v_b - v_a)
            return Crossing(crossed=True, t_theta=t, n_points=n)

    return Crossing(crossed=False, censored_at=series[-1][0], n_points=n)


def private_head_blocking_series(per_edge_rounds, edge_rounds):
    """
    {edge_id: [(r_eff, value|None), ...]}。local_amplification_per_edge 的
    逐轮版——每一轮都算一次 edge_asr[e,r]-client_benign[e,r]，看"私有头挡后门"
    是训练早期就有、后期才出现、还是先强后弱。任一边 None -> 该点 None（不是 0）。
    """
    result = {}
    for round_str, edges in per_edge_rounds.items():
        r_eff = int(round_str) * edge_rounds
        for e in edges:
            eid = e["edge_id"]
            val = _safe_sub(e.get("edge_asr"), e.get("client_benign"))
            result.setdefault(eid, []).append((r_eff, val))
    for eid in result:
        result[eid].sort(key=lambda t: t[0])
    return result


def clean_edge_delay(per_edge_rounds, per_edge_final, edge_rounds, theta):
    """
    delay = 干净 edge 的 T_theta(client_benign) − 污染 edge 的。

    reason 精确区分失败原因，不许塌成一个 0：
      no_clean_edge             —— *_distributed 的情况，零个干净 edge
      no_contaminated_edge      —— 防御性：零个污染 edge
      clean_all_censored        —— 干净组存在，但组内没有 edge 越阈
      contaminated_all_censored —— 污染组存在，但组内没有 edge 越阈
      None                      —— 正常，delay 是真实 float

    n_clean_edges/n_contaminated_edges 始终填数（越阈判定前的原始组大小），
    让"没有干净 edge"和"有干净 edge 但没越阈"可区分。
    """
    clean_ids = [pe["edge_id"] for pe in per_edge_final if pe.get("has_malicious") is False]
    contaminated_ids = [pe["edge_id"] for pe in per_edge_final if pe.get("has_malicious") is True]

    if not clean_ids:
        return CleanEdgeDelay(delay=None, reason=REASON_NO_CLEAN_EDGE,
                               t_clean=None, t_contaminated=None,
                               n_clean_edges=0, n_contaminated_edges=len(contaminated_ids))
    if not contaminated_ids:
        return CleanEdgeDelay(delay=None, reason=REASON_NO_CONTAMINATED_EDGE,
                               t_clean=None, t_contaminated=None,
                               n_clean_edges=len(clean_ids), n_contaminated_edges=0)

    def group_t_theta(ids):
        ts = []
        for eid in ids:
            series = per_edge_round_series(per_edge_rounds, edge_rounds, eid, "client_benign")
            c = first_crossing(series, theta)
            if c.t_theta is not None:
                ts.append(c.t_theta)
        return ts

    clean_ts = group_t_theta(clean_ids)
    cont_ts = group_t_theta(contaminated_ids)

    t_clean = sum(clean_ts) / len(clean_ts) if clean_ts else None
    t_cont = sum(cont_ts) / len(cont_ts) if cont_ts else None

    if t_clean is None:
        return CleanEdgeDelay(delay=None, reason=REASON_CLEAN_ALL_CENSORED,
                               t_clean=None, t_contaminated=t_cont,
                               n_clean_edges=len(clean_ids), n_contaminated_edges=len(contaminated_ids))
    if t_cont is None:
        return CleanEdgeDelay(delay=None, reason=REASON_CONTAMINATED_ALL_CENSORED,
                               t_clean=t_clean, t_contaminated=None,
                               n_clean_edges=len(clean_ids), n_contaminated_edges=len(contaminated_ids))

    return CleanEdgeDelay(delay=t_clean - t_cont, reason=None,
                           t_clean=t_clean, t_contaminated=t_cont,
                           n_clean_edges=len(clean_ids), n_contaminated_edges=len(contaminated_ids))


def participation_by_edge(malicious_participation_by_client, n_clients, n_edges, edge_assignment=None):
    """
    {edge_id: 总参与次数}，按 edge_of 把逐客户端参与轮次聚合。

    edge_assignment 非 None 且不是 "block" 时返回 {}（block 公式不适用于别的
    分配方式）；archive-pre-fix 里 edge_assignment 字段本身缺失（None），
    仍按 block 处理——这是本实验唯一用过的分配方式。
    """
    if not malicious_participation_by_client or not n_clients or not n_edges:
        return {}
    if edge_assignment is not None and edge_assignment != "block":
        return {}
    out = {}
    for cid_str, rounds in malicious_participation_by_client.items():
        eid = edge_of(int(cid_str), n_clients, n_edges)
        out[eid] = out.get(eid, 0) + len(rounds)
    return out


def _ols_fit(xs, ys):
    """
    普通最小二乘线性回归（闭式解），风格照 stopping._ols_slope。
    n<2 或所有 x 相同 -> None，不除以 0 也不编一个假斜率。
    ss_tot==0（y 全同）时 r2 为 None（分母为 0，R² 本身没有定义）。
    """
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    sxx = sum((x - x_bar) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    ss_tot = sum((y - y_bar) ** 2 for y in ys)
    if ss_tot == 0:
        r2 = None
    else:
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r2 = 1 - ss_res / ss_tot
    return OLSFit(slope=slope, intercept=intercept, r2=r2, n=n)


def seed_band(values_by_seed):
    """
    values_by_seed: dict[seed, float|None]。丢弃 None 后算
    {mean,min,max,n_seeds,seeds,single_seed}。全 None -> 全部字段 None/n_seeds=0，
    不是 0。single_seed 标记单 seed 的格子——RESULTS.md 的 Figure 12 post-mortem：
    2-seed 均值曾被当成跟更多 seed 一样可信，产生了假的"趋势"。
    """
    defined = {s: v for s, v in values_by_seed.items() if v is not None}
    mean, mn, mx = _mean_min_max(defined.values())
    seeds = sorted(defined.keys())
    return SeedBand(mean=mean, min=mn, max=mx, n_seeds=len(defined),
                     seeds=seeds, single_seed=(len(defined) == 1))


# ── 聚合 / IO ────────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict:
    """
    {cell: {seed: metrics_dict}}。非递归 glob——天然排除 archive-pre-fix/
    子目录（那批数据已知失效，见其自己的 README），除非显式 --results 指向它。
    """
    out = {}
    for p in sorted(results_dir.glob("*.metrics.json")):
        stem = p.name[: -len(".metrics.json")]
        m = CELL_RE.match(stem)
        if not m:
            print(f"[analyze_exp3] 跳过不匹配 <cell>_seed<N> 命名的文件: {p.name}")
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"[analyze_exp3] 跳过无法解析的 {p.name}: {e}")
            continue
        cell = m.group("cell")
        seed = int(m.group("seed"))
        out.setdefault(cell, {})[seed] = data
    return out


def analyze_run(data: dict) -> dict:
    """
    对单个 seed 的 metrics_dict 跑全部纯计算函数，返回一个 run 的结果字典。
    所有 run.* 读取用 .get(...)：archive-pre-fix 缺字段时该项结果为 None，
    绝不崩溃、绝不编造。
    """
    run = data.get("run") or {}
    final = data.get("final") or {}
    per_edge_final = data.get("per_edge_final") or []
    per_edge_rounds = data.get("per_edge_rounds") or {}
    rounds = data.get("rounds") or []
    malicious_participation_by_client = data.get("malicious_participation_by_client") or {}

    n_clients = run.get("n_clients")
    n_edges = run.get("n_edges")
    # archive-pre-fix 缺 edge_rounds；退化到 1（这批数据本来就已知失效，
    # 这里只求"跑得通"，不求这批数字正确）。
    edge_rounds = run.get("edge_rounds") or 1
    malicious_per_edge = run.get("malicious_per_edge")
    edge_assignment = run.get("edge_assignment")

    delta, n_used, n_dropped = delta_hier(final.get("global_asr"), per_edge_final)
    local_amp = local_amplification_per_edge(per_edge_final)
    dilution = dilution_per_edge(per_edge_final)
    clean_prop = clean_edge_propagation_strength(per_edge_final)

    t_theta = {}
    for mk in METRIC_KEYS:
        series = effective_round_series(rounds, edge_rounds, mk)
        t_theta[mk] = {th: first_crossing(series, th) for th in THETAS}

    clean_delay = {th: clean_edge_delay(per_edge_rounds, per_edge_final, edge_rounds, th)
                   for th in THETAS}

    participation = participation_by_edge(
        malicious_participation_by_client, n_clients, n_edges, edge_assignment)

    head_block_series = private_head_blocking_series(per_edge_rounds, edge_rounds)

    hhi_val = hhi(malicious_per_edge) if malicious_per_edge else None

    local_amp_mean, _, _ = unweighted_mean_over_edges(list(local_amp.values()))
    dilution_mean, _, _ = unweighted_mean_over_edges(list(dilution.values()))

    return {
        "run_meta": {k: run.get(k) for k in (
            "method", "defense", "attack", "n_edges", "n_clients",
            "edge_assignment", "malicious_per_edge", "edge_rounds")},
        "hhi": hhi_val,
        "final_asr": {mk: final.get(mk) for mk in METRIC_KEYS},
        "delta_hier": delta,
        "n_edges_used_hier": n_used,
        "n_edges_dropped_hier": n_dropped,
        "local_amplification_per_edge": local_amp,
        "local_amplification_mean": local_amp_mean,
        "dilution_per_edge": dilution,
        "dilution_mean": dilution_mean,
        "clean_edge_propagation_strength": clean_prop,
        "t_theta": t_theta,
        "clean_edge_delay": clean_delay,
        "participation_by_edge": participation,
        "private_head_blocking_timeseries": head_block_series,
    }


def aggregate_cell(cell: str, runs_by_seed: dict) -> dict:
    """
    逐 seed 调 analyze_run，然后把标量量全部 seed_band。hhi 是配置常数，跨 seed
    不该变，不一致时显式标记（consistent_across_seeds）而不是悄悄取均值。
    """
    per_seed = {seed: analyze_run(data) for seed, data in runs_by_seed.items()}
    seeds = sorted(per_seed.keys())
    run_meta = dict(per_seed[seeds[0]]["run_meta"])

    hhi_values = {v for r in per_seed.values() if (v := r["hhi"]) is not None}
    hhi_consistent = len(hhi_values) <= 1
    hhi_value = next(iter(hhi_values)) if hhi_values else None

    final_asr = {mk: seed_band({s: r["final_asr"][mk] for s, r in per_seed.items()}).as_dict()
                 for mk in METRIC_KEYS}

    delta_hier_band = seed_band({s: r["delta_hier"] for s, r in per_seed.items()}).as_dict()
    n_edges_dropped_band = seed_band(
        {s: r["n_edges_dropped_hier"] for s, r in per_seed.items()}).as_dict()
    local_amp_mean_band = seed_band(
        {s: r["local_amplification_mean"] for s, r in per_seed.items()}).as_dict()
    dilution_mean_band = seed_band(
        {s: r["dilution_mean"] for s, r in per_seed.items()}).as_dict()

    all_edge_ids = sorted({eid for r in per_seed.values() for eid in r["local_amplification_per_edge"]})
    local_amp_per_edge = {
        str(eid): seed_band(
            {s: r["local_amplification_per_edge"].get(eid) for s, r in per_seed.items()}).as_dict()
        for eid in all_edge_ids
    }
    dilution_per_edge_band = {
        str(eid): seed_band(
            {s: r["dilution_per_edge"].get(eid) for s, r in per_seed.items()}).as_dict()
        for eid in all_edge_ids
    }

    clean_prop_values = [v for r in per_seed.values()
                          for v in r["clean_edge_propagation_strength"].values()]
    if clean_prop_values:
        m, mn, mx = _mean_min_max(clean_prop_values)
        clean_prop_band = {"mean": m, "min": mn, "max": mx, "n": len(clean_prop_values)}
    else:
        clean_prop_band = None

    t_theta_report = {}
    for mk in METRIC_KEYS:
        t_theta_report[mk] = {}
        for th in THETAS:
            per_seed_crossings = {s: r["t_theta"][mk][th] for s, r in per_seed.items()}
            band = seed_band({s: c.t_theta for s, c in per_seed_crossings.items()}).as_dict()
            t_theta_report[mk][str(th)] = {
                "band": band,
                "per_seed": {str(s): c.as_dict() for s, c in per_seed_crossings.items()},
            }

    clean_delay_report = {}
    for th in THETAS:
        per_seed_delays = {s: r["clean_edge_delay"][th] for s, r in per_seed.items()}
        band = seed_band({s: d.delay for s, d in per_seed_delays.items()}).as_dict()
        reasons = sorted({d.reason for d in per_seed_delays.values() if d.reason is not None})
        clean_delay_report[str(th)] = {
            "band": band,
            "reasons": reasons,
            "per_seed": {str(s): d.as_dict() for s, d in per_seed_delays.items()},
        }

    participation_ids = sorted({eid for r in per_seed.values() for eid in r["participation_by_edge"]})
    participation_total = {
        str(eid): sum(r["participation_by_edge"].get(eid, 0) for r in per_seed.values())
        for eid in participation_ids
    }

    private_head_series = {
        str(s): {str(eid): series for eid, series in r["private_head_blocking_timeseries"].items()}
        for s, r in per_seed.items()
    }

    return {
        "n_seed_files": len(seeds),
        "seeds_present": seeds,
        "run_meta": run_meta,
        "hhi": {"value": hhi_value, "consistent_across_seeds": hhi_consistent},
        "final_asr": final_asr,
        "delta_hier": delta_hier_band,
        "n_edges_dropped_hier": n_edges_dropped_band,
        "local_amplification": {"mean_across_edges": local_amp_mean_band, "per_edge": local_amp_per_edge},
        "dilution": {"mean_across_edges": dilution_mean_band, "per_edge": dilution_per_edge_band},
        "clean_edge_propagation_strength": clean_prop_band,
        "t_theta": t_theta_report,
        "clean_edge_delay": clean_delay_report,
        "participation_by_edge": participation_total,
        "private_head_blocking_timeseries": private_head_series,
    }


def hhi_regression(cells_report: dict) -> dict:
    """
    final_ASR ~ HHI 的线性回归，local_benign_asr 是主指标（据
    experiments/calibration/RESULTS.md：global_asr 会饱和到 ~1.0，失去分辨力）。
    三个 collocated 格子的重复 x=1.0 点保留，当重复测量用，不去重。
    """
    result = {}
    for mk in ("local_benign_asr", "edge_asr", "global_asr"):
        xs, ys = [], []
        for rep in cells_report.values():
            hhi_val = rep["hhi"]["value"]
            y_val = rep["final_asr"][mk]["mean"]
            if hhi_val is not None and y_val is not None:
                xs.append(hhi_val)
                ys.append(y_val)
        fit = _ols_fit(xs, ys)
        result[mk] = fit.as_dict() if fit is not None else None
    return result


def build_report(results_dir: Path) -> dict:
    """顶层入口：load_results -> 逐 cell aggregate_cell -> hhi_regression -> 组装。"""
    grouped = load_results(results_dir)
    cells = {cell: aggregate_cell(cell, runs_by_seed) for cell, runs_by_seed in grouped.items()}
    regression = hhi_regression(cells)
    return {
        "meta": {
            "results_dir": str(results_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "archive_pre_fix_data": "archive-pre-fix" in results_dir.parts,
            "n_cells": len(cells),
            "min_points_for_interpolation": MIN_POINTS_FOR_INTERPOLATION,
            "thetas": list(THETAS),
            "metric_keys": list(METRIC_KEYS),
        },
        "cells": cells,
        "hhi_regression": regression,
    }


def to_csv_rows(report: dict) -> list:
    """每 cell 一行的汇总列；完整逐轮/逐 edge 明细只留 JSON。"""
    rows = []
    for cell, rep in sorted(report["cells"].items()):
        rm = rep["run_meta"]
        row = {
            "cell": cell,
            "method": rm.get("method"),
            "defense": rm.get("defense"),
            "attack": rm.get("attack"),
            "n_edges": rm.get("n_edges"),
            "n_clients": rm.get("n_clients"),
            "edge_assignment": rm.get("edge_assignment"),
            "n_seed_files": rep["n_seed_files"],
            "single_seed": rep["n_seed_files"] == 1,
            "hhi": rep["hhi"]["value"],
            "hhi_consistent": rep["hhi"]["consistent_across_seeds"],
        }
        for mk in METRIC_KEYS:
            band = rep["final_asr"][mk]
            row[f"{mk}_mean"] = band["mean"]
            row[f"{mk}_min"] = band["min"]
            row[f"{mk}_max"] = band["max"]
        row["delta_hier_mean"] = rep["delta_hier"]["mean"]
        row["delta_hier_min"] = rep["delta_hier"]["min"]
        row["delta_hier_max"] = rep["delta_hier"]["max"]
        row["local_amp_mean"] = rep["local_amplification"]["mean_across_edges"]["mean"]
        row["dilution_mean"] = rep["dilution"]["mean_across_edges"]["mean"]
        cep = rep["clean_edge_propagation_strength"]
        row["clean_edge_propagation_strength_mean"] = cep["mean"] if cep else None
        row["t_theta_local_benign_asr_0.5_mean"] = rep["t_theta"]["local_benign_asr"]["0.5"]["band"]["mean"]
        row["t_theta_edge_asr_0.5_mean"] = rep["t_theta"]["edge_asr"]["0.5"]["band"]["mean"]
        row["t_theta_global_asr_0.5_mean"] = rep["t_theta"]["global_asr"]["0.5"]["band"]["mean"]
        cd05 = rep["clean_edge_delay"]["0.5"]
        row["clean_edge_delay_0.5_mean"] = cd05["band"]["mean"]
        row["clean_edge_delay_0.5_reasons"] = ",".join(cd05["reasons"]) if cd05["reasons"] else ""
        row["participation_total"] = sum(rep["participation_by_edge"].values())
        rows.append(row)
    return rows


def write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_summary(report: dict) -> None:
    print(f"[analyze_exp3] 结果目录: {report['meta']['results_dir']}")
    if report["meta"]["archive_pre_fix_data"]:
        print("[analyze_exp3] ⚠️  指向 archive-pre-fix/ —— 这批数据已知失效，仅用于验证代码跑得通")
    print(f"[analyze_exp3] {report['meta']['n_cells']} 个 cell\n")
    header = (f"{'cell':28s} {'n_edges':>7s} {'seeds':>6s} {'HHI':>6s} "
              f"{'local_benign_asr':>17s} {'delta_hier':>11s}")
    print(header)
    print("-" * len(header))
    for cell, rep in sorted(report["cells"].items()):
        rm = rep["run_meta"]
        hhi_val = rep["hhi"]["value"]
        lba = rep["final_asr"]["local_benign_asr"]["mean"]
        dh = rep["delta_hier"]["mean"]
        flag = "*" if rep["n_seed_files"] == 1 else " "
        hhi_s = f"{hhi_val:.2f}" if hhi_val is not None else "n/a"
        lba_s = f"{lba:.3f}" if lba is not None else "n/a"
        dh_s = f"{dh:.3f}" if dh is not None else "n/a"
        print(f"{cell:28s} {str(rm.get('n_edges')):>7s} "
              f"{rep['n_seed_files']:>5d}{flag} {hhi_s:>6s} {lba_s:>17s} {dh_s:>11s}")
    print("\n(* = 单 seed，数值仅供参考)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 3 metrics.json -> README 四条判据 + T_theta + HHI 回归")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS_DIR),
                     help="metrics.json 目录（不递归；archive-pre-fix/ 不会被误并入）")
    ap.add_argument("--out-json", default=None, help="写出完整 JSON 报告")
    ap.add_argument("--out-csv", default=None, help="写出汇总 CSV（一格一行）")
    args = ap.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_dir():
        print(f"[analyze_exp3] 目录不存在：{results_dir}")
        return

    report = build_report(results_dir)
    if not report["cells"]:
        print(f"[analyze_exp3] {results_dir} 下没有可解析的 <cell>_seed<N>.metrics.json")
        return

    print_summary(report)

    if args.out_json:
        write_json(report, Path(args.out_json))
        print(f"\n[analyze_exp3] JSON -> {args.out_json}")
    if args.out_csv:
        write_csv(to_csv_rows(report), Path(args.out_csv))
        print(f"[analyze_exp3] CSV  -> {args.out_csv}")


if __name__ == "__main__":
    main()
