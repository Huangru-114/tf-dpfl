"""
harness/collect_metrics.py  —  把集群 run 的**全量日志**压成一个小 metrics.json

回程协议：GB 级日志永远留在集群，只有这个 KB 级 json 进 git / 贴给 Claude Code。

用法（集群侧）：
    python harness/collect_metrics.py run.log -o experiments/attack/neurotoxin/exp001.metrics.json

输出字段
────────
    run                   这份数字**到底跑了什么**：config 路径 / run_name / method /
                          defense / attack / n_rounds / malicious_ids。
                          自描述是硬要求 —— `--config` 曾经被静默忽略（见 test_config_cli.py），
                          一份不写明自己跑了哪个配置的 metrics.json 事后无法判读。
    rounds[]              每个**后门评估轮**的 {round, global_asr, edge_asr,
                          local_benign_asr, same_edge_asr, diff_edge_asr, local_malicious_asr}
    final                 最后一个后门评估轮的上述指标
    acc_rounds[]          每个**全局轮**的 {round, gm_acc, em_acc, pm_acc}
                          （pm_acc 只在 eval_interval 轮有，其余为 null）
    per_edge_acc_rounds   逐 edge 精度 {round: [{edge_id, em_acc, pm_acc,
                          n_clients, n_samples}]} —— 全局均值会摊平「被污染的
                          edge 精度掉了多少」
    final_acc             最后一轮的 acc + 最终 PM 加权 C-Acc
    admitted[]            鲁棒聚合每次的接纳数 {defense, admitted, total}
    admitted_count_mean   接纳数均值；defense=none 时为 null（本来就没有判决）
    malicious_selected_rounds / n_malicious_participations
                          恶意客户端实际参与训练的轮次。**按 client_id 统计**，
                          不是按「攻击策略打印行」统计 —— vanilla（badnet/blended/dba）
                          的恶意客户端打印的是普通方法行、没有任何标记，
                          旧实现在这些配置下恒读 0，与「攻击根本没参与」不可区分。
    malicious_participation_by_client
                          每个恶意客户端各自参与的轮次（用来核对 fix-frequency 的 Q）
    client_failures[]     被 _collect_updates_parallel 吞掉的客户端异常。
                          「恶意更新被丢弃 → ASR=0」的直接征兆，必须显式回传。
    errors[]              traceback 首行（最多 5 条）
    log_tail              日志最后 40 行（截断版，供人工判读）
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ── 后门分层评估 ────────────────────────────────────────────────────────────
# 数值字段一律允许 "n/a"：无定义的分组（如全 distributed 布点下的 diff_edge、
# 或 100% 恶意 edge 的 same_edge）打 n/a 而非 0.000，解析成 JSON 的 null。
RE_BD = re.compile(
    r"\[Backdoor\] Round (\d+) \| GM_ASR=([\d.]+|n/a) \| EM_ASR=([\d.]+|n/a) \| "
    r"local_benign=([\d.]+|n/a) \(same_edge=([\d.]+|n/a), diff_edge=([\d.]+|n/a)\) \| "
    r"local_malicious=([\d.]+|n/a)")

# ── 准确率（PM 只在 eval_interval 轮出现，故可选）─────────────────────────────
RE_ACC = re.compile(
    r"\[Cloud\] GM=([\d.]+) \| EM=([\d.]+)(?: PM=([\d.]+))? \|")
# 单轮墙钟。**故意与 RE_ACC 分开**：RE_ACC 有 25 条现有测试挂着，把 time= 并进去
# 会让「老日志少一个字段」变成解析失败。这里在同一行上做一次**可选**搜索，
# 老日志拿到 None，新日志拿到数字。
# 注意口径：这个 elapsed 测的是 CloudServer.run_round 的 t0→elapsed，
# 而 BackdoorCloudServer._backdoor_eval 是在 super().run_round() 返回**之后**
# 才跑的 → round_time **不含**后门评估。两者要相加才是这一轮的墙钟。
RE_ROUND_TIME = re.compile(r"\[Cloud\] GM=[\d.]+ .*?\| time=([\d.]+)s")

# 后门评估的分阶段耗时（BackdoorCloudServer._backdoor_eval 打）。
# 未启用的阶段是 "n/a" 而不是 0.0 —— 与陷阱 #10/#13 同一约定。
RE_TIMING = re.compile(
    r"\[Timing\] Round (\d+) \| asr=([\d.]+|n/a)s \| feature=([\d.]+|n/a)s \| "
    r"forgetting=([\d.]+|n/a)s \| drift=([\d.]+|n/a)s \| total=([\d.]+|n/a)s")
RE_ROUND_HDR = re.compile(r"^\[Round\s+(\d+)\]")
RE_FINAL_PM = re.compile(r"\[Final PM\] weighted C-Acc = ([\d.]+)")

# ── 防御判决 ────────────────────────────────────────────────────────────────
# 统一判决行，由 RobustAggregationMixin._log_decision 在唯一收口处发出，
# 覆盖所有防御 / 所有 PFL 方法 / edge 与 cloud 两层：
#     [Decision] edge0 | flame  | admitted 7/10 | rejected=[3, 5]
#     [Decision] edge0 | median | coordinate-wise | n=10
# 旧实现只认 flame 自己那句 "[Defense:flame] admitted 7/10"，
# 而 multi_krum 打 "selected N clients"、dnc 打 "keep N clients"、
# trimmed_mean/median 什么都不打 → admitted_count 在 5 个防御里只有 1 个读得出来。
RE_DECISION = re.compile(
    r"\[Decision\] (\S+) \| (\S+) \| admitted (\d+)/(\d+) \| rejected=\[([\d,\s]*)\]")
RE_DECISION_COORD = re.compile(
    r"\[Decision\] (\S+) \| (\S+) \| coordinate-wise \| n=(\d+)")
# 兼容旧日志（只有 flame 有这一行）
RE_ADMIT_LEGACY = re.compile(r"\[Defense:(\w+)\] admitted (\d+)/(\d+)")

# ── 恶意客户端身份与参与 ────────────────────────────────────────────────────
RE_MAL_RESOLVED = re.compile(
    r"\[Backdoor\] resolved malicious clients[^:]*:\s*\[([\d,\s]*)\]")
RE_MAL_DECLARED = re.compile(r"\[Backdoor\] Client (\d+) MALICIOUS")
# 每个客户端每轮训练完都会打印的行，与 PFL 方法无关：
#   "  [Client  0] Round 1 | Hier-pFedMe(λ2=15.0, K=5) | loss=1.6438"
RE_CLIENT_ROUND = re.compile(r"\[Client\s*(\d+)\]\s+Round\s+(\d+)\s*\|")

# 逐 edge 面板（Experiment 3：per-edge 传播路径）
RE_PER_EDGE = re.compile(
    r"\[Backdoor\] Round (\d+) \| edge(\d+) \| edge_asr=([\d.]+|n/a) \| "
    r"client_benign=([\d.]+|n/a) \| client_malicious=([\d.]+|n/a) \| "
    r"n_benign=(\d+) \| n_malicious=(\d+) \| has_malicious=(True|False)")

# 漂移（Experiment 3C）：edge 相对本轮起点全局的参数/表示漂移
RE_DRIFT = re.compile(
    r"\[Drift\] Round (\d+) \| param_abs=(nan|[\d.]+) \| param_rel=(nan|[\d.]+) \| "
    r"repr_mean=(nan|[\d.]+) \| repr_median=(nan|[\d.]+) \| n_edges=(\d+)")

# ── 自描述 ──────────────────────────────────────────────────────────────────
RE_CFG_PATH = re.compile(r"\[Config\] loading (\S+)")
RE_RUN_NAME = re.compile(r"\[Config\] run_name = (\S+)")
RE_VALIDATE = re.compile(
    r"\[配置校验\] 通过 \| method=(\S+) \| defense=(\S+) \| attack=(\S+)")
# 设定自描述行（config_validate.py 打印）—— 收口陷阱 #7 的同类：
# client_fraction / poison_ratio 此前不进 run 块，"改成论文值" 无法从 artifact 证实。
RE_SETTINGS = re.compile(
    r"\[设定\] client_fraction=([\d.]+) \| poison_ratio=([\d.]+) \| "
    r"n_clients=(\d+) \| n_edges=(\d+) \| edge_rounds=(\d+) \| n_malicious=(\d+) \| "
    r"forced_participation=(True|False) \| arch=(\S+)")
# 第二条自描述行（config_validate.py 的 [设定2]）。**故意与 RE_SETTINGS 分开**：
# RE_SETTINGS 是全或无的 —— 把新字段塞进去，格式一旦对不上，原有八个字段会
# 一起变成 None 而日志毫无异常。分成两条，老格式的日志照样解析出前八个。
#
# 往 [设定2] 里加字段时同样的道理**又适用一次** —— 所以新字段一律写成
# **可选组** `(?: \| key=…)?`：加了字段的新日志解析得出来，`a66da67` 之前
# 的老日志（没有这个字段）原有八个字段照样解析得出来，两边都不塌。
# 反向锚点：tests/test_run_self_description.py::test_settings2_without_attack_stop_round_still_parses
RE_SETTINGS2 = re.compile(
    r"\[设定2\] malicious_per_edge=(\[[\d,]*\]|n/a) \| placement=(\S+) \| "
    r"edge_assignment=(\S+) \| local_epochs=(\d+) \| plocal_epochs=(\d+) \| "
    r"seed=(\d+|n/a) \| bd_eval_interval=(\d+|n/a) \| acc_eval_interval=(\d+|n/a)"
    r"(?: \| attack_stop_round=(\d+|n/a))?")

# 逐 edge 精度面板（server.py 的 [Acc] 行）。pm_acc 只在 PM 评估轮有，其余是 n/a。
RE_PER_EDGE_ACC = re.compile(
    r"\[Acc\] Round (\d+) \| edge(\d+) \| em_acc=([\d.]+|n/a) \| "
    r"pm_acc=([\d.]+|n/a) \| n_clients=(\d+) \| n_samples=(\d+)")

# ── 失败 / 错误 ─────────────────────────────────────────────────────────────
RE_CLIENT_FAIL = re.compile(r"\[ERROR\] Client (\d+): (.*)")
RE_EDGE_DROP = re.compile(r"\[Edge (\d+)\] (\d+) 个客户端本轮失败并被丢弃: \[([\d,\s]*)\]")
RE_ERR = re.compile(r"^(?:\w+Error|Traceback|tensorflow\.python\.framework\.errors)")


def _int_list(s: str) -> list:
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]


def _opt_str(s: str):
    """`n/a` → None。**不返回空串** —— 空串会被下游当成「有值但是空的」。"""
    return None if s == "n/a" else s


def _opt_int(s: str):
    return None if s == "n/a" else int(s)


def _opt_int_list(s: str):
    """`[10,0,0,0]` → [10,0,0,0]；`n/a` → None。

    `n/a`（本 run 不按 edge 布点）与 `[]` 是两件事，不能压成同一个值。
    """
    if s == "n/a":
        return None
    return _int_list(s.strip("[]"))


def _collect_run_info(log_text: str, lines: list) -> dict:
    cfg = RE_CFG_PATH.search(log_text)
    name = RE_RUN_NAME.search(log_text)
    val = RE_VALIDATE.search(log_text)
    setg = RE_SETTINGS.search(log_text)
    setg2 = RE_SETTINGS2.search(log_text)

    mal_ids = []
    m = RE_MAL_RESOLVED.search(log_text)
    if m:
        mal_ids = _int_list(m.group(1))
    if not mal_ids:                      # 回退：逐条 MALICIOUS 声明
        mal_ids = sorted({int(x.group(1)) for x in RE_MAL_DECLARED.finditer(log_text)})

    # n_rounds：日志里出现过的最大 [Round N]
    round_hdrs = [int(x.group(1)) for ln in lines
                  for x in [RE_ROUND_HDR.match(ln.strip())] if x]

    return {
        "config_path":   cfg.group(1) if cfg else None,
        "run_name":      name.group(1) if name else None,
        "method":        val.group(1) if val else None,
        "defense":       val.group(2) if val else None,
        "attack":        val.group(3) if val else None,
        "n_rounds":      max(round_hdrs) if round_hdrs else None,
        "malicious_ids": mal_ids,
        # 设定自描述：跑的到底是不是论文对齐的 fraction/poison —— 从此可从 artifact 核对
        "client_fraction": float(setg.group(1)) if setg else None,
        "poison_ratio":    float(setg.group(2)) if setg else None,
        "n_clients":       int(setg.group(3)) if setg else None,
        "n_edges":         int(setg.group(4)) if setg else None,
        "edge_rounds":     int(setg.group(5)) if setg else None,
        "n_malicious":     int(setg.group(6)) if setg else None,
        "forced_participation": (setg.group(7) == "True") if setg else None,
        "arch":            setg.group(8) if setg else None,
        # ── [设定2]：布点、训练量、评估网格 ──────────────────────────────
        #   `malicious_per_edge` 是 Experiment 3A 的**自变量本身**；`edge_assignment`
        #   是把 client_id 映射回 edge 的规则（回程分析要靠它把
        #   malicious_participation_by_client 聚合到 edge）；两个 eval_interval 决定
        #   评估网格粗细，下游据此判断「这一格能不能算到阈值的有效轮数」。
        #   缺 [设定2] 的老日志 → 全是 None（不是 0、不是 []）。
        "malicious_per_edge": (_opt_int_list(setg2.group(1)) if setg2 else None),
        "malicious_placement": (_opt_str(setg2.group(2)) if setg2 else None),
        "edge_assignment":   (_opt_str(setg2.group(3)) if setg2 else None),
        "local_epochs":      int(setg2.group(4)) if setg2 else None,
        "plocal_epochs":     int(setg2.group(5)) if setg2 else None,
        "seed":              (_opt_int(setg2.group(6)) if setg2 else None),
        "bd_eval_interval":  (_opt_int(setg2.group(7)) if setg2 else None),
        "acc_eval_interval": (_opt_int(setg2.group(8)) if setg2 else None),
        # 攻击退出轮（持久性协议）。三种情况都归到 None：没有 [设定2] 行、
        # 老格式没有这个字段、以及本 run 从不停止投毒（打的是 n/a）。
        # 三者都是「这一格没有退出轮」，下游读到 None 就不该去切衰减段。
        "attack_stop_round": (_opt_int(setg2.group(9)) if setg2 and setg2.group(9)
                              else None),
    }


def _collect_acc(lines: list) -> list:
    """
    按行扫描：`[Round N] Broadcasting...` 之后的第一条 `[Cloud] GM=...` 属于第 N 轮。

    不靠「第 k 条 Cloud 行 = 第 k 轮」这种隐式假设 —— run 中途崩掉或多打印一行，
    编号就会整体错位，而错位的 acc 曲线看起来完全正常。
    """
    out, cur = [], None
    for ln in lines:
        h = RE_ROUND_HDR.match(ln.strip())
        if h:
            cur = int(h.group(1))
            continue
        a = RE_ACC.search(ln)
        if a:
            t = RE_ROUND_TIME.search(ln)
            out.append({
                "round":  cur,
                "gm_acc": float(a.group(1)),
                "em_acc": float(a.group(2)),
                "pm_acc": float(a.group(3)) if a.group(3) else None,
                # 训练 + 常规评估的墙钟；不含后门评估（见 RE_ROUND_TIME 注释）。
                # 老日志没有 time= 字段时是 None，不是 0.0。
                "round_time": float(t.group(1)) if t else None,
            })
    return out


def _mean_admitted(decisions: list):
    vals = [d["admitted"] for d in decisions if d["admitted"] is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _collect_decisions(log_text: str) -> list:
    """
    收集防御的接纳/剔除判决。

    优先读统一判决行（所有防御都有）；找不到时回退到旧的 flame 专属行，
    这样老日志仍然解析得出来。坐标类防御（trimmed_mean / median）没有客户端级
    判决，记 `admitted=None` 而不是 0 —— 0 会被读成「全部剔除」。
    """
    out = []
    for m in RE_DECISION.finditer(log_text):
        out.append({"layer": m.group(1), "defense": m.group(2),
                    "admitted": int(m.group(3)), "total": int(m.group(4)),
                    "rejected": _int_list(m.group(5))})
    for m in RE_DECISION_COORD.finditer(log_text):
        out.append({"layer": m.group(1), "defense": m.group(2),
                    "admitted": None, "total": int(m.group(3)),
                    "rejected": None})
    if not out:                                  # 老日志回退
        out = [{"layer": None, "defense": m.group(1),
                "admitted": int(m.group(2)), "total": int(m.group(3)),
                "rejected": None}
               for m in RE_ADMIT_LEGACY.finditer(log_text)]
    return out


def _collect_participation(log_text: str, malicious_ids: list) -> tuple:
    """
    按 client_id 统计恶意客户端实际参与的轮次。

    **为什么不按攻击策略的打印行统计**（旧实现）：vanilla 策略
    （badnet/blended/dba）的恶意客户端跑的是普通方法的 local_train，
    打印的是 `Hier-pFedMe(...)` 这类行，没有任何「我是恶意的」标记。
    旧正则只认 Neurotoxin/Bad-PFL/CerP，于是所有 vanilla 配置恒读 0 ——
    而 0 同时也是「攻击根本没被选中」的取值，两者不可区分。
    """
    mal = set(int(i) for i in malicious_ids)
    by_client = {i: set() for i in sorted(mal)}
    for m in RE_CLIENT_ROUND.finditer(log_text):
        cid, rnd = int(m.group(1)), int(m.group(2))
        if cid in mal:
            by_client[cid].add(rnd)
    all_rounds = sorted(set().union(*by_client.values()) if by_client else set())
    return all_rounds, {str(k): sorted(v) for k, v in by_client.items()}


def _collect_per_edge_acc(log_text: str) -> dict:
    """逐 edge 精度面板：{round: [ {edge_id, em_acc, pm_acc, n_clients, n_samples} ]}。

    `pm_acc` 只在 PM 评估轮有值，其余轮是 `None`（**不是 0**，陷阱 #13）。
    """
    by_round = {}
    for m in RE_PER_EDGE_ACC.finditer(log_text):
        rnd = int(m.group(1))
        by_round.setdefault(rnd, []).append({
            "edge_id":   int(m.group(2)),
            "em_acc":    _opt(m.group(3)),
            "pm_acc":    _opt(m.group(4)),
            "n_clients": int(m.group(5)),
            "n_samples": int(m.group(6)),
        })
    for rnd in by_round:
        by_round[rnd].sort(key=lambda d: d["edge_id"])
    return by_round


def _collect_per_edge(log_text: str) -> dict:
    """
    逐 edge 面板：{round: [ {edge_id, edge_asr, client_benign, client_malicious,
    n_benign, n_malicious, has_malicious} ]}，按 edge_id 升序。
    """
    by_round = {}
    for m in RE_PER_EDGE.finditer(log_text):
        rnd = int(m.group(1))
        by_round.setdefault(rnd, []).append({
            "edge_id":          int(m.group(2)),
            "edge_asr":         _opt(m.group(3)),
            "client_benign":    _opt(m.group(4)),
            "client_malicious": _opt(m.group(5)),
            "n_benign":         int(m.group(6)),
            "n_malicious":      int(m.group(7)),
            "has_malicious":    m.group(8) == "True",
        })
    return {r: sorted(v, key=lambda d: d["edge_id"]) for r, v in by_round.items()}


def _opt(s):
    """
    "n/a" → None（写进 JSON 就是 null）。

    绝不返回 0.0：无定义的分组与「后门完全没传过去」在数值上无法区分，
    而后者是个强结论。上游打印见 server/backdoor_server.py 的 _f3。
    """
    return None if s == "n/a" else float(s)


def _f(s):
    return float("nan") if s == "nan" else float(s)


def _collect_drift(log_text: str) -> list:
    """[{round, param_abs, param_rel, repr_mean, repr_median, n_edges}]，按 round 升序。"""
    out = []
    for m in RE_DRIFT.finditer(log_text):
        out.append({"round": int(m.group(1)),
                    "param_abs": _f(m.group(2)), "param_rel": _f(m.group(3)),
                    "repr_mean": _f(m.group(4)), "repr_median": _f(m.group(5)),
                    "n_edges": int(m.group(6))})
    return sorted(out, key=lambda d: d["round"])


def _collect_timing(log_text: str) -> list:
    """[{round, asr_s, feature_s, forgetting_s, drift_s, total_s}]，按 round 升序。"""
    out = []
    for m in RE_TIMING.finditer(log_text):
        out.append({"round": int(m.group(1)),
                    "asr_s": _opt(m.group(2)), "feature_s": _opt(m.group(3)),
                    "forgetting_s": _opt(m.group(4)), "drift_s": _opt(m.group(5)),
                    "total_s": _opt(m.group(6))})
    return sorted(out, key=lambda d: d["round"])


def _timing_summary(acc_rounds: list, timing_rounds: list) -> dict:
    """墙钟拆分：训练+常规评估 vs 后门评估。

    这是标定 (local_epochs, n_rounds, eval_interval) 时唯一要读的数。
    两个来源的口径不同、必须分开累加：
      - `round_time`  每轮都有，来自 [Cloud] 行，**不含**后门评估
      - `total_s`     只在 bd_eval_interval 轮有，来自 [Timing] 行

    任何一侧无数据就留 `None` —— 0.0 会被读成「不花时间」（铁律：无定义的
    指标留空）。`bd_eval_fraction` 只在两侧都有数时才给。
    """
    rt = [a["round_time"] for a in acc_rounds if a.get("round_time") is not None]
    bd = [t["total_s"] for t in timing_rounds if t["total_s"] is not None]
    train_s = round(sum(rt), 1) if rt else None
    bd_s    = round(sum(bd), 1) if bd else None
    frac = (round(bd_s / (train_s + bd_s), 4)
            if (train_s is not None and bd_s is not None and train_s + bd_s > 0)
            else None)

    def _phase(key):
        xs = [t[key] for t in timing_rounds if t[key] is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    return {
        "n_rounds_timed":        len(rt),
        "n_bd_evals":            len(bd),
        "round_time_total_s":    train_s,     # 训练 + GM/EM/PM 评估
        "bd_eval_total_s":       bd_s,        # 后门评估（ASR/特征/遗忘/漂移）
        "wall_total_s":          (round(train_s + bd_s, 1)
                                  if train_s is not None and bd_s is not None
                                  else None),
        "bd_eval_fraction":      frac,
        "round_time_mean_s":     round(train_s / len(rt), 2) if rt else None,
        "bd_eval_mean_s":        round(bd_s / len(bd), 2) if bd else None,
        "bd_phase_mean_s": {
            "asr":        _phase("asr_s"),
            "feature":    _phase("feature_s"),
            "forgetting": _phase("forgetting_s"),
            "drift":      _phase("drift_s"),
        },
    }


def collect(log_text: str) -> dict:
    lines = log_text.splitlines()

    rounds = [{
        "round":               int(m.group(1)),
        "global_asr":          _opt(m.group(2)),
        "edge_asr":            _opt(m.group(3)),
        "local_benign_asr":    _opt(m.group(4)),
        "same_edge_asr":       _opt(m.group(5)),
        "diff_edge_asr":       _opt(m.group(6)),
        "local_malicious_asr": _opt(m.group(7)),
    } for m in RE_BD.finditer(log_text)]

    run = _collect_run_info(log_text, lines)
    acc_rounds = _collect_acc(lines)

    final_pm = RE_FINAL_PM.search(log_text)
    final_acc = dict(acc_rounds[-1]) if acc_rounds else {}
    final_acc["final_pm_weighted"] = float(final_pm.group(1)) if final_pm else None

    admitted = _collect_decisions(log_text)

    mal_rounds, by_client = _collect_participation(log_text, run["malicious_ids"])

    client_failures = [{"client_id": int(m.group(1)), "error": m.group(2).strip()[:200]}
                       for m in RE_CLIENT_FAIL.finditer(log_text)]
    for m in RE_EDGE_DROP.finditer(log_text):
        client_failures.append({"edge_id": int(m.group(1)),
                                "dropped": _int_list(m.group(3))})

    errors = [ln.strip() for ln in lines if RE_ERR.match(ln.strip())][:5]

    per_edge_rounds = _collect_per_edge(log_text)
    per_edge_final = (per_edge_rounds[max(per_edge_rounds)]
                      if per_edge_rounds else [])
    drift_rounds = _collect_drift(log_text)
    timing_rounds = _collect_timing(log_text)
    per_edge_acc_rounds = _collect_per_edge_acc(log_text)
    # 末轮快照取**有 pm_acc 的最后一轮**：em_acc 每轮都有，pm_acc 只在评估轮有，
    # 直接取 max(round) 往往落在一个 pm_acc 全是 None 的轮上。
    _pm_rounds = [r for r, v in per_edge_acc_rounds.items()
                  if any(e["pm_acc"] is not None for e in v)]
    per_edge_acc_final = (per_edge_acc_rounds[max(_pm_rounds)] if _pm_rounds
                          else (per_edge_acc_rounds[max(per_edge_acc_rounds)]
                                if per_edge_acc_rounds else []))

    return {
        "run": run,
        "rounds": rounds,
        "final": rounds[-1] if rounds else None,
        # 逐 edge 传播路径（Experiment 3）：{round: [per-edge...]} + 末轮快照
        "per_edge_rounds": per_edge_rounds,
        "per_edge_final": per_edge_final,
        # 漂移曲线（Experiment 3C）：drift_eval=true 才有；否则空
        "drift_rounds": drift_rounds,
        "drift_final": drift_rounds[-1] if drift_rounds else None,
        "acc_rounds": acc_rounds,
        "final_acc": final_acc,
        # 逐 edge 精度（后门的干净精度代价是逐 edge 的，全局均值会把它摊平）
        "per_edge_acc_rounds": per_edge_acc_rounds,
        "per_edge_acc_final": per_edge_acc_final,
        # 墙钟拆分（标定用）：round_time 不含后门评估，两者要相加，见 RE_ROUND_TIME
        "timing_rounds": timing_rounds,
        "timing_summary": _timing_summary(acc_rounds, timing_rounds),
        "admitted": admitted,
        # 只对有客户端级判决的防御求均值；坐标类（admitted=None）不参与，
        # 全是坐标类或无防御时结果是 None —— 0 会被误读成「全部被剔除」。
        "admitted_count_mean": _mean_admitted(admitted),
        # 被防御剔除过的 client_id 汇总（TPR/FPR 的原料）
        "rejected_ids": sorted({i for a in admitted for i in (a["rejected"] or [])}),
        "malicious_selected_rounds": mal_rounds,
        "n_malicious_participations": len(mal_rounds),
        "malicious_participation_by_client": by_client,
        "client_failures": client_failures[:20],
        "errors": errors,
        "log_tail": lines[-40:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="集群 run 的日志文件")
    ap.add_argument("-o", "--out", required=True, help="输出 metrics.json 路径")
    args = ap.parse_args()

    text = Path(args.log).read_text(errors="replace")
    metrics = collect(text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    run, f, fa = metrics["run"], metrics["final"], metrics["final_acc"]
    print(f"[collect] {out}  ({out.stat().st_size / 1024:.1f} KB)")
    print(f"[collect] run: config={run['config_path']} method={run['method']} "
          f"attack={run['attack']} defense={run['defense']} n_rounds={run['n_rounds']}")
    print(f"[collect] 设定: client_fraction={run['client_fraction']} "
          f"poison_ratio={run['poison_ratio']} n_clients={run['n_clients']} "
          f"n_edges={run['n_edges']} edge_rounds={run['edge_rounds']} "
          f"n_malicious={run['n_malicious']} "
          f"forced_participation={run['forced_participation']} arch={run['arch']}")
    if f:
        print(f"[collect] final round {f['round']}: GM_ASR={f['global_asr']:.3f} "
              f"local_benign={f['local_benign_asr']:.3f} "
              f"local_malicious={f['local_malicious_asr']:.3f}")
    else:
        print("[collect] 日志里没有后门评估行 —— 攻击可能根本没跑起来", file=sys.stderr)
    if fa:
        print(f"[collect] final acc: GM={fa.get('gm_acc')} EM={fa.get('em_acc')} "
              f"PM={fa.get('pm_acc')} final_PM_weighted={fa.get('final_pm_weighted')}")

    # 参与度：这是「接线是否生效」最直接的一个数，和 n_rounds 一起打出来才有意义
    n_part, n_rounds = metrics["n_malicious_participations"], run["n_rounds"]
    if run["malicious_ids"]:
        flag = "" if (n_rounds and n_part == n_rounds) else "  ← 与 n_rounds 不符，请核对 Q"
        print(f"[collect] malicious {run['malicious_ids']} 参与 {n_part}/{n_rounds} 轮{flag}")
        # 注意：恶意端每轮都参与在本仓库是**预期**的 —— 集群的 select_clients 强制
        # 把恶意端固定选进每一轮（force-inclusion），所以 frac<1 也全参与，不是配置漂移。
        # （曾在此处按「均匀抽样」误判成 fraction≈1.0，被 r47/r79 原始日志证伪，已删。）
    if metrics["client_failures"]:
        print(f"[collect] ⚠ {len(metrics['client_failures'])} 条客户端失败/丢弃记录，"
              f"见 json.client_failures", file=sys.stderr)
    if metrics["errors"]:
        print(f"[collect] {len(metrics['errors'])} 个错误行，见 json.errors", file=sys.stderr)


if __name__ == "__main__":
    main()
