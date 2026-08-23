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
RE_BD = re.compile(
    r"\[Backdoor\] Round (\d+) \| GM_ASR=([\d.]+) \| EM_ASR=([\d.]+) \| "
    r"local_benign=([\d.]+) \(same_edge=([\d.]+), diff_edge=([\d.]+)\) \| "
    r"local_malicious=([\d.]+)")

# ── 准确率（PM 只在 eval_interval 轮出现，故可选）─────────────────────────────
RE_ACC = re.compile(
    r"\[Cloud\] GM=([\d.]+) \| EM=([\d.]+)(?: PM=([\d.]+))? \|")
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
    r"\[Backdoor\] Round (\d+) \| edge(\d+) \| edge_asr=([\d.]+) \| "
    r"client_benign=([\d.]+) \| client_malicious=([\d.]+) \| "
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
    r"n_clients=(\d+) \| n_edges=(\d+) \| n_malicious=(\d+) \| "
    r"forced_participation=(True|False) \| arch=(\S+)")

# ── 失败 / 错误 ─────────────────────────────────────────────────────────────
RE_CLIENT_FAIL = re.compile(r"\[ERROR\] Client (\d+): (.*)")
RE_EDGE_DROP = re.compile(r"\[Edge (\d+)\] (\d+) 个客户端本轮失败并被丢弃: \[([\d,\s]*)\]")
RE_ERR = re.compile(r"^(?:\w+Error|Traceback|tensorflow\.python\.framework\.errors)")


def _int_list(s: str) -> list:
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]


def _collect_run_info(log_text: str, lines: list) -> dict:
    cfg = RE_CFG_PATH.search(log_text)
    name = RE_RUN_NAME.search(log_text)
    val = RE_VALIDATE.search(log_text)
    setg = RE_SETTINGS.search(log_text)

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
        "n_malicious":     int(setg.group(5)) if setg else None,
        "forced_participation": (setg.group(6) == "True") if setg else None,
        "arch":            setg.group(7) if setg else None,
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
            out.append({
                "round":  cur,
                "gm_acc": float(a.group(1)),
                "em_acc": float(a.group(2)),
                "pm_acc": float(a.group(3)) if a.group(3) else None,
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
            "edge_asr":         float(m.group(3)),
            "client_benign":    float(m.group(4)),
            "client_malicious": float(m.group(5)),
            "n_benign":         int(m.group(6)),
            "n_malicious":      int(m.group(7)),
            "has_malicious":    m.group(8) == "True",
        })
    return {r: sorted(v, key=lambda d: d["edge_id"]) for r, v in by_round.items()}


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


def collect(log_text: str) -> dict:
    lines = log_text.splitlines()

    rounds = [{
        "round":               int(m.group(1)),
        "global_asr":          float(m.group(2)),
        "edge_asr":            float(m.group(3)),
        "local_benign_asr":    float(m.group(4)),
        "same_edge_asr":       float(m.group(5)),
        "diff_edge_asr":       float(m.group(6)),
        "local_malicious_asr": float(m.group(7)),
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
          f"n_edges={run['n_edges']} n_malicious={run['n_malicious']} "
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
