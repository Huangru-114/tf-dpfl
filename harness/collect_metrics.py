"""
harness/collect_metrics.py  —  把集群 run 的**全量日志**压成一个小 metrics.json

回程协议：GB 级日志永远留在集群，只有这个 KB 级 json 进 git / 贴给 Claude Code。

用法（集群侧）：
    python harness/collect_metrics.py run.log -o experiments/attack/neurotoxin/exp001.metrics.json

输出字段：
    rounds[]              每个评估轮的 {round, global_asr, edge_asr, local_benign_asr,
                                       local_malicious_asr, same_edge, diff_edge}
    final                 最后一轮的上述指标
    admitted[]            FLAME/鲁棒聚合每次接纳数 {admitted, total}
    malicious_rounds      恶意客户端实际被选中的轮次（验接线是否生效）
    errors[]              traceback 首行（最多 5 条）
    log_tail              日志最后 40 行（截断版，供人工判读）
"""

import argparse
import json
import re
import sys
from pathlib import Path

RE_BD = re.compile(
    r"\[Backdoor\] Round (\d+) \| GM_ASR=([\d.]+) \| EM_ASR=([\d.]+) \| "
    r"local_benign=([\d.]+) \(same_edge=([\d.]+), diff_edge=([\d.]+)\) \| "
    r"local_malicious=([\d.]+)")
RE_ADMIT = re.compile(r"\[Defense:(\w+)\] admitted (\d+)/(\d+)")
RE_MAL = re.compile(r"\[Client\s*(\d+)\] Round (\d+) \| (Neurotoxin|Bad-PFL|CerP)")
RE_ERR = re.compile(r"^(?:\w+Error|Traceback|tensorflow\.python\.framework\.errors)")


def collect(log_text: str) -> dict:
    rounds = []
    for m in RE_BD.finditer(log_text):
        rounds.append({
            "round":                int(m.group(1)),
            "global_asr":           float(m.group(2)),
            "edge_asr":             float(m.group(3)),
            "local_benign_asr":     float(m.group(4)),
            "same_edge_asr":        float(m.group(5)),
            "diff_edge_asr":        float(m.group(6)),
            "local_malicious_asr":  float(m.group(7)),
        })

    admitted = [{"defense": m.group(1), "admitted": int(m.group(2)),
                 "total": int(m.group(3))} for m in RE_ADMIT.finditer(log_text)]

    malicious_rounds = sorted({int(m.group(2)) for m in RE_MAL.finditer(log_text)})

    lines = log_text.splitlines()
    errors = [ln.strip() for ln in lines if RE_ERR.match(ln.strip())][:5]

    return {
        "rounds": rounds,
        "final": rounds[-1] if rounds else None,
        "admitted": admitted,
        "admitted_count_mean": (
            round(sum(a["admitted"] for a in admitted) / len(admitted), 2)
            if admitted else None),
        "malicious_selected_rounds": malicious_rounds,
        "n_malicious_participations": len(malicious_rounds),
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

    f = metrics["final"]
    print(f"[collect] {out}  ({out.stat().st_size / 1024:.1f} KB)")
    if f:
        print(f"[collect] final round {f['round']}: GM_ASR={f['global_asr']:.3f} "
              f"local_benign={f['local_benign_asr']:.3f} "
              f"local_malicious={f['local_malicious_asr']:.3f}")
    else:
        print("[collect] 日志里没有后门评估行 —— 攻击可能根本没跑起来", file=sys.stderr)
    if metrics["errors"]:
        print(f"[collect] {len(metrics['errors'])} 个错误行，见 json.errors", file=sys.stderr)


if __name__ == "__main__":
    main()
