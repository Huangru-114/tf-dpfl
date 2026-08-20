"""
harness/collect_matrix.py  —  把矩阵里所有格子的 metrics.json 汇总成一张表

跑完矩阵后在本地（或集群）执行：
    python3 harness/collect_matrix.py
    python3 harness/collect_matrix.py --csv experiments/matrix/summary.csv

输出：
  - 终端一张按 (attack, method, defense) 排列的表
  - 可选 CSV，便于画图 / 进论文
  - **缺格清单**：哪些格子还没跑 / 跑挂了 —— 这比表本身更重要，
    它回答「我能不能开始写结论」。

只读小 json，不碰任何大产物。
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX_DIR = ROOT / "experiments" / "matrix"


def load_cells(matrix_dir: Path):
    cells, broken = [], []
    for path in sorted(matrix_dir.glob("*.metrics.json")):
        try:
            d = json.loads(path.read_text())
        except Exception as e:
            broken.append((path.name, f"json 解析失败: {e}"))
            continue
        meta = d.get("cell")
        if not meta:
            broken.append((path.name, "缺 cell 字段（不是矩阵产物？）"))
            continue
        final = d.get("final") or {}
        cells.append({
            "attack":  meta.get("attack"),
            "method":  meta.get("method"),
            "defense": meta.get("defense"),
            "seed":    meta.get("seed"),
            "exit_code": meta.get("exit_code"),
            "global_asr":       final.get("global_asr"),
            "local_benign_asr": final.get("local_benign_asr"),
            "local_malicious_asr": final.get("local_malicious_asr"),
            "rounds_evaluated": len(d.get("rounds") or []),
            "admitted_mean":   d.get("admitted_count_mean"),
            "mal_participations": d.get("n_malicious_participations"),
            "n_errors": len(d.get("errors") or []),
        })
    return cells, broken


def _fmt(v, nd=3):
    if v is None:
        return "  —  "
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def summarize(cells):
    """按 (attack, method, defense) 聚合多个 seed，报均值和格子数。"""
    groups = defaultdict(list)
    for c in cells:
        groups[(c["attack"], c["method"], c["defense"])].append(c)

    rows = []
    for key, gs in sorted(groups.items()):
        ok = [g for g in gs if g["exit_code"] == 0]
        def mean(field):
            vals = [g[field] for g in ok if g[field] is not None]
            return sum(vals) / len(vals) if vals else None
        rows.append({
            "attack": key[0], "method": key[1], "defense": key[2],
            "n_seeds": len(gs), "n_ok": len(ok),
            "global_asr": mean("global_asr"),
            "local_benign_asr": mean("local_benign_asr"),
            "local_malicious_asr": mean("local_malicious_asr"),
            "admitted_mean": mean("admitted_mean"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(MATRIX_DIR))
    ap.add_argument("--csv", default=None, help="同时写出 CSV")
    args = ap.parse_args()

    matrix_dir = Path(args.dir)
    if not matrix_dir.is_dir():
        print(f"[collect_matrix] 目录不存在：{matrix_dir}")
        return

    cells, broken = load_cells(matrix_dir)
    if not cells:
        print(f"[collect_matrix] {matrix_dir} 下没有可用的 metrics.json")
        return

    rows = summarize(cells)

    hdr = (f"{'attack':<12}{'method':<18}{'defense':<14}"
           f"{'seeds':>6}{'ok':>4}{'GM_ASR':>9}{'benign':>9}{'malic':>9}{'admit':>8}")
    print("\n" + hdr)
    print("─" * len(hdr))
    for r in rows:
        print(f"{r['attack']:<12}{r['method']:<18}{r['defense']:<14}"
              f"{r['n_seeds']:>6}{r['n_ok']:>4}"
              f"{_fmt(r['global_asr']):>9}{_fmt(r['local_benign_asr']):>9}"
              f"{_fmt(r['local_malicious_asr']):>9}{_fmt(r['admitted_mean'], 1):>8}")

    # ── 缺格 / 失败清单（比表更重要）──────────────────────────────────────
    failed = [c for c in cells if c["exit_code"] != 0]
    no_eval = [c for c in cells if c["exit_code"] == 0 and c["rounds_evaluated"] == 0]

    print(f"\n共 {len(cells)} 个格子，{len(rows)} 个 (attack,method,defense) 组合。")
    if failed:
        print(f"\n⚠ {len(failed)} 个格子退出码非 0：")
        for c in failed[:15]:
            print(f"   {c['attack']}/{c['method']}/{c['defense']}/seed{c['seed']} "
                  f"exit={c['exit_code']} errors={c['n_errors']}")
    if no_eval:
        print(f"\n⚠ {len(no_eval)} 个格子跑完但**没有任何后门评估行** "
              f"（攻击可能没挂上）：")
        for c in no_eval[:15]:
            print(f"   {c['attack']}/{c['method']}/{c['defense']}/seed{c['seed']}")
    if broken:
        print(f"\n⚠ {len(broken)} 个文件无法解析：")
        for name, why in broken[:10]:
            print(f"   {name}: {why}")
    if not (failed or no_eval or broken):
        print("所有格子均正常完成。")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[collect_matrix] CSV → {out}")


if __name__ == "__main__":
    main()
