"""
harness/registry.py  —  实验登记表：**声明**每个 run 应该是什么

为什么需要它（experiments/attack/hfl-mechanism/FINDINGS.md F-001 / F-010 / F-011）：
  旧方案里一格的身份只靠文件名，没有任何东西核对「文件名说的」和「实际跑的」。
  6 个 `def_*` 格子文件名写着防御，实际跑的是 none，事后读表的人无从知道。
  登记表把「应该是什么」写下来，`harness/status.py` 拿它逐格去对 metrics.json
  的 run 块（「实际是什么」）。

登记表格式（YAML）：

    study: hfl-mechanism
    protocol: P2                  # 这些 run 属于哪个口径版本（fedavg/utils/provenance.py）
    layout: nested                # nested: results/<protocol>/<group>/<run_id>.metrics.json
                                  # flat:   results/<run_id>.metrics.json（旧方案）
    results_dir: results          # 相对登记表所在目录
    configs_dir: configs          # materialize 的输出目录（nested 才用）
    base: base.yaml               # materialize 的基配置；null = 还没定（materialize 拒绝）
    overlays: [../x/tpl.yaml]     # 可选：按顺序深合并到 base 上（在组/格子的 set 之前）。
                                  # A4：对齐模板 fedavg/config/alignment_p2.yaml 就是这样叠上去的
    run_id_format: "{group}__{cell}__s{seed}"   # 可选，这是默认值
    audit: AUDIT.md               # requires 里写 audit 时，门槛看这个文件
    available: [S3]               # 已经实现的功能会话；requires 里其余的 token 对它检查
    intended_cli:                 # 可选：提交脚本打算在命令行上加的值（旧方案 sbatch 固定传的）
      backdoor.malicious_strategy: badpfl
    groups:
      G2:
        purpose: "..."
        requires: [audit, S5]
        seeds: [42, 43]
        set: {federation.client_fraction: 0.1}       # 组内共用的覆盖
        factors:                                      # 笛卡儿积，cell 名 = 各因素 label 用 "_" 连接
          topology:
            - {label: e2, set: {federation.n_edges: 2}}
        cells:                                        # 显式格子（与 factors 二选一或并用）
          - {cell: flat, set: {federation.n_edges: 1}}
          - {cell: 2edge_collocated, config: 2edge_collocated.yaml}   # 旧方案：直接指向 yaml

纯标准库 + PyYAML，本地与容器里都能跑，不 import TF。
"""

from __future__ import annotations

import copy
import itertools
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))
from utils.provenance import config_sha   # noqa: E402  —— 与 main.py 算的是同一个 hash

DEFAULT_RUN_ID_FORMAT = "{group}__{cell}__s{seed}"

# 登记表里声明的配置 → metrics.json run 块的对应字段。status 逐项核对这些。
# 值是 (run 块键, 配置的点号路径)。seed 单独处理（它来自 run 的 seed，不来自配置）。
EXPECT_KEYS = (
    ("method",             "training.drift_correction"),
    ("defense",            "defense.name"),
    ("attack",             "backdoor.malicious_strategy"),
    ("n_clients",          "federation.n_clients"),
    ("n_edges",            "federation.n_edges"),
    ("edge_rounds",        "federation.edge_rounds"),
    ("client_fraction",    "federation.client_fraction"),
    ("poison_ratio",       "backdoor.poison_ratio"),
    ("malicious_per_edge", "backdoor.malicious_per_edge"),
    ("local_epochs",       "training.local_epochs"),
)

AUDIT_STATUSES = ("open", "align", "deviate", "done")
AUDIT_CLOSED = ("deviate", "done")
_AUDIT_ROW = re.compile(r"^\|\s*([AD]\d{2})\s*\|")
_AUDIT_CELL = re.compile(r"^`(" + "|".join(AUDIT_STATUSES) + r")`$")


class RegistryError(ValueError):
    pass


# ══════════════════════════════════════════════════════════════════════════
# 小工具
# ══════════════════════════════════════════════════════════════════════════

def get_dotted(d: dict, path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def set_dotted(d: dict, path: str, value):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def deep_merge(dst: dict, src: dict) -> dict:
    """把 src 递归并进 dst（就地）：dict 对 dict 往下合并，其余一律以 src 为准。"""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _slug(s) -> str:
    """组名 / 格子名 / 因素 label：只许字母数字 . _ -，且不许出现 `__`
    （默认 run_id 用 `__` 分隔组、格子、seed，出现在名字里就拆不回来了）。"""
    s = str(s)
    if not re.fullmatch(r"[A-Za-z0-9._\-]+", s) or "__" in s:
        raise RegistryError(f"名字只能含字母数字 . _ -，且不能含 '__'：{s!r}")
    return s


# ══════════════════════════════════════════════════════════════════════════
# 审计门槛
# ══════════════════════════════════════════════════════════════════════════

def audit_rows(audit_md: Path) -> dict:
    """AUDIT.md → {row_id: status}。每一行 A##/D## 必须恰好有一个状态格。"""
    rows = {}
    for ln in Path(audit_md).read_text(encoding="utf-8").splitlines():
        m = _AUDIT_ROW.match(ln)
        if not m:
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        found = [c for c in cells if _AUDIT_CELL.match(c)]
        if len(found) != 1:
            raise RegistryError(
                f"AUDIT 行 {m.group(1)} 应当恰好有一个状态格 "
                f"（{'/'.join('`'+s+'`' for s in AUDIT_STATUSES)}），实际 {len(found)} 个")
        if m.group(1) in rows:
            raise RegistryError(f"AUDIT 行 {m.group(1)} 重复出现")
        rows[m.group(1)] = found[0].strip("`")
    if not rows:
        raise RegistryError(f"{audit_md} 里没有找到任何 A##/D## 行 —— 格式变了？")
    return rows


def audit_open_rows(audit_md: Path) -> list:
    """还没关闭的行（状态不是 done / deviate）。空列表 = 门槛放行。"""
    return sorted(k for k, v in audit_rows(audit_md).items() if v not in AUDIT_CLOSED)


# ══════════════════════════════════════════════════════════════════════════
# 登记表加载与展开
# ══════════════════════════════════════════════════════════════════════════

class Registry:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.dir = self.path.parent
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self.raw = raw
        self.study = raw.get("study") or self.dir.name
        self.protocol = raw.get("protocol")
        self.layout = raw.get("layout", "nested")
        if self.layout not in ("nested", "flat"):
            raise RegistryError(f"layout 只能是 nested / flat：{self.layout!r}")
        self.results_dir = (self.dir / raw.get("results_dir", "results")).resolve()
        self.configs_dir = (self.dir / raw.get("configs_dir", "configs")).resolve()
        self.base = raw.get("base")
        self.overlays = list(raw.get("overlays") or [])
        self.run_id_format = raw.get("run_id_format", DEFAULT_RUN_ID_FORMAT)
        self.audit = (self.dir / raw["audit"]) if raw.get("audit") else None
        self.available = set(raw.get("available") or [])
        self.intended_cli = dict(raw.get("intended_cli") or {})
        self.groups = raw.get("groups") or {}
        if not self.groups:
            raise RegistryError(f"{self.path} 没有 groups")

    # ── 门槛 ──────────────────────────────────────────────────────────────
    def unmet_requires(self, group: str) -> list:
        """这个组还缺什么。audit → 列出未关闭的 AUDIT 行；其余 token 对 available 检查。"""
        out = []
        for tok in self.groups[group].get("requires") or []:
            if tok == "audit":
                if self.audit is None:
                    raise RegistryError(f"组 {group} requires audit，但登记表没写 audit 文件")
                open_rows = audit_open_rows(self.audit)
                if open_rows:
                    out.append(f"audit({len(open_rows)} 行未关闭)")
            elif tok not in self.available:
                out.append(tok)
        return out

    # ── 展开 ──────────────────────────────────────────────────────────────
    def cells(self, group: str) -> list:
        """组 → [{cell, set, config}]，顺序确定。"""
        g = self.groups[group]
        out = []
        factors = g.get("factors") or {}
        if factors:
            names = list(factors)
            for combo in itertools.product(*(factors[n] for n in names)):
                st = {}
                for v in combo:
                    st.update(v.get("set") or {})
                out.append({"cell": "_".join(_slug(v["label"]) for v in combo),
                            "set": st, "config": None})
        for c in g.get("cells") or []:
            cell = c.get("cell") or (Path(c["config"]).stem if c.get("config") else None)
            if cell is None:
                raise RegistryError(f"组 {group} 有一个格子既没有 cell 也没有 config")
            out.append({"cell": _slug(cell), "set": dict(c.get("set") or {}),
                        "config": c.get("config")})
        seen = set()
        for c in out:
            if c["cell"] in seen:
                raise RegistryError(f"组 {group} 里格子名重复：{c['cell']}")
            seen.add(c["cell"])
        return out

    def runs(self) -> list:
        """全部 run 的声明，按 登记表顺序 × 格子顺序 × seed 顺序。"""
        out, ids = [], set()
        for group, g in self.groups.items():
            _slug(group)
            seeds = g.get("seeds")
            if not seeds:
                raise RegistryError(f"组 {group} 没写 seeds")
            for c in self.cells(group):
                for seed in seeds:
                    run_id = self.run_id_format.format(group=group, cell=c["cell"],
                                                       seed=seed)
                    if run_id in ids:
                        raise RegistryError(f"run_id 重复：{run_id}")
                    ids.add(run_id)
                    out.append({"run_id": run_id, "group": group, "cell": c["cell"],
                                "seed": int(seed), "set": {**(g.get("set") or {}), **c["set"]},
                                "config": c["config"]})
        return out

    # ── 路径 ──────────────────────────────────────────────────────────────
    def metrics_path(self, run: dict) -> Path:
        if self.layout == "flat":
            return self.results_dir / f"{run['run_id']}.metrics.json"
        return self.results_dir / str(self.protocol) / run["group"] / f"{run['run_id']}.metrics.json"

    def config_path(self, run: dict) -> Path:
        if run.get("config"):
            return (self.dir / run["config"]).resolve()
        return self.configs_dir / f"{run['run_id']}.yaml"

    # ── 配置 ──────────────────────────────────────────────────────────────
    def declared_config(self, run: dict) -> dict:
        """这个 run 的 yaml 原文应当是什么（= main.py 算 config_sha 的那个 dict）。

        旧方案（cell 指向现成 yaml）：直接读那个文件。
        新方案：base + 组/格子的 set + seed + meta 块。
        """
        if run.get("config"):
            path = (self.dir / run["config"]).resolve()
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        if not self.base:
            raise RegistryError(
                f"{self.path.name} 的 base 还没定（base: null）—— 按计划在 A4 之后才有；"
                f"现在不能生成 {run['run_id']} 的配置")
        cfg = copy.deepcopy(yaml.safe_load((self.dir / self.base).read_text(encoding="utf-8")))
        for ov in self.overlays:
            path = (self.dir / ov).resolve()
            if not path.is_file():
                raise RegistryError(f"{self.path.name} 的 overlay 不存在：{ov}")
            deep_merge(cfg, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        for k, v in run["set"].items():
            set_dotted(cfg, k, copy.deepcopy(v))
        cfg["seed"] = run["seed"]
        cfg["meta"] = {"study": self.study, "group": run["group"], "run_id": run["run_id"],
                       "protocol": self.protocol}
        return cfg

    def intended_config(self, run: dict) -> dict:
        """声明 + 提交脚本打算加的 CLI 值 + seed —— 这是「应该实际跑成的样子」。"""
        cfg = copy.deepcopy(self.declared_config(run))
        for k, v in self.intended_cli.items():
            set_dotted(cfg, k, v)
        cfg["seed"] = run["seed"]
        return cfg

    def expected_run_block(self, run: dict) -> dict:
        cfg = self.intended_config(run)
        exp = {rk: get_dotted(cfg, ck) for rk, ck in EXPECT_KEYS}
        exp["seed"] = run["seed"]
        return exp


# ══════════════════════════════════════════════════════════════════════════
# materialize：新方案把每个 run 的完整配置写出来 + INDEX.tsv
# ══════════════════════════════════════════════════════════════════════════

INDEX_COLUMNS = ("run_id", "group", "cell", "seed", "config_sha", "config", "metrics")

_GENERATED_HEADER = (
    "# GENERATED by harness/registry.py —— 不要手改。改 registry.yaml / base 之后重新 materialize。\n"
)


def _rel_to_root(p: Path) -> str:
    """仓库内的路径写成相对仓库根（集群上 submit.sh 拼 $ROOT/<path>）；仓库外（测试）写绝对路径。"""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def materialize(reg: Registry, groups=None) -> list:
    """写出 configs/<run_id>.yaml 与 configs/INDEX.tsv；返回写出的行。

    - 只处理 requires 里**功能会话**都已具备的组（audit 门槛由 submit.sh 在提交时再查，
      这样审计关闭前也能先把配置生成出来看）。
    - config_sha 是对**重新读回来**的 dict 算的：保证与 main.py 读同一个文件算出的相等。
    """
    if reg.layout != "nested":
        raise RegistryError("materialize 只用于 nested 布局（新方案）；旧方案的配置已经是现成 yaml")
    eligible, skipped = [], {}
    for run in reg.runs():
        if groups and run["group"] not in groups:
            continue
        missing = [t for t in reg.unmet_requires(run["group"]) if not t.startswith("audit")]
        if missing:
            skipped[run["group"]] = missing
            continue
        eligible.append(run)
    if not eligible:
        why = "; ".join(f"{g} 缺 {', '.join(m)}" for g, m in skipped.items()) or "没有匹配的组"
        raise RegistryError(f"没有可以生成配置的组（{why}）—— 不写 INDEX.tsv")
    configs = [(run, reg.declared_config(run)) for run in eligible]   # base 未定会在这里报错
    reg.configs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, cfg in configs:
        path = reg.config_path(run)
        path.write_text(_GENERATED_HEADER + yaml.safe_dump(cfg, sort_keys=False,
                                                          allow_unicode=True),
                        encoding="utf-8")
        reloaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        rows.append({"run_id": run["run_id"], "group": run["group"], "cell": run["cell"],
                     "seed": run["seed"], "config_sha": config_sha(reloaded),
                     "config": _rel_to_root(path),
                     "metrics": _rel_to_root(reg.metrics_path(run))})
    index = reg.configs_dir / "INDEX.tsv"
    with index.open("w", encoding="utf-8") as f:
        f.write("\t".join(INDEX_COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in INDEX_COLUMNS) + "\n")
    return rows


def read_index(reg: Registry) -> dict:
    """INDEX.tsv → {run_id: row}；没有就是 {}。"""
    index = reg.configs_dir / "INDEX.tsv"
    if not index.exists():
        return {}
    lines = index.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    return {r["run_id"]: r for r in (dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="展开 / materialize 实验登记表")
    ap.add_argument("registry")
    ap.add_argument("--materialize", action="store_true", help="写出 configs/ 与 INDEX.tsv")
    ap.add_argument("--group", action="append", help="只处理这些组（可重复）")
    args = ap.parse_args(argv)
    reg = Registry(args.registry)
    if args.materialize:
        try:
            rows = materialize(reg, groups=args.group)
        except RegistryError as e:
            print(f"[registry] 未生成：{e}")
            return 2
        print(f"[registry] 写出 {len(rows)} 个配置 → {reg.configs_dir}")
        return 0
    runs = reg.runs()
    by_group = {}
    for r in runs:
        by_group.setdefault(r["group"], []).append(r)
    for g, rs in by_group.items():
        miss = reg.unmet_requires(g)
        print(f"{g:<10} {len(rs):>4} runs   requires 未满足: {', '.join(miss) or '—'}")
    print(f"合计 {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
