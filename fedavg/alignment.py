"""
alignment.py  —  对齐审计（experiments/attack/hfl-mechanism/AUDIT.md）落到代码的开关

A4 会话（2026-09-26）用户拍板：**每个对齐项做成开关，默认 = 现行为**；
对齐项写成**一套配置模板**（`config/alignment_p2.yaml`），P2 配置由它叠加生成。

为什么默认必须是旧行为：冻结的 P1 配置（hfl-propagation/）、matrix、bad-pfl 目录
都还要能按原样重跑；开关一律「不写 = 旧行为」，它们的数值逐字节不变。
为什么要一套模板而不是散落的键：P2 的定义就是「模板里的每一项都打开」。
`config_validate` 对 `meta.protocol == "P2"` 的配置逐键核对模板，少开一项就拒绝启动
—— 否则一格 P2 run 悄悄少了一个对齐项，跑出来与其他格子不可比，而日志毫无异常。

**读开关的唯一入口是 `get_switch(config, key)`**：默认值只在本文件定义一次。
守卫 tests/test_alignment_switches.py 断言模板里每个键都在 `fedavg/` 里经它被读到
（label_smoothing 那种「写在配置里、代码从来不读」的死开关不会再出现，AUDIT A16）。

纯标准库（YAML 懒加载），**不 import TF** —— 本地 L1 秒级。
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

TEMPLATE_PATH = Path(__file__).resolve().parent / "config" / "alignment_p2.yaml"


class Switch(NamedTuple):
    key: str            # 点分路径，如 "backdoor.badpfl_xi"
    legacy: object      # 不写这个键时的值 = 现行为（P1）
    p2: object          # 模板里的值；None = 不进模板（A26 / G7 前提）
    choices: tuple      # 合法取值；() = 不在这里校验（model.arch 由 build_model 校验）
    row: str            # AUDIT 行号
    decision: str       # DECISIONS 编号
    scope: str | None   # 只对哪类配置有意义："hier_fedrep" / "badpfl" / None（全部）


# ── 进模板的：AUDIT 里全部 `align` 行，外加 A08（D-023，候选方案，由 D-029 验证）──
SWITCHES = (
    Switch("backdoor.badpfl_xi",           "fgsm",       "pgd",
           ("fgsm", "pgd"),                         "A01", "D-014", "badpfl"),
    Switch("backdoor.eval_xi_model",       "victim",     "fixed_attacker",
           ("victim", "fixed_attacker"),            "A02", "D-015/D-033", "badpfl"),
    Switch("backdoor.poison_sampling",     "exact_k",    "bernoulli",
           ("exact_k", "bernoulli"),                "A03", "D-016", "badpfl"),
    Switch("backdoor.badpfl_bn_mode",      "inference",  "official",
           ("inference", "official"),               "A05", "D-018", "badpfl"),
    Switch("evaluation.asr_columns",       "filtered",   "four_way",
           ("filtered", "four_way"),                "A06", "D-019", None),
    Switch("backdoor.badpfl_generator",    "legacy",     "official",
           ("legacy", "official"),                  "A14", "D-020", "badpfl"),
    Switch("training.deterministic_ops",   False,        True,
           (False, True),                           "A15", "D-028", None),
    Switch("training.fedrep_poison_phases", "both",      "body",
           ("both", "body"),                        "A24", "D-021", "hier_fedrep"),
    Switch("data.batch_pipeline",          "legacy",     "per_epoch",
           ("legacy", "per_epoch"),                 "A25", "D-026", None),
    Switch("training.fedrep_bn_stats",     "shared",     "private",
           ("shared", "private"),                   "A27", "D-032", "hier_fedrep"),
    Switch("evaluation.pm_model",          "stale",      "fresh",
           ("stale", "fresh"),                      "A28", "D-033", None),
    Switch("model.arch",                   "resnet10",   "resnet10_torch",
           (),                                      "A29", "D-034", None),
    Switch("training.lr_round_axis",       "cloud",      "effective",
           ("cloud", "effective"),                  "A08", "D-023", None),
    Switch("federation.edge_schedule",     "sequential", "interleaved",
           ("sequential", "interleaved"),           "D01", "D-036", None),
    Switch("federation.quota_round_axis",  "cloud",      "effective",
           ("cloud", "effective"),                  "D02", "D-036", None),
)

# ── 不进模板、但同样只经 get_switch 读的 ────────────────────────────────────
EXTRA_SWITCHES = (
    # A26：两种顺序都跑 pilot（D-031），相同则维持 head_first（→ deviate）
    Switch("training.fedrep_order",        "head_first", None,
           ("head_first", "body_first"),            "A26", "D-031", "hier_fedrep"),
    # G7（D-025）的前提：「官方预处理」= 不标准化、不增强。ε/σ 与静态触发器跟随它（F-027）
    Switch("data.normalize",               True,         None,
           (True, False),                           "A10", "D-025", None),
    Switch("data.augment",                 True,         None,
           (True, False),                           "A10", "D-025", None),
)

ALL = {s.key: s for s in SWITCHES + EXTRA_SWITCHES}

# evaluation.pm_model=fresh 需要方法定义「私有部分」（client.private_state）：
# FedAvg 没有私有部分 → fresh-PM = 当前 edge 模型（D-033）；FedRep = head（+ 统计量，A27）。
# 其他方法没定义，config_validate 拒绝 fresh（无从组装，不能静默退化）。
PM_FRESH_METHODS = frozenset({"fedavg", "hierfedavg", "hier_fedrep"})

# model.arch 历来由 build_model 直接读（`config["model"]["arch"]`），不走 get_switch。
READ_DIRECTLY = frozenset({"model.arch"})


def _walk(config: dict, key: str):
    cur = config or {}
    for k in key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return False, None
        cur = cur[k]
    return True, cur


def get_switch(config: dict, key: str):
    """开关的当前值；配置里没写 → 旧值（现行为）。非法取值直接报错，绝不静默回退。"""
    sw = ALL[key]
    found, val = _walk(config, key)
    if not found or val is None:
        return sw.legacy
    if sw.choices and val not in sw.choices:
        raise ValueError(f"{key}={val!r} 不是合法取值 {list(sw.choices)}"
                         f"（AUDIT {sw.row} / {sw.decision}）")
    return val


def invalid_values(config: dict) -> list:
    """[(key, value, choices)]：写了但不合法的开关（config_validate 用它 fail fast）。"""
    out = []
    for sw in ALL.values():
        found, val = _walk(config, sw.key)
        if found and val is not None and sw.choices and val not in sw.choices:
            out.append((sw.key, val, sw.choices))
    return out


def load_template(path=None) -> dict:
    """模板 → 扁平 {key: value}。"""
    import yaml
    raw = yaml.safe_load(Path(path or TEMPLATE_PATH).read_text(encoding="utf-8")) or {}
    flat = {}

    def rec(d, prefix):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                rec(v, p)
            else:
                flat[p] = v
    rec(raw, "")
    return flat


def template_nested(path=None) -> dict:
    """模板原样（嵌套 dict），供 registry 叠加。"""
    import yaml
    return yaml.safe_load(Path(path or TEMPLATE_PATH).read_text(encoding="utf-8")) or {}


def effective_values(config: dict) -> dict:
    """全部开关（含不进模板的）的有效值，按 ALL 的顺序。"""
    out = {}
    for key, sw in ALL.items():
        if key in READ_DIRECTLY:
            found, val = _walk(config, key)
            out[key] = val if found else None
        else:
            out[key] = get_switch(config, key)
    return out


def p2_mismatches(config: dict, template: dict | None = None) -> list:
    """[(key, 模板值, 实际值)]：P2 配置里与模板不一致的开关。空 = 全部打开。"""
    tpl = template if template is not None else load_template()
    eff = effective_values(config)
    return [(k, v, eff.get(k)) for k, v in tpl.items() if eff.get(k) != v]


def template_state(config: dict) -> str:
    """p2（模板全开）/ legacy（全是旧值）/ mixed（部分打开：消融或写错）。"""
    eff = effective_values(config)
    on = [eff[s.key] == s.p2 for s in SWITCHES]
    if all(on):
        return "p2"
    if not any(on):
        return "legacy"
    return "mixed"


def short_name(key: str) -> str:
    return key.split(".")[-1]


def describe_fields(config: dict) -> dict:
    """`[设定4]` 行的字段：template=… 在前，然后每个开关（短名）。"""
    fields = {"template": template_state(config)}
    for key, val in effective_values(config).items():
        fields[short_name(key)] = val
    return fields
