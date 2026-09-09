"""
tests/test_run_self_description.py  —  `[设定2]` 自描述行的两侧

**为什么要两侧分开测**（陷阱 #10 的教训）：解析器认得某个格式 ≠ 代码真的会打印它。
  - 上游侧：`config_validate.validate_config` 真的打印了 `[设定2]`，字段齐全；
  - 下游侧：`collect_metrics.collect` 真的把它解析进 `run` 块。

**这条行回答什么**：metrics.json 的 `run` 块此前**不含** `malicious_per_edge`
（Experiment 3A 的自变量本身）、`edge_assignment`（client_id → edge 的映射规则）、
`local_epochs`（Stage B 之后会变）、`seed`（只在 run_name 字符串里）。
于是「这份 json 到底是哪一格」只能靠文件名 —— 而文件名不是 artifact 的一部分。

**为什么另起一行而不是扩 `[设定]`**：`RE_SETTINGS` 是一条**全或无**的正则。
往里加字段，格式一旦对不上，原有八个字段会**一起变成 None**，而日志毫无异常。
`test_old_log_without_settings2_still_parses_the_first_line` 就是锁这条性质的反向锚点。

下游侧纯 stdlib，本地秒级；上游侧要 numpy（config_validate 惰性 import defense 包），
本机自动 skip、集群上真跑。
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

from collect_metrics import collect            # noqa: E402
from config_validate import validate_config    # noqa: E402


# ── 上游侧 ────────────────────────────────────────────────────────────────
def _cfg(**over):
    """一份最小的合法 config；子字典用 over 覆盖（浅合并到二级）。"""
    cfg = {
        "seed": 42,
        "training": {"drift_correction": "hier_fedrep",
                     "local_epochs": 3, "plocal_epochs": 1},
        "defense": {"name": "none"},
        "federation": {"n_clients": 100, "n_edges": 4, "edge_rounds": 5,
                       "client_fraction": 0.1, "edge_assignment": "block"},
        "data": {"num_classes": 10},
        "model": {"arch": "resnet10"},
        "evaluation": {"eval_interval": 2},
        "backdoor": {"enabled": True, "malicious_strategy": "vanilla",
                     "trigger": "badnet", "target_label": 0, "poison_ratio": 0.2,
                     "n_malicious": 10, "eval_interval": 2,
                     "malicious_placement": "by_edge",
                     "malicious_per_edge": [10, 0, 0, 0]},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def _stdout_of(cfg):
    # `config_validate` 自己没有 import，但 `_configured_layers` 会惰性
    # `from defense import configured_layers`，而 defense 包 import numpy
    # → 上游侧在没有 numpy 的本机跑不了，集群上真跑。
    # （下游侧的解析测试是纯 stdlib，本地照跑。）
    pytest.importorskip("numpy")
    buf = io.StringIO()
    with redirect_stdout(buf):
        validate_config(cfg)
    return buf.getvalue()


def _line(text, prefix):
    for ln in text.splitlines():
        if ln.startswith(prefix):
            return ln
    return None


def test_settings2_line_is_printed_with_every_field():
    ln = _line(_stdout_of(_cfg()), "[设定2]")
    assert ln is not None, "validate_config 没有打印 [设定2] 行"
    for field in ("malicious_per_edge=[10,0,0,0]", "placement=by_edge",
                  "edge_assignment=block", "local_epochs=3", "plocal_epochs=1",
                  "seed=42", "bd_eval_interval=2", "acc_eval_interval=2",
                  "attack_stop_round=n/a"):
        assert field in ln, f"[设定2] 缺 {field}\n  实际: {ln}"


def test_attack_stop_round_is_reported_when_set():
    """持久性格必须自证退出轮 —— 否则衰减段是从哪一轮开始的事后无法判读。"""
    cfg = _cfg(backdoor={"malicious_strategy": "badpfl", "attack_stop_round": 40},
               federation={"n_rounds": 80})
    ln = _line(_stdout_of(cfg), "[设定2]")
    assert "attack_stop_round=40" in ln, ln


def test_never_stopping_prints_na_not_zero():
    """`n/a`（一直投到跑完）与 `0`（一轮都不投）语义相反，不能长一样。"""
    ln = _line(_stdout_of(_cfg()), "[设定2]")
    assert "attack_stop_round=n/a" in ln
    assert "attack_stop_round=0" not in ln


def test_settings_line_is_untouched():
    """扩新行不能碰老行 —— 老行一坏，八个字段一起变 None。"""
    ln = _line(_stdout_of(_cfg()), "[设定]")
    assert ln is not None
    for field in ("client_fraction=0.1", "poison_ratio=0.2", "n_clients=100",
                  "n_edges=4", "edge_rounds=5", "n_malicious=10",
                  "forced_participation=False", "arch=resnet10"):
        assert field in ln, f"[设定] 缺 {field}\n  实际: {ln}"


def test_no_by_edge_placement_prints_na_not_empty_list():
    """`n/a`（不按 edge 布点）与 `[]`（按 edge 布点但每 edge 零个）是两件事。"""
    cfg = _cfg(backdoor={"malicious_placement": "spread",
                         "malicious_per_edge": None})
    ln = _line(_stdout_of(cfg), "[设定2]")
    assert "malicious_per_edge=n/a" in ln
    assert "malicious_per_edge=[]" not in ln


def test_attack_off_reports_na_placement():
    cfg = _cfg(backdoor={"enabled": False})
    ln = _line(_stdout_of(cfg), "[设定2]")
    assert "malicious_per_edge=n/a" in ln and "placement=n/a" in ln


# ── 两个 eval_interval 的一致性警告 ────────────────────────────────────────
def _warns(cfg):
    return "\n".join(ln for ln in _stdout_of(cfg).splitlines()
                     if ln.startswith("[配置校验警告]"))


def test_mismatched_eval_intervals_warn():
    w = _warns(_cfg(backdoor={"eval_interval": 5}, evaluation={"eval_interval": 2}))
    assert "eval_interval" in w and "无法逐点配对" in w


def test_missing_backdoor_eval_interval_warns_about_the_default_50():
    cfg = _cfg()
    cfg["backdoor"].pop("eval_interval")
    assert "50" in _warns(cfg)


def test_matching_eval_intervals_do_not_warn():
    """反向锚点：正常配置不能刷警告，否则警告就没人看了。"""
    assert "eval_interval" not in _warns(_cfg())


# ── 下游侧 ────────────────────────────────────────────────────────────────
_SETTINGS = ("[设定] client_fraction=0.1 | poison_ratio=0.2 | n_clients=100 | "
             "n_edges=4 | edge_rounds=5 | n_malicious=10 | "
             "forced_participation=False | arch=resnet10")
_SETTINGS2 = ("[设定2] malicious_per_edge=[10,0,0,0] | placement=by_edge | "
              "edge_assignment=block | local_epochs=3 | plocal_epochs=1 | "
              "seed=42 | bd_eval_interval=2 | acc_eval_interval=2")


def _log(*extra):
    return "\n".join([
        "[Config] loading experiments/attack/hfl-propagation/4edge_collocated.yaml",
        "[Config] run_name = hier_fedavg_fedrep_noniid_badpfl_seed42",
        "[配置校验] 通过 | method=hier_fedrep | defense=none | attack=badpfl | 0 个警告",
        *extra,
        "[Round   1] Broadcasting to 4 edges...",
        "  [Cloud] GM=0.1000 | EM=0.5000 PM=0.6000 | loss=2.0 | time=10.0s | "
        "comm=1.0MB (total=1MB)",
    ])


def test_collect_parses_settings2_into_the_run_block():
    run = collect(_log(_SETTINGS, _SETTINGS2))["run"]
    assert run["malicious_per_edge"] == [10, 0, 0, 0]
    assert run["malicious_placement"] == "by_edge"
    assert run["edge_assignment"] == "block"
    assert run["local_epochs"] == 3
    assert run["plocal_epochs"] == 1
    assert run["seed"] == 42
    assert run["bd_eval_interval"] == 2
    assert run["acc_eval_interval"] == 2


def test_settings2_without_attack_stop_round_still_parses():
    """**加字段时同一个坑的第二次**（collect_metrics.py 的 RE_SETTINGS2 注释指向这条）。

    `attack_stop_round` 是后加的，所以在正则里是**可选组**：`a66da67`~`<本次>`
    之间跑的日志有 [设定2] 但没有这个字段。它们的原有八个字段**必须照常解析**，
    新字段是 None。写成必需组的话，这八个会一起变 None 而日志毫无异常。
    """
    run = collect(_log(_SETTINGS, _SETTINGS2))["run"]        # _SETTINGS2 不含新字段
    assert run["local_epochs"] == 3 and run["seed"] == 42    # 老字段没塌
    assert run["malicious_per_edge"] == [10, 0, 0, 0]
    assert run["attack_stop_round"] is None                  # 新字段留空


def test_attack_stop_round_round_trips():
    run = collect(_log(_SETTINGS, _SETTINGS2 + " | attack_stop_round=40"))["run"]
    assert run["attack_stop_round"] == 40
    assert run["acc_eval_interval"] == 2                     # 前八个字段不受影响


def test_attack_stop_round_na_is_none_not_zero():
    """`n/a` = 从不停止。读成 0 会被下游当成「第 0 轮就退出」，方向正好相反。"""
    run = collect(_log(_SETTINGS, _SETTINGS2 + " | attack_stop_round=n/a"))["run"]
    assert run["attack_stop_round"] is None
    assert run["attack_stop_round"] != 0


def test_old_log_without_settings2_still_parses_the_first_line():
    """**这条是两行设计的全部理由。**

    老日志（[设定2] 出现之前跑的）里没有第二行。新字段应当是 None，
    而原有八个字段**必须照常解析出来** —— 如果当初把新字段塞进 RE_SETTINGS，
    这里八个字段会一起变 None。
    """
    run = collect(_log(_SETTINGS))["run"]
    assert run["n_edges"] == 4 and run["edge_rounds"] == 5      # 老字段没受影响
    assert run["client_fraction"] == 0.1 and run["arch"] == "resnet10"
    assert run["malicious_per_edge"] is None                    # 新字段留空
    assert run["local_epochs"] is None and run["seed"] is None


def test_na_round_trips_to_none_not_zero():
    line = _SETTINGS2.replace("malicious_per_edge=[10,0,0,0]", "malicious_per_edge=n/a") \
                     .replace("placement=by_edge", "placement=spread") \
                     .replace("seed=42", "seed=n/a")
    run = collect(_log(_SETTINGS, line))["run"]
    assert run["malicious_per_edge"] is None
    assert run["malicious_per_edge"] != []          # 铁律 #5：无定义不是空也不是 0
    assert run["seed"] is None
    assert run["malicious_placement"] == "spread"
