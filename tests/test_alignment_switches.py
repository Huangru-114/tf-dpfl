"""
tests/test_alignment_switches.py  —  A4：对齐开关 + 「P2 对齐」模板

用户拍板（A4 会话）：每个对齐项做成开关、默认 = 现行为；对齐项写成一套配置模板
（fedavg/config/alignment_p2.yaml），P2 = 模板全开。这里守五件事：

  1. 模板与代码里的开关表（fedavg/alignment.py:SWITCHES）逐键一致；
  2. 每个开关的 P2 值 ≠ 旧值（否则「打开」是空操作）；
  3. 每个开关在 fedavg/ 里真的被读到（label_smoothing 式死开关，AUDIT A16）；
  4. config_validate：P2 少开一项 → 拒绝；取值拼错 → 拒绝；P1 配置照过；
  5. `[设定4]` 能打印、能解析；老日志没有这一行 → None。

纯 python，不 import TF。
"""

import copy
import re
import sys
from pathlib import Path

import pytest
import yaml

import alignment as AL
from config_validate import ConfigError, validate_config
from utils.kvline import format_kv, parse_kv

ROOT = Path(__file__).resolve().parent.parent
FEDAVG = ROOT / "fedavg"
sys.path.insert(0, str(ROOT / "harness"))
import registry as R                    # noqa: E402
from collect_metrics import collect     # noqa: E402

P1_CELL = ROOT / "experiments/attack/hfl-propagation/2edge_distributed.yaml"


def _p1():
    """P1 锚点格，按它实际跑的样子：run_exp3.sh 传 --attack_method badpfl（→ strategy=badpfl）。"""
    cfg = yaml.safe_load(P1_CELL.read_text(encoding="utf-8"))
    cfg["backdoor"]["malicious_strategy"] = "badpfl"
    return cfg


def _with_template(cfg):
    R.deep_merge(cfg, AL.template_nested())
    return cfg


# ══════════════════════════════════════════════════════════════════════════
# 1–2. 模板 ↔ 开关表
# ══════════════════════════════════════════════════════════════════════════
def test_template_keys_equal_switch_table():
    tpl = AL.load_template()
    assert set(tpl) == {s.key for s in AL.SWITCHES}
    for s in AL.SWITCHES:
        assert tpl[s.key] == s.p2, f"{s.key}: 模板 {tpl[s.key]!r} ≠ 开关表 {s.p2!r}"


def test_every_p2_value_differs_from_legacy():
    for s in AL.SWITCHES:
        assert s.p2 != s.legacy, f"{s.key} 的 P2 值与旧值相同 —— 打开它什么也不改变"


def test_extra_switches_are_not_in_template():
    tpl = AL.load_template()
    for s in AL.EXTRA_SWITCHES:
        assert s.key not in tpl and s.p2 is None


def test_every_switch_has_an_audit_row_in_the_audit_file():
    audit = R.audit_rows(ROOT / "experiments/attack/hfl-mechanism/AUDIT.md")
    for s in AL.SWITCHES + AL.EXTRA_SWITCHES:
        assert s.row in audit, f"{s.key} 指向不存在的 AUDIT 行 {s.row}"


# ══════════════════════════════════════════════════════════════════════════
# 3. 每个开关都被代码读到（死开关守卫）
# ══════════════════════════════════════════════════════════════════════════
# 按提交逐步接线：这个集合只许变小，最后一个提交之后必须为空。
NOT_YET_WIRED = {
    "backdoor.eval_xi_model", "evaluation.asr_columns",
    "training.deterministic_ops", "evaluation.pm_model",
    "training.lr_round_axis", "federation.edge_schedule", "federation.quota_round_axis",
}


def _reads():
    """fedavg/ 里 get_switch(…, "key") 的全部 key（alignment.py / config_validate 不算）。"""
    pat = re.compile(r"""get_switch\(\s*[^,]+,\s*["']([\w.]+)["']""")
    out = {}
    for p in FEDAVG.rglob("*.py"):
        if p.name in ("alignment.py", "config_validate.py"):
            continue
        for k in pat.findall(p.read_text(encoding="utf-8", errors="replace")):
            out.setdefault(k, []).append(p.relative_to(FEDAVG).as_posix())
    return out


def _arch_registered(name):
    src = (FEDAVG / "models/cnn.py").read_text(encoding="utf-8")
    return f'"{name}"' in src


def test_every_wired_switch_is_read_by_code():
    reads = _reads()
    for key in AL.ALL:
        if key in NOT_YET_WIRED:
            continue
        if key in AL.READ_DIRECTLY:
            assert _arch_registered(AL.ALL[key].p2), f"{key}: build_model 没有注册 {AL.ALL[key].p2}"
            continue
        assert key in reads, f"{key} 在模板/开关表里，但 fedavg/ 里没有任何 get_switch 读它"


def test_not_yet_wired_list_is_not_stale():
    """NOT_YET_WIRED 里的键必须**确实**还没被读 —— 接好线就要从集合里删掉。"""
    reads = _reads()
    for key in NOT_YET_WIRED:
        if key in AL.READ_DIRECTLY:
            assert not _arch_registered(AL.ALL[key].p2), f"{key} 已接线，请从 NOT_YET_WIRED 删掉"
        else:
            assert key not in reads, f"{key} 已接线（{reads[key]}），请从 NOT_YET_WIRED 删掉"


# ══════════════════════════════════════════════════════════════════════════
# get_switch 的默认值与非法取值
# ══════════════════════════════════════════════════════════════════════════
def test_get_switch_defaults_to_legacy():
    for s in AL.SWITCHES + AL.EXTRA_SWITCHES:
        if s.key in AL.READ_DIRECTLY:
            continue
        assert AL.get_switch({}, s.key) == s.legacy


def test_get_switch_rejects_misspelled_values():
    with pytest.raises(ValueError):
        AL.get_switch({"backdoor": {"badpfl_xi": "PGD"}}, "backdoor.badpfl_xi")


# ══════════════════════════════════════════════════════════════════════════
# 4. config_validate
# ══════════════════════════════════════════════════════════════════════════
def test_p1_cell_still_validates_and_is_legacy():
    cfg = _p1()
    validate_config(cfg)
    assert AL.template_state(cfg) == "legacy"


def test_p2_with_full_template_validates():
    cfg = _with_template(_p1())
    cfg["meta"] = {"protocol": "P2"}
    validate_config(cfg)
    assert AL.template_state(cfg) == "p2"
    assert AL.p2_mismatches(cfg) == []


@pytest.mark.parametrize("key", [s.key for s in AL.SWITCHES])
def test_p2_missing_any_switch_is_rejected(key):
    """反向锚点：P2 配置里把任一开关改回旧值 → 拒绝启动。"""
    cfg = _with_template(_p1())
    cfg["meta"] = {"protocol": "P2"}
    R.set_dotted(cfg, key, AL.ALL[key].legacy)
    with pytest.raises(ConfigError, match=re.escape(key)):
        validate_config(cfg)


def test_ablation_is_allowed_outside_p2():
    """消融（模板 + 改回一项）在 pilot 表（P1 口径）里是合法的，状态记为 mixed。"""
    cfg = _with_template(_p1())
    cfg["meta"] = {"protocol": "P1"}
    R.set_dotted(cfg, "training.fedrep_bn_stats", "shared")
    validate_config(cfg)
    assert AL.template_state(cfg) == "mixed"


def test_misspelled_switch_value_is_rejected():
    cfg = _p1()
    cfg["training"]["fedrep_order"] = "bodyfirst"
    with pytest.raises(ConfigError, match="fedrep_order"):
        validate_config(cfg)


def test_per_epoch_pipeline_rejects_static_poisoning():
    """per_epoch 从原始数组取数，会绕过静态投毒的数据集 → 恶意端静默变良性。"""
    cfg = _p1()
    cfg["backdoor"]["malicious_strategy"] = "vanilla"
    cfg["data"]["batch_pipeline"] = "per_epoch"
    with pytest.raises(ConfigError, match="per_epoch"):
        validate_config(cfg)


def test_fresh_pm_requires_a_method_with_private_state():
    cfg = _p1()
    cfg["training"]["drift_correction"] = "hier_ditto"
    cfg["evaluation"]["pm_model"] = "fresh"
    with pytest.raises(ConfigError, match="private_state"):
        validate_config(cfg)


# ══════════════════════════════════════════════════════════════════════════
# 5. [设定4] 打印 ↔ 解析
# ══════════════════════════════════════════════════════════════════════════
def test_settings4_line_is_printed_and_parsed(capsys):
    cfg = _with_template(_p1())
    validate_config(cfg)
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("[设定4]"))
    d = parse_kv(line, "[设定4]")
    assert d["template"] == "p2"
    for s in AL.SWITCHES + AL.EXTRA_SWITCHES:
        assert d[AL.short_name(s.key)] == AL.effective_values(cfg)[s.key]
    run = collect(out)["run"]
    assert run["alignment"]["template"] == "p2"
    assert run["alignment"]["pm_model"] == "fresh"
    assert run["alignment"]["deterministic_ops"] is True


def test_old_log_without_settings4_gives_none():
    run = collect("[Round   1] Broadcasting to 2 edges...\n")["run"]
    assert run["alignment"] is None and run["eval_attacker"] is None


def test_kvline_round_trip_keeps_none_and_bools():
    line = format_kv("[X]", {"a": None, "b": True, "c": 0.5, "d": 3, "e": "pgd"},
                     round_idx=7, edge_id=2)
    assert line == "[X] Round 7 | edge2 | a=n/a | b=true | c=0.5000 | d=3 | e=pgd"
    assert parse_kv(line, "[X]") == {"round": 7, "edge_id": 2, "a": None, "b": True,
                                     "c": 0.5, "d": 3, "e": "pgd"}


# ══════════════════════════════════════════════════════════════════════════
# registry：overlays 在 base 之后、set 之前
# ══════════════════════════════════════════════════════════════════════════
def test_registry_overlays_merge_between_base_and_set(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(
        {"a": {"x": 1, "y": 1}, "b": 1}))
    (tmp_path / "ov.yaml").write_text(yaml.safe_dump({"a": {"y": 2, "z": 2}}))
    (tmp_path / "reg.yaml").write_text(yaml.safe_dump({
        "study": "t", "protocol": "P1", "base": "base.yaml", "overlays": ["ov.yaml"],
        "groups": {"G": {"seeds": [1], "cells": [{"cell": "c", "set": {"a.z": 3}}]}}}))
    reg = R.Registry(tmp_path / "reg.yaml")
    cfg = reg.declared_config(reg.runs()[0])
    assert cfg["a"] == {"x": 1, "y": 2, "z": 3} and cfg["b"] == 1


def test_registry_missing_overlay_is_an_error(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\n")
    (tmp_path / "reg.yaml").write_text(yaml.safe_dump({
        "study": "t", "protocol": "P1", "base": "base.yaml", "overlays": ["nope.yaml"],
        "groups": {"G": {"seeds": [1], "cells": [{"cell": "c"}]}}}))
    reg = R.Registry(tmp_path / "reg.yaml")
    with pytest.raises(R.RegistryError, match="overlay"):
        reg.declared_config(reg.runs()[0])


def test_template_overlay_does_not_touch_anything_but_switches():
    """模板叠到 P1 格子上，改动的叶子恰好是开关表里的键（不多不少）。"""
    from utils.provenance import flat_diff
    before = _p1()
    after = _with_template(copy.deepcopy(before))
    changed = {d[0] for d in flat_diff(before, after)}
    assert changed == {s.key for s in AL.SWITCHES}
