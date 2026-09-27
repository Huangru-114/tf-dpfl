"""
tests/test_eval_downsampling.py  —  评估降频（D-050 / D-054）：开关、配置落点、终值口径、计时

  · 三个开关在 EXTRA_SWITCHES（预算旋钮，不是对齐项 → 不进模板），缺省 = 旧行为；
  · P2 的值写在 hfl-mechanism/base.yaml：白盒关、陈旧 ASR / 陈旧 pm_acc 隔点；pilot 不受影响；
  · config_validate 显式查类型（1 == True），且隔点时两个 eval_interval 必须相等；
  · 终值：副列按「末 10 个评估点窗口」取（隔点时约 5 点），不能先丢 None 再往回够；
  · [TimingAcc]：精度评估分项计时 → acc_rounds[] + timing_summary.acc_split_total_s。

纯标准库 + numpy + pyyaml。TF 侧的接线在 tests/test_eval_integration.py。
"""

import copy
import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import alignment as AL                                          # noqa: E402
import registry as R                                            # noqa: E402
from collect_metrics import collect                             # noqa: E402
from config_validate import ConfigError, validate_config        # noqa: E402
from runs_table import last_k_mean, summarize_run, window_mean  # noqa: E402

MECH = ROOT / "experiments" / "attack" / "hfl-mechanism"
KEYS = ("evaluation.whitebox_asr", "evaluation.stale_asr_every", "evaluation.stale_pm_every")


# ══════════════════════════════════════════════════════════════════════════
# 开关与配置落点
# ══════════════════════════════════════════════════════════════════════════
def test_switches_are_extra_and_default_to_old_behaviour():
    extra = {s.key: s for s in AL.EXTRA_SWITCHES}
    assert set(KEYS) <= set(extra)
    assert not set(KEYS) & {s.key for s in AL.SWITCHES}          # 不进模板
    assert [AL.get_switch({}, k) for k in KEYS] == [True, 1, 1]


def test_p2_base_turns_whitebox_off_and_halves_both_stale_columns():
    reg = R.Registry(MECH / "registry.yaml")
    for run in reg.runs():
        ev = reg.declared_config(run)["evaluation"]
        assert (ev["whitebox_asr"], ev["stale_asr_every"], ev["stale_pm_every"]) == (False, 2, 2)


def test_pilot_registry_is_untouched():
    """pilot 只叠模板、不叠 base.yaml → 降频不改它的配置（它已判完，sha 不能动）。"""
    reg = R.Registry(MECH / "pilot" / "registry.yaml")
    for run in reg.runs():
        ev = reg.declared_config(run).get("evaluation") or {}
        assert not set(ev) & {k.split(".")[1] for k in KEYS}


def _g7_cfg():
    reg = R.Registry(MECH / "registry.yaml")
    return reg.declared_config([r for r in reg.runs() if r["group"] == "G7"][0])


def test_declared_p2_config_validates():
    validate_config(copy.deepcopy(_g7_cfg()))


@pytest.mark.parametrize("bad", [{"whitebox_asr": 1}, {"whitebox_asr": "false"},
                                 {"stale_asr_every": True}, {"stale_asr_every": 0},
                                 {"stale_pm_every": 1.5}])
def test_wrong_types_are_rejected(bad):
    cfg = _g7_cfg()
    cfg["evaluation"].update(bad)
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_downsampling_requires_asr_and_acc_on_the_same_grid():
    cfg = _g7_cfg()
    cfg["evaluation"]["eval_interval"] = 5                      # backdoor.eval_interval 仍是 1
    with pytest.raises(ConfigError, match="同一批点"):
        validate_config(cfg)
    cfg["evaluation"].update(stale_asr_every=1, stale_pm_every=1)
    validate_config(cfg)                                         # 不隔点时照旧只是警告


def test_every_switch_is_read_through_get_switch_in_the_servers():
    srv = (ROOT / "fedavg/server/server.py").read_text()
    bd = (ROOT / "fedavg/server/backdoor_server.py").read_text()
    assert 'get_switch(config, "evaluation.stale_pm_every")' in srv
    assert 'get_switch(self.config, "evaluation.whitebox_asr")' in bd
    assert 'get_switch(self.config, "evaluation.stale_asr_every")' in bd
    # 两个陈旧列共用 CloudServer 的评估点序号
    assert "self._side_due(getattr(self, \"stale_pm_every\", 1))" in srv
    assert "self._side_due(getattr(self, \"stale_asr_every\", 1))" in bd


# ══════════════════════════════════════════════════════════════════════════
# 终值口径：末 10 个评估点窗口
# ══════════════════════════════════════════════════════════════════════════
def test_window_mean_equals_last_k_mean_when_every_point_is_defined():
    vals = [0.1 * i for i in range(15)]
    rows = [{"x": v} for v in vals]
    assert window_mean(rows, "x") == last_k_mean(vals)


def test_window_mean_does_not_reach_back_past_the_last_ten_points():
    # 20 个评估点，陈旧列只在偶数点有值 → 末 10 点窗口里只有 5 个
    rows = [{"x": float(i) if i % 2 == 0 else None} for i in range(20)]
    mean, n = window_mean(rows, "x")
    assert n == 5 and mean == (10 + 12 + 14 + 16 + 18) / 5
    mean_old, n_old = last_k_mean([r["x"] for r in rows])
    assert n_old == 10 and mean_old != mean                      # 反向锚点：旧口径往回够了 20 点


def test_window_mean_anchor_counts_only_evaluated_rows():
    # acc_rounds 每云轮一行；pm_acc 每 2 轮评一次；陈旧 pm_acc 再隔一个评估点
    rows = []
    for r in range(1, 41):
        ev = r % 2 == 0
        idx = r // 2 - 1
        rows.append({"round": r, "pm_acc": 0.5 if ev else None,
                     "pm_acc_stale": float(r) if ev and idx % 2 == 0 else None})
    mean, n = window_mean(rows, "pm_acc_stale", anchor="pm_acc")
    assert n == 5
    assert mean == sum([22.0, 26.0, 30.0, 34.0, 38.0]) / 5


def _metrics(n_points=20):
    rounds = [{"round": r, "local_benign_asr": 0.5,
               "local_benign_asr_stale": (0.9 if r % 2 == 1 else None)}
              for r in range(1, n_points + 1)]
    acc = [{"round": r, "pm_acc": 0.8, "pm_acc_stale": (0.85 if r % 2 == 1 else None)}
           for r in range(1, n_points + 1)]
    return {"run": {"edge_rounds": 5, "provenance": {"protocol": "P2"}},
            "rounds": rounds, "acc_rounds": acc}


def test_runs_table_side_columns_use_the_window():
    row = summarize_run(_metrics(), name="x", source="x")
    assert row["local_benign_asr_n"] == 10
    assert row["local_benign_asr_stale_n"] == 5 and row["local_benign_asr_stale_last10"] == 0.9
    assert row["pm_acc_stale_n"] == 5 and row["pm_acc_n"] == 10


# ══════════════════════════════════════════════════════════════════════════
# [TimingAcc]
# ══════════════════════════════════════════════════════════════════════════
def _log(timing=True):
    L = []
    for r in (1, 2):
        L.append(f"[Round   {r}] Broadcasting to 2 edges...")
        pm = " PM=0.5000" if r == 2 else ""
        L.append(f"  [Cloud] GM=0.1000 | EM=0.2000{pm} | loss=2.3000 | time=10.0s | comm=1.0MB (total=2MB)")
        if timing:
            L.append(f"[TimingAcc] Round {r} | gm=1.50 | em=2.00 | "
                     f"pm={'3.00' if r == 2 else 'n/a'} | pm_stale={'1.25' if r == 2 else 'n/a'}")
    return "\n".join(L) + "\n"


def test_timing_acc_lands_in_acc_rounds_and_the_summary():
    m = collect(_log())
    a = {r["round"]: r for r in m["acc_rounds"]}
    assert (a[1]["acc_gm_s"], a[1]["acc_pm_s"], a[1]["acc_pm_stale_s"]) == (1.5, None, None)
    assert (a[2]["acc_em_s"], a[2]["acc_pm_s"], a[2]["acc_pm_stale_s"]) == (2.0, 3.0, 1.25)
    assert m["timing_summary"]["acc_split_total_s"] == {"gm": 3.0, "em": 4.0, "pm": 3.0,
                                                        "pm_stale": 1.2}


def test_old_log_without_timing_acc_is_null_not_zero():
    m = collect(_log(timing=False))
    assert m["timing_summary"]["acc_split_total_s"] == {"gm": None, "em": None, "pm": None,
                                                        "pm_stale": None}
    assert all(r["acc_gm_s"] is None for r in m["acc_rounds"])


def test_timing_acc_is_a_separate_line_and_cloud_line_is_unchanged():
    src = (ROOT / "fedavg/server/server.py").read_text()
    assert 'format_kv("[TimingAcc]"' in src
    assert re.search(r'print\(f"  \[Cloud\] GM=\{global_acc:\.4f\} \| EM=\{avg_em_acc:\.4f\}\{pm_str\} \| "',
                     src)
