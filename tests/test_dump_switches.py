"""
tests/test_dump_switches.py  —  S9 / D-073：logits 存盘与分析快照两个开关（纯标准库 + numpy + pyyaml）

  · 两个开关在 EXTRA_SWITCHES（不进模板），缺省 = 关（与加它之前逐字节一致）；
  · config_validate 拦下所有静默写错（bool、负数、YAML 列表、超过 n_rounds、超过 3 个、乱序）；
  · `snapshot_rounds` 写成 "30/70"：[设定4] 往返逐字相同（非 legacy 值也测）；
  · 写盘小工具：落盘位置、带作业号的目录名、[Dump] manifest 行可解析。
TF 侧（真跑一轮、快照复原 fresh-PM）在 tests/test_eval_detail_tf.py。
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import alignment as AL                                          # noqa: E402
import registry as R                                            # noqa: E402
from collect_metrics import collect                             # noqa: E402
from config_validate import ConfigError, validate_config        # noqa: E402
from utils import dumps as D                                    # noqa: E402
from utils.kvline import parse_kv                               # noqa: E402

MECH = ROOT / "experiments" / "attack" / "hfl-mechanism"
KEYS = ("evaluation.dump_logits_every", "evaluation.snapshot_rounds")


def _g6a_cfg():
    reg = R.Registry(MECH / "registry.yaml")
    run = next(r for r in reg.runs() if r["run_id"] == "G6__a__s42")
    return copy.deepcopy(reg.declared_config(run))


# ── 开关表 ──────────────────────────────────────────────────────────────────
def test_switches_are_extra_and_off_by_default():
    extra = {s.key for s in AL.EXTRA_SWITCHES}
    assert set(KEYS) <= extra
    assert not set(KEYS) & {s.key for s in AL.SWITCHES}
    assert [AL.get_switch({}, k) for k in KEYS] == [0, None]


def test_base_yaml_does_not_turn_them_on():
    """开关只在组的 set: 里开（D-073）—— 写进 base.yaml 会让所有 P2 组的 sha 变。"""
    base = yaml.safe_load((MECH / "base.yaml").read_text(encoding="utf-8"))
    ev = base.get("evaluation") or {}
    assert "dump_logits_every" not in ev and "snapshot_rounds" not in ev


# ── config_validate ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("ok", [{"dump_logits_every": 0}, {"dump_logits_every": 2},
                                {"snapshot_rounds": "30/60"}, {"snapshot_rounds": 60},
                                {"snapshot_rounds": "1/30/60"}])
def test_valid_values_pass(ok):
    cfg = _g6a_cfg()
    cfg["evaluation"].update(ok)
    validate_config(cfg)


@pytest.mark.parametrize("bad", [{"dump_logits_every": True}, {"dump_logits_every": -1},
                                 {"dump_logits_every": 1.5},
                                 {"snapshot_rounds": [30, 60]},       # YAML 列表
                                 {"snapshot_rounds": "60/30"},        # 乱序
                                 {"snapshot_rounds": "30/30"},        # 重复
                                 {"snapshot_rounds": "0/30"},         # 轮号从 1 起
                                 {"snapshot_rounds": "30/61"},        # > n_rounds（60）
                                 {"snapshot_rounds": "10/20/30/40"},  # > 3 个
                                 {"snapshot_rounds": True},
                                 {"snapshot_rounds": "a/b"}])
def test_silent_mistakes_are_rejected(bad):
    cfg = _g6a_cfg()
    cfg["evaluation"].update(bad)
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_snapshots_with_adaptive_stopping_warn():
    cfg = _g6a_cfg()
    cfg["stopping"] = {"criteria": ["pm_acc_plateau"], "floor_effective": 150,
                       "cap_effective": 300}
    cfg["evaluation"]["snapshot_rounds"] = "30"
    warnings = validate_config(cfg)
    assert any("静默跳过" in w for w in warnings)


def test_settings4_round_trips_non_legacy_values(capsys):
    cfg = _g6a_cfg()
    cfg["evaluation"].update(dump_logits_every=1, snapshot_rounds="30/60")
    validate_config(cfg)
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[设定4]"))
    d = parse_kv(line, "[设定4]")
    assert d["dump_logits_every"] == 1
    assert d["snapshot_rounds"] == "30/60"                     # 字符串原样往返
    assert collect(line + "\n")["run"]["alignment"]["snapshot_rounds"] == "30/60"


# ── utils/dumps ─────────────────────────────────────────────────────────────
def test_parse_rounds():
    assert D.parse_rounds(None) == ()
    assert D.parse_rounds(70) == (70,)
    assert D.parse_rounds("30/70") == (30, 70)
    for bad in ("70/30", "3/3", "0", True, [30], "x"):
        with pytest.raises(ValueError):
            D.parse_rounds(bad)


def test_dump_location_and_run_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("TFDPFL_DUMPDIR", raising=False)
    assert D.dump_root() == ROOT.parent / "tfdpfl-dumps"            # 与 tfdpfl-logs 同级
    monkeypatch.setenv("TFDPFL_DUMPDIR", str(tmp_path))
    assert D.dump_root() == tmp_path
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert D.run_dir_name({"meta": {"run_id": "G8__a__s42"}}) == "G8__a__s42.12345"
    monkeypatch.delenv("SLURM_JOB_ID")
    assert D.run_dir_name({}) == "norun.local"


def test_write_npz_and_manifest_line(tmp_path):
    info = D.write_npz(tmp_path, "r.local/x.npz", {"a": np.arange(3, dtype=np.int32)})
    assert (tmp_path / "r.local/x.npz").stat().st_size == info["bytes"]
    assert len(info["sha"]) == 12 and info["path"] == "r.local/x.npz"
    np.testing.assert_array_equal(np.load(tmp_path / info["path"])["a"], [0, 1, 2])
    d = parse_kv(D.dump_line(7, "logits", info), "[Dump]")
    assert d["round"] == 7 and d["kind"] == "logits" and d["bytes"] == info["bytes"]
    assert d["path"] == "r.local/x.npz" and d["sha"] == info["sha"]


def test_snapshot_estimate_for_g8_is_well_under_budget():
    est = D.estimate_snapshot_bytes(2, 5, 4_909_002, 100, 10_890)
    assert 150e6 < est < 250e6                                     # 约 2 × 98 MB
