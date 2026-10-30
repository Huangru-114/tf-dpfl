"""
阶段三 SA1（D-108）：`defense.edge_harden` 的配置校验、自描述行与日志解析、P1 的登记配置。纯 python，本地秒级。
"""

import copy
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fedavg"))
sys.path.insert(0, str(ROOT / "harness"))
from config_validate import validate_config                 # noqa: E402
import collect_metrics as CM                                # noqa: E402

HERE = ROOT / "experiments/defense/edge-native"
P1 = HERE / "configs/P1__col-pre__s42.yaml"
SNAP = HERE / "configs/SNAP__collocated__s42.yaml"


def _cfg():
    return yaml.safe_load(P1.read_text(encoding="utf-8"))


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return out


def test_p1_config_validates_and_announces_itself(capsys):
    validate_config(_cfg())
    assert "[设定11] edge_harden=sau-e8-train-high" in capsys.readouterr().out


def test_p1_differs_from_snap_collocated_only_in_the_switch_and_snapshots():
    a, b = _flat(_cfg()), _flat(yaml.safe_load(SNAP.read_text(encoding="utf-8")))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    assert diff - {"meta.run_id", "meta.group", "meta.cell"} == {"defense.edge_harden", "evaluation.snapshot_rounds"}


@pytest.mark.parametrize("mutate,needle", [
    (lambda c: c["defense"].__setitem__("edge_harden", "pgd-e99-train-low"), "不在"),
    (lambda c: c["defense"].__setitem__("edge_harden", 3), "字符串"),
    (lambda c: c["federation"].__setitem__("edge_schedule", "sequential"), "interleaved"),
    (lambda c: c["federation"]["design"].__setitem__("clean_per_edge", 0), "clean_per_edge"),
    (lambda c: c["defense"].__setitem__("name", "median"), "鲁棒聚合"),
])
def test_validation_refuses(mutate, needle):
    c = _cfg()
    mutate(c)
    with pytest.raises(Exception, match=needle):
        validate_config(c)


def test_switch_off_announces_none(capsys):
    c = _cfg()
    del c["defense"]["edge_harden"]
    validate_config(c)
    assert "[设定11] edge_harden=none" in capsys.readouterr().out


def test_collect_parses_settings_harden_and_gate_lines():
    log = "\n".join([
        "[设定11] edge_harden=sau-e8-train-high",
        "[Harden] round=1 | edge=0 | cfg=sau-e8-train-high | steps=75 | diverged_at=n/a | loss_first=1.2 | "
        "loss_last=0.9 | probe_acc=0.98 | ref_acc=0.97 | lr=0.0001 | dw_body_absmax=0.01 | seconds=55.1",
        "[Harden] round=1 | edge=1 | cfg=sau-e8-train-high | steps=75 | diverged_at=12 | loss_first=1.2 | "
        "loss_last=n/a | probe_acc=0.98 | ref_acc=0.97 | lr=0.0001 | dw_body_absmax=n/a | seconds=20.0",
        "[HardenAgg] round=1 | unhardened=0123456789ab",
        "[HardenAgg] round=2 | unhardened=1234e5678901",          # 长得像数字的哈希：按字符串读
    ])
    m = CM.collect(log)
    assert m["schema_version"] == 15 and m["run"]["edge_harden"] == "sau-e8-train-high"
    assert [h["diverged_at"] for h in m["harden"]] == [None, 12] and m["harden"][0]["steps"] == 75
    assert m["harden_agg"] == [{"round": 1, "unhardened": "0123456789ab"}, {"round": 2, "unhardened": "1234e5678901"}]
    old = CM.collect("[Round   1] Broadcasting to 4 edges...")
    assert old["run"]["edge_harden"] is None and old["harden"] == [] and old["harden_agg"] == []
