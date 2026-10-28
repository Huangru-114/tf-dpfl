"""
阶段三 SA1（D-108）：在线上传前加固（`defense.edge_harden` → `hardening/online.py` + `CloudServer.collect_and_aggregate`）。需要 TF。

小 HFL（tests/test_eval_integration._setup：2 edge × 3 个 FedRep 客户端，交错调度 R=2）上：
  ① 开关开：每个云轮每个 edge 一行 [Harden]、一行 [HardenAgg]；
  ② **P1 的闸**：第 1 云轮 [HardenAgg] 的 unhardened（没洗过的上传物聚合）= 开关关的同一 run 的 [Checksum] Round 1；
  ③ 加固真的改了上传物：开关开的 [Checksum] Round 1 ≠ 开关关的；
  ④ 一次加固不推进 Python random；发散 → 上传原权重；
  ⑤ 开关关：一行 [Harden] / [HardenAgg] / 加固器都没有。
"""

import random
import re

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="L1 需要 TF；本地无 TF 时在集群跑")

import test_eval_integration as TEI                         # noqa: E402

FED = {"edge_schedule": "interleaved", "edge_rounds": 2, "n_rounds": 2}
CFG_ID = "sau-e8-train-high"


def _world(monkeypatch, harden: str | None):
    orig = TEI._cfg

    def cfg(aligned):
        c = orig(aligned)
        if harden is not None:
            c["defense"] = {"name": "none", "edge_harden": harden}
        return c
    monkeypatch.setattr(TEI, "_cfg", cfg)
    random.seed(42)
    c, cloud, clients, edges = TEI._setup(True, federation=dict(FED))
    r = np.random.default_rng(7)
    for e in edges:                                          # main.py 挂的 edge 干净集（这里 64 张合成图）
        e.clean_x = r.normal(size=(64, TEI.IMG, TEI.IMG, 3)).astype(np.float32)
        e.clean_y = (np.arange(64) % TEI.NCLS).astype(np.int64)
    monkeypatch.setattr(TEI, "_cfg", orig)
    return c, cloud, edges


def _lines(out, tag):
    return [ln for ln in out.splitlines() if ln.startswith(tag)]


@pytest.fixture(scope="module")
def runs():
    mp = pytest.MonkeyPatch()
    res = {}
    import io
    import contextlib
    for name, h in (("off", None), ("on", CFG_ID)):
        c, cloud, edges = _world(mp, h)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cloud.run_round(1)
            cloud.run_round(2)
        res[name] = {"out": buf.getvalue(), "cloud": cloud, "edges": edges}
    mp.undo()
    return res


def _checksum(out, rnd):
    m = re.search(rf"\[Checksum\] Round {rnd} \| global=([0-9a-f]+)", out)
    return m.group(1)


def test_harden_lines_per_round_and_edge(runs):
    out = runs["on"]["out"]
    h = _lines(out, "[Harden]")
    assert len(h) == 2 * 2 and all(f"cfg={CFG_ID}" in ln for ln in h)
    assert len(_lines(out, "[HardenAgg]")) == 2
    assert all("diverged_at=n/a" in ln for ln in h)


def test_gate_unhardened_round1_equals_the_switch_off_checksum(runs):
    agg = _lines(runs["on"]["out"], "[HardenAgg]")[0]
    unhardened = re.search(r"unhardened=([0-9a-f]+)", agg).group(1)
    assert unhardened == _checksum(runs["off"]["out"], 1)
    assert _checksum(runs["on"]["out"], 1) != _checksum(runs["off"]["out"], 1)       # 加固真的改了上传物


def test_switch_off_has_no_hardener_and_no_lines(runs):
    assert runs["off"]["cloud"]._harden is None
    assert not _lines(runs["off"]["out"], "[Harden]") and not _lines(runs["off"]["out"], "[HardenAgg]")


def test_one_upload_hardening_keeps_python_random_and_divergence_falls_back(runs, monkeypatch):
    cloud, edges = runs["on"]["cloud"], runs["on"]["edges"]
    hz = cloud._harden
    w = edges[0].model.get_weights()
    st = random.getstate()
    new, info = hz.harden_upload(edges[0], w, 3)
    assert random.getstate() == st and info["diverged_at"] is None
    assert any(not np.array_equal(a, b) for a, b in zip(new, w))
    monkeypatch.setattr(hz.hardener, "harden", lambda *a, **k: ([x * np.nan for x in w], {"diverged_at": 1}))
    back, info = hz.harden_upload(edges[0], w, 3)
    assert info["diverged_at"] == 1 and all(a is b for a, b in zip(back, w))


def test_missing_clean_set_is_an_error(runs):
    cloud, edges = runs["on"]["cloud"], runs["on"]["edges"]

    class E:
        edge_id, clean_x, clean_y = 9, None, None
    with pytest.raises(ValueError, match="干净集"):
        cloud._harden.harden_upload(E(), edges[0].model.get_weights(), 1)
