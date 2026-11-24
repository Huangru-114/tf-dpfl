"""
harness/atk_route.py（攻击基线第一阶段的分流，PREREG-v2 第七节 A1；D-141）。不需要 TF。

脚本写于数据之前 → 用合成数据覆盖三个分支、每一道闸、种子汇总的每个状态；真实数据只用 Bad-PFL 对照
SNAP__collocated__s42 作正锚点（受害边缘 0.995 → 若套用分流规则是 propagates）。
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

import atk_route as A                                            # noqa: E402

SNAP = ROOT / "experiments/defense/edge-native/results/P2/SNAP/SNAP__collocated__s42.metrics.json"


@pytest.fixture(scope="module")
def snap():
    return json.loads(SNAP.read_text(encoding="utf-8"))


def _as_static(m, trigger="badnet", e0=None, victim=None):
    """把真实的 SNAP 文件改写成「一个 static run」：只动分流读的量与闸查的字段。"""
    m = copy.deepcopy(m)
    m["run"].update(attack="static", static=True, trigger=trigger)
    mal = m["run"]["malicious_ids"]
    m["poison_counts"] = [{"client": c, "reports": 30, "n_poisoned": 1500, "n_seen": 7500} for c in mal]
    for rows in m["per_edge_rounds"].values():
        for row in rows:
            if e0 is not None and row["edge_id"] == 0:
                row["client_benign"] = e0
            if victim is not None and row["edge_id"] in (1, 2, 3):
                row["client_benign"] = victim
    return m


# ── 规则本身 ─────────────────────────────────────────────────────────────────
def test_threshold_is_the_preregistered_value():
    assert A.ROUTE_THRESHOLD == 0.30            # 改它就是改预注册（PREREG-v2 第七节）


@pytest.mark.parametrize("b0,v,want", [
    (0.9, 0.9, "propagates"),
    (0.1, 0.5, "propagates"),                   # 表按顺序取：受害边缘过线就是 propagates
    (0.9, 0.29, "local_only"),
    (0.30, 0.0, "local_only"),
    (0.29, 0.0, "no_implant"),
    (0.9 - 0.6, 0.0, "local_only"),             # 0.30000000000000004 / 0.29999…：带 1e-9 容差
    (0.0, 0.7 - 0.4, "propagates"),
    (None, 0.5, None), (0.5, None, None),
])
def test_classify(b0, v, want):
    assert A.classify(b0, v) == want


# ── 每个 run：真实 SNAP 改写 ─────────────────────────────────────────────────
def test_snap_control_routes_as_propagates(snap):
    q = A.run_quantities(snap)
    assert q["V"] == pytest.approx(0.995, abs=0.001) and q["B0"] > 0.9
    assert A.classify(q["B0"], q["V"]) == "propagates"


@pytest.mark.parametrize("e0,victim,want", [(None, None, "propagates"),
                                            (0.8, 0.05, "local_only"),
                                            (0.1, 0.05, "no_implant")])
def test_route_run_three_branches(snap, e0, victim, want):
    r = A.route_run(_as_static(snap, e0=e0, victim=victim), "badnet")
    assert r["route"] == want, r
    assert r["poison"]["rate"] == 0.2 and r["poison"]["n_clients"] == 10
    assert r["yt_clean_pooled"] is not None and r["yt_clean_victims"] is not None


@pytest.mark.parametrize("mutate,needle", [
    (lambda m: m["run"].update(attack="badpfl"), "run.attack"),
    (lambda m: m["run"].update(trigger="blended"), "run.trigger"),
    (lambda m: m["run"].update(static=None), "run.static"),
    (lambda m: m.update(poison_counts=[]), "poison_counts 为空"),
    (lambda m: m["poison_counts"][0].update(n_poisoned=0), "一张都没投毒"),
    (lambda m: m["poison_counts"].append({"client": 99, "reports": 1, "n_poisoned": 3, "n_seen": 32}),
     "不在 malicious_ids"),
    (lambda m: m.update(client_failures=[{"client_id": 6}]), "client_failures"),
    (lambda m: m["run"].update(malicious_per_edge=[3, 3, 2, 2]), "malicious_per_edge"),
    (lambda m: m["run"].update(defense="median"), "run.defense"),
    (lambda m: m["run"].update(edge_shared_blocks=1), "edge_shared_blocks"),
])
def test_gate(snap, mutate, needle):
    m = _as_static(snap)
    mutate(m)
    r = A.route_run(m, "badnet")
    assert r["route"] == "invalid" and any(needle in x for x in r["reasons"]), r.get("reasons")


def test_missing_file_is_missing_not_invalid():
    assert A.route_run(None, "dba") == {"route": "missing"}


# ── 种子汇总 ─────────────────────────────────────────────────────────────────
def _r(route):
    return {"route": route}


def test_conclude_states():
    assert A.conclude({42: _r("missing")})["status"] == "insufficient"
    c = A.conclude({42: _r("local_only")})
    assert c["status"] == "routed_on_s42" and c["next"] == A.NEXT_STEP["local_only"]
    c = A.conclude({42: _r("no_implant"), 43: _r("no_implant"), 44: _r("no_implant")})
    assert c["status"] == "conclusion" and c["branch"] == "no_implant"
    c = A.conclude({42: _r("no_implant"), 43: _r("local_only"), 44: _r("no_implant")})
    assert c["status"] == "single_seed_observation" and c["branch"] is None
    assert A.conclude({42: _r("propagates"), 43: _r("invalid")})["status"] == "invalid_present"


def test_run_all_on_a_results_tree(tmp_path, snap):
    d = tmp_path / "ATK"
    d.mkdir()
    (d / "ATK__badnet__s42.metrics.json").write_text(json.dumps(_as_static(snap)), encoding="utf-8")
    (d / "ATK__dba__s42.metrics.json").write_text(
        json.dumps(_as_static(snap, trigger="dba", e0=0.05, victim=0.02)), encoding="utf-8")
    ctrl = tmp_path / "SNAP"
    ctrl.mkdir()
    (ctrl / "SNAP__collocated__s42.metrics.json").write_text(json.dumps(snap), encoding="utf-8")
    res = A.run_all(tmp_path)
    assert res["attacks"]["badnet"]["branch"] == "propagates"
    assert res["attacks"]["dba"]["branch"] == "no_implant"
    assert res["attacks"]["blended"]["status"] == "insufficient"
    assert set(res["attacks"]["blended"]["runs"]) == {"42"}             # 只有 s42 缺时列出来
    assert res["control_badpfl"]["route_if_applied"] == "propagates"
    out = tmp_path / "x.json"
    assert A.main(["--results", str(tmp_path), "--out", str(out)]) == 0 and out.exists()


# ── 探路组 ATKP 的放行检查（--probe；不是预注册判定）─────────────────────────
def _as_probe(snap, trigger, tamper=False):
    m = _as_static(snap, trigger=trigger)
    m["run"]["n_rounds"] = 4
    for k in ("acc_rounds", "rounds", "checksums", "post_agg_rounds"):
        m[k] = [r for r in m.get(k) or [] if r["round"] <= 4]
    m["per_edge_rounds"] = {r: v for r, v in m["per_edge_rounds"].items() if int(r) <= 4}
    if tamper:
        m["checksums"][2]["global"] = "000000000000"
    return m


def _probe_tree(tmp_path, snap, tamper=False, drop=None):
    d = tmp_path / "ATKP"
    d.mkdir()
    for cell in A.PROBE_CELLS:
        if cell == drop:
            continue
        m = _as_probe(snap, cell.replace("-rep", ""), tamper=tamper and cell == "badnet-rep")
        (d / f"ATKP__{cell}__s42.metrics.json").write_text(json.dumps(m), encoding="utf-8")
    return tmp_path


def test_probe_passes_when_all_four_are_valid_and_the_repeat_is_identical(tmp_path, snap):
    res = A.probe_check(_probe_tree(tmp_path, snap))
    assert res["verdict"] == "pass", res["reasons"]
    assert res["repeat"]["n_checksums"] == 4 and res["repeat"]["n_diffs"] == 0
    assert all(c["n_post_agg"] == 3 for c in res["cells"].values())       # 第 2–4 云轮


def test_probe_fails_on_a_checksum_difference(tmp_path, snap):
    res = A.probe_check(_probe_tree(tmp_path, snap, tamper=True))
    assert res["verdict"] == "fail" and any("不逐位相同" in r for r in res["reasons"])


def test_probe_fails_on_a_missing_cell_and_on_the_60_round_gate(tmp_path, snap):
    res = A.probe_check(_probe_tree(tmp_path, snap, drop="dba"))
    assert res["verdict"] == "fail" and res["cells"]["dba"]["status"] == "missing"
    m = _as_probe(snap, "blended")
    assert not any("跑满" in r for r in A.gate(m, "blended", n_rounds=4))
    assert any("跑满 60" in r for r in A.gate(m, "blended"))
