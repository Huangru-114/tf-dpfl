"""
tests/test_pack.py  —  一卡多跑（D-047）：作业脚本的接线 + 预注册判定

  · pack.sbatch：显存按需增长、逐个 wait、每个 run 都 collect 并写 exit_code、只传 --config、绑核；
  · harness/pack_test.py：checksum 必须等于 DET 参照（硬门槛）、D-044 的有效性闸、加速比 ≥ 1.5 才采用。

纯 python，不 import TF。
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import pack_test as P        # noqa: E402

MECH = ROOT / "experiments/attack/hfl-mechanism"
PACK = MECH / "pack.sbatch"
SUBMIT = MECH / "pilot/submit_pack_test.sh"


def _code(path: Path) -> str:
    return "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                     if not ln.lstrip().startswith("#"))


# ══════════════════════════════════════════════════════════════════════════
# 作业脚本
# ══════════════════════════════════════════════════════════════════════════
def test_pack_enables_gpu_memory_growth():
    """代码没开显存按需增长：不设这个，第一个进程占满显存，第二个起不来。"""
    assert "export TF_FORCE_GPU_ALLOW_GROWTH=true" in _code(PACK)


def test_pack_waits_on_every_pid_and_collects_every_run():
    code = _code(PACK)
    assert re.search(r'wait "\$\{PIDS\[\$i\]\}"', code), "要逐个 wait 每个子进程并取它的退出码"
    assert "harness/collect_metrics.py" in code and 'd["exit_code"] = rc' in code
    # 收尾在 for 循环里，对每个 run 都做
    assert re.search(r"for \(\(i = 0; i < K; i\+\+\)\); do\s+mkdir -p", code)


def test_pack_passes_only_config_to_main():
    """与 cell.sbatch 同一条规矩：只传 --config（F-001：CLI 静默盖掉 yaml）。"""
    lines = [ln for ln in _code(PACK).splitlines() if "main.py" in ln]
    assert len(lines) == 1
    assert re.findall(r"--[\w-]+", lines[0]) == ["--config"]


def test_pack_pins_each_run_to_exactly_four_cores():
    """核数会改变结果（F-047）：每个 run 必须看到与单跑（cell.sbatch 的 -c 4）相同的核数。"""
    code = _code(PACK)
    assert 'PER="${TFDPFL_CORES_PER_RUN:-4}"' in code
    assert 'taskset -c "$CORES" $PY main.py' in code
    assert "Cpus_allowed_list" in code
    # 核不够 / 没有 taskset → 拒绝启动，而不是退化成「核数随 K 变」
    assert re.search(r'-lt \$\(\( K \* PER \)\) \]; then\s+echo[^\n]*\n?[^\n]*exit 2', code)
    assert "没有 taskset" in code


def test_single_run_job_script_still_uses_four_cores():
    """cell.sbatch 的 -c 4 是一卡一跑的核数；pack 的每 run 核数缺省必须与它相同。"""
    head = (MECH / "cell.sbatch").read_text(encoding="utf-8")
    assert re.search(r"^#SBATCH -c 4$", head, re.M)


def test_pack_exit_code_reflects_any_failed_run():
    code = _code(PACK)
    assert 'WORST=${RCS[$i]}' in code and 'exit "$WORST"' in code


def test_submit_pack_test_requests_four_cores_per_run_and_uses_det_config():
    code = _code(SUBMIT)
    assert "sbatch -c $((4 * K))" in code
    assert "DET__rep1__s42.yaml" in code
    assert "pack.sbatch" in code


# ══════════════════════════════════════════════════════════════════════════
# 判定
# ══════════════════════════════════════════════════════════════════════════
REF_CS = [f"{r:012x}" for r in range(1, 6)]


def _m(cs=REF_CS, train=900.0, bd=300.0, rc=0, fails=None):
    return {"exit_code": rc, "client_failures": fails or [],
            "checksums": [{"round": r, "global": h} for r, h in enumerate(cs, 1)],
            "timing_summary": {"round_time_total_s": train, "bd_eval_total_s": bd}}


def _ref():
    return P.reference([_m(), _m(train=920.0, bd=280.0)])


def test_thresholds_are_the_decision_values():
    assert P.MIN_SPEEDUP == 1.5 and P.DET_ROUNDS == 5 and P.EXPECTED_KS == (2, 3)


def test_reference_needs_two_agreeing_det_runs():
    assert _ref()["verdict"] == "ok" and _ref()["t_solo_s"] == pytest.approx(1200.0)
    bad = _m(cs=REF_CS[:4] + ["ffffffffffff"])
    assert P.reference([_m(), bad])["verdict"] == "missing"
    assert P.reference([_m(), None])["verdict"] == "missing"


def test_eligible_when_checksums_match_and_fast_enough():
    # K=3，每份 1500 s → 加速比 3 × 1200 / 1500 = 2.4
    r = P.judge_pack(3, [_m(train=1200.0, bd=300.0)] * 3, _ref())
    assert r["verdict"] == "eligible"
    assert r["speedup"] == pytest.approx(2.4) and r["gpu_h_per_run_vs_solo"] == pytest.approx(0.417, abs=1e-3)


def test_too_slow_below_threshold():
    # K=2，每份 1800 s → 2 × 1200 / 1800 = 1.33 < 1.5
    r = P.judge_pack(2, [_m(train=1500.0, bd=300.0)] * 2, _ref())
    assert r["verdict"] == "too-slow"


def test_speedup_uses_the_slowest_slot():
    runs = [_m(train=1200.0, bd=300.0), _m(train=2100.0, bd=300.0)]    # 1500 / 2400
    r = P.judge_pack(2, runs, _ref())
    assert r["t_pack_max_s"] == pytest.approx(2400.0) and r["speedup"] == pytest.approx(1.0)


def test_diverged_checksum_blocks_adoption_even_if_fast():
    """反向锚点：快但不确定 → 不能采用（确定性是硬门槛，A15）。"""
    fast = _m(train=600.0, bd=100.0)
    bad = _m(cs=REF_CS[:2] + ["ffffffffffff"] + REF_CS[3:], train=600.0, bd=100.0)
    r = P.judge_pack(2, [fast, bad], _ref())
    assert r["verdict"] == "diverged" and r["diverged_slots"] == [2]


def test_swallowed_client_failure_is_invalid():
    r = P.judge_pack(2, [_m(), _m(fails=[{"client_id": 5, "error": "x"}])], _ref())
    assert r["verdict"] == "invalid" and r["reasons"][0].startswith("s2: ")


def test_missing_slot_is_missing():
    assert P.judge_pack(3, [_m(), None, _m()], _ref())["verdict"] == "missing"


def test_judge_all_picks_the_fastest_eligible_k():
    packs = {2: [_m(train=1000.0, bd=200.0)] * 2,        # 2.0
             3: [_m(train=1300.0, bd=300.0)] * 3}        # 2.25
    res = P.judge_all(packs, _ref())
    assert res["adopt_k"] == 3 and "K=3" in res["decision"]
    packs[3] = [_m(cs=["x" * 12] * 5)] * 3               # K=3 不确定 → 退回 K=2
    assert P.judge_all(packs, _ref())["adopt_k"] == 2


def test_judge_all_rejects_when_nothing_is_eligible():
    packs = {2: [_m(train=1600.0, bd=400.0)] * 2,        # 1.2
             3: [_m(train=2000.0, bd=600.0)] * 3}        # 1.38
    res = P.judge_all(packs, _ref())
    assert res["adopt_k"] is None and "不采用" in res["decision"]


def test_real_det_reference_reads_from_the_pilot_files():
    """参照不手抄：从 DET 第二轮的 metrics 读出来就是 F-045 里的那串。"""
    ref = P.reference([json.loads(p.read_text()) for p in P.DET_FILES])
    assert ref["verdict"] == "ok"
    assert ref["checksums"][0] == "d259128657fd" and ref["checksums"][-1] == "931d1867fbac"
    assert ref["t_solo_s"] == pytest.approx(1202.25)
