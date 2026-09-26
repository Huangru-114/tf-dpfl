"""
tests/test_eval_pm_and_asr_columns.py  —  A4：评估口径（AUDIT A06 / A28，纯 python 部分）

  A28 / D-033：fresh-PM = [当前 edge 权重, 客户端私有部分]（utils/pm.py:compose_pm）
  A06 / D-019：ASR 四列（过滤 / 不过滤 × 仅良性 / 全体），attack/asr_counting.py
  回程：[ASR4] [ASRwb] [StaleASR] [Stale] [Checksum] [设定5] → metrics.json（schema 3）

不 import TF。TF 端的集成断言（主 ASR 与主 pm_acc 同一个组装模型）在
tests/test_eval_integration.py。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from utils.pm import compose_pm
from attack.asr_counting import asr_counts, mean_defined, rates_from_counts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from collect_metrics import collect   # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# A28：compose_pm
# ══════════════════════════════════════════════════════════════════════════
def test_compose_pm_replaces_only_private_positions(rng):
    edge = [rng.normal(size=s).astype(np.float32) for s in ((3, 3), (3,), (4,), (4,), (3, 2), (2,))]
    priv_idx = [2, 3, 4, 5]                      # 统计量 ×2 + head ×2
    priv = [np.full_like(edge[i], 7.0) for i in priv_idx]
    before = [w.copy() for w in edge]
    out = compose_pm(edge, priv, priv_idx)
    for i, w in enumerate(out):
        want = priv[priv_idx.index(i)] if i in priv_idx else edge[i]
        np.testing.assert_array_equal(w, want)
    for a, b in zip(edge, before):               # 不修改 edge 权重
        np.testing.assert_array_equal(a, b)


def test_compose_pm_without_private_part_is_the_edge_model(rng):
    edge = [rng.normal(size=(2, 2)).astype(np.float32)]
    out = compose_pm(edge, [], [])
    assert out[0] is edge[0]


def test_compose_pm_rejects_mismatches(rng):
    edge = [np.zeros((2, 2), np.float32), np.zeros(2, np.float32)]
    with pytest.raises(ValueError):
        compose_pm(edge, [np.zeros(3)], [1])
    with pytest.raises(IndexError):
        compose_pm(edge, [np.zeros(2)], [5])
    with pytest.raises(ValueError):
        compose_pm(edge, [], [1])


# ══════════════════════════════════════════════════════════════════════════
# A06：四列的解析算例
# ══════════════════════════════════════════════════════════════════════════
T = 0


def _probe(t, m):
    """t 个目标类样本 + m 个其他类样本。"""
    return np.array([T] * t + [1 + (i % 9) for i in range(m)])


def test_always_predict_target_gives_one_everywhere():
    y = _probe(3, 7)
    f, u = rates_from_counts(asr_counts(np.full(10, T), y, T))
    assert f == 1.0 and u == 1.0


def test_always_predict_true_label():
    """恒预测真标签：不过滤 = t/(t+m)（目标类样本「命中」），过滤 = 0。"""
    y = _probe(3, 7)
    f, u = rates_from_counts(asr_counts(y.copy(), y, T))
    assert f == 0.0 and u == pytest.approx(3 / 10)


def test_probe_with_only_target_class_is_undefined_not_zero():
    y = _probe(4, 0)
    f, u = rates_from_counts(asr_counts(y.copy(), y, T))
    assert f is None and u == 1.0


def test_benign_vs_all_client_means():
    """良性端过滤 ASR 0.2 / 0.4，恶意端 1.0 → 仅良性 0.3，全体 (0.2+0.4+1.0)/3。"""
    benign = [0.2, 0.4]
    mal = [1.0]
    assert mean_defined(benign) == pytest.approx(0.3)
    assert mean_defined(benign + mal) == pytest.approx(1.6 / 3)
    assert mean_defined([None, 0.5]) == 0.5 and mean_defined([None]) is None


# ══════════════════════════════════════════════════════════════════════════
# 回程：新行 → metrics.json
# ══════════════════════════════════════════════════════════════════════════
LOG = """\
[设定5] eval_attacker=17 | eval_attacker_edge=1 | eval_xi_model=fixed_attacker
[Round   1] Broadcasting to 2 edges...
[Stale] Round 1 | pm_acc=0.6123
  [Cloud] GM=0.5000 | EM=0.5100 PM=0.6400 | loss=1.2000 | time=10.0s | comm=1.0MB (total=1MB)
[Checksum] Round 1 | global=0123456789ab
[Backdoor] Round 1 | GM_ASR=0.100 | EM_ASR=0.200 | local_benign=0.300 (same_edge=0.300, diff_edge=n/a) | local_malicious=0.900
[ASR4] Round 1 | benign_filtered=0.3000 | benign_unfiltered=0.3500 | all_filtered=0.3600 | all_unfiltered=0.4000 | global_unfiltered=n/a
[ASRwb] Round 1 | local_benign=0.5000
[StaleASR] Round 1 | local_benign=0.2500 | local_malicious=0.8000
[Round   2] Broadcasting to 2 edges...
  [Cloud] GM=0.5000 | EM=0.5100 | loss=1.2000 | time=10.0s | comm=1.0MB (total=2MB)
[Checksum] Round 2 | global=1234e5678901
"""


def test_side_columns_land_in_rounds_and_acc_rounds():
    m = collect(LOG)
    assert m["schema_version"] == 3
    r = m["rounds"][0]
    assert r["local_benign_asr"] == 0.3
    assert r["local_benign_asr_unfiltered"] == 0.35
    assert r["local_all_asr"] == 0.36 and r["local_all_asr_unfiltered"] == 0.4
    assert r["global_asr_unfiltered"] is None
    assert r["local_benign_asr_whitebox"] == 0.5
    assert r["local_benign_asr_stale"] == 0.25 and r["local_malicious_asr_stale"] == 0.8
    acc = {a["round"]: a for a in m["acc_rounds"]}
    assert acc[1]["pm_acc"] == 0.64 and acc[1]["pm_acc_stale"] == 0.6123
    assert acc[2]["pm_acc_stale"] is None                          # 非评估轮：null，不是 0
    assert m["run"]["eval_attacker"] == 17 and m["run"]["eval_attacker_edge"] == 1
    assert m["checksums"] == [{"round": 1, "global": "0123456789ab"},
                              {"round": 2, "global": "1234e5678901"}]   # 长得像数字也是字符串


def test_old_log_has_null_side_columns():
    old = "\n".join(ln for ln in LOG.splitlines()
                    if not ln.startswith(("[ASR4]", "[ASRwb]", "[StaleASR]", "[Stale]",
                                          "[设定5]", "[Checksum]")))
    m = collect(old)
    r = m["rounds"][0]
    for k in ("local_benign_asr_unfiltered", "local_benign_asr_whitebox",
              "local_benign_asr_stale", "local_all_asr"):
        assert r[k] is None
    assert m["acc_rounds"][0]["pm_acc_stale"] is None
    assert m["checksums"] == [] and m["run"]["eval_attacker"] is None
