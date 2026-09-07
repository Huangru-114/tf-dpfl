"""`experiments/METRICS.md` 里的常数必须与代码/配置一致（防陷阱 #14）。

陷阱 #14 的教训：`experiments/` 下的文档会比数据旧，而且**骗过了一份报告**。
口径说明页是要直接贴进论文的东西 —— 它一旦和代码对不上，错的就是论文里的
方法学描述。所以这里把文档里的每个数字回头对一遍代码。

与 Bad-PFL 侧的 `diag/tests/test_metrics_doc.py` 是**同一份文档的两半**：
那边管攻击/触发器的常数，这边管 HFL 侧的数据划分与 ASR 探针。

纯 stdlib，本地秒级。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "experiments" / "METRICS.md"
BD_EVAL = ROOT / "fedavg" / "attack" / "backdoor_eval.py"
BD_SERVER = ROOT / "fedavg" / "server" / "backdoor_server.py"
ANCHOR = ROOT / "experiments" / "attack" / "hfl-propagation" / "2edge_distributed.yaml"

DOC_TEXT = DOC.read_text(encoding="utf-8")
EXP3_CONFIGS = sorted((ROOT / "experiments" / "attack" / "hfl-propagation").glob("*.yaml"))
CALIB_CONFIGS = sorted((ROOT / "experiments" / "calibration").glob("*.yaml"))


def _yaml_values(path: Path, key: str) -> set:
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = re.match(rf"^\s+{re.escape(key)}:\s*(.+)$", line)
        if m:
            out.add(m.group(1).split("#")[0].strip())
    return out


def test_doc_exists_and_is_the_same_page_as_the_bad_pfl_copy():
    assert DOC.exists()
    assert "两库各存一份，内容相同" in DOC_TEXT, "少了同步提示 —— 两份会各改各的"
    assert "Bad-PFL/diag/METRICS.md" in DOC_TEXT


def test_asr_is_filtered_on_this_side():
    """tf-dpfl 一直是 filtered（排除目标类）—— 文档据此说两库的分歧在哪。"""
    src = BD_EVAL.read_text(encoding="utf-8")
    assert re.search(r"mask\s*=\s*\(y_test\s*!=\s*int\(target_label\)\)", src), \
        "ASR 的分母不再排除目标类了 —— 文档里 filtered/unfiltered 那节要重写"
    assert "filtered" in DOC_TEXT


def test_empty_group_returns_none_not_zero():
    """无定义 ≠ 0（陷阱 #13）。文档把它写成了一条通用约定。"""
    src = BD_EVAL.read_text(encoding="utf-8")
    body = src[src.index("def compute_asr("):]
    body = body[:body.index("\ndef ")] if "\ndef " in body else body
    assert "return None" in body, "compute_asr 无合格样本时不再返回 None"
    assert "无定义 ≠ 0" in DOC_TEXT or "绝不填 0" in DOC_TEXT


@pytest.mark.parametrize("layer,needle", [
    # 三层的探针名各自出现在 backdoor_eval.py 里：global 是传进来的形参
    # `global_test_ds`（实参 self.test_dataset 在 backdoor_server 那边，
    # 由 test_the_probe_is_not_the_raw_x_test_anymore 管），edge 与 client
    # 是各自取自己的那份。第一版在 backdoor_eval 里找 self.test_dataset，
    # 找错了文件。
    ("global", "global_test_ds"),
    ("edge", "edge.get_test_dataset()"),
    ("client", "c.test_dataset"),
])
def test_each_level_is_scored_on_its_own_held_out_shard(layer, needle):
    """文档里那张「哪一层用哪个探针」的表，逐行对代码。

    这三条是 2026-09 那次修复的全部内容；退回去用原始 x_test 就是在训练过的
    图上测 ASR，而文档还会说是留出分片。
    """
    src = BD_EVAL.read_text(encoding="utf-8")
    assert needle in src, f"{layer} 层的探针变了（找不到 {needle}）"


def test_the_probe_is_not_the_raw_x_test_anymore():
    """反向锚点：`_backdoor_eval` 不能再把 x_test 当分层 ASR 的探针。"""
    src = BD_SERVER.read_text(encoding="utf-8")
    body = src[src.index("def _backdoor_eval("):]
    body = body[:body.index("\n    def _eval_drift")]
    call = body[body.index("evaluate_hierarchical_asr("):]
    call = call[:call.index(")\n")]
    assert "self.x_test" not in call, "分层 ASR 又用回原始 x_test 了"
    assert "self.test_dataset" in call


def test_per_client_test_ratio_matches_the_doc():
    """文档写 0.25 -> 每客户端约 150 张留出、约 135 张进 ASR 分母。"""
    ratios = _yaml_values(ANCHOR, "per_client_test_ratio")
    assert ratios == {"0.25"}, f"锚点 config 的留出比例是 {ratios}，文档写的是 0.25"
    for number in ("0.25", "150", "135", "600"):
        assert number in DOC_TEXT, f"文档里少了 {number}"


def test_target_label_is_zero_in_every_config():
    """两库统一到 target=0。任何一个 config 漏改，跨库比较就不成立。"""
    bad = {}
    for cfg in EXP3_CONFIGS + CALIB_CONFIGS:
        vals = _yaml_values(cfg, "target_label")
        if vals and vals != {"0"}:
            bad[cfg.name] = vals
    assert not bad, f"这些 config 的 target_label 不是 0：{bad}"
    assert "0 = airplane" in DOC_TEXT


def test_asr_max_samples_matches_the_doc():
    vals = _yaml_values(ANCHOR, "asr_max_samples")
    assert vals == {"2000"}, f"asr_max_samples = {vals}，文档写的是 2000"
    assert "asr_max_samples" in DOC_TEXT and "2000" in DOC_TEXT


def test_doc_says_the_merge_is_not_the_bug():
    """守住那次误判：合并 train+test 是 PFLlib 口径，**不是**泄漏。
    文档必须把话说清楚，否则下一个人又会去"修"它（`test_no_test_leakage.py`
    里也有一条同样目的的守卫）。"""
    assert "合并本身不造成泄漏" in DOC_TEXT
    assert "全局不相交" in DOC_TEXT


def test_doc_does_not_claim_a_direction_for_the_probe_fix():
    """影响幅度未知 —— 文档不能替它下结论（我们撤回过一次这种话）。"""
    assert "影响幅度未知" in DOC_TEXT
    for forbidden in ("整体抬高", "只会抬高", "相对趋势仍然成立"):
        assert forbidden not in DOC_TEXT, f"文档又给结论了：{forbidden}"


def test_english_section_has_no_cjk():
    """导师是英文读者；那一节要能直接贴进报告。"""
    assert "Threat model." in DOC_TEXT
    english = DOC_TEXT[DOC_TEXT.index("Threat model."):]
    cjk = [ch for ch in english if "一" <= ch <= "鿿"]
    assert not cjk, f"英文段里混进了中文：{''.join(cjk[:20])}"
