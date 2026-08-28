"""
tests/test_probe_shell_guard.py  —  守卫 run_probe.sh 里那条会静默炸掉训练的写法。

**为什么值得一条测试**（这是真发生过的事，见 CLAUDE.md 陷阱 #11）：
在 run_probe.sh 里 `export TF_DETERMINISTIC_OPS=1` 看起来完全合理 —— paired
counterfactual 确实需要确定性。但那个 export 是**作用于整个作业**的，于是训练阶段
也带上了它，而 Bad-PFL 的 FGSM 要在 `training=False` 下对输入求梯度、模型又带 BN，
TF 没有这个形状的确定性 GPU kernel：

    UnimplementedError: A deterministic GPU implementation of fused batch-norm
    backprop, when training is disabled, is not currently available.

脚本本身没有语法错、`bash -n` 通过、前几十轮的 setup 日志全是正常的 —— 一直跑到
第一次后门评估才炸，而报错信息（fused batch-norm）与真实起因（探针的确定性开关）
之间隔着三层调用栈。正是本仓库最贵的那类「静默跑错 / 迟到爆炸」。

确定性现在由 `probe.run_probe --determinism` 在**探针进程内**处理。

纯文本检查，不需要 TF，本地秒级。
"""
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "run_probe.sh"

# TF 的确定性开关。它们**只**允许出现在注释里（解释为什么不能设），不允许是活代码。
DETERMINISM_VARS = ("TF_DETERMINISTIC_OPS", "TF_CUDNN_DETERMINISTIC")


@pytest.fixture(scope="module")
def lines():
    assert SCRIPT.exists(), f"{SCRIPT} 不存在"
    return SCRIPT.read_text(encoding="utf-8").splitlines()


def _code_lines(lines):
    """去掉注释行与空行，只留可执行的部分。"""
    return [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


@pytest.mark.parametrize("var", DETERMINISM_VARS)
def test_no_job_wide_determinism_export(lines, var):
    """
    作业级 export 会波及**训练阶段** → UnimplementedError。这是原样复现过的 bug。
    """
    offenders = [ln for ln in _code_lines(lines)
                 if re.search(rf"\b{var}\b", ln) and "=" in ln]
    assert not offenders, (
        f"run_probe.sh 里有活的 {var} 设定：{offenders}\n"
        f"它会作用于整个作业（含训练阶段），而 Bad-PFL 的 FGSM + BN 在 GPU 上没有"
        f"确定性实现 → 训练跑到后门评估时抛 UnimplementedError。\n"
        f"确定性请走 `probe.run_probe --determinism`（探针进程内），见 CLAUDE.md 陷阱 #11。")


def test_the_reason_is_documented_in_the_script(lines):
    """
    只删掉不够 —— 下一个人会把它加回来。脚本里必须留着「为什么不能设」。
    """
    text = "\n".join(ln for ln in lines if ln.lstrip().startswith("#"))
    assert any(v in text for v in DETERMINISM_VARS), \
        "run_probe.sh 的注释里应当说明为什么不能设 TF_DETERMINISTIC_OPS，否则会被加回来"
    assert "陷阱 #11" in text, "注释应指向 CLAUDE.md 陷阱 #11"


def test_probe_stage_passes_a_determinism_mode(lines):
    """确定性没有被简单丢掉，而是移交给了探针进程。"""
    code = "\n".join(_code_lines(lines))
    assert "--determinism" in code, "阶段 2 应当把 --determinism 传给 probe.run_probe"
    assert "PROBE_DETERMINISM" in code, "应当留一个环境变量开关覆盖缺省模式"


def test_training_stage_invocation_carries_no_determinism_flag(lines):
    """训练阶段那一行本身也不能夹带确定性开关。"""
    for ln in _code_lines(lines):
        if "main.py" in ln:
            assert not any(v in ln for v in DETERMINISM_VARS), \
                f"训练阶段的调用行不该带确定性开关: {ln}"


def test_script_still_sources_cluster_env(lines):
    """
    CLAUDE.md：不要在新脚本里硬写容器路径，一律经 cluster_env.sh 解析出 $PY。
    """
    code = "\n".join(_code_lines(lines))
    assert "cluster_env.sh" in code
    assert "apptainer" not in code, "容器路径应当收口在 cluster_env.sh，不要硬写"


# ── 判据汇总的 schema 契约 ─────────────────────────────────────────────────
def test_summary_printer_matches_the_json_writer_actually_produces(lines, tmp_path):
    """
    run_probe.sh 末尾那段判据汇总是**手写**的 JSON 取键（`v['signal']['signal_over_floor']`
    之类）。改了 `writer.verdicts` 的字段名而忘了改它，脚本会在跑完几小时之后、
    在最后一步抛 KeyError —— 那时 CSV/JSON 其实都已经好了，但作业以非零码结束。

    本仓库对这类「两边各写一份格式」的情况一贯分开测（见陷阱 #10：
    「解析器认得格式 ≠ 代码会打印它」）。这里把脚本里那段 python **原样抠出来跑**，
    喂给 `writer` 真实产出的 payload。
    """
    import json
    import subprocess
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fedavg"))
    from probe.writer import verdicts, write_summary

    summaries = [
        {"role": "benign", "bd_norm": 0.0, "stochastic_norm": 1.0},
        {"role": "malicious", "bd_norm": 9.0, "stochastic_norm": 1.0},
    ]
    payload = {
        "run": {"round": 7, "determinism": "cpu"},
        "verdicts": verdicts(summaries, determinism="cpu"),
        "aborted": None,
        "client_failures": [],
        "summaries": summaries,
    }
    js = write_summary(tmp_path / "probe_summary_t.json", payload)

    # 抠出脚本里判据汇总那段 python。脚本里有**多个** PYEOF heredoc
    # （另一个是 checkpoint 目录解析），按内容挑读 verdicts 的那个。
    text = "\n".join(lines)
    blocks = re.findall(r"<<'PYEOF'\n(.*?)\nPYEOF", text, re.S)
    cands = [b for b in blocks if "verdicts" in b]
    assert len(cands) == 1, f"期望恰好 1 段判据汇总 python，找到 {len(cands)} 段"
    snippet = cands[0]

    r = subprocess.run([sys.executable, "-c", snippet, str(js)],
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        f"判据汇总段跑不动 —— writer.verdicts 的字段名与 run_probe.sh 对不上了。\n"
        f"stderr:\n{r.stderr}")
    assert "round=7" in r.stdout
    assert "determinism=cpu" in r.stdout
    # 判据 1 pass / 判据 2 pass 都要真的被打出来，不能是 None
    assert "pass=True" in r.stdout
    assert "None" not in r.stdout.replace("determinism=cpu", "")
