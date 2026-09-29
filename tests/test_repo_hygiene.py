"""
tests/test_repo_hygiene.py  —  CLAUDE.md 的红线：什么永远不进 git（2026-09-29）

历史里 96% 的体积（约 280 MiB）是早已删除的 *.h5 权重、误提交的 venv、wandb 运行记录
（`python3 harness/git_size_report.py` 复现）。`.gitignore` 挡得住 `git add .`，挡不住 `git add -f`
和改名后的路径 —— 这里对**当前跟踪的文件**再查一遍，入库那一刻 L1 就红。

纯标准库 + `git ls-files`；不是 git 仓库（例如解压的源码包）时 skip。
"""

import fnmatch
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

MAX_BYTES = 10 * 2 ** 20                          # CLAUDE.md：单文件 > 10 MB → 回去查 .gitignore

# (规则名, 判定函数)；路径是 git ls-files 给的仓库相对路径（正斜杠）
FORBIDDEN = [
    ("checkpoint", lambda p: p.endswith((".h5", ".pt", ".pth", ".ckpt"))),
    ("array", lambda p: p.endswith((".npz", ".npy")) and not p.startswith("tests/fixtures/")),
    ("wandb", lambda p: p.startswith("wandb/") or "/wandb/" in p),
    ("venv", lambda p: fnmatch.fnmatch(p, "lib/python*") or fnmatch.fnmatch(p, "*/lib/python*")
     or "site-packages/" in p or p.endswith("pyvenv.cfg") or p.startswith(("bin/", "lib/", "include/"))),
    ("dumps", lambda p: p.startswith("tfdpfl-dumps/") or "/tfdpfl-dumps/" in p),   # S9 存盘只留集群
]


def violations(paths, sizes=None):
    """[(path, rule)]。sizes：{path: bytes}（缺的按 0）。"""
    out = []
    for p in paths:
        for name, bad in FORBIDDEN:
            if bad(p):
                out.append((p, name))
        if sizes and sizes.get(p, 0) > MAX_BYTES:
            out.append((p, "size>10MB"))
    return out


def _tracked():
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("不是 git 仓库或没有 git")
    return [p for p in r.stdout.decode("utf-8", "replace").split("\0") if p]


def test_no_tracked_file_breaks_the_red_lines():
    paths = _tracked()
    sizes = {p: (ROOT / p).stat().st_size for p in paths if (ROOT / p).is_file()}
    assert violations(paths, sizes) == []


def test_the_scan_is_not_vacuous():
    """反向自检：确实扫到了东西（零个文件会让上一条空洞地通过）。"""
    paths = _tracked()
    assert len(paths) > 100 and "CLAUDE.md" in paths and "fedavg/main.py" in paths


@pytest.mark.parametrize("path,rule", [
    ("fedavg/wrn28_4_cifar100-sweep-3.h5", "checkpoint"),       # 历史上真实入过库的 7 个之一
    ("experiments/x/model.pt", "checkpoint"),
    ("results/probe_logits.npz", "array"),
    ("fedavg/wandb/run-2026/files/output.log", "wandb"),
    ("lib/python3.10/site-packages/pyarrow/libarrow.so", "venv"),  # 历史上的 venv
    ("bin/python3", "venv"),
    ("pyvenv.cfg", "venv"),
    ("tfdpfl-dumps/G8__s42.123/round30.npz", "array"),
])
def test_rules_catch_the_historical_junk(path, rule):
    """反向锚点：历史里真实出现过的那几类路径，规则都拦得住。"""
    assert (path, rule) in violations([path])


def test_allowed_paths_pass():
    ok = ["tests/fixtures/tiny_probe.npz", "fedavg/models/cnn.py", "harness/figures.py",
          "experiments/attack/hfl-mechanism/results/P2/G8F/G8F__std__s42.metrics.json",
          "fedavg/binning.py", "fedavg/library.py"]
    assert violations(ok) == []


def test_size_rule():
    assert violations(["a.json"], {"a.json": MAX_BYTES + 1}) == [("a.json", "size>10MB")]
    assert violations(["a.json"], {"a.json": MAX_BYTES}) == []
