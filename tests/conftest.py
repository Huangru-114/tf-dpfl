"""
tests/conftest.py

集群与本地都以 `cwd=fedavg` 运行主程序（import 形如 `from client.x import ...`），
所以测试也把 fedavg/ 放进 sys.path，保持与运行时一致的 import 语义。

跑法（仓库根目录）：
    pytest tests/ -v

纯 numpy 的测试本地就能跑；需要 TF 的测试用 pytest.importorskip("tensorflow")
标记，本地自动 skip，在集群上跑。
"""

import sys
from pathlib import Path

import pytest

FEDAVG_ROOT = Path(__file__).resolve().parent.parent / "fedavg"
if str(FEDAVG_ROOT) not in sys.path:
    sys.path.insert(0, str(FEDAVG_ROOT))


@pytest.fixture
def rng():
    """所有测试用固定种子的独立 RNG——不碰全局 np.random。"""
    import numpy as np
    return np.random.default_rng(20260819)
