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


# ── L1 一律跑在 CPU 上 ──────────────────────────────────────────────────────
# 三个理由，按重要性排：
#   1. `tests/test_probe_determinism.py` 的断言是**逐比特**的（同 order_seed 的两条
#      clean twin 必须给出同一个数）。GPU 上 cuDNN 的部分 kernel 默认非确定性，
#      那些断言会因为环境变红 —— 而那是环境问题不是回归，会污染 L1 门禁。
#   2. **不能靠打开 TF 的确定性开关在 GPU 上救它**：Bad-PFL 的 FGSM 要在
#      training=False 下对输入求梯度、模型又带 BN，TF 没有这个形状的确定性 GPU
#      kernel，请求确定性会直接抛 UnimplementedError（CLAUDE.md 陷阱 #11）。
#   3. L1 是算法不变量测试，本来就不需要 GPU（CLAUDE.md：本地秒级）。把它钉在 CPU 上
#      顺带保证了「登录节点跑」和「GPU 作业里跑」得到同一份结果。
#
# 必须在任何 GPU 被初始化**之前**调用，所以放在 conftest 的模块级（pytest 保证
# conftest 先于所有测试模块导入），而不是某个测试文件里 —— 放在测试文件里的话，
# 先导入的那个文件若已经碰过 GPU，这里就会抛 RuntimeError。
#
# 刻意**不**调 `tf.config.experimental.enable_op_determinism()`：它会把所有
# **未播种**的随机 op 变成硬错误（RuntimeError: Random ops require a seed），
# 而进程级的全局开关会波及整个测试会话里的其他文件。CPU 上不开它也已经逐比特一致
# （实测 maxdiff = 0.000e+00），所以没必要付这个代价。
try:
    import tensorflow as _tf
    _tf.config.set_visible_devices([], "GPU")
except Exception:                      # noqa: BLE001
    # 无 TF（本地）→ 需要 TF 的测试本来就整体 skip；
    # GPU 已初始化 → 保持原状，让测试自己去暴露问题，而不是在收集阶段就炸。
    pass


@pytest.fixture
def rng():
    """所有测试用固定种子的独立 RNG——不碰全局 np.random。"""
    import numpy as np
    return np.random.default_rng(20260819)
