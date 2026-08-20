#!/bin/bash
# run_l1.sh  —  L1 算法不变量：一条命令，客观、自动、无需人肉判读
#
#   bash run_l1.sh              全部
#   bash run_l1.sh flame        只跑匹配 flame 的
#
# 纯 numpy 的测试哪儿都能跑；需要 TF 的会自动 skip（本地）/ 执行（集群）。

set -uo pipefail
cd "$(dirname "$0")"

PATTERN="${1:-}"
if [ -n "$PATTERN" ]; then
    python3 -m pytest tests/ -v --tb=short -k "$PATTERN"
else
    python3 -m pytest tests/ -v --tb=short
fi
RC=$?

echo
if [ $RC -eq 0 ]; then
    echo "[L1] PASS —— 可以进 L2（bash run_smoke.sh ...）"
else
    echo "[L1] FAIL —— 不 review、不跑通，就不 commit。先看上面第一个失败的断言。"
fi
exit $RC
