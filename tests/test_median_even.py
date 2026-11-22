"""
tests/test_median_even.py  —  逐坐标中位数在补充实验 D4（N-015）里的实际规模（纯 numpy，不 import TF）

D4 的两个臂：边缘层中位数（每台边缘每个边缘轮只有 2 或 3 个上传，交替）、云层中位数（4 个边缘模型）。
这里用手造张量钉住 `MedianDefense.aggregate` 在 n = 2 / 3 / 4 时的解析值：
  n = 2 → 两者的**不加权**平均；n = 3 → 中间那个；n = 4 → 中间两个的平均；
  不看样本数（坐标级防御按等权，文献如此）；逐坐标独立（私有分类头的坐标不影响主干坐标，陷阱 #9 不咬中位数）；
  dtype 不变（float32 进、float32 出）。
"""

import numpy as np
import pytest

from defense.trimmed_mean import MedianDefense


def _upd(vals, n=1):
    """一个上传 = ([body, head], n_samples, loss, t)；body / head 各一个 float32 张量。"""
    body, head = vals
    return ([np.asarray(body, np.float32), np.asarray(head, np.float32)], n, 0.0, 0.0)


AGG = MedianDefense({}).aggregate


def test_two_uploads_give_the_unweighted_mean():
    out = AGG([_upd(([1.0, 4.0], [0.0]), n=1), _upd(([3.0, 8.0], [2.0]), n=99)], None)
    assert np.array_equal(out[0], np.array([2.0, 6.0], np.float32))       # 样本数 1 vs 99 不影响
    assert np.array_equal(out[1], np.array([1.0], np.float32))


def test_three_uploads_give_the_middle_one_per_coordinate():
    ups = [_upd(([1.0, 9.0], [5.0])), _upd(([7.0, 2.0], [1.0])), _upd(([4.0, 5.0], [3.0]))]
    out = AGG(ups, None)
    assert np.array_equal(out[0], np.array([4.0, 5.0], np.float32))       # 逐坐标各取中间，不是取同一个上传
    assert np.array_equal(out[1], np.array([3.0], np.float32))


def test_four_uploads_give_the_mean_of_the_middle_two():
    """云层臂：4 个边缘模型，集中布点时只有 E0 被污染 —— 一个离群值完全不进结果。"""
    ups = [_upd(([100.0, -100.0], [0.0])), _upd(([1.0, 2.0], [0.0])),
           _upd(([3.0, 4.0], [0.0])), _upd(([5.0, 6.0], [0.0]))]
    out = AGG(ups, None)
    assert np.array_equal(out[0], np.array([4.0, 3.0], np.float32))       # (3+5)/2、(2+4)/2
    assert out[0].dtype == np.float32 and out[1].dtype == np.float32


@pytest.mark.parametrize("n", [2, 3, 4])
def test_head_coordinates_do_not_move_body_coordinates(n):
    """逐坐标运算：把分类头换成任意值，主干的结果逐位不变（FedRep 边缘服务器之后只取主干索引）。"""
    rng = np.random.default_rng(n)
    bodies = rng.normal(size=(n, 6)).astype(np.float32)
    a = AGG([_upd((b, rng.normal(size=3))) for b in bodies], None)
    b = AGG([_upd((b, 1e6 * rng.normal(size=3))) for b in bodies], None)
    assert np.array_equal(a[0], b[0])
