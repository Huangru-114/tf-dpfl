# current-focus：<方法名>

> 复制这个模板到 `experiments/<axis>/<method>/current-focus.md`，开工前填完。
> 这是你的锚：一个会话只回答这**一个**问题，其余全部记到「已排除」或台账里。

## 现在要回答的唯一问题

（一句话。不是「让 neurotoxin работать」，而是「mask 的 ratio 到底是保留比例还是屏蔽比例」。）

## 判定「完成」的客观标准

- L1：`pytest tests/test_<x>.py` 全绿
- L2：`experiments/<axis>/<method>/expNNN.metrics.json` 里 `final.local_benign_asr` ≥ ___
      且 `final.…acc` 不低于 baseline ___ 个点

（必须是能从 metrics.json 直接读出的数字，不能是「看起来对了」。）

## 参考实现

- 源仓库：
- clone 的 commit：
- 关键文件 / 函数：

## 语义 diff 表

| 论文公式/步骤 | 官方实现 | 本仓库实现 | 差异 | 怎么验证 |
|---|---|---|---|---|
|  |  |  |  |  |

## 已排除的可能

- （每排除一条就记一条，附证据。避免下个会话重走同一条死路。）

## 进展日志

| 日期 | 做了什么 | 证据（metrics/测试） | 结论 |
|---|---|---|---|
|  |  |  |  |
