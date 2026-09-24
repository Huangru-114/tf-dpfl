# current-focus —— Experiment 3（改版）

> 本文件是 `CLAUDE.md`「新会话开场第 3 步」要读的那一份。
> **写于 2026-09-24**（S1 结束时）。下一会话 = **A1**。

## 下一会话唯一要回答的问题

**A1：攻击部分的实现与官方 Bad-PFL 逐行对齐了吗？**

范围是 `AUDIT.md` 的 A01–A06 和 A14：ξ 的构造、评估时 ξ 用哪个模型、投毒量、生成器训练数据、BN 模式、ASR 定义、生成器结构。

## 客观判据

- 这 7 行的状态都从 `open` 变成 `align` 或 `deviate`，并由用户逐行拍板，写进 `DECISIONS.md`。
- 每个 `align` 都写好 L1 测试的断言（本会话只写断言设计，实现放 A4）。
- A02 里「用哪个恶意端的模型」有明确答案（官方是循环里最后一个恶意端；本仓库 `shared_generator=true`）。

## 开工前

1. 参考源：`raw.githubusercontent.com/fmy266/Bad-PFL/main/<file>`，核对 `fba.py` 的 etag 是否仍是 `d65deddbd62a…`（AUDIT.md 顶部）。若已变化，先重读再对比。
2. `bash run_l1.sh 2>&1 | tail -3` 记下当天的基线。
3. **不跑任何实验**（D-006）。

## 之后的顺序

A2 训练协议 → A3 FedRep / ResNet / 论文（**需要用户提供 FedRep 参考实现与论文 PDF**）→ A4 改代码并升 P2 → S3…S8，见 `PLAN.md` §5。
