# current-focus —— attack / IBA（第一版：打通端到端）

> 官方：`sail-research/iba`（NeurIPS'23）。IBA = LIRA 可学习触发器
> `noise=G(x)·eps; x_p=clip(x+noise)` + target-network 双缓冲 + 跨轮"逐步植入"。
> 本会话把 IBA 作为攻击轴的一个基线接入，走已被 Bad-PFL exp007 验证的
> 「可学习触发器 + 共享持久生成器」路径。

## 本会话唯一要回答的问题

**IBA 的端到端接线是否成立？** 即：在最小 smoke 上，恶意端每轮参与、无被吞异常、
run 段自证跑的是 iba，且 `local_malicious_asr` 起得来（恶意端自己的后门成立）。
**迁移性 / irreversible 不在本会话判据内**——那要全长 ResNet run（见文末）。

## 客观判据

**L1（已完成，本地/容器秒级）** —— `tests/test_iba_trigger.py`，10 条不变量全绿：
扰动预算 `|G(x)·eps|≤eps`、clip 边界、投毒比例/标签精确、seeded RNG 可复现、
target-network 同步、净增1 缩放（γ=1 恒等 / γ=k 精确放大 / 仅恶意端）、共享生成器注入。
> 已在 TF 2.21 + tf-keras（Keras2 兼容）实测：`16 passed`（IBA 10 + BadPFL 6）；
> 全量 `tests/` = 2 failed(既有陷阱#4 neurotoxin，与 IBA 无关) / 256 passed。

**L2-接线（待集群 smoke）** —— 判 PASS 的充要条件：
- `client_failures == []`、`errors == []`；
- 两恶意端每轮都在 `malicious_selected_rounds`；
- `run.config_path` 自证 = smoke-base.yaml、`run.attack == "iba"`（非陷阱 #7）；
- `final.local_malicious_asr` 明显 > 0（恶意端本地后门成立，是有效信号）。

判 **不可判/继续** 而非"攻击无效"的情形（照 Bad-PFL exp001 的纪律）：
- `local_benign_asr` / `global_asr` 低 → **不下"迁移性弱"结论**，5 轮 + fedavg_cnn 太短太小。

## 怎么跑

```bash
# L1（本地/容器）
bash run_l1.sh
# L2 接线 smoke（集群，GPU，约 3 分钟）
sbatch run_smoke.sh attack iba iba none exp001 hier_fedavg_fedrep
```

## ⚠️ 第一版为打通端到端做的忠实性妥协（务必记得回补）

| # | 第一版 | 忠实版（待回补，单变量增强） | 官方依据 |
|---|---|---|---|
| C① | `on_batch` 批内混合、单前向（poison_ratio≈1-α） | `on_extra_loss` 双损失 `α·CE(clean)+(1-α)·CE(poison)` | `fl_trainer.py:457` |
| D① | eps 固定 `iba_eps`、直接投毒 | 两阶段 started_poisoning 门 + 指数退火 `cur_eps=max(test_eps, atk_eps·decay^t)` + alternative_training | `fl_trainer.py:1034-1041` |
| 净增1 | `iba_scale_weights_poison` 固定 γ（默认关）、锚 edge、只替换 edge 层 | 自适应 γ=N/n（edge server 下行分母）+ 全局替换 + eps 门 | `fl_trainer.py:1118` |
| 共享生成器 | `iba_shared_generator` 默认 **false**（每端独立） | 全长 run 应设 **true**（对齐官方、跨轮持久，exp007 教训） | `use_our_attack` |

## 迁移性/irreversible 的真正判据（后续会话）

照 Bad-PFL exp004→exp007 的方法论：**ResNet10**（别用 fedavg_cnn，否则后门被聚合洗掉）、
`defense=none`、`iba_shared_generator=true`、`n_rounds×edge_rounds ≈ 300~500`，
看 `global_asr` / `local_benign_asr` 的爬升曲线。γ=1 先取基线，再单变量开固定 γ / 回补 C②D②。
