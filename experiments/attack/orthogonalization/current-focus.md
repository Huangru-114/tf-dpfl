# current-focus：客户端行为组合机制（陷阱 #1 正交化 + 主动防御接口）

> 状态：**七个决策点已确认，等「开始改」。**
> 基线已核对：`bash run_l1.sh` → 94 passed / 3 skipped / 3 xfailed ✓

## 现在要回答的唯一问题

把「客户端在一轮里做什么」从**类继承**（只能有一个）改成**mixin 组合**（可以有多个），
使攻击轴、防御轴都能叠加在 PFL 方法轴之上，而不是替换它。

不在本会话范围内：Neurotoxin mask 方向（陷阱 #4）、FLAME 实现（陷阱 #3）、
Bad-PFL 归一化常数（陷阱 #5）、任何具体的主动防御算法。
本会话只负责让这些问题的数字**变得可解释**，不负责让 ASR 变高。

---

## 一、轴 × 层 矩阵（正确的心智模型）

「轴在哪注入」是错的问法。正确的是**每条轴在三层各有没有插槽**：

| | client 层 | edge 层 | cloud 层 |
|---|---|---|---|
| **PFL 方法** | `local_train()` ✅ | `run_edge_round()` ✅ | ❌ 写死 FedAvg（分支全注释） |
| **攻击** | ❌ 替换类（陷阱 #1） | 猴补丁 `select_clients` | `BackdoorCloudServer`（仅评估） |
| **防御** | ❌ 不存在 | `robust_mean()` ✅ | ❌ 不存在 |

**3 个空洞 + 1 个坏插槽。**

### 上行的窄腰是好的，下行的是烂的

`robust_mean`（上行）：**6/6** 个 edge server 都走，`tests/test_defense_coverage.py` AST 守着。

`broadcast_to_clients`（下行）：**只有 1/6** 在用。

| edge server | 下行 |
|---|---|
| `edge_server_fedavg.py:28` | `self.broadcast_to_clients(...)` ✅ |
| `edge_server_pfedme.py:72` | 内联 `for client in selected: client.set_weights(...)` ❌ |
| `hier_ditto.py:35` | 内联 ❌ |
| `hier_ditto_rep.py:56` | 内联 ❌ |
| `hier_fedrep.py:50` | 内联 ❌ |
| `hier_pfedme_rep.py` | 继承 ditto_rep ❌ |

**后果**：任何需要「服务器下发额外东西给客户端」的主动防御（裁剪上界、挑战样本、
全局参考统计量），写在 `broadcast_to_clients` 里会在 5/6 的 PFL 方法下**静默送不到**，
而日志照样打印 `[Defense] enabled`。与陷阱 #3 同病，方向相反，且没有守卫。

---

## 二、决策记录（2026-08-19 已确认）

| # | 决策点 | 选定 |
|---|---|---|
| A | Neurotoxin 掩码碰到 Rep 家族 upload 的 head 槽位 | 不管它，文档写明惰性 |
| B | 投毒覆盖哪些训练阶段 | 全投（贴合威胁模型），日志记录阶段数 |
| C | CerP 的 α/β 正则项 | 加 `on_extra_loss()` 钩子，只挂产出 upload 的 step |
| D | 测试里的 `hier_perfedavg` 假绿格 | 从 `PFL_METHODS` 删掉 |
| E | 主动防御 mixin 挂给谁 | **所有客户端，攻击在外层** |
| F | 下行/上行额外载荷 | **现在建空壳 + AST 守卫** |
| G | cloud 层接口 | **抽 `RobustAggregationMixin` + `aggregate_edges` 空壳**，不动 `_select_method_classes` |

---

## 三、接口设计

### 3.1 客户端钩子协议（`FLClientBase`，默认全部无操作）

**命名是中性的 `on_*`，不是 `_atk_*`** —— CerP 的正则项和未来某个防御的近端项是同一个钩子。
这个命名必须在写代码前定死，晚了要再动一遍全部 6 个方法文件。

| 钩子 | 签名 | 调用点 | 谁用 |
|---|---|---|---|
| `set_control(control)` | `dict -> None` | `broadcast_to_clients` | 主动防御下行 |
| `on_round_start(r)` | `-> None` | `local_train` 开头，收到 edge 权重后、训练前 | Neurotoxin 算 mask / Bad-PFL 训 generator / CerP 调 trigger |
| `on_batch(x, y)` | `-> (x, y)` | 每个训练循环体首行，**`@tf.function` 之外** | Bad-PFL / CerP 动态投毒 |
| `on_extra_loss()` | `-> tf.Tensor` 标量，默认 0 | 产出 upload 的那个 step 的 tape 内 | CerP 的 α/β |
| `on_upload(upload, r)` | `-> upload` | `return` 之前 | Neurotoxin 掩码投影 |
| `get_aux()` | `-> dict` | `_finalize_updates` | 主动防御上行 |

### 3.2 组合

```python
compose_client_class(MethodCls, DefenseMixin, AttackMixin)   # 按 (类元组) 缓存
# MRO: AttackMixin → DefenseMixin → MethodCls → FLClientBase
```

三条规则：

1. **防御 mixin 挂所有客户端**，攻击 mixin 只挂恶意的。防御方分不出谁是恶意的
   —— 这才是主动防御的真实语义（决策 E）。
2. **攻击在外层**：恶意客户端有权把防御要求的客户端侧动作做假或跳过。
   「调不调 `super()`」= 「是否假装守协议」，做成 config 开关而非写死，它是研究变量。
3. **`defense.name=none` 时不组合任何防御 mixin**，类对象等于 `MethodCls`
   —— L2「没跑坏」的论证依赖这一条。

### 3.3 防御声明自己碰哪几层

照抄 `METHOD_SUPPORTS_DEFENSE` 那套「单一事实来源 + 启动前 fail fast」：

```python
class BaseDefense:
    layers = frozenset({"edge"})     # ⊆ {"client", "edge", "cloud", "post_hoc"}
    client_mixin = None              # 声明了 "client" 就必须给一个
    def make_control(self, edge, round_idx) -> dict: return {}
```

`config_validate` 检查「声明的层」与「实际接线的层」一致，不一致拒绝启动。

### 3.4 cloud 层（只留接口，决策 G）

`robust_mean` 从 `EdgeServerBase` 抽成 `RobustAggregationMixin`，edge 与 cloud 共用：

```python
# CloudServer
def aggregate_edges(self, edge_updates, prev_global):
    return self.robust_mean(edge_updates, prev_global)   # 无防御 → fedavg_aggregate
```

```yaml
defense:
  name: flame
  layers: [edge]          # 默认，行为不变；未来可 [edge, cloud]
```

本会话**不**动 `_select_method_classes`（不加第三个 CloudMixin），
**不**实现任何方法专属 cloud 聚合（`beta_hier` 式 9、FedDyn h_g 仍是死配置）。

---

## 四、四个坑，逐个怎么处理

### 坑 1 —— 上传结构因方法而异

六个方法里**五个的 upload ≠ `self.model`**：

| 方法 | upload | `self.model` 结束时 |
|---|---|---|
| FedAvg | 完整权重 | = upload |
| pFedMe | 锚点 Θ（`_local_model`） | θ̃ |
| Ditto | 阶段 A 的 w_k | v_k |
| FedRep | 完整列表，backbone=w_k，head=私有 | [φ_e, 私有 head] |
| Ditto-Rep | 同上 | [v_k, 私有 head] |
| pFedMe-Rep | 锚点 backbone + 私有 head | [θ̃_bb, 私有 head] |

`on_upload` 只接收并返回那个 `upload` 列表，钩子里永远不去 `self.model.get_weights()`
自己找上传物。

**顺带修一个真 bug**：Neurotoxin 现在用 `self.model` 训练前后之差算掩码，
非 FedAvg 下那是**个性化模型**，与真正上传的 Θ / w_k 无关。基准应改为 `self.edge_weights`
—— `upload − edge_weights` 才是聚合端看到的 delta，且对六个方法**全都有定义**。

> 残留（决策 A）：Rep 家族上传列表里的 head 槽位会被一起掩码。那些槽位上游平均后
> 客户端收到时被私有 head 覆盖 → **惰性**，不影响结果。不写「等价」，只写「惰性」。

### 坑 2 —— 个性化模型不能被污染

**(a) 实现层面的写回 —— 是 bug，必须修。**
Neurotoxin 现在 `self.model.set_weights(w_proj)`，在 PFL 下把掩码后的上传物盖到
个性化模型上，破坏 PM 评估 / 遗忘曲线 / local ASR。
修法：`on_upload` 是**纯函数**，一个 `set_weights` 都不许有。L1 断言调用前后逐元素相等。

**(b) 投毒流进攻击者自己的 v_k / 私有 head —— 不是 bug。**
威胁模型里恶意客户端完全控制自己的客户端，`local_asr_benign` 度量的是良性客户端。不修。

**(c) 投毒覆盖哪些阶段** —— 决策 B：全投。会让「Ditto 下的攻击强度」与
「FedAvg 下的攻击强度」不可直接比较（前者经过的投毒 epoch 更多），日志记录阶段数备查。

### 坑 3 —— mixin `__init__` 协作

1. **签名照抄**：`(self, client_id, dataset, model, config, n_samples=None)`，
   第一句 `super().__init__(...)` 位置传参。六个方法类都是这个签名且都向上位置传参。
2. **自己的初始化在 `super()` 之后**：要用 `self.model` / `self.rng` / `self.lr_schedule`。
3. **属性按 mixin 加前缀**：攻击 `_atk_`，防御 `_def_`。现有方法类已占了
   `_tv_w_idx`（CerP）、`_base_tv_idx`、`_anchor_model`、`_head_opt`、`_lam_var`；
   Neurotoxin 的 `_tv_to_w_idx` 与 CerP 的 `_tv_w_idx` 只差一个字符 —— 撞上是**静默**的。
4. **`is_malicious`**：main.py 在构造之后才赋值 → 依赖它的一切懒构建；
   mixin `__init__` 里先 `self.is_malicious = False` 兜底。
5. **组合类必须缓存**：不缓存则 N 个客户端生成 N 个不同类对象 → `@tf.function` 重复
   retrace，`isinstance` 不可预测。缓存 key = 类元组；`__name__` 显式设成可读的组合名。

### 坑 4 —— 每个方法的 `_train_step` 形态不同

**不统一它们。** 十来个 `@tf.function` step 原样不动，钩子挂在 Python 循环体上：

```python
for x, y in self._shuffled_batches():
    x, y = self.on_batch(x, y)          # ← 新增，基类恒等
    loss = self._plain_step(x, y)
```

共 9 处（fedavg 1 / pfedme 1 / ditto 2 / fedrep 2 / ditto_rep 3 / pfedme_rep 2，
实施时逐一核对）。基类 `on_batch` 直接 `return x, y` → **良性路径数值零变化**。

投毒保持 eager（Bad-PFL 对输入求 FGSM 梯度、CerP 更新 trigger 变量），
与 `@tf.function` 的 step 隔离在不同语句里，`n_workers=1` 的现有约束照旧。

`on_extra_loss()`（决策 C）只加到**产出 upload 的那个 step**：
FedAvg `_train_step` / pFedMe `_inner_step` / Ditto `_plain_step` /
FedRep `_train_backbone_step` / Ditto-Rep `_train_backbone_plain_step` / pFedMe-Rep `_inner_step`。
CerP 的 peer 列表长度逐轮变化会触发 retrace → peer 权重塞进固定形状 `tf.Variable`。

### 坑 5（新）—— 下行不统一

5 个内联的 `for client in selected: client.set_weights(...)` 改回
`self.broadcast_to_clients(selected, global_weights=...)`，循环体里加 `client.set_control(...)`。

配 AST 守卫 `tests/test_broadcast_coverage.py`（写法同 `test_defense_coverage.py`）：
**任何 edge server 的 `run_edge_round` 里不许出现裸的 `client.set_weights`**。

> 良性数值会变吗？不会 —— 5 份内联的循环体与 `broadcast_to_clients` 逐字相同
> （都是 `set_weights(global_weights=self._global_weights_ref, edge_weights=<edge 模型权重>)`）。
> 唯一差别是 `edge_server_pfedme.py:75-85` 那段打印 client 测试集分布的调试代码，
> 它每轮遍历一遍测试集，是纯开销，一并删掉。

---

## 五、判定「完成」的客观标准

| 级别 | 判据 | 现状 |
|---|---|---|
| L1-a | `tests/test_attack_method_orthogonality.py` 全绿（需 TF，集群） | 本地 skip |
| L1-b | `bash run_l1.sh` 仍 94 passed / 3 skipped / 3 xfailed（+ 新增用例） | 基线已核对 ✓ |
| L1-c | 新增 `test_client_hooks.py`：`on_upload` 纯函数性（`self.model` 逐元素不变） | 未写 |
| L1-d | 新增 `test_broadcast_coverage.py`：AST 断言 6/6 走 `broadcast_to_clients` | 未写 |
| L1-e | 新增 `test_cloud_aggregate_default.py`：`aggregate_edges` 默认路径与现在逐元素相等 | 未写 |
| L2 | `badnet × hierpfedme` smoke 跑通，`acc` 与改动前同量级、`admitted_count` 不变 | 未跑 |

L2 跑法：

```bash
cd fedavg && python main.py --config ../experiments/smoke-base.yaml \
    --framework hier_pfedme --attack_method badnet --defense none 2>&1 | tee /tmp/smoke.log
python ../harness/collect_metrics.py /tmp/smoke.log \
    -o ../experiments/attack/orthogonalization/exp001.metrics.json
```

> **改动前的对照数字目前还没有。** 严格说 L2 需要先跑一次 baseline 存下来再改。
> 本地无 TF，这一步只能在集群做。

---

## 六、附带改动

1. **`main.py`**：新增 `resolve_client_classes(config) -> (benign_cls, malicious_cls)`，
   把类选择收成一个函数。
   **`tests/test_attack_method_orthogonality.py` 的 `_resolve` 现在是手抄副本**
   —— 改成直接调这个函数。两条 `assert` 一个字不动。
   > 「改动作为验收标准的测试」是危险信号，所以说清楚：改的是 `_resolve` 这个**夹具**。
2. **测试里的 `hier_perfedavg`**（决策 D）：从 `PFL_METHODS` 删掉。该方法已移除，
   `_select_method_classes` 对它返回默认 `PFedMeClient` → 这一格测的其实是 pfedme，是假绿。
3. **`config_validate.py`**：删掉 `NON_ORTHOGONAL_STRATEGIES` 告警分支（已不成立）；
   新增「防御声明的 layers vs 实际接线」校验。
4. **`run_smoke.sh`**：加可选 `--framework` 透传，否则跑不出 `× hierpfedme` 那一格。
5. **`aggregation/client_update.py`**：`ClientUpdate` 加 `.aux: dict`（决策 F），
   手法同已有的 `.client_id`，旧解包写法全部照旧。

---

## 七、本会话不碰、但已记录的发现

以下三条读了代码路径确认，**但都没有数值证据**，不在本会话范围：

1. **cloud 层聚合分支全被注释**（`server/server.py:108-151`）→ 不管 `drift_correction`
   是什么，cloud 永远是朴素样本加权 FedAvg。`beta_hier`（Hier-pFedMe 式 9 的 β）
   是死配置，读都没读。本会话只留 `aggregate_edges` 接口，不改这个行为。
2. **Rep 家族的私有 head 进了防御的距离计算** —— `hier_fedrep.py:62` 把**完整**权重列表
   传给 `robust_mean`，而 `flame.py:37-39`、`multi_krum.py:42` 都 `flatten_weights(upd[0])`
   展平全部。私有 head 逐客户端 warm-start、从不同步，是全部权重里发散最快的部分
   → 余弦/距离矩阵可能被 head 主导，而 head 恰恰是防御不该看的（连聚合结果都不用）。
   与陷阱 #1 同类但独立，正交化修好了它还在。
   验证方法：构造 backbone 相同、head 随机发散的一组更新，断言 FLAME 接纳集合不变。
3. **`defense/flame.py:76` 用全局 `np.random.normal`** —— 陷阱 #2 列的四处
   （select_clients / 投毒选样 / DnC 投影 / ASR 子采样）之外的**第五处**，
   且在训练循环内（每个 edge round 调一次）。`defense=flame` 的格子固定种子重跑对不上，
   而这正是 `tests/test_flame.py` 有 3 个 xfail 的那个防御。修法一行：改用 seeded RNG。

## 进展日志

| 日期 | 做了什么 | 证据 | 结论 |
|---|---|---|---|
| 2026-08-19 | 核对基线、通读 6 方法类 + 3 攻击类 + 6 edge server + 5 防御 + main.py | `run_l1.sh` 94/3/3 | 方案 v1（仅攻击轴），4 个决策点确认 |
| 2026-08-19 | 发现下行广播 1/6 覆盖率；按「主动防御需客户端配合」重做设计 | `grep` 6 个 edge server 的 `client.set_weights` | 方案 v2（客户端行为组合机制），7 个决策点确认 |
