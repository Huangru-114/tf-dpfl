# current-focus：攻击轴 × PFL 方法轴正交化（CLAUDE.md 陷阱 #1）

> 状态：**四个决策点已确认，等「开始改」。**
> 基线已核对：`bash run_l1.sh` → 94 passed / 3 skipped / 3 xfailed ✓

## 决策记录（2026-08-19 已确认）

| # | 决策点 | 选定 |
|---|---|---|
| A | Neurotoxin 掩码碰到 Rep 家族 upload 的 head 槽位 | **不管它**，文档写明惰性；不加 `_aggregated_weight_indices()` |
| B | `_atk_batch` 覆盖哪些训练阶段 | **全投**（贴合威胁模型）；日志记录每方法的投毒阶段数 |
| C | CerP 的 α/β 正则项 | **加第四个钩子 `_atk_extra_loss()`**，只挂到产出 upload 的那个 step |
| D | 测试里的 `hier_perfedavg` 假绿格 | **从 `PFL_METHODS` 删掉** |

于是钩子共**四个**（下表三个 + `_atk_extra_loss()`）。

## 现在要回答的唯一问题

恶意客户端如何在**保持自己是该 PFL 方法的客户端**的前提下叠加投毒 ——
即把 `use_cls = MalCls or ClientCls`（替换）改成 `type(名字, (AttackMixin, MethodCls), {})`（组合），
且组合后**上传语义**与**个性化模型**都不被攻击代码破坏。

不在本会话范围内：Neurotoxin mask 方向（陷阱 #4）、FLAME（陷阱 #3）、Bad-PFL 归一化常数（陷阱 #5）。
正交化只负责让这些问题的 ASR 数字**变得可解释**，不负责让 ASR 变高。

## 判定「完成」的客观标准

| 级别 | 判据 | 现状 |
|---|---|---|
| L1-a | `pytest tests/test_attack_method_orthogonality.py` 全绿（需 TF，集群跑） | 本地 skip |
| L1-b | `bash run_l1.sh` 仍是 94 passed / 3 skipped / 3 xfailed（+ 新增用例） | 基线已核对 ✓ |
| L1-c | 新增 `tests/test_attack_hooks.py`：`on_upload` 纯函数性（`self.model` 逐元素不变） | 未写 |
| L2 | `badnet × hierpfedme` smoke 跑通，`acc` 与改动前同量级、`admitted_count` 不变 | 未跑 |

L2 跑法（badnet = vanilla 策略，走 `MalMixin is None` 分支，是「没跑坏」的对照）：

```bash
cd fedavg && python main.py --config ../experiments/smoke-base.yaml \
    --framework hier_pfedme --attack_method badnet --defense none 2>&1 | tee /tmp/smoke.log
python ../harness/collect_metrics.py /tmp/smoke.log \
    -o ../experiments/attack/orthogonalization/exp001.metrics.json
```

> `run_smoke.sh` 目前不透传 `--framework`，需要加一个可选参数（见「附带改动」）。

---

## 一、总体形态

```python
# client/attack_mixin.py
class AttackMixin:                      # ← 不继承 FedAvgClient，不继承任何方法类
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)      # 先把方法类初始化完
        ...                             # 再做自己的（全部 _atk_ 前缀）

# main.py
use_cls = compose_malicious_client(MalMixin, ClientCls)   # 带缓存
# MRO: NeurotoxinBadPFL... → AttackMixin 子类 → MethodCls → FLClientBase
```

三个钩子（基类默认全部是恒等/空操作，良性路径数值零变化）：

| 钩子 | 签名 | 调用点 | 谁用 |
|---|---|---|---|
| `_atk_round_start(round_idx)` | `-> None` | `local_train` 开头，**收到 edge 权重之后、训练之前** | Neurotoxin 算 mask、Bad-PFL 训 generator、CerP 调 trigger + 算 θ_clean |
| `_atk_batch(x, y)` | `-> (x, y)` | 每个方法训练循环体首行，**在 `@tf.function` step 之外** | Bad-PFL / CerP 动态投毒 |
| `_atk_upload(upload, round_idx)` | `-> upload` | `local_train` `return` 之前 | Neurotoxin 掩码投影 |

---

## 二、四个坑，逐个怎么处理

### 坑 1 —— 上传结构因方法而异

各方法 `local_train` 返回的第一个元素根本不是同一个东西：

| 方法 | upload 是什么 | 与 `self.model` 的关系 |
|---|---|---|
| FedAvg | `self.model.get_weights()` | 相同 |
| pFedMe | `self._local_model.get_weights()`（锚点 Θ） | **不同**（`self.model` 是 θ̃） |
| Ditto | 阶段 A 结束时的 w_k | **不同**（`self.model` 结束时是 v_k） |
| FedRep | 完整列表，backbone=w_k，head=私有（上游忽略 head） | **不同**（`self.model` 停在 [φ_e, head]） |
| Ditto-Rep | 同上 | **不同**（停在 [v_k, head]） |
| pFedMe-Rep | `_anchor_model` 的 backbone + 私有 head | **不同**（停在 [θ̃_bb, head]） |

**处理**：`_atk_upload` 只接收并返回那个 `upload` **列表**，攻击代码永远不去
`self.model.get_weights()` 自己找上传物。

**顺带修掉一个真 bug**：Neurotoxin 现在算的是
`w_before = self.model.get_weights()`（训练前）→ `w_after = self.model.get_weights()`（训练后）。
在任何非 FedAvg 方法下 `w_after` 是**个性化模型**，与真正上传的 Θ / w_k 无关 →
掩码作用在了错误的张量上。

**基准应改为 `self.edge_weights`**：`upload − edge_weights` 才是聚合端真正看到的 delta，
且对上表**每一个**方法都有定义（每个方法的 `set_weights` 都写了 `self.edge_weights`）。

```python
def _atk_upload(self, upload, round_idx):
    mask = self._atk_mask                      # 在 _atk_round_start 里算好
    ref  = self.edge_weights                   # 聚合端看到的基准，方法无关
    return [r + (u - r) * m for r, u, m in zip(ref, upload, mask)]
```

> 已知残留：FedRep 家族上传列表里的 head 槽位会被一起掩码。那些槽位上游平均后
> 客户端收到时会被私有 head 覆盖 → **惰性的，不影响结果**，但我不打算假装它「对」。
> 若你要求干净，可在基类加 `_aggregated_weight_indices()`（默认全部，Rep 家族返回
> `_base_w_idx`）。**这是决策点 A。**

### 坑 2 —— 个性化模型不能被污染

要分清两种「污染」，我认为只有第一种是 bug：

**(a) 实现层面的写回 —— 是 bug，必须修。**
Neurotoxin 现在做 `self.model.set_weights(w_proj)`。在 PFL 下这一步把
**掩码后的上传物**盖到**个性化模型**上，直接破坏 PM 评估、遗忘曲线、local ASR
三处指标。修法：`_atk_upload` 是**纯函数** —— 只返回新列表，一个 `set_weights` 都不许有。
L1-c 就断言这一条（调用前后 `self.model.get_weights()` 逐元素相等）。

**(b) 投毒流进攻击者自己的个性化状态（v_k / 私有 head / Θ）—— 不是 bug。**
威胁模型里恶意客户端完全控制自己的客户端，它的个性化模型本来就该被后门污染
（`local_benign_asr` 度量的是**良性**客户端，不受影响）。我不打算「修」它。

**(c) 投毒作用在哪些训练阶段 —— 需要你拍板。**
`_atk_batch` 挂在每个循环体上，则 Ditto 的阶段 A（上传）和阶段 B（个性化）、
FedRep 的阶段 1（head）和阶段 2（backbone）**全部**被投毒。
默认我按「全投」实现（攻击者控制自己客户端的全部本地计算，最贴近威胁模型），
但这会让「Ditto 下的攻击强度」与「FedAvg 下的攻击强度」不可直接比较。
**这是决策点 B**：全投 / 只投产出 upload 的阶段。我倾向全投 + 在 metrics 里记录阶段数。

### 坑 3 —— mixin `__init__` 协作

四条硬规则：

1. **签名照抄**：`def __init__(self, client_id, dataset, model, config, n_samples=None)`，
   第一句 `super().__init__(client_id, dataset, model, config, n_samples)`。
   所有方法类都是这个签名且都向上位置传参，MRO 能一路走到 `FLClientBase`。
2. **自己的初始化必须在 `super()` 之后**：攻击代码要用 `self.model` / `self.rng` /
   `self.lr_schedule` / `self.config`，这些都是父类建的。
3. **属性一律 `_atk_` 前缀**。现有方法类已经占了 `_tv_w_idx`（CerP）、`_base_tv_idx`、
   `_anchor_model`、`_head_opt`、`_lam_var` 等名字，Neurotoxin 的 `_tv_to_w_idx` 和
   CerP 的 `_tv_w_idx` 只差一个字符 —— 不加前缀迟早撞上，而撞上是**静默**的。
4. **`is_malicious`**：main.py 在**构造之后**才赋值，所以 mixin `__init__` 里
   一切依赖它的东西必须懒构建。mixin 里先 `self.is_malicious = False` 兜底，
   把现在满仓库的 `getattr(self, "is_malicious", False)` 变成可以直接读的属性。

**组合类必须缓存**。`compose_malicious_client` 用 `(Mixin, MethodCls)` 做 key：
不缓存的话 10~100 个恶意客户端会生成 10~100 个**不同的类对象**，`@tf.function`
按类持有 trace 缓存 → 重复 retrace，且 `isinstance` 行为不可预测。
组合类显式设 `__name__ = f"{Mixin.__name__}_{MethodCls.__name__}"` 方便读日志。

### 坑 4 —— 每个方法的 `_train_step` 形态不同

现状：`_train_step` / `_inner_step`+`_moreau_step` / `_plain_step`+`_prox_step` /
`_train_head_step`+`_train_backbone_step` / `_train_backbone_plain_step`+`_train_backbone_ditto_step`
—— 六个文件、十来个不同名的步骤函数，全是 `@tf.function`。

**不去统一它们。** 攻击只需要在 `@tf.function` **之外**改数据，所以钩子挂在
Python 循环体上，每个方法一行：

```python
for x, y in self._shuffled_batches():
    x, y = self._atk_batch(x, y)        # ← 新增这一行，基类是恒等
    loss = self._plain_step(x, y)
```

改动点共 9 处（fedavg 1、pfedme 1、ditto 2、fedrep 2、ditto_rep 3，pfedme_rep 2 —— 实施时逐一核对）。
基类 `_atk_batch` 直接 `return x, y`，**良性路径数值零变化**，这是 L2「没跑坏」的依据。

投毒保持 eager（Bad-PFL 要对输入求 FGSM 梯度、CerP 要更新 trigger 变量），
与 `@tf.function` 的 step 隔离在不同语句里 → 不比现状更糟，`n_workers=1` 的现有约束照旧。

**明确一条我处理不了的**：CerP 的两个正则项 `α‖θ−θ_clean‖₂` 和 `β·Σcos(θ,θ_peer)`
是**损失项**，必须在 tape 里面，`_atk_batch` 够不着。三个钩子做不到 CerP 的完整语义。
两个选择，**决策点 C → 选 C1**：

- **C1（已选定）**：加第四个钩子 `_atk_extra_loss()`（基类返回 `tf.constant(0.0)`），
  只加到**产出 upload 的那个 step** 里：FedAvg `_train_step`、pFedMe `_inner_step`、
  Ditto `_plain_step`、FedRep `_train_backbone_step`、Ditto-Rep `_train_backbone_plain_step`、
  pFedMe-Rep `_inner_step`。语义上说得通（CerP 的正则就是为了让**上传物**躲过鲁棒聚合）。
  代价：peer 列表长度逐轮变化会触发 `@tf.function` retrace，得把 peer 权重塞进
  固定形状的 `tf.Variable`，或给该 step 关掉 `jit`。
- **C2**：本会话 CerP 只接 `_atk_batch`（触发器投毒生效，α/β 正则**失效**），
  在 `config_validate` 里把 `cerp` 标成「正则项未接线」并告警。

无论选哪个，我都不会在文档或注释里写「应该等价」。

---

## 三、附带改动（都很小，但会碰到验收面）

1. **`main.py`**：新增 `resolve_client_classes(config) -> (benign_cls, malicious_cls)`，
   把类选择逻辑收成**一个**函数。
   **`tests/test_attack_method_orthogonality.py` 的 `_resolve` 现在是「复刻 main.py:386-395」
   的手抄副本** —— 改成直接调这个函数。断言一个字不动，但测的从「副本」变成「真货」。
   > 我知道「改动作为验收标准的测试」是危险信号，所以说清楚：改的是 `_resolve` 这个
   > **夹具**，两条 `assert` 原样保留。你若不同意，我就保留手抄形式，代价是它以后还会漂。
2. **测试里的 `hier_perfedavg`**：`PFL_METHODS` 含它，但 CLAUDE.md 说该方法已移除，
   `config_validate.REMOVED_METHODS` 也拒绝它。`_select_method_classes` 对它返回默认的
   `PFedMeClient` → 这一格会**假绿**（测的其实是 pfedme）。建议从列表删掉。**决策点 D。**
3. **`config_validate.py`**：正交化完成后删掉 `NON_ORTHOGONAL_STRATEGIES` 的告警分支
   （否则每次 run 都打一条已经不成立的警告）。若选 C2，改成 CerP 专属的正则告警。
4. **`run_smoke.sh`**：加可选 `--framework` 透传，否则跑不出 `× hierpfedme` 那一格。

---

## 四、语义 diff 表

| 步骤 | 应有语义 | 本仓库现实现 | 差异 | 怎么验证 |
|---|---|---|---|---|
| 恶意客户端的训练算法 | = 良性客户端的 PFL 方法 | `use_cls = MalCls or ClientCls`，攻击类**替换**方法类 | **架构级错误** | `test_malicious_client_preserves_pfl_method` |
| 恶意客户端的 `local_train` | 解析到方法类 | 解析到 `FedAvgClient.local_train` | 个性化/近端项全丢 | `test_..._does_not_silently_downgrade_to_fedavg` |
| Neurotoxin 掩码作用对象 | 上传 delta（`upload − edge_weights`） | `self.model` 训练前后之差 | 非 FedAvg 下作用在个性化模型上 | 新增 `test_mask_applies_to_upload_delta` |
| Neurotoxin 掩码后写回 | 只改返回值 | `self.model.set_weights(w_proj)` | **污染个性化模型** | 新增 `test_upload_hook_is_pure`（L1-c） |
| 投毒时机 | 每个 batch，`@tf.function` 之外 | 恶意类自带整套 eager 训练循环 | 与方法的训练循环互斥 | L2 acc 不崩 |
| CerP α/β 正则 | tape 内损失项 | 恶意类自己的 tape | 三钩子够不着 | 决策点 C |
| 良性客户端路径 | 数值零变化 | — | 新增恒等钩子 | L2：badnet × hierpfedme |

## 已排除的可能

- ~~攻击类只是「顺序」问题，调 MRO 就行~~ —— 不行。`NeurotoxinClient(FedAvgClient)`
  这条继承链本身就是错的，`type("X", (NeurotoxinClient, PFedMeClient), {})` 会因为
  `FedAvgClient` 和 `PFedMeClient` 同为 `FLClientBase` 的兄弟而 MRO 冲突/语义混乱。
  必须先把攻击类**拆掉基类**变成 mixin。

## 进展日志

| 日期 | 做了什么 | 证据 | 结论 |
|---|---|---|---|
| 2026-08-19 | 核对基线、通读 6 个方法类 + 3 个攻击类 + main.py 类选择 | `run_l1.sh` 94/3/3 | 方案成文，待确认 A/B/C/D 四个决策点 |
