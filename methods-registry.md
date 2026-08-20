# 方法台账 / 研究看板

**一次只推进一个方法。** 开工前把该行状态改成 `porting`，`reference/` 里只 clone 这一个源库；
收工后 `rm -rf reference/<x>`，状态改成 `parity-passed` / `experimenting` / `done` / `rejected`。

状态取值：`backlog`（只在台账里）/ `porting`（正在移植）/ `parity-passed`（L1 算法不变量过）
/ `experimenting`（L2 smoke 过，正在跑实验）/ `done` / `rejected`。

> 源仓库 URL 标 `?` 的表示尚未核实——**clone 前先确认，不要凭印象写**。
> commit 列在 clone 当天填入 hash，保证以后能复现同一份参考实现。

---

## 攻击轴（`backdoor.malicious_strategy` × `backdoor.trigger`）

| 方法 | 源仓库 | commit | 一句话 | 状态 | 备注 |
|---|---|---|---|---|---|
| badnet | — | — | 右下角固定方块触发器 + 改标签 | done | 静态投毒，走 `build_poisoned_dataset` |
| blended | — | — | 整图混合图案（hello_kitty，α=0.2） | done | 静态投毒 |
| dba | github.com/AI-secure/DBA | ? | 分布式触发器：每个恶意端一个局部 pattern，评估用全局 | done | pattern 取自官方 `cifar_params.yaml` |
| **neurotoxin** | github.com/jhcknzzm/Federated-Learning-Backdoor | ? | 把后门藏进良性梯度最小的坐标 → 持久 | **porting** | 陷阱 #4：mask 语义疑似反向；陷阱 #1：接线 |
| cerp | ? | ? | 可训练触发器 + 模型距离 + peer 余弦正则 | backlog | 动态投毒；官方仓库待确认 |
| **bad-pfl** | github.com/fmy266/Bad-PFL | ? | 生成器扰动 δ + FGSM 破坏性噪声 ξ，T(x)=x+ξ+δ | **porting** | 陷阱 #5 + 陷阱 #1 |

## 防御轴（`defense.name`）

| 方法 | 源仓库 / 文献 | commit | 一句话 | 状态 | 备注 |
|---|---|---|---|---|---|
| trimmed_mean | Yin et al., ICML'18 | — | 逐坐标两端截尾均值 | done | |
| median | Yin et al., ICML'18 | — | 逐坐标中位数 | done | |
| multi_krum | Blanchard et al., NIPS'17 | — | 选距离最近的 m−f−2 邻居和最小者 | done | |
| **flame** | Nguyen et al., USENIX Sec'22 | ? | HDBSCAN 余弦聚类 + 中位数范数裁剪 + 高斯噪声 | **porting** | 陷阱 #3：现为 majority-cosine 近似，非文献方案 |
| dnc | Shejwalkar & Houmansadr, NDSS'21 | ? | 随机投影 + 谱方向离群剔除 | backlog | 未验证 |
| simple_tuning | xtLyu/PFedBA | ? | 客户端侧后处理：重置头 + 干净微调 | backlog | 未验证 |

## PFL 方法轴（`training.drift_correction`）

| 方法 | 参考 | 状态 | 备注 |
|---|---|---|---|
| hierfedavg | — | done | baseline |
| fedprox / feddyn / scaffold | — | done | |
| pfedme / hierpfedme | PFLlib | done | 双 lr（`personal_lr` / `moreau_lr`）已对齐 PFLlib |
| hier_perfedavg | — | done | |
| hier_ditto / hier_ditto_rep | — | done | |
| hier_fedrep / hier_pfedme_rep | — | done | head 用独立低 lr `head_lr_rep` + `rep_dim` 瓶颈 |

---

## 当前焦点

**修复顺序（先地基后方法）：**

1. ~~仓库瘦身 + hub 骨架~~ ✅
2. ~~统一聚合信息流（防御轴覆盖全部 PFL 方法）+ 上传带 client_id + 随机性播种
   + 配置兼容性校验 + 矩阵提交器~~ ✅
3. **陷阱 #1：攻击轴 / PFL 方法轴正交化**（三个方法共同病根，必须先修）
4. 陷阱 #3：FLAME 照 USENIX'22 重写（HDBSCAN + 无权平均）
5. 陷阱 #4：Neurotoxin mask 语义（先 clone 官方实现证伪）
6. 陷阱 #5：Bad-PFL 生成器迁移性 + 归一化常数

> 已移除：**Hier-PerFedAvg**。它的 edge 层聚合的是元梯度而非权重，与防御接口
> （对 W_i − G_{t-1} 做范数裁剪）语义不兼容，在三维矩阵里只会产出无法解释的格子。
