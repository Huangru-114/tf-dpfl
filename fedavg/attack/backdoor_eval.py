"""
attack/backdoor_eval.py  –  后门评估（C-Acc 本地 / ASR 全局）

对每个良性客户端的本地（个性化）模型 client.model：
  C-Acc : 在该 client 的本地干净测试集上的分类准确率（衡量攻击隐蔽性）
  ASR   : 在完整 CIFAR-10 测试集上，取非目标类样本加 trigger 后被分类为 target_label
          的比例（衡量后门有效性）。排除恶意客户端自身，只统计良性客户端。
"""

import numpy as np
import tensorflow as tf

from attack.drift_metrics import param_drift, repr_shift


def _f3(v):
    """None → "n/a"，否则三位小数。无定义与 0 必须在日志里就分得开。"""
    return "n/a" if v is None else f"{v:.3f}"


def _acc_on_dataset(model, ds):
    tc = tn = 0
    for x, y in ds:
        p = model(x, training=False).numpy()
        tc += np.sum(np.argmax(p, 1) == y.numpy())
        tn += x.shape[0]
    return float(tc / tn) if tn else 0.0


def _acc_on_numpy(model, x, y, batch_size=256):
    """在已标准化的 numpy 测试集上算干净分类精度。"""
    y = np.asarray(y).reshape(-1)
    n = len(y)
    if n == 0:
        return 0.0
    correct = 0
    for i in range(0, n, batch_size):
        p = model(x[i:i + batch_size], training=False).numpy()
        correct += int(np.sum(np.argmax(p, 1) == y[i:i + batch_size]))
    return float(correct / n)


def compute_asr(model, x_test, y_test, trigger_fn, target_label, batch_size=256):
    """
    ASR = 加 trigger 的非目标类样本被预测为 target_label 的比例。
    x_test/y_test 为已标准化的完整测试集 numpy。

    评估侧 trigger 约定：trigger_fn(model, x, y) -> 加触发器后的 x。
    静态触发器（badnet/blended/dba 全局）忽略 model/y；动态/model-dependent
    触发器（CerP 用固定 trigger 变量；Bad-PFL 用 model + generator + FGSM，需要 y）
    用这两个参数生成。投毒侧触发器仍是 trigger_fn(x)，见 build_poisoned_dataset。
    """
    y_test = np.asarray(y_test).reshape(-1)
    mask = (y_test != int(target_label))
    xt = x_test[mask]
    yt = y_test[mask]
    n = len(xt)
    if n == 0:
        # 无定义（这个评估集里没有非目标类样本），不是「ASR 为零」。
        # 非 IID 下这是常态：pathological/Dirichlet 划分会让一些客户端的留出
        # 分片全是目标类。返回 0.0 会被读成「攻击完全失败」——那是个强结论。
        return None
    xp = trigger_fn(model, xt, yt)
    hits = 0
    for i in range(0, n, batch_size):
        p = model(xp[i:i + batch_size], training=False).numpy()
        hits += np.sum(np.argmax(p, 1) == int(target_label))
    return float(hits / n)


def _collect_eligible(ds, target_label, max_samples=0):
    """从 tf.data.Dataset 里收集**真实标签非目标类**的样本，返回 (x, y) numpy。

    转成 numpy 再交给 trigger_fn，是为了与 `compute_asr` 的调用约定完全一致 ——
    静态触发器（badnet/blended/dba）在 numpy 上做切片赋值，喂 tf 张量会炸。

    测试集的 dataset 是 `make_client_dataset(..., shuffle=False)` 建的，顺序固定，
    所以「取前 max_samples 个」是确定性的，不需要 RNG。
    """
    xs, ys, got = [], [], 0
    for x, y in ds:
        xb = x.numpy()
        yb = np.asarray(y.numpy()).reshape(-1)
        m = yb != int(target_label)
        if m.any():
            xs.append(xb[m])
            ys.append(yb[m])
            got += int(m.sum())
        if max_samples and got >= max_samples:
            break
    if not xs:
        return None, None
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    if max_samples and len(Y) > max_samples:
        X, Y = X[:max_samples], Y[:max_samples]
    return X, Y


def dataset_to_numpy(ds, max_samples=0):
    """把 tf.data.Dataset 摊平成 (x, y) numpy。**不做**目标类过滤。

    给还在吃 numpy 的旧接口用（如 `evaluate_forgetting_curve`，它内部自己调
    `compute_asr`、自己做过滤）。测试集的 dataset 是 shuffle=False 建的，
    顺序固定，所以「取前 max_samples 个」是确定性的。
    """
    xs, ys, got = [], [], 0
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(np.asarray(y.numpy()).reshape(-1))
        got += len(ys[-1])
        if max_samples and got >= max_samples:
            break
    if not xs:
        return None, None
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    if max_samples and len(Y) > max_samples:
        X, Y = X[:max_samples], Y[:max_samples]
    return X, Y


def compute_asr_on_dataset(model, ds, trigger_fn, target_label,
                           max_samples=0, batch_size=256):
    """在**留出分片**上算 ASR。返回 ``(asr 或 None, 合格样本数)``。

    与 `compute_asr` 的区别只是取数来源：那个吃 numpy 全量测试集，这个吃
    per-client / per-edge / 全局的 `tf.data.Dataset`。判据（排除真实标签已是
    目标类的样本、统计 argmax == target）逐字相同。

    **为什么要有这个函数**：本仓库按 PFLlib 口径把 train+test 合并后分区，
    于是原始 `x_test` 数组里的图已经分给客户端了，再拿它当 ASR 探针就是在
    训练过的图上测攻击成功率。而「所有客户端留出分片的并集」与「所有训练数据」
    **全局不相交**（分区使每个 index 只归一个客户端），才是真正的留出集。
    顺带地，ASR 与 `pm_acc` 这下测在同一个 population 上了。

    合格样本数为 0 → 返回 ``(None, 0)``：无定义，不是「ASR 为零」。
    """
    X, Y = _collect_eligible(ds, target_label, max_samples)
    if X is None or len(Y) == 0:
        return None, 0
    xp = trigger_fn(model, X, Y)
    hits = 0
    for i in range(0, len(Y), batch_size):
        p = model(xp[i:i + batch_size], training=False).numpy()
        hits += int(np.sum(np.argmax(p, 1) == int(target_label)))
    return float(hits / len(Y)), int(len(Y))


def evaluate_backdoor(clients, x_test, y_test, trigger_fn, target_label,
                      malicious_ids, fallback_test_ds=None, batch_size=256,
                      round_idx=None, verbose=True):
    """
    对所有良性客户端计算 C-Acc / ASR，逐客户端打印并返回平均值。

    Returns:
        {"c_acc": mean_c_acc, "asr": mean_asr, "per_client": [(id, cacc, asr), ...]}
    """
    malicious_ids = set(int(i) for i in (malicious_ids or set()))
    benign = [c for c in clients if int(c.client_id) not in malicious_ids]

    caccs, asrs, per_client = [], [], []
    for c in benign:
        ds = c.test_dataset if getattr(c, "test_dataset", None) is not None else fallback_test_ds
        cacc = _acc_on_dataset(c.model, ds) if ds is not None else float("nan")
        asr = compute_asr(c.model, x_test, y_test, trigger_fn, target_label, batch_size)
        caccs.append(cacc)
        asrs.append(asr)
        per_client.append((int(c.client_id), cacc, asr))
        if verbose:
            print(f"Round {round_idx} | Client {c.client_id} | "
                  f"C-Acc: {cacc:.3f} | ASR: {_f3(asr)}")

    # compute_asr 现在在「没有非目标类样本」时返回 None（无定义）——
    # 本函数当前无人调用（扁平路径的遗留），但留着不处理就是个地雷。
    defined = [v for v in asrs if v is not None]
    mc = float(np.nanmean(caccs)) if caccs else None
    ma = float(np.mean(defined)) if defined else None
    print(f"Round {round_idx} | AVG over {len(benign)} benign clients | "
          f"C-Acc: {_f3(mc)} | ASR: {_f3(ma)}")
    return {"c_acc": mc, "asr": ma, "per_client": per_client}


# ════════════════════════════════════════════════════════════════════════════
# 任务2：分层 ASR —— 分别评估 global / edge / local 三层模型的 ASR 与 ACC
# ════════════════════════════════════════════════════════════════════════════

def evaluate_hierarchical_asr(global_model, edge_servers, clients,
                              global_test_ds, trigger_fn, target_label,
                              malicious_ids, fallback_test_ds=None,
                              batch_size=256, local_model_fn=None,
                              asr_max_samples=0):
    """
    分别评估三层模型的 ASR 与干净精度（ACC）。

    层级结构特有指标：local_asr_same_edge 与 local_asr_diff_edge 的差值，
    用于判断后门是否溢出到与恶意客户端**不同**边缘节点的良性客户端。

    Args:
        global_model  : 全局模型
        edge_servers  : List[EdgeServer]，每个含 .model / .edge_id
        clients       : List[FLClient]，每个含 .model / .client_id /
                        .is_malicious / .assigned_edge / .test_dataset
        global_test_ds: 全局留出集（= 所有客户端留出分片的并集，
                        `merge_test_datasets(edge_servers, ...)`）。
                        **不是**官方 x_test —— 本仓库按 PFLlib 口径合并了
                        train+test 再分区，官方 test 的图已经分给客户端了，
                        拿它当探针就是在训练过的图上测 ASR。留出分片的并集
                        与全部训练数据全局不相交，才是真正的留出集。
        asr_max_samples: 每个探针最多取多少合格样本（0 = 全用）。限评估开销。
        malicious_ids : 恶意客户端 id 集合
        local_model_fn: 可选 callable(client) -> model，对本地模型评估前做变换
                        （Simple-Tuning 防御：返回重置头+干净微调后的模型）。
                        None 时直接用 client.model。

    Returns:
        dict（见 log_round_metrics 使用的全部字段）
    """
    malicious_ids = set(int(i) for i in (malicious_ids or set()))

    # ── 全局模型：探针 = 所有客户端留出分片的并集 ─────────────────────────
    global_asr, _ = compute_asr_on_dataset(
        global_model, global_test_ds, trigger_fn, target_label,
        asr_max_samples, batch_size)
    global_acc = _acc_on_dataset(global_model, global_test_ds)

    # ── 边缘模型 ──────────────────────────────────────────────────────────
    edge_ids, edge_asr_per_node, edge_accs = [], [], []
    malicious_edges = set()
    for edge in edge_servers:
        # 探针 = 该 edge 下所有客户端留出分片的并集（与 em_acc 同一个集合）
        e_ds = edge.get_test_dataset() or global_test_ds
        e_asr, _ = compute_asr_on_dataset(
            edge.model, e_ds, trigger_fn, target_label,
            asr_max_samples, batch_size)
        e_acc = _acc_on_dataset(edge.model, e_ds)
        edge_ids.append(int(edge.edge_id))
        edge_asr_per_node.append(e_asr)
        edge_accs.append(e_acc)
    # 找出恶意客户端所在的 edge（用于 same/diff edge 划分）
    for c in clients:
        if int(c.client_id) in malicious_ids:
            malicious_edges.add(int(getattr(c, "assigned_edge", -1)))

    # ── 本地（个性化）模型，按群体分组 ────────────────────────────────────
    local_asrs, local_accs = [], []
    benign_asrs, malicious_asrs = [], []
    same_edge_asrs, diff_edge_asrs = [], []
    # 逐 edge 分组（Experiment 3：判 amplification / dilution / cross-edge cancellation
    # 都是逐 edge 的比较，聚合均值会把它们全抹平）
    benign_by_edge = {eid: [] for eid in edge_ids}
    malicious_by_edge = {eid: [] for eid in edge_ids}

    for c in clients:
        cid = int(c.client_id)
        # Simple-Tuning 等后处理防御：评估前对本地模型做变换（默认用 c.model 原样）
        eval_model = local_model_fn(c) if local_model_fn is not None else c.model
        # 探针 = **这个客户端自己**的留出分片，与它的 pm_acc 同一个集合。
        # 于是 ASR 与 MTA 终于测在同一个 population 上（非 IID 下这很重要：
        # 全局均匀的官方测试集与某个客户端的类别分布可以差得很远）。
        ds = c.test_dataset if getattr(c, "test_dataset", None) is not None \
            else fallback_test_ds
        if ds is None:
            asr, acc = None, float("nan")
        else:
            asr, _ = compute_asr_on_dataset(
                eval_model, ds, trigger_fn, target_label,
                asr_max_samples, batch_size)
            acc = _acc_on_dataset(eval_model, ds)
        local_asrs.append(asr)
        local_accs.append(acc)

        edge_id = int(getattr(c, "assigned_edge", -1))
        if cid in malicious_ids:
            malicious_asrs.append(asr)
            malicious_by_edge.setdefault(edge_id, []).append(asr)
        else:
            benign_asrs.append(asr)
            benign_by_edge.setdefault(edge_id, []).append(asr)
            if edge_id in malicious_edges:
                same_edge_asrs.append(asr)
            else:
                diff_edge_asrs.append(asr)

    # 空组 = **该指标在本配置下无定义**，返回 None 而不是 0.0。
    #
    # 0.0 会被读成「后门完全没传过去」，与真实的强结论无法区分。实际发生过：
    #   · 所有 *distributed* 布点（每个 edge 都有恶意端）→ 不存在「无恶意 edge 的
    #     良性端」→ diff_edge_asrs 为空 → 旧代码报 diff_edge_asr=0.000；
    #   · 10edge_collocated（E0 全是恶意端，没有良性端）→ same_edge_asrs 为空
    #     → 旧代码报 same_edge_asr=0.000，per_edge[0].client_benign 也是 0.000。
    # 后者还会被画进逐 edge 图里，把那条线拉到底。
    #
    # 与防御判决的约定一致（RobustAggregationMixin 对坐标类防御记 admitted=None
    # 而非 0）：**无定义就留空，绝不用 0 填充后当数值参与统计**。
    # 先滤掉 None 再求均值：单个客户端的 ASR 可能**无定义**（它的留出分片里
    # 没有非目标类样本 —— 非 IID 下这是常态，不是异常）。把 None 混进 np.mean
    # 会直接炸；把它当 0 参与平均则会把「无定义」读成「攻击失败」。
    # 全组都无定义 → 整个指标无定义 → None。
    def _defined(xs):
        return [v for v in xs if v is not None]

    def _mean(xs):
        vals = _defined(xs)
        return float(np.mean(vals)) if vals else None

    def _std(xs):
        vals = _defined(xs)
        return float(np.std(vals)) if vals else None

    # 逐 edge 面板：edge 模型 ASR / 该 edge 内良性、恶意个性化 ASR 均值 / 计数 / 是否含恶意
    per_edge = []
    edge_asr_by_id = dict(zip(edge_ids, edge_asr_per_node))
    edge_acc_by_id = dict(zip(edge_ids, edge_accs))
    for eid in edge_ids:
        per_edge.append({
            "edge_id":         eid,
            "edge_asr":        edge_asr_by_id[eid],
            "edge_acc":        edge_acc_by_id[eid],
            "client_benign":   _mean(benign_by_edge.get(eid, [])),
            "client_malicious": _mean(malicious_by_edge.get(eid, [])),
            "n_benign":        len(benign_by_edge.get(eid, [])),
            "n_malicious":     len(malicious_by_edge.get(eid, [])),
            "has_malicious":   eid in malicious_edges,
        })

    return {
        "global_asr":             global_asr,
        "global_acc":             global_acc,
        "edge_asr_mean":          _mean(edge_asr_per_node),
        "edge_asr_std":           _std(edge_asr_per_node),
        "edge_asr_per_node":      edge_asr_per_node,
        "edge_ids":               edge_ids,
        "edge_acc_mean":          _mean(edge_accs),
        "local_asr_mean":         _mean(local_asrs),
        "local_asr_std":          _std(local_asrs),
        "local_asr_benign_mean":  _mean(benign_asrs),
        "local_asr_malicious_mean": _mean(malicious_asrs),
        "local_asr_same_edge":    _mean(same_edge_asrs),
        "local_asr_diff_edge":    _mean(diff_edge_asrs),
        "local_acc_mean":         float(np.nanmean(local_accs)) if local_accs else None,
        "per_edge":               per_edge,
    }


# ════════════════════════════════════════════════════════════════════════════
# 任务2：后门遗忘曲线 —— 个性化微调过程中每个 epoch 后记录 ASR
# ════════════════════════════════════════════════════════════════════════════

def evaluate_forgetting_curve(model, local_epochs, local_dataset,
                              x_test, y_test, trigger_fn, target_label,
                              loss_fn=None, lr=0.02, batch_size=256):
    """
    在本地训练过程中，每个 epoch 后记录一次 ASR，用于画后门遗忘曲线。

    重要：本函数**只做评估**，不影响实际训练状态——评估前保存模型权重，
    评估后恢复（TF 版用 get_weights/set_weights）。使用全新 SGD，避免污染
    调用方的优化器状态。

    Returns:
        长度为 local_epochs 的 ASR 列表。
    """
    if loss_fn is None:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    w0 = model.get_weights()                      # 保存原始状态
    opt = tf.keras.optimizers.SGD(learning_rate=lr)

    curve = []
    for _ in range(int(local_epochs)):
        for x, y in local_dataset:
            with tf.GradientTape() as tape:
                loss = loss_fn(y, model(x, training=True))
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        curve.append(
            compute_asr(model, x_test, y_test, trigger_fn, target_label, batch_size)
        )

    model.set_weights(w0)                          # 恢复，绝不污染训练态
    return curve


# ════════════════════════════════════════════════════════════════════════════
# 任务2：特征空间分离度（替代 T-SNE 的量化指标）
# ════════════════════════════════════════════════════════════════════════════

def _build_feature_extractor(model):
    """构建提取分类头之前一层（penultimate）特征的子模型。"""
    # 函数式模型最后一层是分类 Dense(softmax)，其输入张量即 penultimate 特征。
    return tf.keras.Model(model.inputs, model.layers[-1].input)


def evaluate_drift(edge_servers, global_model, anchor_weights, base_idx,
                   x_probe, batch_size=256):
    """
    Experiment 3C：每个 edge 相对「本轮起点全局」anchor 的 **参数漂移 + 表示漂移**。

      参数漂移（只在 backbone 索引 base_idx 上）：abs=‖Δ‖₂、rel=‖Δ‖/‖anchor‖₂
      表示漂移：固定干净探针 x_probe 上，penultimate 特征的 (1−cos)，对样本取 mean 与 median

    anchor 的特征：临时把 global_model 的权重设为 anchor 抽一次，再还原（单线程 eval 阶段安全）。
    对 edge 取均值上报，同时保留 per-edge。

    Returns: dict（见 backdoor_server._backdoor_eval 的使用）。
    """
    # anchor 侧特征：mutate-extract-restore（避免额外建一个模型）
    cur = global_model.get_weights()
    global_model.set_weights(anchor_weights)
    E_anchor = _extract_features(_build_feature_extractor(global_model), x_probe, batch_size)
    global_model.set_weights(cur)

    per_edge = []
    for edge in edge_servers:
        ew = edge.model.get_weights()
        p_abs, p_rel = param_drift(ew, anchor_weights, base_idx)
        E_e = _extract_features(_build_feature_extractor(edge.model), x_probe, batch_size)
        r_mean, r_med = repr_shift(E_e, E_anchor)
        per_edge.append({
            "edge_id":      int(edge.edge_id),
            "param_abs":    p_abs,
            "param_rel":    p_rel,
            "repr_mean":    r_mean,
            "repr_median":  r_med,
        })

    def _m(key):
        xs = [d[key] for d in per_edge if not np.isnan(d[key])]
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "param_abs":    _m("param_abs"),
        "param_rel":    _m("param_rel"),
        "repr_mean":    _m("repr_mean"),
        "repr_median":  _m("repr_median"),
        "per_edge":     per_edge,
    }


def _extract_features(feat_model, x, batch_size=256):
    feats = []
    for i in range(0, len(x), batch_size):
        feats.append(feat_model(x[i:i + batch_size], training=False).numpy())
    f = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1))
    return f.reshape(f.shape[0], -1)               # 展平为 (N, D)


def _cosine(a, b):
    """两个向量的余弦相似度。"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_feature_separation(model, poisoned_samples, poisoned_true_labels,
                                clean_samples_by_class, target_label,
                                batch_size=256):
    """
    提取分类头之前一层的特征向量，量化后门是否被植入：

      backdoor_to_target_sim : 后门样本特征 与 目标类干净样本特征 的余弦相似度均值
      backdoor_to_true_sim   : 后门样本特征 与 其真实类干净样本特征 的余弦相似度均值
      separation_score       : backdoor_to_target_sim − backdoor_to_true_sim
          > 0 说明后门被成功植入（特征空间更像目标类）
          < 0 说明后门被消除（特征空间仍像真实类）

    Args:
        poisoned_samples       : 加了 trigger 的样本 numpy (N,H,W,C)
        poisoned_true_labels   : 对应的真实类别 (N,)
        clean_samples_by_class : dict{class_id: numpy(Nc,H,W,C)} 干净样本
        target_label           : 后门目标类
    """
    feat_model = _build_feature_extractor(model)

    # 各类干净样本的特征质心
    centroids = {}
    for c, xs in clean_samples_by_class.items():
        if len(xs) == 0:
            continue
        feats = _extract_features(feat_model, xs, batch_size)
        if len(feats):
            centroids[int(c)] = feats.mean(axis=0)

    bd_feats = _extract_features(feat_model, poisoned_samples, batch_size)
    poisoned_true_labels = np.asarray(poisoned_true_labels).reshape(-1)

    target_centroid = centroids.get(int(target_label))
    to_target, to_true = [], []
    for i, f in enumerate(bd_feats):
        if target_centroid is not None:
            to_target.append(_cosine(f, target_centroid))
        true_c = int(poisoned_true_labels[i])
        if true_c in centroids:
            to_true.append(_cosine(f, centroids[true_c]))

    bt = float(np.mean(to_target)) if to_target else 0.0
    bf = float(np.mean(to_true)) if to_true else 0.0
    return {
        "backdoor_to_target_sim": bt,
        "backdoor_to_true_sim":   bf,
        "separation_score":       bt - bf,
    }
