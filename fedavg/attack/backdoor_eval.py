"""
attack/backdoor_eval.py  –  后门评估（C-Acc 本地 / ASR 全局）

对每个良性客户端的本地（个性化）模型 client.model：
  C-Acc : 在该 client 的本地干净测试集上的分类准确率（衡量攻击隐蔽性）
  ASR   : 在完整 CIFAR-10 测试集上，取非目标类样本加 trigger 后被分类为 target_label
          的比例（衡量后门有效性）。排除恶意客户端自身，只统计良性客户端。
"""

import numpy as np
import tensorflow as tf


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

    评估侧 trigger 约定：trigger_fn(model, x) -> 加触发器后的 x。
    静态触发器（badnet/blended/dba 全局）忽略 model；动态/model-dependent
    触发器（CerP/Bad-PFL）用 model 生成。投毒侧触发器仍是 trigger_fn(x)，
    见 attack/backdoor.build_poisoned_dataset。
    """
    y_test = np.asarray(y_test).reshape(-1)
    mask = (y_test != int(target_label))
    xt = x_test[mask]
    n = len(xt)
    if n == 0:
        return 0.0
    xp = trigger_fn(model, xt)
    hits = 0
    for i in range(0, n, batch_size):
        p = model(xp[i:i + batch_size], training=False).numpy()
        hits += np.sum(np.argmax(p, 1) == int(target_label))
    return float(hits / n)


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
                  f"C-Acc: {cacc:.3f} | ASR: {asr:.3f}")

    mc = float(np.nanmean(caccs)) if caccs else 0.0
    ma = float(np.mean(asrs)) if asrs else 0.0
    print(f"Round {round_idx} | AVG over {len(benign)} benign clients | "
          f"C-Acc: {mc:.3f} | ASR: {ma:.3f}")
    return {"c_acc": mc, "asr": ma, "per_client": per_client}


# ════════════════════════════════════════════════════════════════════════════
# 任务2：分层 ASR —— 分别评估 global / edge / local 三层模型的 ASR 与 ACC
# ════════════════════════════════════════════════════════════════════════════

def evaluate_hierarchical_asr(global_model, edge_servers, clients,
                              x_test, y_test, trigger_fn, target_label,
                              malicious_ids, fallback_test_ds=None,
                              batch_size=256):
    """
    分别评估三层模型的 ASR 与干净精度（ACC）。

    层级结构特有指标：local_asr_same_edge 与 local_asr_diff_edge 的差值，
    用于判断后门是否溢出到与恶意客户端**不同**边缘节点的良性客户端。

    Args:
        global_model  : 全局模型
        edge_servers  : List[EdgeServer]，每个含 .model / .edge_id
        clients       : List[FLClient]，每个含 .model / .client_id /
                        .is_malicious / .assigned_edge / .test_dataset
        x_test,y_test : 已标准化的全量测试集 numpy（ASR 与 global ACC 用）
        malicious_ids : 恶意客户端 id 集合

    Returns:
        dict（见 log_round_metrics 使用的全部字段）
    """
    y_test = np.asarray(y_test).reshape(-1)
    malicious_ids = set(int(i) for i in (malicious_ids or set()))

    # ── 全局模型 ──────────────────────────────────────────────────────────
    global_asr = compute_asr(global_model, x_test, y_test, trigger_fn,
                             target_label, batch_size)
    global_acc = _acc_on_numpy(global_model, x_test, y_test, batch_size)

    # ── 边缘模型 ──────────────────────────────────────────────────────────
    edge_asr_per_node, edge_accs = [], []
    malicious_edges = set()
    for edge in edge_servers:
        e_asr = compute_asr(edge.model, x_test, y_test, trigger_fn,
                            target_label, batch_size)
        e_acc = _acc_on_numpy(edge.model, x_test, y_test, batch_size)
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

    for c in clients:
        cid = int(c.client_id)
        asr = compute_asr(c.model, x_test, y_test, trigger_fn,
                          target_label, batch_size)
        ds = c.test_dataset if getattr(c, "test_dataset", None) is not None \
            else fallback_test_ds
        acc = _acc_on_dataset(c.model, ds) if ds is not None else float("nan")
        local_asrs.append(asr)
        local_accs.append(acc)

        if cid in malicious_ids:
            malicious_asrs.append(asr)
        else:
            benign_asrs.append(asr)
            edge_id = int(getattr(c, "assigned_edge", -1))
            if edge_id in malicious_edges:
                same_edge_asrs.append(asr)
            else:
                diff_edge_asrs.append(asr)

    def _mean(xs):
        return float(np.mean(xs)) if len(xs) else 0.0

    def _std(xs):
        return float(np.std(xs)) if len(xs) else 0.0

    return {
        "global_asr":             global_asr,
        "global_acc":             global_acc,
        "edge_asr_mean":          _mean(edge_asr_per_node),
        "edge_asr_std":           _std(edge_asr_per_node),
        "edge_asr_per_node":      edge_asr_per_node,
        "edge_acc_mean":          _mean(edge_accs),
        "local_asr_mean":         _mean(local_asrs),
        "local_asr_std":          _std(local_asrs),
        "local_asr_benign_mean":  _mean(benign_asrs),
        "local_asr_malicious_mean": _mean(malicious_asrs),
        "local_asr_same_edge":    _mean(same_edge_asrs),
        "local_asr_diff_edge":    _mean(diff_edge_asrs),
        "local_acc_mean":         float(np.nanmean(local_accs)) if local_accs else 0.0,
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
