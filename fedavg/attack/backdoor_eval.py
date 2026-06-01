"""
attack/backdoor_eval.py  –  后门评估（C-Acc 本地 / ASR 全局）

对每个良性客户端的本地（个性化）模型 client.model：
  C-Acc : 在该 client 的本地干净测试集上的分类准确率（衡量攻击隐蔽性）
  ASR   : 在完整 CIFAR-10 测试集上，取非目标类样本加 trigger 后被分类为 target_label
          的比例（衡量后门有效性）。排除恶意客户端自身，只统计良性客户端。
"""

import numpy as np


def _acc_on_dataset(model, ds):
    tc = tn = 0
    for x, y in ds:
        p = model(x, training=False).numpy()
        tc += np.sum(np.argmax(p, 1) == y.numpy())
        tn += x.shape[0]
    return float(tc / tn) if tn else 0.0


def compute_asr(model, x_test, y_test, trigger_fn, target_label, batch_size=256):
    """
    ASR = 加 trigger 的非目标类样本被预测为 target_label 的比例。
    x_test/y_test 为已标准化的完整测试集 numpy。
    """
    y_test = np.asarray(y_test).reshape(-1)
    mask = (y_test != int(target_label))
    xt = x_test[mask]
    n = len(xt)
    if n == 0:
        return 0.0
    xp = trigger_fn(xt)
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
