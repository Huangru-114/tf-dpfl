import numpy as np


def aggregate(client_updates: list):
    """
    FedAvg 聚合：按样本数加权平均各客户端的模型权重。

    Args:
        client_updates: List of (weights, n_samples, loss, train_time)
            weights:    list of np.ndarray，模型每层的权重
            n_samples:  该客户端的本地样本数
            loss:       本地训练 loss
            train_time: 本地训练耗时（聚合不用，server 用于 logging）

    Returns:
        aggregated_weights: list of np.ndarray，聚合后的全局权重
    """
    total_samples = sum(n for _, n, _, _ in client_updates)
    aggregated    = [np.zeros_like(w) for w in client_updates[0][0]]

    for weights, n_samples, _, _ in client_updates:
        coeff = n_samples / total_samples
        for i, layer_w in enumerate(weights):
            aggregated[i] += coeff * layer_w

    return aggregated
