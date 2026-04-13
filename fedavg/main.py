import yaml
import numpy as np
import tensorflow as tf
import argparse

from data.dataset       import load_cifar10
from data.partition     import extract_numpy, iid_partition, noniid_partition
from data.clustering    import random_assignment, warmup_gradient_assignment, histogram_assignment
from models.cnn         import build_model
from models.model_utils import clone_model
from client.client      import FLClient
from server.edge_server import EdgeServer
from server.server      import CloudServer   # 原来是 FLServer
from utils.logger       import FLLogger


def load_config(path: str = "config/config.yaml") -> dict:
    """
    加载 config.yaml，支持命令行 --override key=value 覆盖任意字段。
    key 用点号表示嵌套，例如 federation.alpha=0.1
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default=path)
    parser.add_argument("--override", type=str, action="append", default=[])
    args, _ = parser.parse_known_args()

    # 应用覆盖
    for override in args.override:
        key_path, value = override.split("=", 1)
        keys = key_path.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]
        # 尝试转换类型
        try:
            value = yaml.safe_load(value)   # 自动识别 bool/int/float/string
        except Exception:
            pass
        d[keys[-1]] = value
        print(f"[Config] Override: {key_path} = {value}")

    return config


def set_seed(seed: int):
    """
    固定随机种子，保证实验可复现。
    FL 的随机性来自三处：numpy（分区/采样）、tensorflow（初始化）。
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[Setup] Random seed: {seed}")


def build_clients(images_np, labels_np, global_model, config):
    """
    数据分区 + 批量实例化所有客户端。

    每个客户端拿到：
      - 自己的 dataset partition
      - 全局模型的独立副本（clone_model，不共享权重对象）
    """
    n_clients = config["federation"]["n_clients"]
    partition = config["federation"]["partition"]
    alpha     = config["federation"].get("alpha", 0.5)

    if partition == "iid":
        client_datasets, client_indices = iid_partition(images_np, labels_np, n_clients, config)
    elif partition == "noniid":
        client_datasets, client_indices = noniid_partition(
            images_np, labels_np, n_clients, config, alpha=alpha
        )
    else:
        raise ValueError(f"Unknown partition type: {partition}")

    clients = []
    for i, (ds, indices) in enumerate(zip(client_datasets, client_indices)):
        client_model = clone_model(global_model)
        client = FLClient(client_id=i, dataset=ds, model=client_model, config=config,n_samples=len(indices))
        clients.append(client)

    print(f"[Setup] {len(clients)} clients built "
          f"({partition}"
          f"{f', α={alpha}' if partition == 'noniid' else ''})")
    return clients

def build_edge_servers(clients, global_model, config):
    """
    根据 config 选择分配策略，把客户端分配给 Edge Server。

    两个分配策略：
        random:   随机均匀分配（baseline）
        gradient: warm-up 后基于梯度相似度聚类分配

    分配完成后，EdgeServer 内部用 client_fraction 控制每轮参与比例。
    """
    n_edges  = config["federation"]["n_edges"]
    strategy = config["federation"].get("edge_assignment", "random")

    # ── Step 1：决定分配方案 ──────────────────────────────
    if strategy == "random":
        assignments = random_assignment(clients, n_edges)

    elif strategy == "gradient":
        assignments = warmup_gradient_assignment(
            clients, global_model, n_edges, config
        )

    elif strategy == "histogram":
        assignments = histogram_assignment(
            clients, n_edges, config
        )

    else:
        raise ValueError(f"Unknown edge_assignment: {strategy}")

    # ── Step 2：按分配方案构建 EdgeServer ─────────────────
    client_groups = [[] for _ in range(n_edges)]
    for client_idx, edge_idx in enumerate(assignments):
        client_groups[edge_idx].append(clients[client_idx])

    edge_servers = []
    for i, group in enumerate(client_groups):
        if len(group) == 0:
            # 分配不均时的保护
            print(f"  Warning: Edge {i} has 0 clients, "
                  f"check n_edges vs n_clients")
            continue
        edge_model = clone_model(global_model)
        edge       = EdgeServer(
            edge_id=i,
            clients=group,
            model=edge_model,
            config=config
        )
        edge_servers.append(edge)

    print(f"\n[Setup] Built {len(edge_servers)} edge servers "
          f"(strategy={strategy})")
    for e in edge_servers:
        print(f"  Edge {e.edge_id}: "
              f"{len(e.clients)} clients | "
              f"{e.n_samples} samples")

    return edge_servers

def run_experiment(config_path="config/config.yaml"):
    config       = load_config(config_path)
    set_seed(config.get("seed", 42))

    print("[Setup] Loading CIFAR-10...")
    train_ds, test_ds, x_train, y_train = load_cifar10(config)

    print("[Setup] Building global model...")
    global_model = build_model(
        input_shape=(config["data"]["img_size"],
                     config["data"]["img_size"], 3),
        num_classes=config["data"]["num_classes"],
        arch=config["model"]["arch"]
    )
    global_model.summary()

    print("[Setup] Building clients...")
    clients = build_clients(x_train, y_train, global_model, config)

    # ← 新增：把 clients 分组给 Edge Server
    print("[Setup] Building edge servers...")
    edge_servers = build_edge_servers(clients, global_model, config)

    # ← 原来是 FLServer，现在是 CloudServer
    cloud = CloudServer(
        global_model=global_model,
        edge_servers=edge_servers,
        test_dataset=test_ds,
        config=config
    )

    logger = None
    if config.get("wandb", {}).get("enabled", False):
        logger = FLLogger(config)

    print("\n" + "=" * 52)
    print(" HierFAVG Training Start")
    print("=" * 52)
    history = cloud.run(logger=logger)

    if logger is not None:
        logger.finish()

    _print_summary(history)
    return history

def _print_summary(history: dict):
    best_idx  = int(np.argmax(history["global_acc"]))
    best_acc  = history["global_acc"][best_idx]
    best_round = history["round"][best_idx]
    final_acc = history["global_acc"][-1]

    print("\n" + "=" * 52)
    print(" Experiment Summary")
    print("=" * 52)
    print(f"  Best  global_acc : {best_acc:.4f}  (round {best_round})")
    print(f"  Final global_acc : {final_acc:.4f}")
    print(f"  Total rounds     : {len(history['round'])}")
    print("=" * 52)


if __name__ == "__main__":
    run_experiment()
