import yaml
import numpy as np
import tensorflow as tf
import argparse

from data.dataset       import load_cifar10
from data.partition     import extract_numpy, iid_partition, noniid_partition
from models.resnet         import build_model
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
        client_datasets = iid_partition(images_np, labels_np, n_clients, config)
    elif partition == "noniid":
        client_datasets = noniid_partition(
            images_np, labels_np, n_clients, config, alpha=alpha
        )
    else:
        raise ValueError(f"Unknown partition type: {partition}")

    clients = []
    for i, ds in enumerate(client_datasets):
        client_model = clone_model(global_model)
        client = FLClient(client_id=i, dataset=ds, model=client_model, config=config)
        clients.append(client)

    print(f"[Setup] {len(clients)} clients built "
          f"({partition}"
          f"{f', α={alpha}' if partition == 'noniid' else ''})")
    return clients

def build_edge_servers(clients: list, global_model, config):
    """
    把客户端列表均分给各 Edge Server，
    每个 Edge Server 拿到一个独立的模型副本。
    """
    n_edges  = config["federation"]["n_edges"]
    # 均分客户端到各 Edge
    splits   = np.array_split(clients, n_edges)

    edge_servers = []
    for i, client_group in enumerate(splits):
        edge_model = clone_model(global_model)
        edge = EdgeServer(
            edge_id=i,
            clients=list(client_group),
            model=edge_model,
            config=config
        )
        edge_servers.append(edge)

    print(f"[Setup] {n_edges} edge servers built, "
          f"~{len(clients) // n_edges} clients each")
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
