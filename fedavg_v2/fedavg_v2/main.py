import yaml
import argparse
import numpy as np
import tensorflow as tf

from data.dataset       import load_cifar10
from data.partition     import iid_partition, noniid_partition
from models.cnn         import build_model
from models.model_utils import clone_model
from client.client      import FLClient
from server.server      import FLServer
from utils.logger       import FLLogger


def load_config(path: str = "config/config.yaml") -> dict:
    """
    加载 config.yaml，支持命令行 --override key=value 覆盖任意字段。
    key 用点号表示嵌套，例如：
        --override federation.alpha=0.1
        --override wandb.run_name=my-run
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default=path)
    parser.add_argument("--override", type=str, action="append", default=[])
    args, _ = parser.parse_known_args()

    for override in args.override:
        key_path, value = override.split("=", 1)
        keys = key_path.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]
        try:
            value = yaml.safe_load(value)
        except Exception:
            pass
        d[keys[-1]] = value
        print(f"[Config] Override: {key_path} = {value}")

    return config


def set_seed(seed: int):
    """固定随机种子，保证实验可复现。"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[Setup] Random seed: {seed}")


def build_clients(x_train, y_train, global_model, config):
    """
    数据分区 + 批量实例化所有客户端。
    每个客户端拿到自己的 dataset partition 和独立的模型副本。
    """
    n_clients = config["federation"]["n_clients"]
    partition = config["federation"]["partition"]
    alpha     = config["federation"].get("alpha", 0.5)

    if partition == "iid":
        client_datasets = iid_partition(x_train, y_train, n_clients, config)
    elif partition == "noniid":
        client_datasets = noniid_partition(
            x_train, y_train, n_clients, config, alpha=alpha
        )
    else:
        raise ValueError(f"Unknown partition type: {partition}")

    clients = []
    for i, ds in enumerate(client_datasets):
        client_model = clone_model(global_model)
        client = FLClient(
            client_id=i,
            dataset=ds,
            model=client_model,
            config=config
        )
        clients.append(client)

    print(f"[Setup] {len(clients)} clients built "
          f"({partition}"
          f"{f', α={alpha}' if partition == 'noniid' else ''})")
    return clients


def run_experiment(config_path: str = "config/config.yaml"):
    # ── 1. 配置与随机种子 ─────────────────────────────────────
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    # ── 2. 数据加载 ──────────────────────────────────────────
    print("[Setup] Loading CIFAR-10...")
    train_ds, test_ds, x_train, y_train = load_cifar10(config)

    # ── 3. 全局模型 ──────────────────────────────────────────
    print("[Setup] Building global model...")
    global_model = build_model(
        input_shape=(config["data"]["img_size"],
                     config["data"]["img_size"], 3),
        num_classes=config["data"]["num_classes"]
    )
    global_model.summary()

    # ── 4. 客户端 ────────────────────────────────────────────
    print("[Setup] Building clients...")
    clients = build_clients(x_train, y_train, global_model, config)

    # ── 5. 服务端 ────────────────────────────────────────────
    server = FLServer(
        global_model=global_model,
        clients=clients,
        test_dataset=test_ds,
        config=config
    )

    # ── 6. Logger ────────────────────────────────────────────
    logger = None
    if config.get("wandb", {}).get("enabled", False):
        logger = FLLogger(config)
        print("[Setup] wandb enabled")
    else:
        print("[Setup] wandb disabled")

    # ── 7. 训练 ──────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(" Federated Training Start")
    print("=" * 52)
    history = server.run(logger=logger)

    # ── 8. 收尾 ──────────────────────────────────────────────
    if logger is not None:
        logger.finish()

    _print_summary(history)
    return history


def _print_summary(history: dict):
    best_idx   = int(np.argmax(history["global_acc"]))
    best_acc   = history["global_acc"][best_idx]
    best_round = history["round"][best_idx]
    final_acc  = history["global_acc"][-1]
    total_comm = history["comm_bytes_total"][-1] / 1024 / 1024

    print("\n" + "=" * 52)
    print(" Experiment Summary")
    print("=" * 52)
    print(f"  Best  global_acc  : {best_acc:.4f}  (round {best_round})")
    print(f"  Final global_acc  : {final_acc:.4f}")
    print(f"  Total rounds      : {len(history['round'])}")
    print(f"  Total comm        : {total_comm:.1f} MB")
    print("=" * 52)


if __name__ == "__main__":
    run_experiment()
