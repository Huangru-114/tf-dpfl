import yaml
import argparse
import numpy as np
import tensorflow as tf

from data.dataset       import load_cifar10
from data.partition     import (iid_partition, noniid_partition,
                                pathological_noniid_partition,
                                make_per_client_test_datasets)
from models.cnn         import build_model
from models.model_utils import clone_model
from client.client      import FLClient
from client.client_pfedme import PFedMeClient
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


def build_clients(x_train, y_train, x_test, y_test, global_model, config):
    """
    数据分区 + 批量实例化所有客户端。

    Args:
        x_train, y_train : 训练集 numpy 数组
        x_test,  y_test  : 测试集 numpy 数组（用于生成 per-client 测试集）
        global_model     : 全局模型，每个客户端从中 clone 独立副本
        config           : 实验配置

    Returns:
        clients : List[PFedMeClient]
    """
    n_clients  = config["federation"]["n_clients"]
    partition  = config["federation"]["partition"]
    alpha      = config["federation"].get("alpha", 0.5)

    # ── 训练集分区 ────────────────────────────────────────────────────────
    if partition == "iid":
        client_datasets, client_indices = iid_partition(
            x_train, y_train, n_clients, config
        )
    elif partition == "noniid":
        client_datasets, client_indices = noniid_partition(
            x_train, y_train, n_clients, config, alpha=alpha
        )
    elif partition == "pathological":
        client_datasets, client_indices = pathological_noniid_partition(
            x_train, y_train, n_clients, config,
            classes_per_client=config["federation"].get("classes_per_client", 2)
        )
    else:
        raise ValueError(f"Unknown partition type: {partition}")

    # ── Per-client 测试集生成 ─────────────────────────────────────────────
    # 根据每个客户端训练集中实际出现的类别，从测试集里过滤出同类别样本。
    # 三种分区方式均支持：
    #   pathological → 每客户端只有少数类别，测试集同样只含这些类别
    #   noniid       → 每客户端持有的类别由 Dirichlet 决定，过滤掉样本数为 0 的类别
    #   iid          → 每客户端持有几乎所有类别，测试集接近全局测试集
    client_test_datasets = make_per_client_test_datasets(
        x_test_np=x_test,
        y_test_np=y_test,
        client_train_indices=client_indices,
        y_train_np=y_train,
        config=config,
    )

    # ── 客户端实例化 ──────────────────────────────────────────────────────
    clients = []
    for i, (ds, test_ds) in enumerate(zip(client_datasets, client_test_datasets)):
        client_model = clone_model(global_model)
        client = PFedMeClient(
            client_id=i,
            dataset=ds,
            model=client_model,
            config=config,
        )
        if test_ds is not None:
            client.set_test_dataset(test_ds)
        clients.append(client)

    print(
        f"[Setup] {len(clients)} clients built "
        f"({partition}"
        f"{f', α={alpha}' if partition == 'noniid' else ''})"
    )
    return clients


def run_experiment(config_path: str = "config/config.yaml"):
    # ── 1. 配置与随机种子 ─────────────────────────────────────
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    # ── 2. 数据加载 ──────────────────────────────────────────
    # load_cifar10 返回 (train_ds, test_ds, x_train_np, y_train_np,
    #                    x_test_np, y_test_np)
    # 若原 load_cifar10 只返回4个值，需同步修改 dataset.py 一并返回测试集 numpy。
    print("[Setup] Loading CIFAR-10...")
    train_ds, test_ds, x_train, y_train, x_test, y_test = load_cifar10(config)

    # ── 3. 全局模型 ──────────────────────────────────────────
    print("[Setup] Building global model...")
    global_model = build_model(
        input_shape=(config["data"]["img_size"],
                     config["data"]["img_size"], 3),
        num_classes=config["data"]["num_classes"]
    )
    global_model.summary()

    # ── 4. 客户端（含 per-client 测试集注入） ─────────────────
    print("[Setup] Building clients...")
    clients = build_clients(x_train, y_train, x_test, y_test, global_model, config)

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