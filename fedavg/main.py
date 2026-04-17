import yaml
import numpy as np
import tensorflow as tf
import argparse

from data.dataset       import load_cifar10, load_cifar100, load_imagenet
from data.partition     import extract_numpy, iid_partition, noniid_partition, pathological_noniid_partition, make_per_client_test_datasets
from data.clustering    import random_assignment, warmup_gradient_assignment, histogram_assignment, semantic_assignment
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


def build_clients(images_np, labels_np, global_model, config,
                  x_test_np=None, y_test_np=None):
    """
    数据分区 + 批量实例化所有客户端。

    每个客户端拿到：
      - 自己的 dataset partition
      - 全局模型的独立副本（clone_model，不共享权重对象）
      - per-client 同分布测试集（x_test_np/y_test_np 存在时注入，论文评估方式）
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
    elif partition == "pathological":
        client_datasets, client_indices = pathological_noniid_partition(
            images_np, labels_np, n_clients, config, classes_per_client=10
        )
    else:
        raise ValueError(f"Unknown partition type: {partition}")

    # per-client 同分布测试集：与训练分区相同的类别过滤，对应论文评估方式
    client_test_datasets = None
    if x_test_np is not None and y_test_np is not None:
        client_test_datasets = make_per_client_test_datasets(
            x_test_np, y_test_np, client_indices, labels_np, config
        )

    clients = []
    for i, (ds, indices) in enumerate(zip(client_datasets, client_indices)):
        client_model = clone_model(global_model)
        client = FLClient(client_id=i, dataset=ds, model=client_model,
                          config=config, n_samples=len(indices))
        if client_test_datasets is not None:
            client.set_test_dataset(client_test_datasets[i])
        clients.append(client)

    print(f"[Setup] {len(clients)} clients built "
          f"({partition}"
          f"{f', α={alpha}' if partition == 'noniid' else ''})"
          f"{' | per-client test sets injected' if client_test_datasets else ''}")
    return clients


def build_clients_imagenet(train_ds, y_train_np, global_model, config,
                            x_test_np=None, y_test_np=None):
    """
    ImageNet 专用客户端构建：train_ds 是完整 tf.data.Dataset，
    不做逐样本分区，而是按 shard 均匀切分给各 client。
    测试集同样按 per-client 类别过滤（如果 y_test_np 存在）。
    """
    n_clients = config["federation"]["n_clients"]

    # 用 y_train_np 做 pathological/noniid 分区得到每个 client 的样本索引
    # 然后用这些索引从 train_ds 切片（ImageNet 太大，改为 shard 方式）
    partition = config["federation"]["partition"]
    if partition == "pathological":
        from data.partition import pathological_noniid_partition
        # 为了得到 client_indices，需要 labels_np
        # 用 shard 代替 numpy 索引：先做索引分配，再用 tf.data.Dataset.filter
        num_classes = config["data"]["num_classes"]
        total_shards     = n_clients * 2
        shards_per_class = max(1, total_shards // num_classes)
        all_shards = []
        import numpy as np
        for c in range(num_classes):
            class_idx = np.where(y_train_np == c)[0]
            np.random.shuffle(class_idx)
            all_shards.extend(np.array_split(class_idx, shards_per_class))
        shard_order    = np.random.permutation(len(all_shards))
        client_indices = [[] for _ in range(n_clients)]
        for rank, sid in enumerate(shard_order):
            client_indices[rank % n_clients].extend(all_shards[sid].tolist())
    else:
        import numpy as np
        total   = len(y_train_np)
        indices = np.random.permutation(total)
        client_indices = [arr.tolist()
                          for arr in np.array_split(indices, n_clients)]

    # 构建 per-client dataset via index filtering
    client_test_datasets = None
    if x_test_np is not None and y_test_np is not None:
        from data.partition import make_per_client_test_datasets
        import numpy as np
        client_test_datasets = make_per_client_test_datasets(
            x_test_np, y_test_np,
            [np.array(idx) for idx in client_indices],
            y_train_np, config
        )

    clients = []
    for i, indices in enumerate(client_indices):
        idx_arr    = np.array(indices, dtype=np.int64)
        idx_set    = set(idx_arr.tolist())
        # 用 enumerate + filter 从全局 dataset 中筛选该 client 的样本
        counter    = [0]
        def make_filter(s):
            def f(img, lbl):
                idx = tf.py_function(lambda: counter.__setitem__(0, counter[0]+1) or (counter[0]-1) in s, [], tf.bool)
                return idx
            return f
        # 简化：用 shard 方式代替 filter（ImageNet 已经很大，filter 效率低）
        # 每个 client 取 train_ds 的第 i 个 shard
        client_ds = train_ds.shard(num_shards=n_clients, index=i)
        client_model = clone_model(global_model)
        client = FLClient(client_id=i, dataset=client_ds, model=client_model,
                          config=config, n_samples=len(indices))
        if client_test_datasets is not None:
            client.set_test_dataset(client_test_datasets[i])
        clients.append(client)

    print(f"[Setup] {n_clients} ImageNet clients built (shard-based)")
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

    elif strategy == "semantic":
        # 语义分配：基于客户端数据的标签分布与 Edge 服务器的类别分布的匹配度来分配。
        # 需要先统计每个客户端的数据标签分布，再根据预设的 Edge 服务器类别分布进行匹配分配。
        assignments = semantic_assignment(clients, n_edges, config)

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

    print("[Setup] Loading dataset...")
    dataset_name = config["data"].get("dataset", "cifar10").lower()
    if dataset_name == "cifar10":
        train_ds, test_ds, x_train, y_train, x_test, y_test = load_cifar10(config)
    elif dataset_name == "cifar100":
        train_ds, test_ds, x_train, y_train, x_test, y_test = load_cifar100(config)
    elif dataset_name in ("imagenet", "imagenet1k"):
        train_ds, test_ds, x_train, y_train, x_test, y_test = load_imagenet(config)
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. "
                         "Supported: cifar10, cifar100, imagenet")

    print("[Setup] Building global model...")
    global_model = build_model(
        input_shape=(config["data"]["img_size"],
                     config["data"]["img_size"], 3),
        num_classes=config["data"]["num_classes"],
        arch=config["model"]["arch"]
    )
    global_model.summary()

    print("[Setup] Building clients...")
    # ImageNet 不返回 x_train numpy 数组（内存不够），只用 y_train 做分区索引
    # partition 函数只需要 labels_np 来统计类别分布，images 用 train_ds 代替
    if x_train is None:
        # ImageNet：直接用 train_ds，不做 per-image 分区
        # 退化为 IID 分区（ImageNet 本身类别均衡，可接受）
        print("[Setup] ImageNet detected: using IID partition over tf.data pipeline")
        clients = build_clients_imagenet(train_ds, y_train, global_model, config,
                                         x_test_np=x_test, y_test_np=y_test)
    else:
        clients = build_clients(x_train, y_train, global_model, config,
                                x_test_np=x_test, y_test_np=y_test)

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