import yaml
import numpy as np
import tensorflow as tf
import argparse

from data.dataset       import load_cifar10, load_cifar100, load_imagenet
from data.partition     import (extract_numpy, iid_partition, noniid_partition,
                                pathological_noniid_partition,
                                split_client_train_test,
                                make_client_dataset,
                                superclass_edge_partition,
                                make_per_edge_test_datasets,
                                merge_test_datasets)
from data.clustering    import random_assignment, warmup_gradient_assignment, histogram_assignment, semantic_assignment
from models.cnn         import build_model
from models.model_utils import clone_model
from client.client_pfedme      import PFedMeClient
from client.hier_ditto_rep      import HierDittoRepClient
from client.hier_pfedme_rep     import HierPFedMeRepClient
from server.edge_server_pfedme import PFedMeEdgeServer
from server.hier_ditto_rep      import HierDittoRepEdgeServer
from server.hier_pfedme_rep     import HierPFedMeRepEdgeServer
from server.server      import CloudServer   # 原来是 FLServer


def _select_method_classes(config):
    """
    按 config["training"]["drift_correction"] 选择 client / edge 类。

    新增的 Hier-Rep 方法走各自的子类，其余方法（pfedme / hierpfedme 等）
    仍走默认的 PFedMeClient / PFedMeEdgeServer，行为不变。
    """
    method = config["training"].get("drift_correction", "pfedme")
    client_cls = {
        "hier_ditto_rep":  HierDittoRepClient,
        "hier_pfedme_rep": HierPFedMeRepClient,
    }.get(method, PFedMeClient)
    edge_cls = {
        "hier_ditto_rep":  HierDittoRepEdgeServer,
        "hier_pfedme_rep": HierPFedMeRepEdgeServer,
    }.get(method, PFedMeEdgeServer)
    return client_cls, edge_cls
from utils.logger       import FLLogger
from utils.report       import generate_report


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

    Returns:
        clients     : List[FLClient]
        assignments : List[int] or None
                      仅 superclass_pathological 分区时非 None，
                      供 build_edge_servers 直接使用，跳过再分配步骤。
        edge_fine_classes : List[set] or None
                      仅 superclass_pathological 时非 None，供构建 per-edge 测试集。
    """
    n_clients = config["federation"]["n_clients"]
    partition = config["federation"]["partition"]
    alpha     = config["federation"].get("alpha", 0.5)

    assignments       = None
    edge_fine_classes = None

    if partition == "iid":
        client_datasets, client_indices = iid_partition(images_np, labels_np, n_clients, config)

    elif partition == "noniid":
        client_datasets, client_indices = noniid_partition(
            images_np, labels_np, n_clients, config, alpha=alpha
        )

    elif partition == "pathological":
        client_datasets, client_indices = pathological_noniid_partition(
            images_np, labels_np, n_clients, config,
            classes_per_client=config["federation"].get("classes_per_client", 10)
        )

    elif partition == "superclass_pathological":
        # 导入聚类模块，构建每个 edge 的细粒度类别集合
        from data.clustering import (
            _CIFAR10_SEMANTIC_GROUPS,
            _CIFAR100_SEMANTIC_GROUPS,
            _build_cifar100_edge_groups,
            _superclass_groups_to_fineclasses,
        )
        n_edges  = config["federation"]["n_edges"]
        dataset  = config["data"]["dataset"].lower()

        if dataset == "cifar10" and n_edges in _CIFAR10_SEMANTIC_GROUPS:
            edge_fine_classes = _CIFAR10_SEMANTIC_GROUPS[n_edges]
        elif dataset == "cifar100":
            if n_edges in _CIFAR100_SEMANTIC_GROUPS:
                sc_groups = _CIFAR100_SEMANTIC_GROUPS[n_edges]
            else:
                sc_groups = _build_cifar100_edge_groups(n_edges)
            edge_fine_classes = _superclass_groups_to_fineclasses(sc_groups)
        else:
            # Tiny-ImageNet 等：按 ID 区间均匀划分
            num_classes = config["data"]["num_classes"]
            cpe = num_classes // n_edges
            edge_fine_classes = [
                set(range(e * cpe, (e + 1) * cpe if e < n_edges - 1 else num_classes))
                for e in range(n_edges)
            ]

        client_datasets, client_indices, assignments = superclass_edge_partition(
            images_np, labels_np, n_clients, config, edge_fine_classes
        )

    else:
        raise ValueError(f"Unknown partition type: {partition}")

    # per-client 测试集：从每个 client 自身的训练数据中划出 test_ratio（PFLlib 做法）
    test_ratio = config.get("data", {}).get("per_client_test_ratio", 0.2)
    client_datasets, client_indices, client_test_datasets = split_client_train_test(
        images_np, labels_np, client_indices, config, test_ratio=test_ratio
    )

    ClientCls, _ = _select_method_classes(config)
    clients = []
    for i, (ds, indices) in enumerate(zip(client_datasets, client_indices)):
        client_model = clone_model(global_model)
        client = ClientCls(client_id=i, dataset=ds, model=client_model,
                          config=config, n_samples=len(indices))
        if client_test_datasets is not None:
            client.set_test_dataset(client_test_datasets[i])
        # Store training class set for per-edge test dataset construction later.
        client.held_classes = set(np.unique(labels_np[indices]).tolist())
        clients.append(client)

    print(f"[Setup] {len(clients)} clients built "
          f"({partition}"
          f"{f', α={alpha}' if partition == 'noniid' else ''})"
          f"{' | per-client test sets injected' if client_test_datasets else ''}")
    return clients, assignments, edge_fine_classes


def build_edge_servers(clients, global_model, config,
                        precomputed_assignments=None):
    """
    根据 config 选择分配策略，把客户端分配给 Edge Server。

    两个分配策略：
        random:   随机均匀分配（baseline）
        gradient: warm-up 后基于梯度相似度聚类分配

    分配完成后，EdgeServer 内部用 client_fraction 控制每轮参与比例。

    Args:
        precomputed_assignments: 若非 None，直接使用该分配（跳过策略选择）。
                                 由 superclass_pathological 分区预先计算。
    """
    n_edges  = config["federation"]["n_edges"]
    strategy = config["federation"].get("edge_assignment", "random")

    # ── Step 1：决定分配方案 ──────────────────────────────
    if precomputed_assignments is not None:
        assignments = precomputed_assignments
        print(f"[Assignment] Using pre-baked superclass assignments | {n_edges} edges")

    elif strategy == "random":
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
        _, EdgeCls = _select_method_classes(config)
        edge       = EdgeCls(
            edge_id=i,
            clients=group,
            model=edge_model,
            config=config
        )
        edge_servers.append(edge)

    effective_strategy = "superclass_baked" if precomputed_assignments is not None else strategy
    print(f"\n[Setup] Built {len(edge_servers)} edge servers "
          f"(strategy={effective_strategy})")
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
    baked_assignments = None
    edge_fine_classes = None

    # 合并全量数据后分区：确保 per-client train/test 同分布（PFLlib 标准做法）
    # x_test/y_test 保留用于 edge/GM 评估，不受影响。
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    clients, baked_assignments, edge_fine_classes = build_clients(
        x_all, y_all, global_model, config,
        x_test_np=x_test, y_test_np=y_test
    )

    # 把 clients 分组给 Edge Server
    print("[Setup] Building edge servers...")
    edge_servers = build_edge_servers(clients, global_model, config,
                                      precomputed_assignments=baked_assignments)

    # 为每个 edge 注入与其训练分布匹配的测试集（EM 评估用）。
    # superclass_pathological：使用预定义的超类细粒度类集合。
    # 其他分区（pathological/noniid/iid）：从 client.held_classes 推断每个 edge 实际训练的类别。
    # 不注入时，edge 退化为在全体测试集上评估，会导致 loss 和 acc 正相关的假象
    # （模型对训练类自信，对未训练类高度错误 → 全集 loss 远高于随机，全集 acc 虚低）。
    print("[Setup] Building per-edge test datasets (superclass-aware)...")

    for edge in edge_servers:
        e_test_ds = merge_test_datasets(edge.clients, config["data"]["batch_size"])
        edge.set_test_dataset(e_test_ds)

    g_test_ds = merge_test_datasets(edge_servers, config["data"]["batch_size"])

    cloud = CloudServer(
        global_model=global_model,
        edge_servers=edge_servers,
        test_dataset=g_test_ds,
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

    method   = config["training"].get("drift_correction", "fedavg")
    run_name = config.get("wandb", {}).get("run_name", method)
    pm_steps = int(config.get("evaluation", {}).get("pm_steps", 1))

    # GM
    print("\n[Final Report] GM")
    generate_report(
        model        = cloud.global_model,
        test_dataset = test_ds,
        save_path    = f"report_GM_{run_name}.txt"
    )

    # EM（每个 edge 单独报告）
    for edge in edge_servers:
        print(f"\n[Final Report] EM — Edge {edge.edge_id}")
        generate_report(
            model        = edge.model,
            test_dataset = edge.get_test_dataset(),
            save_path    = f"report_EM_edge{edge.edge_id}_{run_name}.txt"
        )

    # PM（所有客户端汇总）
    print("\n[Final Report] PM — collecting predictions from all clients...")
    all_labels, all_preds, all_probs = [], [], []

    for edge in edge_servers:
        for client in edge.clients:
            if method == "hier_perfedavg":
                client.personalize_and_evaluate(
                    edge.model.get_weights(),
                    steps=pm_steps,
                    fallback_dataset=edge.get_test_dataset()
                )
            if client.test_dataset is not None:
                ds = client.test_dataset
                print(f"  [Client {client.client_id:>2}] Using per-client test dataset "
                      f"({len(ds)} batches)")
            else:
                ds = edge.get_test_dataset()
                print(f"  [Client {client.client_id:>2}] Using edge-level test dataset "
                      f"({len(ds)} batches)")
            for x, y in ds:
                probs = client.model(x, training=False).numpy()
                all_probs.append(probs)
                all_preds.append(np.argmax(probs, axis=1))
                all_labels.append(y.numpy())

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    all_probs  = np.concatenate(all_probs)

    generate_report(
        model        = None,           # PM 直接传预计算结果
        test_dataset = None,
        all_labels   = all_labels,
        all_preds    = all_preds,
        all_probs    = all_probs,
        save_path    = f"report_PM_{run_name}.txt"
    )

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