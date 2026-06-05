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
from client.client_fedavg       import FedAvgClient
from client.hier_ditto_rep      import HierDittoRepClient
from client.hier_pfedme_rep     import HierPFedMeRepClient
from client.hier_ditto          import HierDittoClient
from client.hier_perfedavg      import HierPerFedAvgClient
from client.hier_fedrep         import HierFedRepClient
from server.edge_server_pfedme import PFedMeEdgeServer
from server.edge_server_fedavg import FedAvgEdgeServer
from server.hier_ditto_rep      import HierDittoRepEdgeServer
from server.hier_pfedme_rep     import HierPFedMeRepEdgeServer
from server.hier_ditto          import HierDittoEdgeServer
from server.hier_perfedavg      import HierPerFedAvgEdgeServer
from server.hier_fedrep         import HierFedRepEdgeServer
from server.server      import CloudServer   # 原来是 FLServer
from server.backdoor_server import BackdoorCloudServer
from attack.triggers        import build_trigger
from attack.backdoor        import (get_malicious_ids, resolve_malicious_ids,
                                    build_poisoned_dataset,
                                    make_dba_local_triggers,
                                    install_forced_participation)
from client.client_neurotoxin import NeurotoxinClient
from client.client_cerp        import CerPClient
from client.client_badpfl      import BadPFLClient


def _select_method_classes(config):
    """
    按 config["training"]["drift_correction"] 选择 client / edge 类。

    新增的 Hier-FedAvg / Hier-Rep / Hier-Ditto / Hier-PerFedAvg 方法走各自的
    子类，其余方法（pfedme / hierpfedme 等）仍走默认的
    PFedMeClient / PFedMeEdgeServer，行为不变。
    """
    method = config["training"].get("drift_correction", "pfedme")
    client_cls = {
        "fedavg":          FedAvgClient,
        "hierfedavg":      FedAvgClient,
        "hier_ditto_rep":  HierDittoRepClient,
        "hier_pfedme_rep": HierPFedMeRepClient,
        "hier_ditto":      HierDittoClient,
        "hier_perfedavg":  HierPerFedAvgClient,
        "hier_fedrep":     HierFedRepClient,
    }.get(method, PFedMeClient)
    edge_cls = {
        "fedavg":          FedAvgEdgeServer,
        "hierfedavg":      FedAvgEdgeServer,
        "hier_ditto_rep":  HierDittoRepEdgeServer,
        "hier_pfedme_rep": HierPFedMeRepEdgeServer,
        "hier_ditto":      HierDittoEdgeServer,
        "hier_perfedavg":  HierPerFedAvgEdgeServer,
        "hier_fedrep":     HierFedRepEdgeServer,
    }.get(method, PFedMeEdgeServer)
    return client_cls, edge_cls
from utils.logger       import FLLogger
from utils.report       import generate_report


# ── 高层实验维度 → 现有 config 字段的映射（任务3：sweep / CLI 用） ───────────
# 本轮仅实现 4 个方法；FedBN / 纯 FedRep 暂缓（见计划文件）。
FRAMEWORK_MAP = {
    "hier_fedavg":    {"drift_correction": "hierfedavg", "final_finetune": False},
    "hier_fedavg_ft": {"drift_correction": "hierfedavg", "final_finetune": True},
    "hier_pfedme":    {"drift_correction": "hierpfedme", "final_finetune": False},
    "hier_ditto":     {"drift_correction": "hier_ditto", "final_finetune": False},
    "hier_fedavg_fedrep": {"drift_correction": "hier_fedrep", "final_finetune": False},
    # TODO: "hier_fedavg_fedbn"
}

# ── attack_method 友好别名 → 两个正交轴 (trigger × malicious_strategy) ────────
# 配置内部始终是两正交轴；CLI/sweep 用单一 attack_method 别名，便于网格扫描。
ATTACK_METHOD_MAP = {
    "none":      {"enabled": False, "trigger": "badnet",     "strategy": "vanilla"},
    "badnet":    {"enabled": True,  "trigger": "badnet",     "strategy": "vanilla"},
    "blended":   {"enabled": True,  "trigger": "blended",    "strategy": "vanilla"},
    "dba":       {"enabled": True,  "trigger": "dba",        "strategy": "vanilla"},
    "neurotoxin":{"enabled": True,  "trigger": "badnet",     "strategy": "neurotoxin"},
    # Phase 2（动态投毒 + model-dependent 评估，触发器在 local_train 内动态生成，
    # 故 trigger 字段仅作标签用；投毒不走 build_poisoned_dataset）：
    "cerp":      {"enabled": True,  "trigger": "badnet",     "strategy": "cerp"},
    "badpfl":    {"enabled": True,  "trigger": "badnet",     "strategy": "badpfl"},
}


def select_malicious_client_class(strategy: str):
    """
    恶意客户端用的「攻击策略类」（继承 FedAvgClient，与 PFL 方法正交）。
    返回 None 表示用方法类（vanilla：badnet/blended/dba 走方法类 + 投毒数据，
    保持与既有实验一致的行为）。
    """
    return {
        "neurotoxin": NeurotoxinClient,
        "cerp":       CerPClient,
        "badpfl":     BadPFLClient,
    }.get(str(strategy).lower())


def build_eval_trigger(bd_cfg, static_trigger, clients, config):
    """
    构建评估侧触发器 trigger_fn(model, x, y=None) -> 加触发器的 numpy x。

    - 静态策略（vanilla/neurotoxin：badnet/blended/dba）：忽略 model/y，套用静态触发器。
    - CerP / Bad-PFL（动态/model-dependent）：用首个对应恶意客户端的触发器/生成器。
      CerP 用其固定 _trigger 变量；Bad-PFL 用被评估模型自身 + 该客户端 generator。
    """
    strategy = str(bd_cfg.get("malicious_strategy", "vanilla")).lower()
    mal = [c for c in clients if getattr(c, "is_malicious", False)]
    if strategy in ("cerp", "badpfl") and mal:
        c0 = mal[0]
        return lambda model, x, y=None: c0.eval_trigger(model, x, y)
    return lambda model, x, y=None: static_trigger(x)


def _set_nested(config: dict, key_path: str, value):
    keys = key_path.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_experiment_args(config: dict, args) -> dict:
    """
    把高层实验维度（framework / distribution_config / attack_method / seed）
    展开为现有 config 字段，并按规范生成 wandb run 名：
        {framework}_{distribution}_{attack}_seed{seed}
    例：hier_fedavg_ft_D4_dir_05_badnet_seed42
    """
    from experiments.distributions import resolve_distribution

    if args.seed is not None:
        config["seed"] = int(args.seed)

    if args.framework is not None:
        if args.framework not in FRAMEWORK_MAP:
            raise ValueError(
                f"Unknown framework: {args.framework!r}. "
                f"Available: {list(FRAMEWORK_MAP)}")
        fm = FRAMEWORK_MAP[args.framework]
        _set_nested(config, "training.drift_correction", fm["drift_correction"])
        _set_nested(config, "evaluation.final_finetune", fm["final_finetune"])

    if args.distribution_config is not None:
        dist = resolve_distribution(args.distribution_config)
        for k, v in dist.items():
            _set_nested(config, f"federation.{k}", v)

    if args.attack_method is not None:
        am = args.attack_method.lower()
        if am not in ATTACK_METHOD_MAP:
            raise ValueError(
                f"Unknown attack_method: {args.attack_method!r}. "
                f"Available: {list(ATTACK_METHOD_MAP)}")
        spec = ATTACK_METHOD_MAP[am]
        _set_nested(config, "backdoor.enabled", spec["enabled"])
        if spec["enabled"]:
            # 友好别名展开为两个正交轴：trigger × malicious_strategy
            _set_nested(config, "backdoor.trigger", spec["trigger"])
            _set_nested(config, "backdoor.malicious_strategy", spec["strategy"])

    # run 命名规范（仅当提供了高层维度时覆盖）
    if any(v is not None for v in
           (args.framework, args.distribution_config, args.attack_method)):
        fw = args.framework or config["training"].get("drift_correction", "fl")
        dist = args.distribution_config or config["federation"].get("partition", "dist")
        atk = args.attack_method or (
            config["backdoor"].get("trigger", "badnet")
            if config.get("backdoor", {}).get("enabled") else "none")
        seed = config.get("seed", 42)
        _set_nested(config, "wandb.run_name", f"{fw}_{dist}_{atk}_seed{seed}")
        print(f"[Config] run_name = {config['wandb']['run_name']}")

    return config


def load_config(path: str = "config/config.yaml") -> dict:
    """
    加载 config.yaml，支持：
      --override key=value          覆盖任意字段（点号表示嵌套，如 federation.alpha=0.1）
      --framework / --distribution_config / --attack_method / --seed
                                    高层实验维度（展开为现有字段 + 生成 run_name）
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default=path)
    parser.add_argument("--override", type=str, action="append", default=[])
    parser.add_argument("--framework",           type=str, default=None)
    parser.add_argument("--distribution_config", type=str, default=None)
    parser.add_argument("--attack_method",       type=str, default=None)
    parser.add_argument("--seed",                type=str, default=None)
    args, _ = parser.parse_known_args()

    # 先应用高层维度（可被后续显式 --override 进一步覆盖）
    config = apply_experiment_args(config, args)

    # 应用 --override
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

    elif partition == "hierarchical":
        # 层级两段式划分（任务1 核心）：先 inter_edge 分到 edge，再 intra_edge 分到 client。
        # 返回 assignments，走 baked 通道（与 superclass_pathological 同款）。
        from data.hierarchical_partition import hierarchical_partition
        n_edges   = config["federation"]["n_edges"]
        inter_cfg = config["federation"]["inter_edge"]
        intra_cfg = config["federation"]["intra_edge"]
        min_samp  = int(config["federation"].get("min_samples", 10))
        client_datasets, client_indices, assignments = hierarchical_partition(
            images_np, labels_np, n_clients, n_edges,
            inter_cfg, intra_cfg, config,
            min_samples=min_samp, seed=config.get("seed", 42),
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

    # ── 后门攻击：恶意客户端用投毒数据集替换原本地数据集 ───────────────────
    bd_cfg        = config.get("backdoor", {})
    bd_enabled    = bool(bd_cfg.get("enabled", False))
    # 解析恶意客户端（默认跨 edge 分散，需要 assignments；非 baked 分区退化为等距）。
    # 解析结果写回 bd_cfg["malicious_ids"]，保证 run_experiment 后续调用一致。
    malicious_ids = resolve_malicious_ids(
        bd_cfg, n_clients, assignments=assignments, seed=config.get("seed", 42))
    if bd_enabled:
        bd_cfg["malicious_ids"] = sorted(int(i) for i in malicious_ids)
        print(f"[Backdoor] resolved malicious clients "
              f"(placement={bd_cfg.get('malicious_placement', 'spread')}): "
              f"{bd_cfg['malicious_ids']}")
    bd_trigger    = build_trigger(bd_cfg, img_size=config["data"]["img_size"]) if bd_enabled else None
    bd_target     = int(bd_cfg.get("target_label", 9))
    bd_poison     = float(bd_cfg.get("poison_ratio", 0.5))
    strategy      = str(bd_cfg.get("malicious_strategy", "vanilla")).lower()
    trigger_kind  = bd_cfg.get("trigger", "badnet")

    # DBA：每个恶意客户端分到一个**局部**触发器用于投毒（评估侧用全局触发器）
    dba_triggers  = (make_dba_local_triggers(bd_cfg, malicious_ids)
                     if (bd_enabled and trigger_kind == "dba") else {})
    # 动态投毒策略（Phase 2 CerP/Bad-PFL）：恶意客户端保留 clean 数据，训练时动态投毒
    dynamic_poison = strategy in ("cerp", "badpfl")

    # 安全防护：CerP/Bad-PFL 的恶意客户端用自定义 eager 训练（可训练触发器 / generator），
    # 与良性客户端的 @tf.function 路径在线程池里并发会互相污染 TF 图（reshape/形状错乱）。
    # 故对动态策略强制串行收集（n_workers=1）。Neurotoxin 已复用 @tf.function 路径，无需串行。
    if bd_enabled and dynamic_poison:
        config["federation"]["n_workers"] = 1
        print(f"[Backdoor] strategy={strategy}: forcing serial client collection "
              f"(federation.n_workers=1) to avoid eager/@tf.function 线程冲突。")

    ClientCls, _ = _select_method_classes(config)
    MalCls       = select_malicious_client_class(strategy)  # None => 用方法类

    clients = []
    for i, (ds, indices) in enumerate(zip(client_datasets, client_indices)):
        is_mal  = bd_enabled and (i in malicious_ids)
        use_cls = ClientCls
        if is_mal:
            use_cls = MalCls or ClientCls
            if not dynamic_poison:
                # 静态投毒：构造 client 前替换数据集（DBA 用局部触发器，否则用统一触发器）
                poison_trig = dba_triggers.get(i, bd_trigger)
                ds = build_poisoned_dataset(images_np, labels_np, indices, config,
                                            poison_trig, bd_target, bd_poison)
            print(f"[Backdoor] Client {i} MALICIOUS | strategy={strategy} | "
                  f"trigger={trigger_kind} | target={bd_target} | poison={bd_poison}"
                  f"{' | dynamic-poison (clean kept)' if dynamic_poison else ''}")
        client_model = clone_model(global_model)
        client = use_cls(client_id=i, dataset=ds, model=client_model,
                         config=config, n_samples=len(indices))
        client.is_malicious = is_mal
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

    # ── Step 2a：（可选）划分 train / test client ─────────────────
    # test-client 评估设定：把一定比例的 client 整体留出，不参与任何 FL 训练，
    # 其数据（含 per-client test 部分）在训练中从不泄露；训练结束后这些 test
    # client 接收其分配 edge 的模型、做一组微调，再在留出的 test 数据上评估，
    # 衡量模型对「新客户端」的泛化能力。train client 行为不变。
    eval_cfg   = config.get("evaluation", {})
    tc_enabled = bool(eval_cfg.get("test_client_eval", False))
    tc_ratio   = float(eval_cfg.get("test_client_ratio", 0.2))
    n_clients  = len(clients)
    is_test    = np.zeros(n_clients, dtype=bool)
    if tc_enabled and tc_ratio > 0:
        n_test = max(1, int(n_clients * tc_ratio))
        test_idx = np.random.choice(n_clients, n_test, replace=False)
        is_test[test_idx] = True
        print(f"[Test-client holdout] {n_test}/{n_clients} clients held out "
              f"(ratio={tc_ratio}); excluded from FL training.")

    # ── Step 2b：按分配方案构建 EdgeServer（仅 train client 入 edge） ──────
    test_clients  = []
    client_groups = [[] for _ in range(n_edges)]
    for client_idx, edge_idx in enumerate(assignments):
        c = clients[client_idx]
        if is_test[client_idx]:
            c.is_test_client = True
            c.assigned_edge  = int(edge_idx)
            test_clients.append(c)
        else:
            c.is_test_client = False
            c.assigned_edge  = int(edge_idx)
            client_groups[edge_idx].append(c)

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
    if test_clients:
        print(f"  [Held-out] {len(test_clients)} test clients "
              f"(not in any edge training set)")

    return edge_servers, test_clients

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
    edge_servers, test_clients = build_edge_servers(
        clients, global_model, config,
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

    # ── 后门攻击：用 BackdoorCloudServer + fix-frequency 强制参与 ───────────
    bd_cfg     = config.get("backdoor", {})
    bd_enabled = bool(bd_cfg.get("enabled", False))
    if bd_enabled:
        malicious_ids = get_malicious_ids(bd_cfg)
        # 投毒侧静态触发器（badnet/blended，或 DBA 全局触发器）
        bd_trigger    = build_trigger(bd_cfg, img_size=config["data"]["img_size"])
        # 评估侧 trigger 统一为 (model, x, y)：静态触发器忽略 model/y。
        # CerP/Bad-PFL 动态触发器在此处替换为真正依赖 model/y 的评估触发器。
        eval_trigger  = build_eval_trigger(bd_cfg, bd_trigger, clients, config)
        cloud = BackdoorCloudServer(
            global_model=global_model,
            edge_servers=edge_servers,
            test_dataset=g_test_ds,
            config=config,
            bd_cfg=bd_cfg,
            x_test=x_test, y_test=y_test,
            trigger_fn=eval_trigger,
            malicious_ids=malicious_ids,
        )
        # fix-frequency：强制恶意客户端每 Q 轮参与一次
        Q = int(bd_cfg.get("attack_freq_Q", 10))
        for mc in clients:
            if int(mc.client_id) in malicious_ids:
                install_forced_participation(edge_servers, mc, Q)
    else:
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
    eval_cfg = config.get("evaluation", {})
    pm_steps = int(eval_cfg.get("pm_steps", 1))

    # 最后微调选项：PM 评估前对下发的 edge 模型做少步本地微调
    # （主要用于 Hier-FedAvg 等无个性化模型的方法，得到「FedAvg + 微调」baseline）。
    do_final_ft = bool(eval_cfg.get("final_finetune", False))
    ft_steps    = int(eval_cfg.get("final_finetune_steps", pm_steps))
    ft_lr       = float(eval_cfg.get("final_finetune_lr",
                                     config["training"]["learning_rate"]))

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
    if do_final_ft:
        print(f"  [Final fine-tuning] enabled | steps={ft_steps}, lr={ft_lr}")
    all_labels, all_preds, all_probs = [], [], []
    pm_accs, pm_ns = [], []

    for edge in edge_servers:
        for client in edge.clients:
            if method == "hier_perfedavg":
                client.personalize_and_evaluate(
                    edge.model.get_weights(),
                    steps=pm_steps,
                    fallback_dataset=edge.get_test_dataset()
                )
            elif do_final_ft:
                # 下发 edge 模型 + 少步本地微调，微调后的模型留在 client.model
                _finetune_client_model(client, edge.model.get_weights(),
                                       ft_steps, ft_lr)
            ds = client.test_dataset if client.test_dataset is not None \
                else edge.get_test_dataset()
            c_correct = c_total = 0
            for x, y in ds:
                probs = client.model(x, training=False).numpy()
                preds = np.argmax(probs, axis=1)
                yn    = y.numpy()
                all_probs.append(probs)
                all_preds.append(preds)
                all_labels.append(yn)
                c_correct += int(np.sum(preds == yn))
                c_total   += int(len(yn))
            c_acc = c_correct / c_total if c_total else 0.0
            pm_accs.append(c_acc)
            pm_ns.append(c_total)
            tag = "finetuned " if do_final_ft else ""
            print(f"  [Client {client.client_id:>2}] {tag}C-Acc={c_acc:.4f} (n={c_total})")

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    all_probs  = np.concatenate(all_probs)

    _tot  = sum(pm_ns)
    _wacc = sum(a * n / _tot for a, n in zip(pm_accs, pm_ns)) if _tot else 0.0
    _ftlabel = " (after final fine-tuning)" if do_final_ft else ""
    print(f"\n[Final PM]{_ftlabel} weighted C-Acc = {_wacc:.4f} "
          f"over {len(pm_ns)} clients / {_tot} samples")

    generate_report(
        model        = None,           # PM 直接传预计算结果
        test_dataset = None,
        all_labels   = all_labels,
        all_preds    = all_preds,
        all_probs    = all_probs,
        save_path    = f"report_PM_{run_name}.txt"
    )

    # Held-out test-client 评估（仅当启用了 test-client 留出设定时）
    if test_clients:
        _evaluate_test_clients(test_clients, edge_servers, global_model,
                               config, test_ds)

    return history

def _finetune_client_model(client, src_weights, steps, lr):
    """
    把下发模型 src_weights 载入 client.model，在 client 自身训练数据上微调
    steps 个 epoch（每次用全新 SGD，避免跨 client 状态与优化器变量集冲突）。
    微调后的模型留在 client.model 供后续评估。
    """
    client.model.set_weights(src_weights)
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    for _ in range(int(steps)):
        for x, y in client.dataset:
            with tf.GradientTape() as tape:
                loss = client.loss_fn(y, client.model(x, training=True))
            grads = tape.gradient(loss, client.model.trainable_variables)
            opt.apply_gradients(zip(grads, client.model.trainable_variables))


def _evaluate_test_clients(test_clients, edge_servers, global_model, config,
                           fallback_test_ds):
    """
    Held-out test-client 评估（衡量对新客户端的泛化）：

      每个 test client 接收其分配 edge 的模型（下发模型），在自身 train 数据上
      微调 test_finetune_steps 个 epoch，再在训练中从未泄露的 per-client test
      数据上评估。结果按样本数加权汇总后直接打印到日志（不生成报告文件）。

    train client 不受影响；test client 全程未参与任何 FL 训练。
    """
    eval_cfg = config.get("evaluation", {})
    steps    = int(eval_cfg.get("test_finetune_steps",
                                eval_cfg.get("pm_steps", 1)))
    lr       = float(eval_cfg.get("test_finetune_lr",
                                  config["training"]["learning_rate"]))
    loss_fn  = tf.keras.losses.SparseCategoricalCrossentropy()
    edge_by_id = {e.edge_id: e for e in edge_servers}

    print("\n" + "=" * 52)
    print(f" Held-out Test-Client Evaluation "
          f"(finetune {steps} epoch(s), lr={lr})")
    print("=" * 52)

    accs, losses, ns = [], [], []
    ft_model = clone_model(global_model)

    for tc in test_clients:
        edge  = edge_by_id.get(getattr(tc, "assigned_edge", None))
        src_w = edge.model.get_weights() if edge is not None \
            else global_model.get_weights()
        ft_model.set_weights(src_w)

        # 微调：在 test client 自身 train 数据上（每个 client 用全新 SGD）
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
        for _ in range(steps):
            for x, y in tc.dataset:
                with tf.GradientTape() as tape:
                    loss = loss_fn(y, ft_model(x, training=True))
                grads = tape.gradient(loss, ft_model.trainable_variables)
                opt.apply_gradients(zip(grads, ft_model.trainable_variables))

        # 评估：在留出的 per-client test 数据上（从未参与训练）
        ds = tc.test_dataset if tc.test_dataset is not None else fallback_test_ds
        tl = tcorr = tn = 0
        for x, y in ds:
            p = ft_model(x, training=False).numpy()
            tl    += loss_fn(y, p).numpy() * x.shape[0]
            tcorr += np.sum(np.argmax(p, 1) == y.numpy())
            tn    += x.shape[0]
        if tn == 0:
            continue
        accs.append(tcorr / tn)
        losses.append(tl / tn)
        ns.append(tn)
        print(f"  [TestClient {tc.client_id:>2}] edge={getattr(tc,'assigned_edge','-')} "
              f"| n={tn} | acc={tcorr/tn:.4f} loss={tl/tn:.4f}")

    if not ns:
        print("  [Held-out] no test client produced predictions.")
        return

    total = sum(ns)
    wacc  = sum(a * n / total for a, n in zip(accs, ns))
    wloss = sum(l * n / total for l, n in zip(losses, ns))
    print(f"\n  Held-out test-client: weighted acc={wacc:.4f} | loss={wloss:.4f} "
          f"over {len(ns)} clients / {total} samples")


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