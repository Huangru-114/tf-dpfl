"""
experiments/distributions.py  –  数据分布配置注册表

集中定义两类分布配置，供 main.py / sweep.yaml 通过名字引用：

  DISTRIBUTION_CONFIGS              非层级（边缘节点内外用同一分布）D1–D6
  HIERARCHICAL_DISTRIBUTION_CONFIGS 层级（intra-edge × inter-edge）H1–H4

`resolve_distribution(name)` 把高层名展开为可写回 config 的字段：
  - 非层级 →  {"partition": <iid|noniid|pathological>, "alpha": .., "classes_per_client": ..}
  - 层级   →  {"partition": "hierarchical",
               "inter_edge": {...}, "intra_edge": {...}}

注：本仓库内部 partition 名映射：
  type "dirichlet"   → partition "noniid"（data/partition.noniid_partition）
  type "pathological"→ partition "pathological"
  type "iid"         → partition "iid"
"""

# ── 非层级配置（边缘节点内外用同一分布）────────────────────────────────────
DISTRIBUTION_CONFIGS = {
    "D1_patho_2": {
        "type": "pathological", "classes_per_client": 2,
        "description": "PFL 社区标准配置，预期 ASR≈0",
    },
    "D2_patho_5": {
        "type": "pathological", "classes_per_client": 5,
        "description": "pathological 临界点探索",
    },
    "D3_dir_01": {
        "type": "dirichlet", "alpha": 0.1,
        "description": "高异质性",
    },
    "D4_dir_05": {
        "type": "dirichlet", "alpha": 0.5,
        "description": "文献标准配置",
    },
    "D5_dir_10": {
        "type": "dirichlet", "alpha": 1.0,
        "description": "低异质性",
    },
    "D6_iid": {
        "type": "iid",
        "description": "IID 上界基线",
    },
}

# ── 层级配置（核心贡献点）：intra=边缘节点内部分布，inter=跨边缘节点分布 ──────
HIERARCHICAL_DISTRIBUTION_CONFIGS = {
    "H1_patho_in_dir_out": {
        "intra_edge": {"type": "pathological", "classes_per_client": 2},
        "inter_edge": {"type": "dirichlet", "alpha": 0.5},
        "description": "节点内极度异质，跨节点中等异质",
    },
    "H2_dir_in_patho_out": {
        # CIFAR-10 + n_edges=2 下用 classes_per_edge=5 覆盖全部 10 类，避免类被丢弃
        "intra_edge": {"type": "dirichlet", "alpha": 0.5},
        "inter_edge": {"type": "pathological", "classes_per_edge": 5},
        "description": "节点内中等异质，跨节点截然分离",
    },
    "H3_high_both": {
        "intra_edge": {"type": "dirichlet", "alpha": 0.1},
        "inter_edge": {"type": "dirichlet", "alpha": 0.5},
        "description": "双层均高异质",
    },
    "H4_iid_in_dir_out": {
        "intra_edge": {"type": "iid"},
        "inter_edge": {"type": "dirichlet", "alpha": 0.1},
        "description": "节点内同质，跨节点高异质",
    },
}

# 仓库内部 type → partition 名映射
_TYPE_TO_PARTITION = {
    "iid": "iid",
    "dirichlet": "noniid",
    "pathological": "pathological",
}


def is_hierarchical(name: str) -> bool:
    return name in HIERARCHICAL_DISTRIBUTION_CONFIGS


def resolve_distribution(name: str) -> dict:
    """
    把分布名展开为可写回 config["federation"] 的字段字典。

    返回示例：
      非层级 D4_dir_05 → {"partition":"noniid", "alpha":0.5}
      层级   H1_...    → {"partition":"hierarchical",
                          "inter_edge":{...}, "intra_edge":{...}}
    """
    if name in HIERARCHICAL_DISTRIBUTION_CONFIGS:
        cfg = HIERARCHICAL_DISTRIBUTION_CONFIGS[name]
        return {
            "partition": "hierarchical",
            "inter_edge": dict(cfg["inter_edge"]),
            "intra_edge": dict(cfg["intra_edge"]),
        }

    if name in DISTRIBUTION_CONFIGS:
        cfg = DISTRIBUTION_CONFIGS[name]
        out = {"partition": _TYPE_TO_PARTITION[cfg["type"]]}
        if cfg["type"] == "dirichlet":
            out["alpha"] = float(cfg["alpha"])
        elif cfg["type"] == "pathological":
            out["classes_per_client"] = int(cfg["classes_per_client"])
        return out

    raise KeyError(
        f"Unknown distribution name: {name!r}. "
        f"Choose from {list(DISTRIBUTION_CONFIGS) + list(HIERARCHICAL_DISTRIBUTION_CONFIGS)}"
    )
