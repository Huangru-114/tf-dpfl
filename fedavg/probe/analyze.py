"""
probe/analyze.py  —  把 paired.probe_client 的 Δ 变成指标行（**纯 numpy，不 import tf**）。

分组信息（层名 → 张量下标、backbone/head 切分）由调用方传进来（TF 侧的
probe/layermap.py 与 models/cnn.get_base_head_indices 产出），于是本模块可以本地 L1。

scope（陷阱 #9 的落实）
──────────────────────
所有指标在 **backbone 与 head 上分别算**，绝不把两者拼进同一个展平向量。
私有 head 逐客户端 warm-start、从不同步，是全部权重里发散最快的部分；混进去会
主导任何余弦/范数，把「后门位移集中在哪」这个问题答成「head 发散得最厉害」。

行的粒度 = (checkpoint, client, weight_space, scope, layer)。
`layer == "__all__"` 是该 scope 的整体行。
"""
import numpy as np

from .flatten import flatten, build_groups
from . import param_metrics as pm
from . import occupation as occ


def _scope_rows(deltas: dict, weights_ref, tensor_groups: dict, indices,
                *, base_row: dict, scope: str, topk=(0.01, 0.05, 0.10),
                n_bins: int = 10) -> tuple:
    """
    对一个 scope（backbone 或 head）算全部指标。

    Returns: (rows, summary)  —— rows 是 layer 级 + 一条 "__all__"；summary 含
             top-k% 能量、occupation 关系曲线、以及**判据用的 bd/stochastic 比值**。
    """
    if not indices:
        return [], None

    flat = {k: flatten(v, indices) for k, v in deltas.items()}
    d_clean, d_pois = flat["clean"], flat["poison"]
    d_bd, d_stoch = flat["bd"], flat["stochastic"]
    groups = build_groups(weights_ref, tensor_groups, indices)

    o_rank = occ.occupation_rank(d_clean)
    o_abs = occ.occupation_abs(d_clean)

    all_idx = np.arange(d_bd.size)
    norms_bd = pm.group_norms(d_bd, groups)
    nf = pm.norm_fractions(norms_bd)
    ef = pm.energy_fractions(norms_bd)

    rows = []
    for layer, idx in list(groups.items()) + [("__all__", all_idx)]:
        cr, cn = pm.conflict_ratio(d_bd[idx], d_clean[idx])
        rows.append({
            **base_row, "scope": scope, "layer": layer,
            "n_params": int(idx.size),
            "clean_update_norm": pm.norm(d_clean[idx]),
            "poison_update_norm": pm.norm(d_pois[idx]),
            "bd_update_norm": pm.norm(d_bd[idx]),
            "stochastic_norm": pm.norm(d_stoch[idx]),
            # Metric A（辅助）：恶意端整体更新是否仍沿正常任务方向
            "cosine_poison_clean": pm.cosine(d_pois[idx], d_clean[idx]),
            # Metric B（主）：poison 额外诱导的位移与主任务方向的关系
            "cosine_bd_clean": pm.cosine(d_bd[idx], d_clean[idx]),
            "conflict_ratio": cr, "conflict_n": cn,
            "bd_norm_frac": nf.get(layer, 1.0 if layer == "__all__" else float("nan")),
            "bd_energy_frac": ef.get(layer, 1.0 if layer == "__all__" else float("nan")),
            "occupation_abs_mean": float(np.mean(o_abs[idx])) if idx.size else float("nan"),
            "occupation_rank_mean": float(np.mean(o_rank[idx])) if idx.size else float("nan"),
        })

    n_stoch = pm.norm(d_stoch)
    n_bd = pm.norm(d_bd)
    summary = {
        **base_row, "scope": scope,
        "n_params": int(d_bd.size),
        "bd_norm": n_bd, "stochastic_norm": n_stoch,
        # ★ Stage 1 的核心判据：BD 位移是否显著超出纯 SGD 随机性
        "bd_over_stochastic": float(n_bd / n_stoch) if n_stoch > 1e-12 else float("nan"),
        "cosine_bd_clean": pm.cosine(d_bd, d_clean),
        "cosine_poison_clean": pm.cosine(d_pois, d_clean),
        "topk_bd_energy": pm.topk_energy_fraction(d_bd, ks=topk),
        "bd_norm_frac_by_layer": nf,
        "bd_energy_frac_by_layer": ef,
        "occupation_vs_bd": occ.occupation_vs_bd(o_rank, d_bd, n_bins=n_bins,
                                                 groups=groups),
    }
    return rows, summary


def analyze_probe(result: dict, weights_ref, tensor_groups: dict,
                  base_indices, head_indices, *, checkpoint: str = "",
                  topk=(0.01, 0.05, 0.10), n_bins: int = 10) -> dict:
    """
    Args:
        result:         paired.probe_client 的返回值
        weights_ref:    任一同形权重列表（只用来读 shape 定位下标区间）
        tensor_groups:  {层名: [get_weights 下标]}，来自 layermap.layer_groups
        base_indices / head_indices: models/cnn.get_base_head_indices 的切分
    Returns:
        {"rows": [...], "summaries": [...]}
    """
    base_row = {
        "checkpoint": checkpoint,
        "round": result["round"],
        "client_id": result["client_id"],
        "role": "malicious" if result["was_malicious"] else "benign",
    }
    rows, summaries = [], []
    for space in ("upload", "personal"):
        d = result["deltas"][space]
        for scope, idx in (("backbone", list(base_indices)), ("head", list(head_indices))):
            r, s = _scope_rows(d, weights_ref, tensor_groups, idx,
                               base_row={**base_row, "weight_space": space},
                               scope=scope, topk=topk, n_bins=n_bins)
            rows.extend(r)
            if s is not None:
                summaries.append(s)
    return {"rows": rows, "summaries": summaries}


ROW_COLUMNS = [
    "checkpoint", "round", "client_id", "role", "weight_space", "scope", "layer",
    "n_params", "clean_update_norm", "poison_update_norm", "bd_update_norm",
    "stochastic_norm", "cosine_poison_clean", "cosine_bd_clean",
    "conflict_ratio", "conflict_n", "bd_norm_frac", "bd_energy_frac",
    "occupation_abs_mean", "occupation_rank_mean",
]
