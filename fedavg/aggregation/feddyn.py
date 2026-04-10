"""
aggregation/feddyn.py
─────────────────────
Server-side FedDyn aggregation (Acar et al., ICLR 2021).

Both EdgeServer and CloudServer call `feddyn_aggregate` after collecting
local models.  The function maintains – and mutates in-place – a persistent
correction vector `h` that tracks cumulative gradient deviation, then
subtracts it from the naive mean to yield a de-biased server model.

Usage
─────
    from aggregation.feddyn import feddyn_aggregate, init_h

    # Initialise once (lazily, on first call):
    h = init_h(server_model.get_weights())

    # Every round:
    new_weights = feddyn_aggregate(
        h_state       = h,
        active_weights= [w for w, *_ in updates],
        prev_anchor   = weights_broadcast_before_training,
        alpha         = cfg["training"]["alpha_feddyn"],
        total_n       = total_participants,   # ← ALL, not just active!
    )
    server_model.set_weights(new_weights)
"""

import numpy as np
from typing import List


def feddyn_aggregate(
    h_state: List[np.ndarray],
    active_weights: List[List[np.ndarray]],
    prev_anchor: List[np.ndarray],
    alpha: float,
    total_n: int,
) -> List[np.ndarray]:
    """
    One FedDyn server aggregation step.

    Update rules (Algorithm 1, Acar et al. 2021)
    ─────────────────────────────────────────────
        h  ←  h  −  (α / N)  ·  Σ_{k ∈ active} (θ_k^new − θ_anchor)
        θ_server^new  =  mean_{k ∈ active}(θ_k^new)  −  (1/α) · h

    ⚠️  Critical: `total_n` must be the *total* participant count (active +
    inactive), **not** `len(active_weights)`.  Using only the active count
    introduces a systematic bias in h that accumulates over rounds and
    prevents convergence.  See §2 and Theorem 1 in the paper.

    Args:
        h_state        : Persistent correction vector. Mutated in-place.
                         Same layer structure as the model weights.
        active_weights : Model weights uploaded by participants this round.
                         Shape: List[ List[np.ndarray] ]
        prev_anchor    : Server model that was broadcast *before* local
                         training began (θ^{t-1} in the paper).
        alpha          : Regularisation coefficient.  Must match the α used
                         inside the clients' local FedDyn objectives.
        total_n        : Total number of participants (active + stale).

    Returns:
        new_server_weights : List[np.ndarray]
    """
    assert len(h_state) == len(prev_anchor), (
        f"h_state has {len(h_state)} layers but prev_anchor has "
        f"{len(prev_anchor)}.  They must match the model architecture."
    )
    assert len(active_weights) > 0, "active_weights must be non-empty."
    assert total_n >= len(active_weights), (
        "total_n must be >= number of active participants."
    )

    n_layers = len(h_state)

    # ── Step 1: update correction state h ─────────────────────────────────
    # h  ←  h  −  (α / N)  ·  Σ_k (θ_k^new − θ_anchor)
    for i in range(n_layers):
        delta_sum = sum(w[i] - prev_anchor[i] for w in active_weights)
        h_state[i] = h_state[i] - (alpha / total_n) * delta_sum

    # ── Step 2: unweighted average of active models ────────────────────────
    # FedDyn uses an unweighted mean (consistent with the paper's formulation).
    # For strongly unbalanced data, consider switching to a sample-weighted
    # mean here, but note that the convergence proof assumes equal weights.
    mean_w = [
        np.mean([w[i] for w in active_weights], axis=0)
        for i in range(n_layers)
    ]

    # ── Step 3: apply h correction ─────────────────────────────────────────
    # θ_server^new  =  mean(θ_k^new)  −  (1/α) · h
    new_w = [m - (1.0 / alpha) * h for m, h in zip(mean_w, h_state)]
    return new_w


def init_h(ref_weights: List[np.ndarray]) -> List[np.ndarray]:
    """
    Return a zeroed h state whose layer shapes match `ref_weights`.
    Call once before the first training round.
    """
    return [np.zeros_like(w, dtype=np.float32) for w in ref_weights]
