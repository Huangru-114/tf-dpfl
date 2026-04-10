"""
aggregation/scaffold.py
───────────────────────
SCAFFOLD server-side aggregation (Karimireddy et al., ICML 2020).

Each EdgeServer and (optionally) the CloudServer call `scaffold_aggregate`
after collecting local models.  Unlike FedDyn, SCAFFOLD carries an explicit
control-variate correction term that is updated by both client and server.

Key formulas (Option II)
────────────────────────
  Client uploads:
      y_i           – local model after K corrected gradient steps
      Δc_i = c_i^new − c_i
           = −c_server + (x − y_i) / (K·η)

  Server aggregates:
      x^new = mean(y_i)                         [unweighted; η_s = 1]
      c^new = c + (1/N) · Σ_{i ∈ active} Δc_i  [N = TOTAL count, not |active|]

Usage
─────
    from aggregation.scaffold import scaffold_aggregate, init_cv

    c_e = init_cv(model.get_weights())   # once

    new_w = scaffold_aggregate(
        c_server      = c_e,
        active_weights= [w for w, *_ in updates],
        c_deltas      = [client.c_delta for client in active_clients],
        total_n       = len(all_clients),
    )
    model.set_weights(new_w)
"""

import numpy as np
from typing import List


def scaffold_aggregate(
    c_server: List[np.ndarray],
    active_weights: List[List[np.ndarray]],
    c_deltas: List[List[np.ndarray]],
    total_n: int,
) -> List[np.ndarray]:
    """
    One SCAFFOLD server aggregation step.

    Side-effects
    ────────────
    Mutates `c_server` in-place.

    Args:
        c_server      : Server control variate (same shape as model weights).
        active_weights: Model weights from active participants this round.
        c_deltas      : Δc_i from each active participant.
                        Only include clients whose c_delta is not None.
        total_n       : TOTAL participants (active + inactive).
                        ⚠️  Must not be replaced with len(active_weights).
    Returns:
        new_weights: unweighted mean of active models.
    """
    assert len(active_weights) > 0, "active_weights must be non-empty."
    assert total_n >= len(active_weights), "total_n < active count – check caller."
    n_layers = len(c_server)

    # ── 1. Update server control variate ──────────────────────────────────
    # c^new = c + (1/N) · Σ Δc_i
    if c_deltas:
        for i in range(n_layers):
            delta_sum = sum(cd[i] for cd in c_deltas)
            c_server[i] = c_server[i] + delta_sum / total_n

    # ── 2. Unweighted model average ────────────────────────────────────────
    new_w = [
        np.mean([w[i] for w in active_weights], axis=0)
        for i in range(n_layers)
    ]
    return new_w


def init_cv(ref_weights: List[np.ndarray]) -> List[np.ndarray]:
    """Return a zeroed control variate matching the shape of ref_weights."""
    return [np.zeros_like(w, dtype=np.float32) for w in ref_weights]
