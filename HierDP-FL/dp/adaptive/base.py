"""
dp/adaptive/base.py
────────────────────────────────────────────────────────────────────────────
Abstract interface for adaptive differential-privacy budget / noise allocators.

Any new adaptive strategy must subclass AdaptiveDPAllocator and implement:

  • get_noise_params(...)  → NoiseParams
  • update(metrics)       → None  (called after each round to update state)

The HierDP-FL training loop calls `allocator.get_noise_params(...)` before
each client's local training to obtain the noise_multiplier and clip_norm
for that client/round/epoch.  After aggregation it calls `allocator.update()`
with the round metrics so the allocator can adapt its internal state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class NoiseParams:
    """
    Per-step DP noise specification returned by any adaptive allocator.

    Fields
    ------
    noise_multiplier : float
        σ (noise multiplier); actual per-step noise std = σ * clip_norm / B.
    clip_norm : float
        Global L2 gradient clipping bound C.
    layer_clip_norms : dict[str, float] or None
        Per-layer clip norms.  If provided, overrides `clip_norm` for matching
        layer names.  Used by the layer-wise adaptive strategy.
    description : str
        Human-readable description (for logging / debugging).
    """
    noise_multiplier: float
    clip_norm: float
    layer_clip_norms: Optional[Dict[str, float]] = None
    description: str = ""

    def effective_noise_std(self, batch_size: int) -> float:
        """Noise standard deviation per gradient coordinate."""
        return self.noise_multiplier * self.clip_norm / batch_size


class AdaptiveDPAllocator(ABC):
    """
    Abstract base class for adaptive DP noise / budget allocation.

    Subclasses implement one or more of the following dimensions:
      - Layer-wise  : different clip_norm per layer
      - Client-wise : different noise_multiplier per client
      - Epoch-wise  : noise schedule over training epochs
      - Round-wise  : noise schedule over global communication rounds

    The allocator is stateful: `update()` is called after every global round
    so it can adapt based on observed training dynamics.
    """

    @abstractmethod
    def get_noise_params(
        self,
        client_id: int,
        round_num: int,
        epoch: int,
        layer_names: Optional[list] = None,
    ) -> NoiseParams:
        """
        Return the DP noise parameters to use for a specific client / round / epoch.

        Parameters
        ----------
        client_id   : int   – client identifier
        round_num   : int   – global communication round (0-indexed)
        epoch       : int   – local epoch within the current round (0-indexed)
        layer_names : list  – names of the model's trainable layers (for layer-wise)

        Returns
        -------
        NoiseParams
        """
        ...

    def update(self, metrics: Dict) -> None:
        """
        Update the allocator's internal state after one global round.

        Override this in adaptive strategies that adjust noise based on
        training dynamics (e.g. gradient norms, loss convergence, etc.).

        Parameters
        ----------
        metrics : dict  – e.g. {
            'round': 3,
            'global_accuracy': 0.72,
            'edge_losses': [...],
            'client_grad_norms': {client_id: float, ...},
            'privacy_spent': {client_id: float, ...},
        }
        """
        pass  # default: stateless (uniform strategy)

    def summary(self) -> str:
        """Human-readable description of the allocator's current state."""
        return f"{self.__class__.__name__}"
