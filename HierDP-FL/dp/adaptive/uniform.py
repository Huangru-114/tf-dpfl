"""
dp/adaptive/uniform.py
────────────────────────────────────────────────────────────────────────────
Uniform DP-SGD allocator — the standard DP-SGD baseline.

All clients, all rounds, all layers receive the same noise_multiplier and
clip_norm.  This is the starting point against which adaptive strategies are
benchmarked.
"""

from __future__ import annotations

from typing import Dict, Optional

from dp.adaptive.base import AdaptiveDPAllocator, NoiseParams


class UniformDPAllocator(AdaptiveDPAllocator):
    """
    Standard DP-SGD: uniform noise_multiplier and clip_norm for every
    client × round × epoch × layer combination.

    Parameters
    ----------
    noise_multiplier : float
    clip_norm        : float
    """

    def __init__(self, noise_multiplier: float, clip_norm: float):
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm

    def get_noise_params(
        self,
        client_id: int,
        round_num: int,
        epoch: int,
        layer_names: Optional[list] = None,
    ) -> NoiseParams:
        return NoiseParams(
            noise_multiplier=self.noise_multiplier,
            clip_norm=self.clip_norm,
            layer_clip_norms=None,  # uniform → no per-layer override
            description=f"uniform σ={self.noise_multiplier:.3f} C={self.clip_norm:.3f}",
        )

    def summary(self) -> str:
        return (
            f"UniformDPAllocator("
            f"noise_multiplier={self.noise_multiplier}, "
            f"clip_norm={self.clip_norm})"
        )
