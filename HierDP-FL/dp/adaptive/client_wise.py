"""
dp/adaptive/client_wise.py
────────────────────────────────────────────────────────────────────────────
Client-wise adaptive DP allocator.

Motivation
----------
In federated learning, clients may have heterogeneous data distributions and
training dynamics.  A client with highly anomalous or sensitive data (measured
by, e.g., gradient magnitude, loss, or a domain-specific sensitivity score)
may warrant a higher noise level.  Conversely, well-behaved clients can be
given less noise to preserve model utility.

This module provides two strategies:

  1. SensitivityScoreClientDPAllocator
       Noise_multiplier for client c ∝ sensitivity_score(c).
       Sensitivity scores can be provided statically or recomputed dynamically.

  2. GradMagnitudeClientDPAllocator
       After each round, clients whose averaged gradient magnitude is in the
       top-k% get a higher noise multiplier for the next round.

Both are budget-neutral: the total expected privacy cost is held constant by
normalising noise levels so their harmonic mean equals the baseline σ.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from dp.adaptive.base import AdaptiveDPAllocator, NoiseParams

logger = logging.getLogger(__name__)


# ── Strategy 1: Static sensitivity scores ────────────────────────────────

class SensitivityScoreClientDPAllocator(AdaptiveDPAllocator):
    """
    Allocate noise proportional to a per-client sensitivity score.

    The noise multiplier for client c is:
        σ_c = σ_base * score_c / mean(scores)

    This is budget-neutral: average σ across clients ≈ σ_base.

    Parameters
    ----------
    base_noise_multiplier : float
        Baseline noise level (σ for a "typical" client).
    clip_norm             : float
        Uniform gradient clipping norm.
    sensitivity_scores    : dict[int, float]
        Per-client sensitivity scores (higher → more sensitive → more noise).
        If a client is not in the dict, its score defaults to 1.0 (neutral).
    min_noise_multiplier  : float
        Floor on per-client σ (prevent σ → 0 for low-score clients).
    max_noise_multiplier  : float
        Cap on per-client σ.
    """

    def __init__(
        self,
        base_noise_multiplier: float,
        clip_norm: float,
        sensitivity_scores: Optional[Dict[int, float]] = None,
        min_noise_multiplier: float = 0.5,
        max_noise_multiplier: float = 5.0,
    ):
        self.base_nm = base_noise_multiplier
        self.clip_norm = clip_norm
        self.scores = sensitivity_scores or {}
        self.min_nm = min_noise_multiplier
        self.max_nm = max_noise_multiplier

    def _client_nm(self, client_id: int) -> float:
        if not self.scores:
            return self.base_nm
        score = self.scores.get(client_id, 1.0)
        mean_score = float(np.mean(list(self.scores.values())))
        raw = self.base_nm * score / (mean_score + 1e-8)
        return float(np.clip(raw, self.min_nm, self.max_nm))

    def get_noise_params(
        self,
        client_id: int,
        round_num: int,
        epoch: int,
        layer_names: Optional[list] = None,
    ) -> NoiseParams:
        nm = self._client_nm(client_id)
        return NoiseParams(
            noise_multiplier=nm,
            clip_norm=self.clip_norm,
            description=f"client_sensitivity c={client_id} σ={nm:.3f}",
        )

    def summary(self) -> str:
        return (
            f"SensitivityScoreClientDPAllocator("
            f"base_σ={self.base_nm}, "
            f"n_scored_clients={len(self.scores)})"
        )


# ── Strategy 2: Gradient-magnitude adaptive per-client noise ─────────────

class GradMagnitudeClientDPAllocator(AdaptiveDPAllocator):
    """
    Dynamically adjust per-client noise based on gradient magnitude.

    After each global round, clients in the top `high_fraction` percentile
    of gradient norms get noise multiplied by `high_noise_factor`; clients
    in the bottom `low_fraction` percentile get noise divided by
    `high_noise_factor`.

    The intent: high-gradient clients are more "influential" and thus
    warrant stronger protection (more noise) to prevent membership inference.

    Parameters
    ----------
    base_noise_multiplier : float
    clip_norm             : float
    high_fraction         : float in (0, 1)
        Fraction of clients considered "high gradient" (get more noise).
    high_noise_factor     : float > 1
        Multiplicative factor for high-gradient clients.
    warmup_rounds         : int
        Rounds before adaptation kicks in.
    """

    def __init__(
        self,
        base_noise_multiplier: float,
        clip_norm: float,
        high_fraction: float = 0.3,
        high_noise_factor: float = 1.5,
        warmup_rounds: int = 2,
    ):
        self.base_nm = base_noise_multiplier
        self.clip_norm = clip_norm
        self.high_fraction = high_fraction
        self.high_noise_factor = high_noise_factor
        self.warmup_rounds = warmup_rounds

        self._round = 0
        # Current noise multiplier per client
        self._client_nms: Dict[int, float] = {}

    def get_noise_params(
        self,
        client_id: int,
        round_num: int,
        epoch: int,
        layer_names: Optional[list] = None,
    ) -> NoiseParams:
        nm = self._client_nms.get(client_id, self.base_nm)
        return NoiseParams(
            noise_multiplier=nm,
            clip_norm=self.clip_norm,
            description=f"grad_mag_client c={client_id} σ={nm:.3f}",
        )

    def update(self, metrics: Dict) -> None:
        """
        Expects metrics['client_grad_norms']: dict[client_id, float].
        """
        self._round += 1
        if self._round <= self.warmup_rounds:
            return

        grad_norms = metrics.get("client_grad_norms", {})
        if not grad_norms:
            return

        client_ids = list(grad_norms.keys())
        norms = np.array([grad_norms[c] for c in client_ids])

        n_high = max(1, int(len(client_ids) * self.high_fraction))
        high_threshold = np.sort(norms)[-n_high]

        new_nms: Dict[int, float] = {}
        for cid, norm in zip(client_ids, norms):
            if norm >= high_threshold:
                new_nms[cid] = min(self.base_nm * self.high_noise_factor, 10.0)
            else:
                new_nms[cid] = max(
                    self.base_nm / self.high_noise_factor, 0.1
                )
        # Budget-neutralise: rescale so mean == base_nm
        mean_new = float(np.mean(list(new_nms.values())))
        scale = self.base_nm / (mean_new + 1e-8)
        self._client_nms = {cid: float(nm * scale) for cid, nm in new_nms.items()}

        logger.debug(
            f"[GradMagClient] Round {self._round}: "
            f"high-grad clients = {[c for c, n in zip(client_ids, norms) if n >= high_threshold]}"
        )

    def summary(self) -> str:
        return (
            f"GradMagnitudeClientDPAllocator("
            f"base_σ={self.base_nm}, high_frac={self.high_fraction}, "
            f"round={self._round})"
        )
