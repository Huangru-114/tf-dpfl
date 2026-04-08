"""
dp/adaptive/epoch_wise.py
────────────────────────────────────────────────────────────────────────────
Epoch-wise and round-wise adaptive DP noise scheduling.

Motivation
----------
Early in training, the model parameters carry little information about
individual training examples (the loss landscape is still rough).  As
training progresses and the model converges, gradients become increasingly
informative about individual data points.

An epoch-decaying noise schedule (higher noise early, lower later) can:
  • Speed up initial convergence (less noise when gradients are large anyway)
  • Maintain strong privacy guarantees at convergence (when gradients are small)

However, varying σ over time requires careful privacy accounting under
heterogeneous composition — the total ε is the sum over all rounds.

Provided schedules
------------------
1. LinearDecayEpochAllocator    – σ decreases linearly from σ_init to σ_final
2. CosineAnnealNoiseAllocator   – σ follows a cosine annealing schedule
3. ConvergenceAdaptiveAllocator – σ decreases when loss improvement slows
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from dp.adaptive.base import AdaptiveDPAllocator, NoiseParams

logger = logging.getLogger(__name__)


# ── Strategy 1: Linear decay ─────────────────────────────────────────────

class LinearDecayEpochAllocator(AdaptiveDPAllocator):
    """
    Linearly decrease noise_multiplier from σ_init to σ_final over
    `total_rounds` global communication rounds.

    σ(t) = σ_init - (σ_init - σ_final) * t / total_rounds

    Note on privacy accounting
    --------------------------
    The total privacy cost for T rounds with varying σ_t is computed via
    RDP composition:  ε_total = Σ_t ε(σ_t, q, 1 step).
    The PrivacyAccountant supports this via per-step `noise_multiplier`
    arguments to `.step()`.

    Parameters
    ----------
    sigma_init    : float  – noise multiplier at round 0
    sigma_final   : float  – noise multiplier at round `total_rounds - 1`
    clip_norm     : float
    total_rounds  : int    – T (total global communication rounds)
    """

    def __init__(
        self,
        sigma_init: float,
        sigma_final: float,
        clip_norm: float,
        total_rounds: int,
    ):
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.clip_norm = clip_norm
        self.total_rounds = max(total_rounds - 1, 1)

    def _sigma(self, round_num: int) -> float:
        t = min(round_num, self.total_rounds)
        return self.sigma_init - (self.sigma_init - self.sigma_final) * t / self.total_rounds

    def get_noise_params(
        self,
        client_id: int,
        round_num: int,
        epoch: int,
        layer_names: Optional[list] = None,
    ) -> NoiseParams:
        sigma = self._sigma(round_num)
        return NoiseParams(
            noise_multiplier=sigma,
            clip_norm=self.clip_norm,
            description=f"linear_decay r={round_num} σ={sigma:.3f}",
        )

    def summary(self) -> str:
        return (
            f"LinearDecayEpochAllocator("
            f"σ_init={self.sigma_init} → σ_final={self.sigma_final}, "
            f"T={self.total_rounds})"
        )


# ── Strategy 2: Cosine annealing ─────────────────────────────────────────

class CosineAnnealNoiseAllocator(AdaptiveDPAllocator):
    """
    Cosine annealing schedule for the noise multiplier.

    σ(t) = σ_final + 0.5 * (σ_init - σ_final) * (1 + cos(π * t / T))

    This mirrors the popular cosine LR schedule, giving a smooth, relatively
    slow decrease at the beginning and end, with faster decrease in the middle.

    Parameters
    ----------
    sigma_init    : float
    sigma_final   : float
    clip_norm     : float
    total_rounds  : int
    n_cycles      : int  – number of cosine cycles (default 1)
    """

    def __init__(
        self,
        sigma_init: float,
        sigma_final: float,
        clip_norm: float,
        total_rounds: int,
        n_cycles: int = 1,
    ):
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.clip_norm = clip_norm
        self.total_rounds = max(total_rounds, 1)
        self.n_cycles = n_cycles

    def _sigma(self, round_num: int) -> float:
        t = round_num % (self.total_rounds // self.n_cycles + 1)
        T = self.total_rounds / self.n_cycles
        return self.sigma_final + 0.5 * (self.sigma_init - self.sigma_final) * (
            1 + math.cos(math.pi * t / T)
        )

    def get_noise_params(
        self,
        client_id: int,
        round_num: int,
        epoch: int,
        layer_names: Optional[list] = None,
    ) -> NoiseParams:
        sigma = self._sigma(round_num)
        return NoiseParams(
            noise_multiplier=sigma,
            clip_norm=self.clip_norm,
            description=f"cosine r={round_num} σ={sigma:.4f}",
        )

    def summary(self) -> str:
        return (
            f"CosineAnnealNoiseAllocator("
            f"σ_init={self.sigma_init} → σ_final={self.sigma_final}, "
            f"T={self.total_rounds}, cycles={self.n_cycles})"
        )


# ── Strategy 3: Convergence-adaptive noise ───────────────────────────────

class ConvergenceAdaptiveAllocator(AdaptiveDPAllocator):
    """
    Reduce noise when training loss improvement plateaus (approximate convergence).

    σ is reduced by `decay_factor` whenever the loss improvement over the
    last `patience` rounds falls below `min_delta`.

    This is analogous to ReduceLROnPlateau for noise.

    Parameters
    ----------
    initial_noise_multiplier : float
    clip_norm                : float
    decay_factor             : float in (0, 1)  – σ ← σ * decay_factor on plateau
    patience                 : int   – number of rounds with small improvement
    min_delta                : float – minimum meaningful loss decrease
    min_noise_multiplier     : float – floor on σ
    """

    def __init__(
        self,
        initial_noise_multiplier: float,
        clip_norm: float,
        decay_factor: float = 0.8,
        patience: int = 3,
        min_delta: float = 1e-3,
        min_noise_multiplier: float = 0.1,
    ):
        self.clip_norm = clip_norm
        self.decay_factor = decay_factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_nm = min_noise_multiplier

        self._current_nm = initial_noise_multiplier
        self._loss_history: List[float] = []
        self._plateau_count = 0

    def get_noise_params(
        self,
        client_id: int,
        round_num: int,
        epoch: int,
        layer_names: Optional[list] = None,
    ) -> NoiseParams:
        return NoiseParams(
            noise_multiplier=self._current_nm,
            clip_norm=self.clip_norm,
            description=f"convergence_adaptive σ={self._current_nm:.4f}",
        )

    def update(self, metrics: Dict) -> None:
        """
        Expects metrics['global_loss'] (or 'train_loss'): float.
        """
        loss = metrics.get("global_loss") or metrics.get("train_loss")
        if loss is None:
            return
        self._loss_history.append(float(loss))

        if len(self._loss_history) < 2:
            return

        improvement = self._loss_history[-2] - self._loss_history[-1]
        if improvement < self.min_delta:
            self._plateau_count += 1
        else:
            self._plateau_count = 0

        if self._plateau_count >= self.patience:
            old = self._current_nm
            self._current_nm = max(self._current_nm * self.decay_factor, self.min_nm)
            self._plateau_count = 0
            logger.info(
                f"[ConvergenceAdaptive] Loss plateau detected: "
                f"σ {old:.4f} → {self._current_nm:.4f}"
            )

    def summary(self) -> str:
        return (
            f"ConvergenceAdaptiveAllocator("
            f"σ={self._current_nm:.4f}, "
            f"plateau_count={self._plateau_count})"
        )
