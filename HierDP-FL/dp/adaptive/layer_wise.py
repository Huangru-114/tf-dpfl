"""
dp/adaptive/layer_wise.py
────────────────────────────────────────────────────────────────────────────
Layer-wise adaptive DP allocator.

Motivation
----------
Different layers in a neural network have different sensitivities: early
convolutional layers that process raw pixel data may require stronger privacy
protection (smaller clip norm → more aggressive clipping) than later,
higher-level feature layers.

Two layer-weighting strategies are provided:

  1. StaticLayerDPAllocator  – user-specified per-layer clip norms and a
     uniform noise_multiplier.  Use this when you have prior knowledge about
     which layers are sensitive.

  2. GradNormAdaptiveLayerDPAllocator – online adaptation: after each round
     the per-layer clip norms are set to a percentile of the observed gradient
     norms for that layer across clients.  This implements a form of
     "private gradient clipping at natural scale".

Extension hook
--------------
Both classes accept `noise_multiplier_fn` as a callable
    (layer_name: str, round_num: int) → float
so you can combine layer-wise clipping with round-based noise scheduling.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np

from dp.adaptive.base import AdaptiveDPAllocator, NoiseParams

logger = logging.getLogger(__name__)


# ── Strategy 1: Static per-layer clip norms ──────────────────────────────

class StaticLayerDPAllocator(AdaptiveDPAllocator):
    """
    Assign a fixed clip norm to each named layer.

    Parameters
    ----------
    layer_clip_norms : dict[str, float]
        Mapping from layer (variable) name to L2 clip norm.
        Layers not in the dict fall back to `default_clip_norm`.
    default_clip_norm : float
        Clip norm for layers not listed in `layer_clip_norms`.
    noise_multiplier : float or callable(layer_name, round_num) → float
        Gaussian noise multiplier.  Can be a constant or a callable for
        combined layer × round adaptation.

    Example
    -------
    >>> alloc = StaticLayerDPAllocator(
    ...     layer_clip_norms={
    ...         "conv1/kernel:0": 0.5,   # early layer, tight clip
    ...         "conv2/kernel:0": 0.8,
    ...         "dense/kernel:0": 1.5,   # top layer, relaxed clip
    ...     },
    ...     default_clip_norm=1.0,
    ...     noise_multiplier=1.1,
    ... )
    """

    def __init__(
        self,
        layer_clip_norms: Dict[str, float],
        default_clip_norm: float = 1.0,
        noise_multiplier: float | Callable = 1.0,
    ):
        self.layer_clip_norms = layer_clip_norms
        self.default_clip_norm = default_clip_norm
        self._nm = noise_multiplier

    def _get_noise_multiplier(self, layer_name: str, round_num: int) -> float:
        if callable(self._nm):
            return self._nm(layer_name, round_num)
        return float(self._nm)

    def get_noise_params(
        self,
        client_id: int,
        round_num: int,
        epoch: int,
        layer_names: Optional[list] = None,
    ) -> NoiseParams:
        # Build per-layer clip norms for all known layers
        resolved: Dict[str, float] = {}
        if layer_names:
            for ln in layer_names:
                resolved[ln] = self.layer_clip_norms.get(ln, self.default_clip_norm)

        # Use the maximum clip norm as the "global" clip for accounting
        max_clip = max(resolved.values()) if resolved else self.default_clip_norm
        nm = self._get_noise_multiplier("global", round_num)

        return NoiseParams(
            noise_multiplier=nm,
            clip_norm=max_clip,
            layer_clip_norms=resolved if resolved else None,
            description=(
                f"layer_static σ={nm:.3f} "
                f"layer_clips={list(resolved.values())[:4]}…"
            ),
        )

    def summary(self) -> str:
        return (
            f"StaticLayerDPAllocator("
            f"{len(self.layer_clip_norms)} custom layers, "
            f"default_clip={self.default_clip_norm})"
        )


# ── Strategy 2: Gradient-norm adaptive per-layer clip norms ──────────────

class GradNormAdaptiveLayerDPAllocator(AdaptiveDPAllocator):
    """
    Adapt per-layer clip norms online based on observed gradient norms.

    After each global round, the clip norm for layer ℓ is set to:
        C_ℓ ← percentile_p( {‖g_ℓ^(c)‖₂  for all clients c} )

    This ensures clipping discards approximately (100 - p)% of gradients.

    Parameters
    ----------
    noise_multiplier : float
        Fixed noise multiplier (σ) applied uniformly across layers.
    init_clip_norm   : float
        Clip norm used for all layers before the first round's stats arrive.
    percentile       : float in (0, 100)
        Percentile of per-client gradient norms used to set C_ℓ.
    min_clip_norm    : float
        Lower bound on C_ℓ (prevents degenerate clipping).
    max_clip_norm    : float
        Upper bound on C_ℓ.
    warmup_rounds    : int
        Use `init_clip_norm` for the first `warmup_rounds` rounds (while
        statistics accumulate).
    """

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        init_clip_norm: float = 1.0,
        percentile: float = 50.0,
        min_clip_norm: float = 0.1,
        max_clip_norm: float = 10.0,
        warmup_rounds: int = 2,
    ):
        self.noise_multiplier = noise_multiplier
        self.init_clip_norm = init_clip_norm
        self.percentile = percentile
        self.min_clip_norm = min_clip_norm
        self.max_clip_norm = max_clip_norm
        self.warmup_rounds = warmup_rounds

        # State updated by update()
        self._round = 0
        self._layer_clip_norms: Dict[str, float] = {}
        # Accumulate per-layer gradient norms: {layer_name: [norm_c0, norm_c1, ...]}
        self._layer_grad_norm_buffer: Dict[str, List[float]] = defaultdict(list)

    def get_noise_params(
        self,
        client_id: int,
        round_num: int,
        epoch: int,
        layer_names: Optional[list] = None,
    ) -> NoiseParams:
        if self._round < self.warmup_rounds or not self._layer_clip_norms:
            clip = self.init_clip_norm
            per_layer = None
            desc = f"grad_norm_adaptive [warmup] σ={self.noise_multiplier:.3f} C={clip:.3f}"
        else:
            clip = max(self._layer_clip_norms.values()) if self._layer_clip_norms else self.init_clip_norm
            per_layer = dict(self._layer_clip_norms)
            desc = (
                f"grad_norm_adaptive σ={self.noise_multiplier:.3f} "
                f"C_max={clip:.3f} p={self.percentile}"
            )
        return NoiseParams(
            noise_multiplier=self.noise_multiplier,
            clip_norm=clip,
            layer_clip_norms=per_layer,
            description=desc,
        )

    def update(self, metrics: Dict) -> None:
        """
        Called after each global round.

        Expects metrics to contain:
            'client_layer_grad_norms': {
                client_id: { layer_name: float (L2 norm), ... },
                ...
            }
        If this key is absent, the allocator stays in warmup mode.
        """
        self._round += 1
        layer_norms = metrics.get("client_layer_grad_norms", {})
        if not layer_norms:
            return

        # Aggregate per-layer norms across clients
        accumulator: Dict[str, List[float]] = defaultdict(list)
        for _cid, layer_dict in layer_norms.items():
            for lname, norm in layer_dict.items():
                accumulator[lname].append(float(norm))

        # Set clip norm per layer
        new_clips = {}
        for lname, norms in accumulator.items():
            raw = float(np.percentile(norms, self.percentile))
            new_clips[lname] = float(
                np.clip(raw, self.min_clip_norm, self.max_clip_norm)
            )
        self._layer_clip_norms = new_clips
        logger.debug(
            f"[GradNormAdaptiveLayer] Round {self._round}: "
            f"updated {len(new_clips)} layer clip norms."
        )

    # ── Utility: collect per-layer gradient norms from DPTrainer ─────────

    @staticmethod
    def collect_layer_grad_norms(
        model,
        x: "tf.Tensor",
        y: "tf.Tensor",
        loss_fn,
    ) -> Dict[str, float]:
        """
        Compute the L2 norm of per-layer gradients on a mini-batch.
        Call this at the end of a client's local round to feed update().
        """
        import tensorflow as tf
        with tf.GradientTape() as tape:
            preds = model(x, training=False)
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        layer_norms = {}
        for var, g in zip(model.trainable_variables, grads):
            if g is not None:
                layer_norms[var.name] = float(tf.norm(g).numpy())
        return layer_norms

    def summary(self) -> str:
        return (
            f"GradNormAdaptiveLayerDPAllocator("
            f"σ={self.noise_multiplier}, p={self.percentile}, "
            f"round={self._round})"
        )
