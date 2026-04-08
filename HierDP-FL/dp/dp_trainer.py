"""
dp/dp_trainer.py
────────────────────────────────────────────────────────────────────────────
Core DP training engine.

Design
------
DPTrainer wraps a Keras model and provides a `train_step()` method that:

  1. Computes per-sample gradients via `tf.vectorized_map` (fast) or
     a loop fallback (compatible with all layer types).
  2. Clips each per-sample gradient vector to `clip_norm` (global or per-layer).
  3. Aggregates clipped gradients.
  4. Adds calibrated Gaussian noise:  g_noisy = g_clip + N(0, (σ * C)² I).
  5. Applies the noisy gradient with the wrapped optimiser.

The interface is deliberately adapter-friendly:
  • `DPTrainer.train_step(x, y)` returns {"loss": ..., "gradients_norm": ...}
  • `DPTrainer.set_dp_params(noise_multiplier, clip_norm)` lets adaptive
    allocators inject fresh parameters before each batch.

Non-DP fallback
---------------
When DPConfig.enabled == False, DPTrainer degrades to a standard Keras
`model.train_on_batch` call with zero overhead.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class DPTrainer:
    """
    DP-SGD trainer for a Keras model.

    Parameters
    ----------
    model           : tf.keras.Model  (already compiled with loss / metrics)
    optimizer       : tf.keras.optimizers.Optimizer
    loss_fn         : callable (y_true, y_pred) → scalar loss
    dp_config       : DPConfig | None
    noise_multiplier: float   – override from dp_config if provided
    clip_norm       : float   – global L2 clip bound C
    num_microbatches: int     – number of per-sample groups (== batch_size → per-sample DP)
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn,
        dp_config=None,
        noise_multiplier: Optional[float] = None,
        clip_norm: Optional[float] = None,
        num_microbatches: Optional[int] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dp_config = dp_config

        # ── DP parameters (can be updated dynamically for adaptive DP) ──
        if dp_config is not None and dp_config.enabled:
            self.noise_multiplier = noise_multiplier or dp_config.noise_multiplier or 1.0
            self.clip_norm = clip_norm or dp_config.max_grad_norm
            self._num_microbatches = num_microbatches or dp_config.num_microbatches
            self._dp_enabled = True
        else:
            self.noise_multiplier = 0.0
            self.clip_norm = 0.0
            self._num_microbatches = None
            self._dp_enabled = False

    # ── Dynamic parameter update (for adaptive allocators) ───────────────

    def set_dp_params(
        self,
        noise_multiplier: Optional[float] = None,
        clip_norm: Optional[float] = None,
    ) -> None:
        """Update DP noise/clip parameters mid-training (adaptive DP hook)."""
        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        if clip_norm is not None:
            self.clip_norm = clip_norm

    def set_layer_clip_norms(self, layer_clip_norms: Dict[str, float]) -> None:
        """
        Set per-layer clipping norms for layer-wise adaptive DP.
        Keys are layer names (model.layers[i].name).
        """
        self._layer_clip_norms = layer_clip_norms

    # ── Train step ───────────────────────────────────────────────────────

    def train_step(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one DP-SGD (or standard SGD) mini-batch update.

        Returns
        -------
        dict with keys: loss, grad_norm
        """
        if not self._dp_enabled:
            return self._plain_train_step(x, y)
        return self._dp_train_step(x, y)

    # ── Internal: plain SGD (DP disabled) ────────────────────────────────

    def _plain_train_step(self, x, y) -> Dict[str, float]:
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        grad_norm = tf.linalg.global_norm(grads).numpy()
        return {"loss": float(loss.numpy()), "grad_norm": float(grad_norm)}

    # ── Internal: DP-SGD with per-sample gradient clipping ───────────────

    def _dp_train_step(self, x, y) -> Dict[str, float]:
        """
        True per-sample DP-SGD via microbatching.

        Each microbatch has size `ceil(batch_size / num_microbatches)`.
        The gradients from each microbatch are clipped independently (simulating
        per-sample clipping when num_microbatches == batch_size).
        """
        batch_size = tf.shape(x)[0]
        num_mb = self._num_microbatches if self._num_microbatches else batch_size

        # Partition batch into microbatches
        mb_losses = []
        summed_clipped_grads: Optional[List[tf.Tensor]] = None

        indices = tf.range(batch_size)
        mb_size = tf.cast(tf.math.ceil(tf.cast(batch_size, tf.float32) / tf.cast(num_mb, tf.float32)), tf.int32)

        for mb_idx in range(num_mb):
            start = mb_idx * mb_size
            end = tf.minimum(start + mb_size, batch_size)
            if start >= batch_size:
                break

            x_mb = x[start:end]
            y_mb = y[start:end]

            with tf.GradientTape() as tape:
                preds = self.model(x_mb, training=True)
                loss_mb = self.loss_fn(y_mb, preds)
                mb_losses.append(loss_mb)

            grads = tape.gradient(loss_mb, self.model.trainable_variables)

            # Per-layer or global clipping
            clipped = self._clip_gradients(grads)

            if summed_clipped_grads is None:
                summed_clipped_grads = clipped
            else:
                summed_clipped_grads = [
                    sg + cg for sg, cg in zip(summed_clipped_grads, clipped)
                ]

        # Average clipped gradients over microbatches
        n_mb_actual = min(num_mb, batch_size)
        avg_grads = [g / tf.cast(n_mb_actual, tf.float32) for g in summed_clipped_grads]

        # Add Gaussian noise:  noise_std = noise_multiplier * clip_norm / batch_size
        # (Division by batch_size because we average, not sum, the gradients)
        noisy_grads = self._add_noise(avg_grads, batch_size=float(n_mb_actual))

        self.optimizer.apply_gradients(zip(noisy_grads, self.model.trainable_variables))

        mean_loss = float(tf.reduce_mean(mb_losses).numpy())
        grad_norm = float(tf.linalg.global_norm(avg_grads).numpy())
        return {"loss": mean_loss, "grad_norm": grad_norm}

    # ── Gradient clipping helpers ─────────────────────────────────────────

    def _clip_gradients(self, grads: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Clip gradient list.

        If `_layer_clip_norms` is set (layer-wise adaptive), clips each
        variable's gradient to the corresponding per-layer norm.
        Otherwise performs global L2 clipping.
        """
        if hasattr(self, "_layer_clip_norms") and self._layer_clip_norms:
            return self._clip_per_layer(grads)
        return self._clip_global(grads)

    def _clip_global(self, grads: List[tf.Tensor]) -> List[tf.Tensor]:
        """Global L2 clipping: scale down g if ‖g‖₂ > C."""
        # Filter None gradients (e.g. frozen layers)
        flat = [g for g in grads if g is not None]
        if not flat:
            return grads
        global_norm = tf.linalg.global_norm(flat)
        scale = tf.minimum(1.0, self.clip_norm / (global_norm + 1e-8))
        return [
            (g * scale if g is not None else None) for g in grads
        ]

    def _clip_per_layer(self, grads: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Per-layer L2 clipping.
        Each variable is clipped to its own norm from `_layer_clip_norms`.
        Falls back to global clip_norm for layers not in the dict.
        """
        clipped = []
        for var, g in zip(self.model.trainable_variables, grads):
            if g is None:
                clipped.append(None)
                continue
            c = self._layer_clip_norms.get(var.name, self.clip_norm)
            layer_norm = tf.norm(g)
            scale = tf.minimum(1.0, c / (layer_norm + 1e-8))
            clipped.append(g * scale)
        return clipped

    # ── Noise injection ───────────────────────────────────────────────────

    def _add_noise(
        self,
        grads: List[tf.Tensor],
        batch_size: float,
    ) -> List[tf.Tensor]:
        """
        Add calibrated Gaussian noise to each gradient tensor.

        Noise std per coordinate = (noise_multiplier * clip_norm) / batch_size
        This is the standard DP-SGD noise calibration.
        """
        noise_std = self.noise_multiplier * self.clip_norm / batch_size
        noisy = []
        for g in grads:
            if g is None:
                noisy.append(None)
                continue
            noise = tf.random.normal(
                shape=tf.shape(g),
                mean=0.0,
                stddev=noise_std,
                dtype=g.dtype,
            )
            noisy.append(g + noise)
        return noisy


# ── TF-Privacy native optimizer factory (alternative approach) ────────────

def make_dp_optimizer(
    optimizer_class,
    noise_multiplier: float,
    l2_norm_clip: float,
    num_microbatches: int,
    **optimizer_kwargs,
):
    """
    Wrap a standard Keras optimizer class with TF-Privacy DP guarantees.

    This uses `tensorflow_privacy.DPKerasSGDOptimizer` (or equivalent) when
    available; otherwise falls back to the plain optimizer.

    Parameters
    ----------
    optimizer_class  : e.g. tf.keras.optimizers.SGD
    noise_multiplier : σ
    l2_norm_clip     : C (gradient clipping bound)
    num_microbatches : B_micro
    **optimizer_kwargs: forwarded to optimizer constructor (lr, momentum, ...)

    Returns
    -------
    optimizer instance
    """
    try:
        import tensorflow_privacy as tfp
        DPClass = tfp.DPKerasSGDOptimizer  # or use make_keras_optimizer_class
        logger.info("Using TF-Privacy native DPKerasSGDOptimizer.")
        return DPClass(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            **optimizer_kwargs,
        )
    except (ImportError, AttributeError):
        logger.warning(
            "TF-Privacy DPKerasSGDOptimizer not available; "
            "DPTrainer._dp_train_step will be used instead."
        )
        return optimizer_class(**optimizer_kwargs)
