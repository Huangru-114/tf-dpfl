"""
client.py
────────────────────────────────────────────────────────────────────────────
FL client with DP-SGD local training.

Mirrors HierFL's Client class but:
  • Uses TensorFlow / Keras instead of PyTorch
  • Integrates DPTrainer for DP-SGD with adaptive noise allocation
  • Exposes layer-level gradient norm collection for adaptive strategies
  • Tracks cumulative privacy budget via PrivacyAccountant

Lifecycle (per communication round):
  1. receive_from_edgeserver() – update local model weights
  2. local_update()            – run τ₁ DP-SGD steps
  3. send_to_edgeserver()      – push updated weights to edge
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from dp.dp_trainer import DPTrainer
from dp.accountant import PrivacyAccountant

logger = logging.getLogger(__name__)


class Client:
    """
    Federated learning client with optional DP-SGD.

    Parameters
    ----------
    client_id        : int
    model            : tf.keras.Model  (shared global model, NOT compiled)
    train_dataset    : tf.data.Dataset
    test_dataset     : tf.data.Dataset
    args             : argparse.Namespace
    dp_config        : DPConfig | None
    dp_allocator     : AdaptiveDPAllocator | None
    num_train_samples: int
    """

    def __init__(
        self,
        client_id: int,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        args,
        dp_config=None,
        dp_allocator=None,
        num_train_samples: int = 0,
    ):
        self.id = client_id
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.dp_config = dp_config
        self.dp_allocator = dp_allocator
        self.num_train_samples = num_train_samples

        # ── Optimizer and loss ───────────────────────────────────────────
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        self._build_optimizer()

        # ── DP accounting ────────────────────────────────────────────────
        if dp_config is not None and dp_config.enabled:
            nm = dp_config.noise_multiplier or 1.0
            self.accountant = PrivacyAccountant(
                dp_config=dp_config,
                num_train_samples=max(num_train_samples, 1),
                batch_size=args.batch_size,
            )
            num_mb = (
                dp_config.num_microbatches
                or args.batch_size
            )
            self.dp_trainer = DPTrainer(
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self._reduced_loss,
                dp_config=dp_config,
                noise_multiplier=nm,
                clip_norm=dp_config.max_grad_norm,
                num_microbatches=num_mb,
            )
        else:
            self.accountant = None
            self.dp_trainer = DPTrainer(
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self._reduced_loss,
                dp_config=None,
            )

        # ── State ────────────────────────────────────────────────────────
        self.epoch = 0                      # local epoch counter (across rounds)
        self._receiver_buffer: Optional[List[np.ndarray]] = None
        self._last_grad_norm: float = 0.0
        self._last_layer_grad_norms: Dict[str, float] = {}

    # ── Loss helpers ─────────────────────────────────────────────────────

    def _reduced_loss(self, y_true, y_pred):
        """Mean reduction (for plain SGD train step)."""
        per_example = self.loss_fn(y_true, y_pred)
        return tf.reduce_mean(per_example)

    # ── Optimizer construction ────────────────────────────────────────────

    def _build_optimizer(self):
        lr = getattr(self.args, "lr", 0.01)
        momentum = getattr(self.args, "momentum", 0.0)
        wd = getattr(self.args, "weight_decay", 0.0)
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            weight_decay=wd if wd > 0 else None,
        )

    def _update_lr(self):
        """Exponential LR decay (mirrors HierFL)."""
        decay = getattr(self.args, "lr_decay", 1.0)
        decay_epoch = getattr(self.args, "lr_decay_epoch", 1)
        if decay < 1.0 and (self.epoch + 1) % decay_epoch == 0:
            old_lr = float(self.optimizer.learning_rate)
            self.optimizer.learning_rate.assign(old_lr * decay)

    # ── Core: local update ────────────────────────────────────────────────

    def local_update(
        self,
        num_iter: int,
        round_num: int = 0,
    ) -> float:
        """
        Run `num_iter` local SGD (or DP-SGD) steps.

        Parameters
        ----------
        num_iter : int   – number of gradient steps (τ₁ in HierFL)
        round_num: int   – global communication round (for adaptive DP)

        Returns
        -------
        float : mean batch loss over the local update
        """
        itered = 0
        total_loss = 0.0
        done = False

        layer_names = [v.name for v in self.model.trainable_variables]

        for _ in range(1000):   # epoch loop (same upper bound as HierFL)
            for x_batch, y_batch in self.train_dataset:
                # ── Get noise params from adaptive allocator ─────────────
                if self.dp_allocator is not None and self.dp_config is not None and self.dp_config.enabled:
                    noise_params = self.dp_allocator.get_noise_params(
                        client_id=self.id,
                        round_num=round_num,
                        epoch=self.epoch,
                        layer_names=layer_names,
                    )
                    self.dp_trainer.set_dp_params(
                        noise_multiplier=noise_params.noise_multiplier,
                        clip_norm=noise_params.clip_norm,
                    )
                    if noise_params.layer_clip_norms:
                        self.dp_trainer.set_layer_clip_norms(noise_params.layer_clip_norms)
                    # Update accountant with current noise multiplier
                    if self.accountant:
                        self.accountant.step(
                            noise_multiplier=noise_params.noise_multiplier,
                            n_steps=1,
                        )

                # ── DP-SGD or plain SGD step ──────────────────────────────
                step_info = self.dp_trainer.train_step(x_batch, y_batch)
                total_loss += step_info["loss"]
                self._last_grad_norm = step_info.get("grad_norm", 0.0)

                itered += 1
                if itered >= num_iter:
                    done = True
                    self.epoch += 1
                    self._update_lr()
                    break

            if done:
                break
            self.epoch += 1
            self._update_lr()

        return total_loss / max(itered, 1)

    # ── Model communication ───────────────────────────────────────────────

    def send_to_edgeserver(self, edge_server) -> None:
        """Push current model weights to the assigned edge server."""
        weights = [w.copy() for w in self.model.get_weights()]
        edge_server.receive_from_client(
            client_id=self.id,
            client_weights=weights,
        )

    def receive_from_edgeserver(self, weights: List[np.ndarray]) -> None:
        """Buffer incoming aggregated weights from edge server."""
        self._receiver_buffer = weights

    def sync_with_edgeserver(self) -> None:
        """Apply buffered weights to local model."""
        if self._receiver_buffer is not None:
            self.model.set_weights(self._receiver_buffer)
            self._receiver_buffer = None

    # ── Evaluation ───────────────────────────────────────────────────────

    def test_model(self) -> Tuple[float, float]:
        """
        Returns
        -------
        (correct, total)
        """
        correct = 0.0
        total = 0.0
        for x_batch, y_batch in self.test_dataset:
            logits = self.model(x_batch, training=False)
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            y_batch = tf.cast(y_batch, tf.int32)
            correct += float(tf.reduce_sum(tf.cast(preds == y_batch, tf.float32)).numpy())
            total += float(y_batch.shape[0])
        return correct, total

    # ── Privacy accounting ────────────────────────────────────────────────

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Return (ε, δ) for this client's accumulated gradient steps."""
        if self.accountant:
            return self.accountant.get_privacy_spent()
        return 0.0, 0.0

    # ── Gradient norm collection (for adaptive strategies) ─────────────────

    def collect_grad_norms(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute gradient norms on one mini-batch (for adaptive DP allocators).

        Returns
        -------
        global_norm   : float
        layer_norms   : dict[str, float]
        """
        try:
            x_batch, y_batch = next(iter(self.train_dataset))
            with tf.GradientTape() as tape:
                preds = self.model(x_batch, training=False)
                loss = self._reduced_loss(y_batch, preds)
            grads = tape.gradient(loss, self.model.trainable_variables)

            layer_norms = {}
            flat_grads = []
            for var, g in zip(self.model.trainable_variables, grads):
                if g is not None:
                    n = float(tf.norm(g).numpy())
                    layer_norms[var.name] = n
                    flat_grads.append(tf.reshape(g, [-1]))

            global_norm = float(tf.norm(tf.concat(flat_grads, 0)).numpy()) if flat_grads else 0.0
            self._last_layer_grad_norms = layer_norms
            return global_norm, layer_norms
        except Exception as e:
            logger.warning(f"Client {self.id}: collect_grad_norms failed: {e}")
            return 0.0, {}
