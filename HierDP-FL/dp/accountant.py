"""
dp/accountant.py
────────────────────────────────────────────────────────────────────────────
Privacy accountant for hierarchical federated learning with DP-SGD.

Uses the Rényi Differential Privacy (RDP) accountant from TF-Privacy to:
  • Compute the privacy cost accumulated by a single client across local steps.
  • Compose costs over global communication rounds.
  • Provide a noise_multiplier → (ε, δ) query and the inverse calibration.

Usage:
    acc = PrivacyAccountant(dp_config, num_train_samples=5000, batch_size=32)
    # After each local training step:
    acc.step()
    eps, delta = acc.get_privacy_spent()
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Try to import TF-Privacy accountant ──────────────────────────────────
try:
    from tensorflow_privacy.privacy.analysis.rdp_accountant import (
        compute_rdp,
        get_privacy_spent,
    )
    _TFP_AVAILABLE = True
except ImportError:
    _TFP_AVAILABLE = False
    logger.warning(
        "tensorflow_privacy not found; falling back to a pure-numpy RDP "
        "accountant (slightly less precise for very small orders)."
    )


# ── Pure-NumPy fallback RDP accountant ───────────────────────────────────

def _compute_rdp_numpy(q: float, noise_multiplier: float, steps: int, orders) -> np.ndarray:
    """
    Vectorised RDP bound for the Gaussian mechanism (subsampled via Poisson).
    Implements Proposition 3 from Mironov et al. (2017) with the subsampling
    amplification from Zhu & Wang (2019).
    """
    orders = np.array(orders, dtype=float)
    rdp = np.zeros_like(orders)
    if q == 0:
        return rdp
    for i, alpha in enumerate(orders):
        if np.isinf(alpha):
            rdp[i] = np.inf
            continue
        # Gaussian mechanism RDP: α / (2 σ²)
        # With subsampling: tight bound (approximate)
        log_term = (alpha - 1) * np.log(
            (1 - q) + q * np.exp((alpha) / (2 * noise_multiplier ** 2))
        )
        rdp[i] = steps * log_term / (alpha - 1) if alpha > 1 else 0.0
    return rdp


def _get_privacy_spent_numpy(orders, rdp, delta: float) -> Tuple[float, float]:
    """Convert RDP to (ε, δ) using the standard conversion formula."""
    orders = np.array(orders, dtype=float)
    eps = rdp - (np.log(delta) + np.log(orders)) / (orders - 1) + np.log((orders - 1) / orders)
    idx = np.nanargmin(eps)
    return float(eps[idx]), float(orders[idx])


# ── Main Accountant class ────────────────────────────────────────────────

class PrivacyAccountant:
    """
    Tracks privacy consumption for a single client.

    Parameters
    ----------
    dp_config       : DPConfig
    num_train_samples : int
        Number of training samples on this client (used to compute sampling rate q).
    batch_size      : int
        Batch size used during local SGD.
    """

    def __init__(self, dp_config, num_train_samples: int, batch_size: int):
        from dp.dp_config import DPConfig  # avoid circular import at module level
        assert isinstance(dp_config, DPConfig)
        self.cfg = dp_config
        self.n = num_train_samples
        self.batch_size = batch_size
        self.q = batch_size / num_train_samples          # Poisson sampling rate
        self.orders = list(dp_config.rdp_orders)
        self._steps = 0                                  # cumulative gradient steps
        self._noise_multiplier = dp_config.noise_multiplier or 1.0  # updated if adaptive

    # ── Public API ───────────────────────────────────────────────────────

    def step(self, noise_multiplier: Optional[float] = None, n_steps: int = 1) -> None:
        """Record that `n_steps` DP-SGD steps were taken with given noise_multiplier."""
        if noise_multiplier is not None:
            self._noise_multiplier = noise_multiplier
        self._steps += n_steps

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Return (ε, δ) for the current accumulated steps."""
        if self._steps == 0:
            return 0.0, 0.0
        if _TFP_AVAILABLE:
            rdp = compute_rdp(
                q=self.q,
                noise_multiplier=self._noise_multiplier,
                steps=self._steps,
                orders=self.orders,
            )
            eps, _ = get_privacy_spent(self.orders, rdp, target_delta=self.cfg.delta)
        else:
            rdp = _compute_rdp_numpy(
                self.q, self._noise_multiplier, self._steps, self.orders
            )
            eps, _ = _get_privacy_spent_numpy(self.orders, rdp, self.cfg.delta)
        return eps, self.cfg.delta

    def reset(self) -> None:
        """Reset the step counter (e.g. between global communication rounds)."""
        self._steps = 0

    @property
    def total_steps(self) -> int:
        return self._steps

    # ── Calibration utility ──────────────────────────────────────────────

    @staticmethod
    def calibrate_noise_multiplier(
        target_epsilon: float,
        delta: float,
        num_train_samples: int,
        batch_size: int,
        total_steps: int,
        orders=None,
        tol: float = 1e-3,
    ) -> float:
        """
        Binary-search for the smallest noise_multiplier σ such that running
        `total_steps` DP-SGD steps yields (ε ≤ target_epsilon, δ).

        Returns
        -------
        float : noise_multiplier
        """
        if orders is None:
            orders = list(range(2, 64)) + [128, 256, 512]
        q = batch_size / num_train_samples

        lo, hi = 0.01, 1000.0
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if _TFP_AVAILABLE:
                rdp = compute_rdp(q=q, noise_multiplier=mid, steps=total_steps, orders=orders)
                eps, _ = get_privacy_spent(orders, rdp, target_delta=delta)
            else:
                rdp = _compute_rdp_numpy(q, mid, total_steps, orders)
                eps, _ = _get_privacy_spent_numpy(orders, rdp, delta)
            if eps <= target_epsilon:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        logger.info(
            f"Calibrated noise_multiplier={hi:.4f} for "
            f"ε={target_epsilon}, δ={delta}, steps={total_steps}"
        )
        return hi


# ── Global / per-experiment accountant ───────────────────────────────────

class GlobalPrivacyAccountant:
    """
    Aggregates privacy consumption across all clients and rounds.
    Uses basic composition (worst-case client).
    """

    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self._client_accountants: dict = {}

    def register(self, client_id: int, accountant: PrivacyAccountant) -> None:
        self._client_accountants[client_id] = accountant

    def get_global_epsilon(self) -> float:
        """Worst-case ε across all clients (basic composition)."""
        if not self._client_accountants:
            return 0.0
        eps_vals = []
        for acc in self._client_accountants.values():
            eps, _ = acc.get_privacy_spent()
            eps_vals.append(eps)
        return max(eps_vals)

    def report(self) -> dict:
        summary = {}
        for cid, acc in self._client_accountants.items():
            eps, delta = acc.get_privacy_spent()
            summary[cid] = {"epsilon": eps, "delta": delta, "steps": acc.total_steps}
        summary["worst_case_epsilon"] = self.get_global_epsilon()
        return summary
