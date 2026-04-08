"""
dp/dp_config.py
────────────────────────────────────────────────────────────────────────────
Central configuration object for differential-privacy hyper-parameters.

All DP-aware modules (optimizer factory, accountant, adaptive allocators)
accept a DPConfig so there is one canonical place to control privacy settings.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DPConfig:
    # ── Master switch ────────────────────────────────────────────────────
    enabled: bool = True

    # ── (ε, δ) privacy budget ────────────────────────────────────────────
    target_epsilon: float = 10.0      # desired ε at end of training
    delta: float = 1e-5               # δ (usually 1/N where N = dataset size)

    # ── Per-step noise parameters ────────────────────────────────────────
    # If noise_multiplier is given it is used directly;
    # otherwise it is calibrated to hit (target_epsilon, delta) at the end
    # of training via the RDP accountant (see dp/accountant.py).
    noise_multiplier: Optional[float] = None   # σ / C  (set None → auto-calibrate)
    max_grad_norm: float = 1.0                  # global L2 clipping bound C

    # ── Micro-batch granularity ──────────────────────────────────────────
    # num_microbatches == batch_size  ↔  true per-sample DP-SGD
    # num_microbatches < batch_size   ↔  mini-batch approximation (faster)
    num_microbatches: Optional[int] = None      # None → set to batch_size

    # ── Adaptive noise-allocation strategy ──────────────────────────────
    # 'uniform'     : standard DP-SGD baseline (same σ / C for all steps)
    # 'layer_wise'  : per-layer adaptive clipping norms
    # 'client_wise' : per-client adaptive noise based on sensitivity score
    # 'epoch_wise'  : noise schedule that decays/grows over training epochs
    # 'hierarchical': combines layer + epoch dimensions
    adaptive_mode: str = "uniform"

    # ── Adaptive strategy hyper-parameters (mode-specific) ──────────────
    # These are forwarded to the corresponding AdaptiveDPAllocator subclass.
    adaptive_kwargs: dict = field(default_factory=dict)

    # ── Accounting ───────────────────────────────────────────────────────
    # Privacy accountant type: 'rdp' (Rényi DP) | 'prv' (f-DP / PRV)
    accountant_type: str = "rdp"
    rdp_orders: tuple = field(
        default_factory=lambda: tuple(range(2, 64)) + (128, 256, 512)
    )

    def __post_init__(self):
        valid_modes = {"uniform", "layer_wise", "client_wise", "epoch_wise", "hierarchical"}
        if self.adaptive_mode not in valid_modes:
            raise ValueError(f"adaptive_mode must be one of {valid_modes}, got '{self.adaptive_mode}'")
        valid_accountants = {"rdp", "prv"}
        if self.accountant_type not in valid_accountants:
            raise ValueError(f"accountant_type must be one of {valid_accountants}")

    def summary(self) -> str:
        lines = [
            "── DPConfig ─────────────────────────────────────",
            f"  enabled          : {self.enabled}",
            f"  target_epsilon   : {self.target_epsilon}",
            f"  delta            : {self.delta}",
            f"  noise_multiplier : {self.noise_multiplier or 'auto-calibrate'}",
            f"  max_grad_norm    : {self.max_grad_norm}",
            f"  adaptive_mode    : {self.adaptive_mode}",
            f"  accountant_type  : {self.accountant_type}",
            "─────────────────────────────────────────────────",
        ]
        return "\n".join(lines)
