"""dp/ – Differential Privacy module for HierDP-FL."""
from dp.dp_config import DPConfig
from dp.accountant import PrivacyAccountant, GlobalPrivacyAccountant
from dp.dp_trainer import DPTrainer


def build_dp_allocator(dp_config: DPConfig, noise_multiplier: float):
    """
    Factory: create an AdaptiveDPAllocator matching dp_config.adaptive_mode.
    """
    from dp.adaptive.uniform import UniformDPAllocator
    from dp.adaptive.layer_wise import StaticLayerDPAllocator, GradNormAdaptiveLayerDPAllocator
    from dp.adaptive.client_wise import SensitivityScoreClientDPAllocator, GradMagnitudeClientDPAllocator
    from dp.adaptive.epoch_wise import LinearDecayEpochAllocator, CosineAnnealNoiseAllocator, ConvergenceAdaptiveAllocator

    mode = dp_config.adaptive_mode
    C = dp_config.max_grad_norm
    kw = dp_config.adaptive_kwargs

    if mode == "uniform":
        return UniformDPAllocator(noise_multiplier=noise_multiplier, clip_norm=C)

    elif mode == "layer_wise":
        strategy = kw.get("layer_strategy", "grad_norm")
        if strategy == "static":
            return StaticLayerDPAllocator(
                layer_clip_norms=kw.get("layer_clip_norms", {}),
                default_clip_norm=C,
                noise_multiplier=noise_multiplier,
            )
        else:  # grad_norm
            return GradNormAdaptiveLayerDPAllocator(
                noise_multiplier=noise_multiplier,
                init_clip_norm=C,
                percentile=kw.get("percentile", 50.0),
                warmup_rounds=kw.get("warmup_rounds", 2),
            )

    elif mode == "client_wise":
        strategy = kw.get("client_strategy", "grad_magnitude")
        if strategy == "sensitivity":
            return SensitivityScoreClientDPAllocator(
                base_noise_multiplier=noise_multiplier,
                clip_norm=C,
                sensitivity_scores=kw.get("sensitivity_scores", None),
            )
        else:  # grad_magnitude
            return GradMagnitudeClientDPAllocator(
                base_noise_multiplier=noise_multiplier,
                clip_norm=C,
                high_fraction=kw.get("high_fraction", 0.3),
                high_noise_factor=kw.get("high_noise_factor", 1.5),
            )

    elif mode == "epoch_wise":
        strategy = kw.get("epoch_strategy", "cosine")
        total_rounds = kw.get("total_rounds", 100)
        sigma_final = kw.get("sigma_final", noise_multiplier * 0.5)
        if strategy == "linear":
            return LinearDecayEpochAllocator(
                sigma_init=noise_multiplier, sigma_final=sigma_final,
                clip_norm=C, total_rounds=total_rounds,
            )
        elif strategy == "convergence":
            return ConvergenceAdaptiveAllocator(
                initial_noise_multiplier=noise_multiplier, clip_norm=C,
                decay_factor=kw.get("decay_factor", 0.8),
                patience=kw.get("patience", 3),
            )
        else:  # cosine
            return CosineAnnealNoiseAllocator(
                sigma_init=noise_multiplier, sigma_final=sigma_final,
                clip_norm=C, total_rounds=total_rounds,
            )

    else:
        raise ValueError(f"Unknown adaptive_mode: {mode}")


__all__ = [
    "DPConfig",
    "PrivacyAccountant",
    "GlobalPrivacyAccountant",
    "DPTrainer",
    "build_dp_allocator",
]
