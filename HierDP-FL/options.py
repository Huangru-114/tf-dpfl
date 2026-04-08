"""
options.py
────────────────────────────────────────────────────────────────────────────
CLI argument parser for HierDP-FL.

Extends HierFL's original options with:
  • --dp_*  arguments for differential-privacy configuration
  • --dirichlet_alpha for Dirichlet non-IID partitioning
  • GTSRB dataset support
"""

import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description="HierDP-FL: Hierarchical Federated Learning with Adaptive DP-SGD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset and model ─────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "cifar100", "gtsrb"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--model", type=str, default="cnn_complex",
        choices=["cnn_complex", "cnn", "resnet18", "convnet", "mobilenet"],
        help="Model architecture.",
    )
    parser.add_argument(
        "--dataset_root", type=str, default="/mimer/NOBACKUP/Datasets/CIFAR",
        help="Root directory for dataset storage.",
    )

    # ── Training hyper-parameters ─────────────────────────────────────────
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Local mini-batch size.",
    )
    parser.add_argument(
        "--num_communication", type=int, default=100,
        help="Number of global communication rounds (T).",
    )
    parser.add_argument(
        "--num_local_update", type=int, default=5,
        help="Number of local SGD steps per client per round (τ₁).",
    )
    parser.add_argument(
        "--num_edge_aggregation", type=int, default=1,
        help="Number of edge aggregation steps per global round (τ₂).",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Initial client learning rate.",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=1.0,
        help="Multiplicative LR decay factor (1.0 = no decay).",
    )
    parser.add_argument(
        "--lr_decay_epoch", type=int, default=1,
        help="Apply LR decay every N local epochs.",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9,
        help="SGD momentum.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="L2 weight decay.",
    )

    # ── Federated learning structure ──────────────────────────────────────
    parser.add_argument(
        "--num_clients", type=int, default=20,
        help="Total number of FL clients.",
    )
    parser.add_argument(
        "--num_edges", type=int, default=4,
        help="Number of edge servers.",
    )
    parser.add_argument(
        "--frac", type=float, default=1.0,
        help="Fraction of clients selected per edge per round.",
    )
    parser.add_argument(
        "--iid", type=int, default=0,
        help="Data partition: 1=IID, 0=shard, -1=Dirichlet.",
    )
    parser.add_argument(
        "--dirichlet_alpha", type=float, default=0.5,
        help="Dirichlet α for non-IID partition (used when --iid=-1).",
    )
    parser.add_argument(
        "--classes_per_client", type=int, default=2,
        help="Classes per client in shard partition (used when --iid=0).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index to use (0-indexed).",
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        choices=[0, 1, 2],
        help="Verbosity level.",
    )

    # ── Differential privacy ──────────────────────────────────────────────
    dp_group = parser.add_argument_group("Differential Privacy")

    dp_group.add_argument(
        "--dp_enabled", action="store_true", default=False,
        help="Enable DP-SGD (disable for non-DP baseline).",
    )
    dp_group.add_argument(
        "--dp_epsilon", type=float, default=10.0,
        help="Target privacy budget ε (used for auto-calibration and accounting).",
    )
    dp_group.add_argument(
        "--dp_delta", type=float, default=1e-5,
        help="Privacy parameter δ (typically 1/N).",
    )
    dp_group.add_argument(
        "--dp_noise_multiplier", type=float, default=0.0,
        help=(
            "Gaussian noise multiplier σ. "
            "Set to 0 to auto-calibrate from --dp_epsilon and --dp_delta."
        ),
    )
    dp_group.add_argument(
        "--dp_max_grad_norm", type=float, default=1.0,
        help="Global gradient L2 clipping norm C.",
    )
    dp_group.add_argument(
        "--dp_num_microbatches", type=int, default=0,
        help=(
            "Number of microbatches per batch. "
            "0 = auto (set to batch_size → per-sample DP). "
            "Values < batch_size trade accuracy for speed."
        ),
    )
    dp_group.add_argument(
        "--dp_adaptive_mode", type=str, default="uniform",
        choices=["uniform", "layer_wise", "client_wise", "epoch_wise"],
        help=(
            "Adaptive DP noise allocation strategy:\n"
            "  uniform     – standard DP-SGD baseline\n"
            "  layer_wise  – adaptive per-layer clipping norms\n"
            "  client_wise – per-client adaptive noise\n"
            "  epoch_wise  – noise schedule over training rounds"
        ),
    )
    dp_group.add_argument(
        "--dp_accountant", type=str, default="rdp",
        choices=["rdp", "prv"],
        help="Privacy accountant type.",
    )

    args = parser.parse_args()

    # ── Derived defaults ──────────────────────────────────────────────────
    if args.dp_num_microbatches == 0:
        args.dp_num_microbatches = args.batch_size  # true per-sample DP

    return args
