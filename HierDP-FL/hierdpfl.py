"""
hierdpfl.py
────────────────────────────────────────────────────────────────────────────
Main training loop for HierDP-FL.

Algorithm (mirrors HierFAVG with DP):
  For each global round t = 1 … T:
    For each edge aggregation step τ₂:
      For each edge e:
        Broadcast edge model → selected clients
        For each selected client c:
          c.local_update(τ₁ steps, DP-SGD)   ← PRIVACY COST ACCUMULATED HERE
          c → edge: push updated weights
        Edge FedAvg aggregation
    Cloud FedAvg aggregation (edge → cloud)
    Log privacy budget, accuracy, loss

Usage:
    python hierdpfl.py [options]
    See options.py for full argument list.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from options import args_parser
from client import Client
from edge import Edge, Cloud
from models import build_model
from datasets import get_dataloaders
from dp import build_dp_allocator
from dp.dp_config import DPConfig
from dp.accountant import PrivacyAccountant, GlobalPrivacyAccountant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("HierDPFL")


# ── Utilities ─────────────────────────────────────────────────────────────

def set_global_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def fast_global_eval(global_model, v_test_dataset) -> float:
    """Evaluate global model on the held-out validation set."""
    correct = 0.0
    total = 0.0
    for x_batch, y_batch in v_test_dataset:
        logits = global_model(x_batch, training=False)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        correct += float(tf.reduce_sum(tf.cast(preds == tf.cast(y_batch, tf.int32), tf.float32)))
        total += float(y_batch.shape[0])
    return correct / total if total > 0 else 0.0


def assign_clients_to_edges(
    clients,
    num_edges: int,
    seed: int,
) -> list:
    """
    Randomly partition clients into edges (uniform).
    Returns list of Edge objects.
    """
    rng = np.random.default_rng(seed)
    all_cids = np.arange(len(clients))
    rng.shuffle(all_cids)
    splits = np.array_split(all_cids, num_edges)

    init_weights = clients[0].model.get_weights()
    edges = []
    for eid, split in enumerate(splits):
        edge = Edge(edge_id=eid, cids=split.tolist(), init_weights=init_weights)
        for cid in split:
            edge.register_client(clients[cid])
        edge.all_train_sample_num = edge.total_samples
        edge.refresh()
        edges.append(edge)
    return edges


def collect_round_metrics(
    clients,
    selected_cids_per_edge,
    edge_losses,
    edge_sample_counts,
    round_num: int,
    global_acc: float,
    dp_enabled: bool,
) -> dict:
    """Build the metrics dict passed to adaptive allocators after each round."""
    metrics = {
        "round": round_num,
        "global_accuracy": global_acc,
        "global_loss": (
            sum(l * s for l, s in zip(edge_losses, edge_sample_counts))
            / max(sum(edge_sample_counts), 1)
        ),
        "edge_losses": edge_losses,
    }
    if dp_enabled:
        all_selected = [cid for cids in selected_cids_per_edge for cid in cids]
        metrics["client_grad_norms"] = {
            cid: clients[cid]._last_grad_norm for cid in all_selected
        }
        metrics["client_layer_grad_norms"] = {
            cid: clients[cid]._last_layer_grad_norms for cid in all_selected
        }
        metrics["privacy_spent"] = {
            cid: clients[cid].get_privacy_spent()[0] for cid in all_selected
        }
    return metrics


# ── Main training loop ────────────────────────────────────────────────────

def HierDPFL(args):
    set_global_seed(args.seed)

    # ── GPU config ────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[args.gpu % len(gpus)], True)
            logger.info(f"Using GPU: {gpus[args.gpu % len(gpus)]}")
        except RuntimeError as e:
            logger.warning(f"GPU config error: {e}")
    else:
        logger.info("No GPU found, running on CPU.")

    # ── DP configuration ──────────────────────────────────────────────────
    dp_config = None
    if args.dp_enabled:
        adaptive_kwargs = {}
        if args.dp_adaptive_mode in ("epoch_wise",):
            adaptive_kwargs["total_rounds"] = args.num_communication
            adaptive_kwargs["sigma_final"] = args.dp_noise_multiplier * 0.5
            adaptive_kwargs["epoch_strategy"] = "cosine"
        elif args.dp_adaptive_mode == "layer_wise":
            adaptive_kwargs["layer_strategy"] = "grad_norm"
            adaptive_kwargs["warmup_rounds"] = 2

        dp_config = DPConfig(
            enabled=True,
            target_epsilon=args.dp_epsilon,
            delta=args.dp_delta,
            noise_multiplier=args.dp_noise_multiplier if args.dp_noise_multiplier > 0 else None,
            max_grad_norm=args.dp_max_grad_norm,
            num_microbatches=args.dp_num_microbatches if args.dp_num_microbatches > 0 else None,
            adaptive_mode=args.dp_adaptive_mode,
            adaptive_kwargs=adaptive_kwargs,
        )
        logger.info(dp_config.summary())

    # ── Datasets ──────────────────────────────────────────────────────────
    train_datasets, test_datasets, v_train_ds, v_test_ds, num_train_samples = (
        get_dataloaders(args)
    )

    # ── Auto-calibrate noise_multiplier if not specified ──────────────────
    noise_multiplier = args.dp_noise_multiplier if args.dp_enabled else 0.0
    if args.dp_enabled and args.dp_noise_multiplier <= 0:
        # Total steps = num_communication × num_edge_aggregation × num_local_update
        total_steps = (
            args.num_communication * args.num_edge_aggregation * args.num_local_update
        )
        avg_samples = int(np.mean(num_train_samples))
        noise_multiplier = PrivacyAccountant.calibrate_noise_multiplier(
            target_epsilon=args.dp_epsilon,
            delta=args.dp_delta,
            num_train_samples=avg_samples,
            batch_size=args.batch_size,
            total_steps=total_steps,
        )
        dp_config.noise_multiplier = noise_multiplier
        logger.info(f"Auto-calibrated noise_multiplier = {noise_multiplier:.4f}")

    # ── Build adaptive allocator ──────────────────────────────────────────
    dp_allocator = None
    if args.dp_enabled:
        dp_allocator = build_dp_allocator(dp_config, noise_multiplier=noise_multiplier)
        logger.info(f"DP Allocator: {dp_allocator.summary()}")

    # ── Initialise clients ────────────────────────────────────────────────
    logger.info("Initialising clients…")
    # Build a reference model first and get its initial weights
    ref_model = build_model(args)
    # Warm-up build to initialise weights
    dummy = tf.zeros((1,) + (32, 32, 3))
    ref_model(dummy, training=False)
    init_weights = ref_model.get_weights()

    clients = []
    for i in range(args.num_clients):
        model_i = build_model(args)
        model_i(dummy, training=False)
        model_i.set_weights([w.copy() for w in init_weights])  # shared initialisation
        client = Client(
            client_id=i,
            model=model_i,
            train_dataset=train_datasets[i],
            test_dataset=test_datasets[i],
            args=args,
            dp_config=dp_config,
            dp_allocator=dp_allocator,
            num_train_samples=num_train_samples[i],
        )
        clients.append(client)

    # ── Global privacy accountant ─────────────────────────────────────────
    global_acc_tracker = GlobalPrivacyAccountant(num_clients=args.num_clients)
    if args.dp_enabled:
        for c in clients:
            if c.accountant:
                global_acc_tracker.register(c.id, c.accountant)

    # ── Assign clients to edges ───────────────────────────────────────────
    edges = assign_clients_to_edges(clients, args.num_edges, seed=args.seed)
    p_clients = []  # per-edge sampling probabilities
    for edge in edges:
        totals = edge.total_samples
        p = np.array([edge.sample_registration.get(cid, 1)
                      for cid in edge.cids], dtype=float)
        p /= p.sum()
        p_clients.append(p)

    cloud = Cloud(init_weights=init_weights)
    for edge in edges:
        cloud.register_edge(edge)
    p_edge = np.array([cloud.sample_registration.get(e.id, 1) for e in edges], dtype=float)
    p_edge /= p_edge.sum()
    cloud.refresh()

    # ── Global model for eval ─────────────────────────────────────────────
    global_model = build_model(args)
    global_model(dummy, training=False)
    global_model.set_weights(init_weights)

    # ── Metrics log ───────────────────────────────────────────────────────
    log_dir = Path(f"runs/{args.dataset}_{args.model}_"
                   f"dp{args.dp_enabled}_eps{args.dp_epsilon}_"
                   f"mode{args.dp_adaptive_mode}_"
                   f"nc{args.num_clients}_ne{args.num_edges}")
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(log_dir))
    clients_per_edge = max(1, args.num_clients // args.num_edges)
    global_step = 0
    last_avg_acc_v = 0.0

    # ══════════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════════
    logger.info(
        f"Starting HierDP-FL | "
        f"Dataset={args.dataset} | Model={args.model} | "
        f"Clients={args.num_clients} | Edges={args.num_edges} | "
        f"Rounds={args.num_communication} | "
        f"τ₁={args.num_local_update} τ₂={args.num_edge_aggregation} | "
        f"DP={args.dp_enabled}"
    )

    for comm_round in tqdm(range(args.num_communication), desc="Global rounds"):
        cloud.refresh()
        for edge in edges:
            cloud.register_edge(edge)

        selected_cids_per_edge = []
        edge_losses = []
        edge_sample_counts = []

        # ── Edge aggregation rounds ───────────────────────────────────────
        for edge_agg_step in range(args.num_edge_aggregation):
            all_correct = 0.0
            all_total = 0.0

            for i, edge in enumerate(edges):
                edge.refresh()

                # Select fraction of clients under this edge
                frac_num = max(1, int(clients_per_edge * args.frac))
                selected_cids = np.random.choice(
                    edge.cids, size=frac_num, replace=False, p=p_clients[i]
                )

                # Re-register selected clients for sample accounting
                for cid in selected_cids:
                    edge.register_client(clients[cid])
                edge.all_train_sample_num = edge.total_samples

                # Broadcast edge model → clients
                for cid in selected_cids:
                    edge.send_to_client(clients[cid])
                    clients[cid].sync_with_edgeserver()

                # Local DP-SGD updates
                client_losses = []
                for cid in selected_cids:
                    loss = clients[cid].local_update(
                        num_iter=args.num_local_update,
                        round_num=comm_round,
                    )
                    client_losses.append(loss)
                    clients[cid].send_to_edgeserver(edge)

                edge.aggregate()

                # Edge-level accuracy (all clients under edge)
                for cid in edge.cids:
                    edge.send_to_client(clients[cid])
                    clients[cid].sync_with_edgeserver()
                    c_correct, c_total = clients[cid].test_model()
                    all_correct += c_correct
                    all_total += c_total

                edge_losses.append(float(np.mean(client_losses)))
                edge_sample_counts.append(edge.total_samples)
                selected_cids_per_edge.append(selected_cids.tolist())

            # ── Log edge-aggregation step metrics ─────────────────────────
            global_step += 1
            avg_loss = sum(l * s for l, s in zip(edge_losses, edge_sample_counts)) / max(sum(edge_sample_counts), 1)
            avg_acc = all_correct / all_total if all_total > 0 else 0.0

            with summary_writer.as_default():
                tf.summary.scalar("train/avg_loss", avg_loss, step=global_step)
                tf.summary.scalar("eval/avg_acc_edge", avg_acc, step=global_step)

            if edge_agg_step == args.num_edge_aggregation - 1:
                logger.info(
                    f"Round {comm_round+1}/{args.num_communication} | "
                    f"Loss={avg_loss:.4f} | EdgeAcc={avg_acc:.4f}"
                )

        # ── Cloud aggregation ─────────────────────────────────────────────
        for edge in edges:
            edge.send_to_cloudserver(cloud)
        cloud.aggregate()
        for edge in edges:
            cloud.send_to_edge(edge)

        # ── Evaluate global model ─────────────────────────────────────────
        global_model.set_weights(cloud.shared_weights)
        avg_acc_v = fast_global_eval(global_model, v_test_ds)
        last_avg_acc_v = avg_acc_v

        # ── Privacy accounting report ─────────────────────────────────────
        if args.dp_enabled:
            privacy_report = global_acc_tracker.report()
            worst_eps = privacy_report.get("worst_case_epsilon", 0.0)
            with summary_writer.as_default():
                tf.summary.scalar("privacy/worst_case_epsilon", worst_eps, step=comm_round + 1)
            logger.info(
                f"  Global acc={avg_acc_v:.4f} | "
                f"Privacy ε≤{worst_eps:.3f} (δ={args.dp_delta})"
            )
        else:
            logger.info(f"  Global acc={avg_acc_v:.4f} (no DP)")

        with summary_writer.as_default():
            tf.summary.scalar("eval/global_acc_cloud", avg_acc_v, step=comm_round + 1)

        # ── Update adaptive allocator state ───────────────────────────────
        if dp_allocator is not None:
            round_metrics = collect_round_metrics(
                clients, selected_cids_per_edge, edge_losses, edge_sample_counts,
                comm_round, avg_acc_v, args.dp_enabled,
            )
            dp_allocator.update(round_metrics)

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete.")
    logger.info(f"Final global accuracy: {last_avg_acc_v:.4f}")
    if args.dp_enabled:
        rpt = global_acc_tracker.report()
        logger.info(f"Final worst-case privacy: ε={rpt['worst_case_epsilon']:.4f}, δ={args.dp_delta}")
    logger.info(f"TensorBoard logs: {log_dir}")
    logger.info(f"{'='*60}\n")

    return last_avg_acc_v


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    args = args_parser()
    HierDPFL(args)


if __name__ == "__main__":
    main()
