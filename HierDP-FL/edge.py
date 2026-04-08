"""
edge.py  /  cloud.py – HierFL server hierarchy (TF/Keras version)
────────────────────────────────────────────────────────────────────────────
Mirrors HierFL's Edge and Cloud classes.

Edge server: aggregates client models within its cluster (τ₂ rounds).
Cloud server: aggregates edge models for the global round (τ₃ = 1).

Aggregation: weighted FedAvg (weight = number of training samples).
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Weighted average ──────────────────────────────────────────────────────

def weighted_average(
    weight_lists: List[List[np.ndarray]],
    sample_counts: List[int],
) -> List[np.ndarray]:
    """
    Compute sample-weighted FedAvg of a list of weight arrays.

    Parameters
    ----------
    weight_lists  : List of per-client weight lists (model.get_weights())
    sample_counts : Per-client training sample counts (weights for averaging)

    Returns
    -------
    List[np.ndarray] – aggregated model weights
    """
    total = sum(sample_counts)
    if total == 0:
        return copy.deepcopy(weight_lists[0])

    result = [np.zeros_like(w) for w in weight_lists[0]]
    for weights, n in zip(weight_lists, sample_counts):
        frac = n / total
        for i, w in enumerate(weights):
            result[i] += frac * w.astype(np.float64)
    return [r.astype(weight_lists[0][i].dtype) for i, r in enumerate(result)]


# ══════════════════════════════════════════════════════════════════════════
# Edge Server
# ══════════════════════════════════════════════════════════════════════════

class Edge:
    """
    Hierarchical FL edge server.

    Attributes
    ----------
    id                  : int
    cids                : list of client IDs under this edge
    shared_weights      : current aggregated model weights (List[np.ndarray])
    receiver_buffer     : {client_id: weights}  – received client updates
    sample_registration : {client_id: int}       – #train samples per client
    id_registration     : list of client IDs that have registered this round
    """

    def __init__(self, edge_id: int, cids: list, init_weights: List[np.ndarray]):
        self.id = edge_id
        self.cids = list(cids)
        self.shared_weights: List[np.ndarray] = copy.deepcopy(init_weights)
        self.receiver_buffer: Dict[int, List[np.ndarray]] = {}
        self.id_registration: List[int] = []
        self.sample_registration: Dict[int, int] = {}
        self.all_train_sample_num: int = 0

    # ── Client management ─────────────────────────────────────────────────

    def register_client(self, client) -> None:
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = client.num_train_samples

    def refresh(self) -> None:
        self.receiver_buffer.clear()
        self.id_registration.clear()
        self.sample_registration.clear()

    # ── Receive / send ────────────────────────────────────────────────────

    def receive_from_client(self, client_id: int, client_weights: List[np.ndarray]) -> None:
        self.receiver_buffer[client_id] = client_weights

    def send_to_client(self, client) -> None:
        client.receive_from_edgeserver(copy.deepcopy(self.shared_weights))

    def send_to_cloudserver(self, cloud) -> None:
        cloud.receive_from_edge(
            edge_id=self.id,
            edge_weights=copy.deepcopy(self.shared_weights),
        )

    def receive_from_cloudserver(self, weights: List[np.ndarray]) -> None:
        self.shared_weights = weights

    # ── Aggregation ───────────────────────────────────────────────────────

    def aggregate(self) -> None:
        """FedAvg over received client weights."""
        client_ids = list(self.receiver_buffer.keys())
        if not client_ids:
            logger.warning(f"Edge {self.id}: aggregate called with empty buffer.")
            return
        weight_lists = [self.receiver_buffer[cid] for cid in client_ids]
        sample_counts = [self.sample_registration.get(cid, 1) for cid in client_ids]
        self.shared_weights = weighted_average(weight_lists, sample_counts)

    @property
    def total_samples(self) -> int:
        return sum(self.sample_registration.values())


# ══════════════════════════════════════════════════════════════════════════
# Cloud Server
# ══════════════════════════════════════════════════════════════════════════

class Cloud:
    """
    Hierarchical FL cloud (global) server.

    Aggregates edge model updates to produce the global model.
    """

    def __init__(self, init_weights: List[np.ndarray]):
        self.shared_weights: List[np.ndarray] = copy.deepcopy(init_weights)
        self.receiver_buffer: Dict[int, List[np.ndarray]] = {}
        self.edge_registration: List[int] = []
        self.sample_registration: Dict[int, int] = {}

    def register_edge(self, edge: Edge) -> None:
        self.edge_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_train_sample_num

    def refresh(self) -> None:
        self.receiver_buffer.clear()
        self.edge_registration.clear()
        self.sample_registration.clear()

    def receive_from_edge(self, edge_id: int, edge_weights: List[np.ndarray]) -> None:
        self.receiver_buffer[edge_id] = edge_weights

    def send_to_edge(self, edge: Edge) -> None:
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_weights))

    def aggregate(self) -> None:
        """FedAvg over received edge weights (weighted by edge's total samples)."""
        edge_ids = list(self.receiver_buffer.keys())
        if not edge_ids:
            logger.warning("Cloud: aggregate called with empty buffer.")
            return
        weight_lists = [self.receiver_buffer[eid] for eid in edge_ids]
        sample_counts = [self.sample_registration.get(eid, 1) for eid in edge_ids]
        self.shared_weights = weighted_average(weight_lists, sample_counts)
