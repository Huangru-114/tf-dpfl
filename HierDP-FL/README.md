# HierDP-FL: Hierarchical Federated Learning with Adaptive DP-SGD

A TensorFlow + TF-Privacy extension of [HierFL](https://github.com/LuminLiu/HierFL.git) that adds differential-privacy guarantees to the three-tier hierarchical FL architecture (Client → Edge → Cloud). The codebase provides a clean, uniform DP-SGD **baseline** and a modular interface for plugging in **adaptive privacy-budget / noise-allocation strategies** across four orthogonal dimensions.

---

## Architecture

```
Client × τ₁ local DP-SGD steps
     ↓  push weights
Edge Server  (τ₂ rounds of FedAvg)
     ↓  push weights
Cloud Server (global FedAvg, 1 round)
     ↓  broadcast global model
```

Privacy is consumed **only at the client** during local DP-SGD updates. The hierarchical aggregation steps are pure averaging (no extra privacy cost under standard FL threat models).

---

## Repository Layout

```
HierDP-FL/
├── hierdpfl.py           ← Main training loop
├── options.py            ← CLI arguments
├── client.py             ← FL client with DP-SGD
├── edge.py               ← Edge + Cloud servers (FedAvg)
│
├── dp/                   ← Differential Privacy module
│   ├── dp_config.py      ← DPConfig dataclass (central knob)
│   ├── accountant.py     ← RDP privacy accountant + calibration
│   ├── dp_trainer.py     ← Core DP-SGD training step
│   └── adaptive/         ← Adaptive allocation strategies
│       ├── base.py       ← Abstract AdaptiveDPAllocator interface
│       ├── uniform.py    ← Baseline: uniform σ and C
│       ├── layer_wise.py ← Per-layer adaptive clipping norms
│       ├── client_wise.py← Per-client adaptive noise
│       └── epoch_wise.py ← Round-decaying noise schedules
│
├── models/
│   ├── cifar_cnn.py      ← CifarCNN3Conv, CifarResNet18 (DP-compatible)
│   └── gtsrb_model.py    ← GTSRBConvNet, GTSRBMobileNet
│
└── datasets/
    ├── cifar10.py        ← CIFAR-10 / CIFAR-100 federated loaders
    └── gtsrb.py          ← GTSRB federated loader
```

---

## Quick Start

### Installation

```bash
pip install tensorflow>=2.12 tensorflow-privacy>=0.9 tensorflow-datasets tqdm numpy scipy
# For GTSRB via torchvision (optional):
# pip install torch torchvision
```

### Non-DP baseline

```bash
python hierdpfl.py \
  --dataset cifar10 --model cnn_complex \
  --num_clients 20 --num_edges 4 \
  --num_communication 100 --num_local_update 5 \
  --batch_size 32 --lr 0.01
```

### DP-SGD baseline (uniform, auto-calibrated to ε=10)

```bash
python hierdpfl.py \
  --dataset cifar10 --model cnn_complex \
  --num_clients 20 --num_edges 4 \
  --num_communication 100 --num_local_update 5 \
  --batch_size 32 --lr 0.01 \
  --dp_enabled \
  --dp_epsilon 10.0 --dp_delta 1e-5 \
  --dp_max_grad_norm 1.0 \
  --dp_adaptive_mode uniform
```

### GTSRB with adaptive layer-wise clipping

```bash
python hierdpfl.py \
  --dataset gtsrb --model convnet \
  --num_clients 20 --num_edges 4 \
  --num_communication 80 \
  --dp_enabled \
  --dp_epsilon 10.0 --dp_delta 1e-5 \
  --dp_adaptive_mode layer_wise
```

### Run all benchmark experiments

```bash
bash run_experiments.sh cifar10
bash run_experiments.sh gtsrb
```

---

## Adaptive DP Strategies

Select with `--dp_adaptive_mode`:

| Mode | Description | Key parameters |
|------|-------------|----------------|
| `uniform` | Standard DP-SGD, same σ/C for all | `--dp_noise_multiplier`, `--dp_max_grad_norm` |
| `layer_wise` | Per-layer clip norms adapt to gradient-norm percentiles | warmup 2 rounds, then p50 of client grad norms |
| `client_wise` | Higher noise for clients with larger gradient magnitudes | `high_fraction=0.3`, `high_noise_factor=1.5` |
| `epoch_wise` | Cosine-annealing noise schedule across rounds | `sigma_init → sigma_final = 0.5 × sigma_init` |

### Extending with a custom strategy

```python
# my_strategy.py
from dp.adaptive.base import AdaptiveDPAllocator, NoiseParams

class MyAdaptiveDPAllocator(AdaptiveDPAllocator):
    def get_noise_params(self, client_id, round_num, epoch, layer_names=None):
        # Your allocation logic here
        return NoiseParams(
            noise_multiplier=self._compute_sigma(client_id, round_num),
            clip_norm=self._compute_clip(layer_names),
        )

    def update(self, metrics: dict):
        # Called after every global round; update internal state from metrics
        pass
```

Then register it in `dp/__init__.py`'s `build_dp_allocator()` factory.

---

## Privacy Accounting

The `PrivacyAccountant` (RDP-based) tracks the cumulative (ε, δ) for each client:

```python
from dp.accountant import PrivacyAccountant

acc = PrivacyAccountant(dp_config, num_train_samples=5000, batch_size=32)
# After each local gradient step:
acc.step()
eps, delta = acc.get_privacy_spent()
print(f"Current privacy: ε={eps:.3f}, δ={delta}")
```

To auto-calibrate σ for a target ε:

```python
sigma = PrivacyAccountant.calibrate_noise_multiplier(
    target_epsilon=10.0, delta=1e-5,
    num_train_samples=5000, batch_size=32,
    total_steps=500,
)
```

---

## DP-SGD Implementation Notes

- **GroupNorm** replaces BatchNorm in all models (BatchNorm's cross-sample statistics are incompatible with per-sample gradient clipping).
- **True per-sample DP-SGD** is achieved by setting `num_microbatches = batch_size` (default). Each microbatch processes one example; gradients are clipped independently before aggregation and noise injection.
- The noise is added once per batch: `noise_std = σ × C / batch_size`.
- If `tensorflow_privacy.DPKerasSGDOptimizer` is available it is used natively for the uniform baseline; otherwise the custom `DPTrainer` provides equivalent guarantees.

---

## TensorBoard

```bash
tensorboard --logdir runs/
```

Logged metrics:
- `train/avg_loss`
- `eval/avg_acc_edge` (per edge-aggregation step)
- `eval/global_acc_cloud` (per global round)
- `privacy/worst_case_epsilon` (DP enabled only)

---

## Datasets

| Dataset | Classes | Train / Test | Partition default |
|---------|---------|-------------|------------------|
| CIFAR-10 | 10 | 50k / 10k | Shard (2 classes/client) |
| CIFAR-100 | 100 | 50k / 10k | Shard (5 classes/client) |
| GTSRB | 43 | ~39k / ~12k | Shard (5 classes/client) |

GTSRB is downloaded automatically via `torchvision` or `tensorflow_datasets`.

---

## Citation

If you use this code, please cite the original HierFL paper:

```bibtex
@article{liu2020client,
  title={Client-edge-cloud hierarchical federated learning},
  author={Liu, Lumin and others},
  journal={ICC 2020},
  year={2020}
}
```
