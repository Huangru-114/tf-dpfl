"""
Quick sanity test:
  - CIFAR-10, 2 classes, 250 samples each
  - sample one batch of size 16
  - train on that single batch for 5 steps
  - evaluate on same-distribution test split (same 2 classes)
"""

import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, "fedavg_v2")
from fedavg_v2.models.cnn import build_model

# ── config ────────────────────────────────────────────────────────────────────
CLASSES      = [3, 5]   # cat=3, dog=5  (change freely)
N_PER_CLASS  = 250
BATCH_SIZE   = 16
TRAIN_STEPS  = 5
LR           = 1e-3
SEED         = 42
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)


def filter_classes(images, labels, classes, n_per_class):
    """Return n_per_class samples for each class, labels remapped to 0..len(classes)-1."""
    xs, ys = [], []
    for new_label, orig_label in enumerate(classes):
        idx = np.where(labels == orig_label)[0]
        chosen = rng.choice(idx, size=n_per_class, replace=False)
        xs.append(images[chosen])
        ys.append(np.full(len(chosen), new_label, dtype=np.int32))
    return np.concatenate(xs), np.concatenate(ys)


def main():
    # ── load data ──────────────────────────────────────────────────────────────
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze().astype(np.int32)
    y_test  = y_test.squeeze().astype(np.int32)

    # normalise to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32)  / 255.0

    # ── filter to selected classes ─────────────────────────────────────────────
    x_sub, y_sub = filter_classes(x_train, y_train, CLASSES, N_PER_CLASS)
    x_test_sub, y_test_sub = filter_classes(x_test,  y_test,  CLASSES, N_TEST := len(np.where(y_test == CLASSES[0])[0]))

    print(f"Train subset : {x_sub.shape}  labels unique={np.unique(y_sub)}")
    print(f"Test  subset : {x_test_sub.shape}  labels unique={np.unique(y_test_sub)}")

    # ── sample one batch ───────────────────────────────────────────────────────
    batch_idx = rng.choice(len(x_sub), size=BATCH_SIZE, replace=False)
    x_batch   = x_sub[batch_idx]
    y_batch   = y_sub[batch_idx]
    print(f"Batch shape  : {x_batch.shape}  label counts={np.bincount(y_batch)}")

    # ── build model ────────────────────────────────────────────────────────────
    model = build_model(input_shape=(32, 32, 3), num_classes=len(CLASSES))
    optimizer = tf.keras.optimizers.Adam(LR)
    loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy()

    x_t = tf.constant(x_batch)
    y_t = tf.constant(y_batch)

    # ── train on the single batch for TRAIN_STEPS steps ───────────────────────
    print(f"\nTraining on batch (size={BATCH_SIZE}) for {TRAIN_STEPS} steps ...")
    for step in range(1, TRAIN_STEPS + 1):
        with tf.GradientTape() as tape:
            logits = model(x_t, training=True)
            loss   = loss_fn(y_t, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        preds   = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
        acc     = np.mean(preds == y_batch)
        print(f"  step {step}/{TRAIN_STEPS}  loss={loss.numpy():.4f}  batch_acc={acc:.2%}")

    # ── evaluate on test split ─────────────────────────────────────────────────
    print(f"\nEvaluating on test subset ({len(x_test_sub)} samples) ...")
    logits_test = model(tf.constant(x_test_sub), training=False)
    preds_test  = tf.argmax(logits_test, axis=1, output_type=tf.int32).numpy()
    test_acc    = np.mean(preds_test == y_test_sub)
    test_loss   = loss_fn(tf.constant(y_test_sub), logits_test).numpy()
    print(f"  test loss={test_loss:.4f}  test acc={test_acc:.2%}")


if __name__ == "__main__":
    main()
