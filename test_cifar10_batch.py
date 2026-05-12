"""
Iterative single-batch training test on CIFAR-10 binary subset:
  - 2 classes, N_PER_CLASS samples each
  - training data wrapped in a cycling batch iterator (batch size = BATCH_SIZE)
  - each round consumes 1 batch, model is updated for STEPS_PER_ROUND gradient steps
  - repeat for TOTAL_ROUNDS rounds, printing test acc each round
"""

import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, "fedavg_v2")
from fedavg_v2.models.cnn import build_model

# ── config ────────────────────────────────────────────────────────────────────
CLASSES          = [3, 5]   # cat=3, dog=5  (change freely)
N_PER_CLASS      = 250
BATCH_SIZE       = 16
STEPS_PER_ROUND  = 5        # gradient steps taken on each batch
TOTAL_ROUNDS     = 50       # total number of rounds (batches consumed)
LR               = 1e-3
SEED             = 42
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


def batch_iterator(x, y, batch_size):
    """Infinite cycling iterator: shuffle the dataset each epoch, yield batches."""
    n = len(x)
    idx = np.arange(n)
    while True:
        rng.shuffle(idx)
        for start in range(0, n - batch_size + 1, batch_size):
            batch = idx[start: start + batch_size]
            yield x[batch], y[batch]


def train_step(model, optimizer, loss_fn, x_batch, y_batch):
    x_t = tf.constant(x_batch)
    y_t = tf.constant(y_batch)
    with tf.GradientTape() as tape:
        logits = model(x_t, training=True)
        loss   = loss_fn(y_t, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss.numpy()


def evaluate(model, loss_fn, x, y):
    logits = model(tf.constant(x), training=False)
    preds  = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
    loss   = loss_fn(tf.constant(y), logits).numpy()
    acc    = np.mean(preds == y)
    return loss, acc


def main():
    # ── load & preprocess ──────────────────────────────────────────────────────
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze().astype(np.int32)
    y_test  = y_test.squeeze().astype(np.int32)
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32)  / 255.0

    # ── filter to selected classes ─────────────────────────────────────────────
    x_sub,      y_sub      = filter_classes(x_train, y_train, CLASSES, N_PER_CLASS)
    n_test_per  = len(np.where(y_test == CLASSES[0])[0])
    x_test_sub, y_test_sub = filter_classes(x_test,  y_test,  CLASSES, n_test_per)

    print(f"Train subset : {x_sub.shape}  (classes {CLASSES})")
    print(f"Test  subset : {x_test_sub.shape}")
    print(f"Rounds={TOTAL_ROUNDS}  batch_size={BATCH_SIZE}  steps_per_round={STEPS_PER_ROUND}\n")

    # ── model & optimizer ──────────────────────────────────────────────────────
    model     = build_model(input_shape=(32, 32, 3), num_classes=len(CLASSES))
    optimizer = tf.keras.optimizers.Adam(LR)
    loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy()

    # ── iterator over training batches ────────────────────────────────────────
    it = batch_iterator(x_sub, y_sub, BATCH_SIZE)

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"{'Round':>6}  {'train_loss':>10}  {'test_loss':>9}  {'test_acc':>8}")
    print("-" * 42)

    history = []
    for rnd in range(1, TOTAL_ROUNDS + 1):
        x_batch, y_batch = next(it)

        # STEPS_PER_ROUND gradient updates on this single batch
        for _ in range(STEPS_PER_ROUND):
            train_loss = train_step(model, optimizer, loss_fn, x_batch, y_batch)

        test_loss, test_acc = evaluate(model, loss_fn, x_test_sub, y_test_sub)
        history.append((rnd, train_loss, test_loss, test_acc))
        print(f"{rnd:>6}  {train_loss:>10.4f}  {test_loss:>9.4f}  {test_acc:>8.2%}")

    # ── summary ───────────────────────────────────────────────────────────────
    best = max(history, key=lambda t: t[3])
    print(f"\nBest test acc: {best[3]:.2%}  at round {best[0]}")
    final = history[-1]
    print(f"Final (round {final[0]}): test_loss={final[2]:.4f}  test_acc={final[3]:.2%}")


if __name__ == "__main__":
    main()
