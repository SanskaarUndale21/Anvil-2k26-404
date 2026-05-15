"""
Train a small MLP: corrupted_query (64) -> precision_weights (64).
Run AFTER generate_data.py.

Architecture: 64 -> 256 -> 128 -> 64 (Softplus output to ensure positivity)
Saves weights as numpy arrays to models/mlp_weights.npz

The trained MLP can be loaded in the adapter as a fallback or primary approach.
"""

import numpy as np
import os


def relu(x):
    return np.maximum(0, x)


def softplus(x):
    return np.log1p(np.exp(np.clip(x, -88, 88)))


def mse(pred, target):
    return np.mean((pred - target) ** 2)


def forward(x, weights):
    w1, b1, w2, b2, w3, b3 = weights
    h1 = relu(x @ w1 + b1)
    h2 = relu(h1 @ w2 + b2)
    out = softplus(h2 @ w3 + b3) + 0.05  # positive output
    return out


def train():
    data_path = os.path.join(os.path.dirname(__file__), "..", "models", "train_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Run generate_data.py first")

    data = np.load(data_path)
    X = data["queries"].astype(np.float64)   # (M, 64)
    Y = data["targets"].astype(np.float64)   # (M, 64)

    # Normalize targets to mean=1 per sample
    Y = Y / (Y.mean(axis=1, keepdims=True) + 1e-8)

    N_samples = X.shape[0]
    rng = np.random.default_rng(0)

    # Xavier init
    def init_layer(n_in, n_out):
        scale = np.sqrt(2.0 / n_in)
        return rng.standard_normal((n_in, n_out)) * scale, np.zeros(n_out)

    w1, b1 = init_layer(64, 256)
    w2, b2 = init_layer(256, 128)
    w3, b3 = init_layer(128, 64)
    weights = [w1, b1, w2, b2, w3, b3]

    lr = 1e-3
    batch_size = 256
    epochs = 50

    print(f"Training on {N_samples} samples for {epochs} epochs...")

    for epoch in range(epochs):
        idx = rng.permutation(N_samples)
        X_shuf, Y_shuf = X[idx], Y[idx]
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_samples, batch_size):
            xb = X_shuf[start:start + batch_size]
            yb = Y_shuf[start:start + batch_size]

            # Forward
            h1 = relu(xb @ w1 + b1)
            h2 = relu(h1 @ w2 + b2)
            logits = h2 @ w3 + b3
            out = softplus(logits) + 0.05

            # Normalize to mean=1 per sample
            out_norm = out / (out.mean(axis=1, keepdims=True) + 1e-8)

            loss = mse(out_norm, yb)
            epoch_loss += loss
            n_batches += 1

            # Backward (MSE on normalized output)
            # dL/d_out_norm
            d_out_norm = 2.0 * (out_norm - yb) / xb.shape[0]

            # Chain through normalization: d_out_norm -> d_out
            m = out.mean(axis=1, keepdims=True) + 1e-8
            d_out = d_out_norm / m - out_norm * d_out_norm.mean(axis=1, keepdims=True) / m

            # Softplus backward
            sig = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))
            d_logits = d_out * sig

            d_w3 = h2.T @ d_logits
            d_b3 = d_logits.sum(axis=0)
            d_h2 = d_logits @ w3.T * (h2 > 0)

            d_w2 = h1.T @ d_h2
            d_b2 = d_h2.sum(axis=0)
            d_h1 = d_h2 @ w2.T * (h1 > 0)

            d_w1 = xb.T @ d_h1
            d_b1 = d_h1.sum(axis=0)

            w1 -= lr * d_w1; b1 -= lr * d_b1
            w2 -= lr * d_w2; b2 -= lr * d_b2
            w3 -= lr * d_w3; b3 -= lr * d_b3

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} -- loss: {epoch_loss/n_batches:.6f}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "..", "models", "mlp_weights.npz")
    np.savez(out_path, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
    print(f"Saved weights to {out_path}")


if __name__ == "__main__":
    train()
