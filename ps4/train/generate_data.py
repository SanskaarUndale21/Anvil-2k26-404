"""
Generate synthetic training data for the optional MLP approach.
Run this BEFORE train_mlp.py.

Output: data/train_data.npz with keys: queries (M, 64), targets (M, 64)
"""

import numpy as np
import os

SEED = 42
K = 16          # number of patterns
N = 64          # state dimension
NOISE_LEVELS = [0.3, 0.5, 0.7, 0.8]
SAMPLES_PER_PATTERN_PER_NOISE = 200


def generate_patterns(k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.choice([-1.0, 1.0], size=(k // 2, n))
    twins = -base  # twin-paired
    return np.vstack([base, twins])


def optimal_precision(corrupted: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """
    Ground truth: high precision where clean and corrupted agree, low where they differ.
    alignment_i = (corrupted_i * clean_i + 1) / 2  -- 1 if same sign, 0 if flipped
    """
    alignment = (corrupted * clean + 1.0) / 2.0  # [0, 1]
    precision = alignment + 0.1
    return precision / precision.mean()


def main():
    rng = np.random.default_rng(SEED)
    patterns = generate_patterns(K, N, rng)  # (K, N)

    queries = []
    targets = []

    for p_idx, pattern in enumerate(patterns):
        for noise in NOISE_LEVELS:
            for _ in range(SAMPLES_PER_PATTERN_PER_NOISE):
                corrupted = pattern.copy()
                flip_mask = rng.random(N) < noise
                corrupted[flip_mask] *= -1.0

                prec = optimal_precision(corrupted, pattern)
                queries.append(corrupted)
                targets.append(prec)

    queries = np.array(queries, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "train_data.npz")
    np.savez(out_path, queries=queries, targets=targets, patterns=patterns)
    print(f"Saved {len(queries)} samples to {out_path}")
    print(f"Query shape: {queries.shape}, Target shape: {targets.shape}")


if __name__ == "__main__":
    main()
