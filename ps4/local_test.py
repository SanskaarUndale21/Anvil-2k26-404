"""
Local test harness -- mimics official self_check.py logic.
Run: python local_test.py [--quick]

p in noise_levels is the fraction of PRESERVED bits (higher p = less noise):
  p=0.5 -> 50% bits correct (hardest)
  p=0.7 -> 70% bits correct (moderate)
  p=0.8 -> 80% bits correct (easiest)
"""

import argparse
import numpy as np
from adapters.myteam import Engine


def generate_hopfield_patterns(K: int, N: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.choice([-1.0, 1.0], size=(K // 2, N))
    return np.vstack([base, -base])  # twin-paired


def pcam_retrieve(
    query: np.ndarray,
    patterns: np.ndarray,
    precision: np.ndarray,
    beta: float = 0.1,   # 1/sqrt(N) for N=64 is the principled Modern Hopfield default
    n_steps: int = 20,
) -> np.ndarray:
    """
    Modern Hopfield / PCAM retrieval with diagonal precision.
    Update: xi_{t+1} = P^T softmax(beta * P * (pi o xi_t))
    """
    state = query.copy().astype(np.float64)
    for _ in range(n_steps):
        prec_state = precision * state          # element-wise precision weighting
        sims = patterns @ prec_state            # (K,) similarities
        # Stabilized softmax
        sims_shifted = beta * sims - beta * sims.max()
        exp_s = np.exp(np.clip(sims_shifted, -88, 88))
        soft = exp_s / exp_s.sum()              # (K,)
        state = patterns.T @ soft               # (N,) new state
    return state


def retrieval_accuracy(retrieved: np.ndarray, target: np.ndarray) -> float:
    # Use element-wise product > 0 (handles continuous retrieved values)
    return (retrieved * target > 0).mean()


def anisotropy_spread(precisions: list) -> float:
    stacked = np.array(precisions)
    dim_means = stacked.mean(axis=0)
    return dim_means.max() / (dim_means.min() + 1e-8)


def evaluate_seed(
    seed: int, K: int, N: int, noise_levels: list, n_queries: int
) -> dict:
    rng = np.random.default_rng(seed)
    patterns = generate_hopfield_patterns(K, N, rng)

    engine = Engine(stored_patterns=patterns)
    identity_precision = np.ones(N)

    baseline_accs, agent_accs, agent_precisions = [], [], []

    queries_per_level = max(1, n_queries // len(noise_levels))

    for p in noise_levels:
        flip_prob = 1.0 - p  # p = fraction preserved -> flip = 1-p
        for _ in range(queries_per_level):
            target_idx = rng.integers(0, K)
            clean = patterns[target_idx]

            corrupted = clean.copy()
            flip = rng.random(N) < flip_prob
            corrupted[flip] *= -1.0
            corrupted = np.sign(corrupted)  # ensure binary

            # Baseline
            base_state = pcam_retrieve(corrupted, patterns, identity_precision)
            baseline_accs.append(retrieval_accuracy(base_state, clean))

            # Agent
            prec = engine.predict_precision(corrupted)
            prec = np.clip(prec, 0.1, 10.0)
            prec = prec / prec.mean()
            agent_state = pcam_retrieve(corrupted, patterns, prec)
            agent_accs.append(retrieval_accuracy(agent_state, clean))
            agent_precisions.append(prec)

    return {
        "seed": seed,
        "baseline_acc": float(np.mean(baseline_accs)),
        "agent_acc": float(np.mean(agent_accs)),
        "delta": float(np.mean(agent_accs) - np.mean(baseline_accs)),
        "spread": anisotropy_spread(agent_precisions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    K, N = 16, 64
    NOISE_LEVELS = [0.5, 0.7, 0.8]   # fraction of bits PRESERVED
    N_QUERIES = 150 if args.quick else 750
    SEEDS = [42, 7] if args.quick else [42, 7, 13, 99, 137]

    print(f"Seeds: {SEEDS} | {N_QUERIES} queries each | preserve rates {NOISE_LEVELS}")
    print("-" * 65)

    all_deltas, all_spreads = [], []
    any_regression = False

    for seed in SEEDS:
        r = evaluate_seed(seed, K, N, NOISE_LEVELS, N_QUERIES)
        flag = "REGRESSION" if r["delta"] < 0 else "OK"
        if r["delta"] < 0:
            any_regression = True
        print(
            f"Seed {r['seed']:4d} | "
            f"Base: {r['baseline_acc']:.4f} | "
            f"Agent: {r['agent_acc']:.4f} | "
            f"Delta: {r['delta']:+.4f} | "
            f"Spread: {r['spread']:.2f}x | "
            f"{flag}"
        )
        all_deltas.append(r["delta"])
        all_spreads.append(r["spread"])

    print("-" * 65)
    mean_delta = float(np.mean(all_deltas))
    mean_spread = float(np.mean(all_spreads))

    print(f"Mean delta: {mean_delta:+.4f}  (need > 0)")
    print(f"Mean spread: {mean_spread:.2f}x  (baseline 1x, full marks 10x)")

    if mean_delta <= 0:
        print("FAIL: mean delta <= 0 -> retrieval score = 0")
    elif any_regression:
        print("WARN: per-seed regression -> retrieval score halved")
    else:
        print("PASS")

    # Rough score estimate
    ret_score = 0.0
    if mean_delta > 0:
        multiplier = 0.5 if any_regression else 1.0
        ret_score = min(70.0, 70.0 * multiplier * (mean_delta / 0.05))

    import math
    spread_score = 0.0
    if mean_spread > 1.0:
        spread_score = min(20.0, 20.0 * math.log10(mean_spread) / math.log10(10))

    print(
        f"\nEstimated: {ret_score:.1f}/70 retrieval + {spread_score:.1f}/20 spread"
        f" = {ret_score + spread_score:.1f}/90 automated"
    )


if __name__ == "__main__":
    main()
