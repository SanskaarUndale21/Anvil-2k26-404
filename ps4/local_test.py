"""
Local test harness -- matches official self_check.py semantics.

Key differences from naive Hopfield harness:
  - Noise is combined mask + Gaussian (not pure bit-flips)
  - model_params = {R, eta, beta} is passed to Engine.__init__
  - Anisotropy is eigenvalue spread of Pi^(1/2) H Pi^(1/2) at each attractor
  - p in noise_levels = fraction of dims PRESERVED (higher p = less noise)

Run:
  python local_test.py [--quick]
"""

import argparse
import numpy as np
from adapters.myteam import Engine


# ── Pattern generation ─────────────────────────────────────────────────────

def make_patterns(K: int, N: int, rng) -> np.ndarray:
    """Twin-paired binary patterns."""
    base = rng.choice([-1.0, 1.0], size=(K // 2, N))
    return np.vstack([base, -base])


def make_R(N: int, rng) -> np.ndarray:
    """
    Generate a random positive-definite structured operator R.
    R = A^T A + epsilon*I ensures PD; diagonal spread creates anisotropy.
    In the real harness R encodes the PCAM memory structure.
    """
    A = rng.standard_normal((N, N)) * 0.5
    R = A.T @ A / N + np.eye(N) * 0.5
    # Add controlled anisotropy so dims have different Hessian curvatures
    scale = np.exp(rng.standard_normal(N) * 0.5)   # log-normal per-dim scale
    R = R * scale[:, None] * scale[None, :]
    return R


# ── Noise model ────────────────────────────────────────────────────────────

def corrupt(clean: np.ndarray, preserve_prob: float, rng) -> np.ndarray:
    """
    Combined mask + Gaussian noise (matches PS4 description).
      preserve_prob = fraction of dims kept (p=0.8 -> 80% clean).
    Masked dims: set to 0 then add small Gaussian noise.
    Clean dims:  keep original value then add small Gaussian noise.
    """
    N = len(clean)
    mask = rng.random(N) < preserve_prob   # True = preserved
    noisy = clean.copy().astype(float)
    noisy[~mask] = 0.0                     # mask out corrupted dims
    noisy += rng.standard_normal(N) * 0.15  # background Gaussian everywhere
    return noisy


# ── PCAM-style retrieval dynamics ──────────────────────────────────────────

def pcam_retrieve(
    query: np.ndarray,
    patterns: np.ndarray,
    precision: np.ndarray,
    beta: float = 0.5,
    n_steps: int = 25,
) -> np.ndarray:
    """
    Modern Hopfield update with per-dimension precision:
      x_{t+1} = P^T @ softmax(beta * P @ (precision * x_t))
    """
    state = query.copy().astype(np.float64)
    for _ in range(n_steps):
        weighted = precision * state
        sims = patterns @ weighted
        shifted = beta * sims - beta * sims.max()
        exp_s = np.exp(np.clip(shifted, -88, 88))
        soft = exp_s / exp_s.sum()
        state = patterns.T @ soft
    return state


def retrieval_accuracy(retrieved: np.ndarray, target: np.ndarray) -> float:
    return float((retrieved * target > 0).mean())


# ── Anisotropy metric ──────────────────────────────────────────────────────

def hessian_at_attractor(
    pattern: np.ndarray,
    patterns: np.ndarray,
    R: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Approximate Hessian of PCAM energy at attractor 'pattern'.
    H ≈ beta^2 * X^T diag(soft) X + R   (diagonal-dominant approximation)
    where soft is softmax at the attractor (peaked at this pattern).

    For the diagonal: H_ii ≈ beta^2 * Var_k[X_ki] * soft_concentration + R_ii
    Simplified: we use R_diag as the dominant term (frozen system contribution).
    """
    # Softmax at this attractor (nearly one-hot)
    sims = patterns @ pattern
    exp_s = np.exp(np.clip(beta * sims - beta * sims.max(), -88, 88))
    soft = exp_s / exp_s.sum()

    # Softmax-weighted second moment of patterns (diagonal of X^T diag(soft) X)
    weighted_sq = (soft[:, None] * patterns ** 2).sum(axis=0)   # (N,)
    weighted_mean_sq = ((soft[:, None] * patterns).sum(axis=0)) ** 2  # (N,)
    var_term = weighted_sq - weighted_mean_sq                    # (N,) variance

    # Hessian diagonal
    H_diag = beta ** 2 * var_term + np.diag(R)
    return np.abs(H_diag) + 1e-8


def anisotropy_spread(
    precision: np.ndarray,
    H_diag: np.ndarray,
) -> float:
    """
    Spread = max_eig / min_eig of Pi^(1/2) H Pi^(1/2).
    For diagonal H: eigenvalue_i = precision_i * H_diag_i.
    Spread = max(prec*H) / min(prec*H).
    """
    eigs = precision * H_diag
    return float(eigs.max() / (eigs.min() + 1e-12))


# ── Per-seed evaluation ────────────────────────────────────────────────────

def evaluate_seed(
    seed: int,
    K: int,
    N: int,
    noise_levels: list,
    n_queries: int,
    beta: float,
) -> dict:
    rng = np.random.default_rng(seed)
    patterns = make_patterns(K, N, rng)
    R = make_R(N, rng)
    eta = 1.0
    model_params = {"beta": beta, "eta": eta, "R": R}

    engine = Engine(stored_patterns=patterns, model_params=model_params)
    identity_precision = np.ones(N)

    b_accs, a_accs = [], []
    b_spreads, a_spreads = [], []

    qpl = max(1, n_queries // len(noise_levels))

    # Pre-compute H_diag at each attractor for anisotropy check
    H_diags = [
        hessian_at_attractor(patterns[k], patterns, R, beta)
        for k in range(K)
    ]

    for p in noise_levels:
        for _ in range(qpl):
            target_idx = rng.integers(0, K)
            clean = patterns[target_idx]
            q = corrupt(clean, p, rng)

            # Baseline
            b_state = pcam_retrieve(q, patterns, identity_precision, beta)
            b_accs.append(retrieval_accuracy(b_state, clean))

            # Agent
            prec = engine.predict_precision(q)
            prec = np.clip(prec, 0.1, 10.0)
            prec = prec / prec.mean()
            a_state = pcam_retrieve(q, patterns, prec, beta)
            a_accs.append(retrieval_accuracy(a_state, clean))

    # Anisotropy: measure across all stored patterns (not per-query)
    for k in range(K):
        # Use a high-confidence query to get a representative precision
        # (use slightly corrupted pattern so agent is confident)
        q_ref = corrupt(patterns[k], 0.9, rng)
        prec = engine.predict_precision(q_ref)
        prec = np.clip(prec, 0.1, 10.0)
        prec = prec / prec.mean()

        H_d = H_diags[k]
        b_spreads.append(anisotropy_spread(identity_precision, H_d))
        a_spreads.append(anisotropy_spread(prec, H_d))

    mean_b_spread = float(np.mean(b_spreads))
    mean_a_spread = float(np.mean(a_spreads))
    # Spread REDUCTION: baseline_spread / agent_spread (want >> 1)
    spread_reduction = mean_b_spread / (mean_a_spread + 1e-10)

    return {
        "seed": seed,
        "baseline_acc": float(np.mean(b_accs)),
        "agent_acc": float(np.mean(a_accs)),
        "delta": float(np.mean(a_accs)) - float(np.mean(b_accs)),
        "baseline_spread": mean_b_spread,
        "agent_spread": mean_a_spread,
        "spread_reduction": spread_reduction,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--beta", type=float, default=0.08)
    args = parser.parse_args()

    K, N = 16, 64
    NOISE_LEVELS = [0.5, 0.7, 0.8]
    N_QUERIES = 150 if args.quick else 750
    SEEDS = [42, 101] if args.quick else [42, 101, 7, 13, 31]
    BETA = args.beta

    print(f"ANVIL P-04  |  K={K}  N={N}  beta={BETA}  seeds={SEEDS}")
    print(f"Noise preserve rates: {NOISE_LEVELS}  |  {N_QUERIES} queries/seed")
    print("=" * 72)
    print(f"{'seed':>6}  {'Pi=I':>6}  {'agent':>6}  {'delta':>7}  "
          f"{'base_sp':>8}  {'agnt_sp':>8}  {'reduc':>7}  status")
    print("-" * 72)

    all_deltas, all_reductions = [], []
    any_regression = False
    any_bad_aniso = False

    for seed in SEEDS:
        r = evaluate_seed(seed, K, N, NOISE_LEVELS, N_QUERIES, BETA)
        status = ""
        if r["delta"] < 0:
            status = "REGRESS"
            any_regression = True
        if r["spread_reduction"] <= 1.0:
            status += " BAD_ANISO"
            any_bad_aniso = True

        print(
            f"{r['seed']:>6}  "
            f"{r['baseline_acc']:.4f}  "
            f"{r['agent_acc']:.4f}  "
            f"{r['delta']:>+.4f}  "
            f"{r['baseline_spread']:>8.2f}  "
            f"{r['agent_spread']:>8.2f}  "
            f"{r['spread_reduction']:>6.2f}x  "
            f"{status}"
        )
        all_deltas.append(r["delta"])
        all_reductions.append(r["spread_reduction"])

    print("=" * 72)
    mean_delta = float(np.mean(all_deltas))
    mean_reduc = float(np.mean(all_reductions))

    print(f"Mean delta accuracy:      {mean_delta:+.4f}  (need > 0 | full marks at 0.05)")
    print(f"Mean spread reduction:{mean_reduc:6.2f}x  (need > 1x | full marks at 10x)")
    print()

    # Score estimate
    import math
    ret_score = 0.0
    if mean_delta > 0:
        mult = 0.5 if any_regression else 1.0
        ret_score = min(70.0, 70.0 * mult * mean_delta / 0.05)

    aniso_score = 0.0
    if mean_reduc > 1.0:
        mult = 0.5 if any_bad_aniso else 1.0
        aniso_score = min(20.0, 20.0 * mult * math.log10(mean_reduc) / math.log10(10))

    total = ret_score + aniso_score
    print(f"Estimated score:  {ret_score:.1f}/70 retrieval  +  "
          f"{aniso_score:.1f}/20 anisotropy  =  {total:.1f}/90 automated")

    if any_regression:
        print("WARNING: retrieval regression on at least one seed -> score halved")
    if any_bad_aniso:
        print("WARNING: anisotropy <= 1x on at least one seed -> aniso score halved")


if __name__ == "__main__":
    main()
