# P-04 PCAM -- Full Deep Dive: Math, Strategy, Max Marks

> "The energy function defines what is remembered. The precision operator controls how retrieval unfolds."

---

## Part 1: ELI5 -- What Is This Problem?

### The Landscape Analogy

Imagine a hilly landscape with 16 valleys. Each valley represents a stored memory (a "pattern"). You blindfold someone, spin them around, and drop them on the hillside at a random noisy location (the "corrupted query"). They stumble downhill following the steepest descent. Eventually they land in a valley. The goal: they should land in the correct valley, not a nearby one.

The problem: some valleys are very close together (the "twin pairs" in the code). Under heavy noise, the starting point sits right on the ridge between two adjacent valleys. Which one you roll into is decided by tiny details of the landscape geometry.

**What is precision?**

Before dropping the person, you get to secretly set the friction on their shoes -- independently for each of 64 directions in space. High friction in direction i = they move cautiously in that direction. Low friction = they zoom. This per-direction friction is the precision vector `pi` (64 numbers).

The baseline agent sets all friction equally to 1 (`pi = ones`). Your job is to set it smarter.

**Why does smarter friction help?**

If you know the person is trying to reach valley A (and not its twin B, which is just to the left), you can:
- Increase friction in the left-right direction (so they don't accidentally drift toward B)
- Decrease friction in the forward direction (toward A) so they arrive faster

That's the whole game.

---

### What Makes It Hard: Twin Pairs

The `data.py` file creates patterns in pairs:
- K/2 "parent" patterns (random unit-norm vectors in 64-dim space)
- K/2 "twins" = parent + small noise, then re-normalized

These twins are like MNIST digits 4 and 9, or 3 and 8: similar shapes, close in pattern space. A corrupted "parent 1" query might roll into "twin 1" valley instead. Precision that tilts the landscape slightly toward parent 1 wins these close calls.

**The noise model** (`data.py:corrupt`):
1. Mask p fraction of dimensions to 0 (simulate missing pixels)
2. Add Gaussian noise sigma/sqrt(N) to all dimensions
3. Re-normalize to unit length

At p=0.8 (80% masked) plus Gaussian noise, the query has very little signal. Precision becomes crucial.

---

## Part 2: The Full Math

### 2.1 Energy Function

```
E(a) = (1/2) * a^T R a  -  (eta/beta) * log( sum_i exp(beta * x_i^T a) )
```

**Variables:**
- `a` in R^64: current system state (starts at corrupted query, evolves toward an attractor)
- `X` = [x_1, ..., x_K] in R^{K x 64}: stored patterns (each row is unit-norm)
- `R` in R^{64x64}: positive definite regularizer matrix (graph Laplacian structure)
  - Built as: `R = alpha*I + gamma*L + delta*11^T` where L is normalized Laplacian of a random graph
  - Default: alpha=0.5, gamma=0.2, delta=0.1
- `eta = 0.5`: coupling strength (how strongly patterns pull the state)
- `beta = 8.0`: inverse temperature (higher = sharper retrieval, more confident)

**The two terms:**
- `(1/2) a^T R a`: confining potential. Prevents the state from exploding. R acts as a geometry-defining metric.
- `-(eta/beta) * lse(beta * Xa)`: pattern recall term. `lse` = log-sum-exp. As beta -> inf, becomes `eta * max_i (x_i^T a)` which selects the single best-aligned pattern.

### 2.2 Softmax Distribution Over Patterns

```
s = softmax(beta * X a)
s_i = exp(beta * x_i^T a) / sum_j exp(beta * x_j^T a)
```

Think of `s_i` as the "probability that pattern i is being recalled right now." At the start (noisy query), s is spread across many patterns. At convergence, s peaks on one pattern.

### 2.3 Gradient

```
grad E(a) = R a  -  eta * X^T s
          = R a  -  eta * (sum_i s_i * x_i)
```

The gradient has two parts:
1. `R a`: pulls the state toward the origin (regularization)
2. `eta * X^T s`: pulls the state toward the softmax-weighted average of all stored patterns

At equilibrium `a*`, these balance: `R a* = eta * X^T s*`. So `a* = eta * R^{-1} * X^T s*`. When retrieval is clean (s_mu -> 1 for correct pattern mu), `a* approx eta * R^{-1} * x_mu`.

### 2.4 Hessian (Local Curvature)

```
H(a) = R  -  eta * beta * X^T (diag(s) - s s^T) X
```

The matrix `D_s = diag(s) - ss^T` is the Jacobian of softmax. Its properties:
- Rank K-1 (not K, because softmax sums to 1 so one eigenvalue is 0)
- All nonzero eigenvalues in (0, s_max) where s_max is the largest softmax value
- Positive semidefinite

In expanded form:
```
H = R  -  eta*beta * (sum_i s_i * x_i x_i^T  -  mu_s * mu_s^T)
```
where `mu_s = X^T s = sum_i s_i * x_i` is the "softmax mean pattern."

**At a clean attractor** (s_mu -> 1): D_s -> 0, so H -> R. The Hessian simplifies to the metric matrix R.

**Diagonal of H in closed form:**
```
H_ii = R_ii  -  eta*beta * Var_s[X[:, i]]
     = R_ii  -  eta*beta * (sum_k s_k * X[k,i]^2  -  (sum_k s_k * X[k,i])^2)
```

This is the softmax-weighted variance of coordinate i across all stored patterns. Dimensions where patterns disagree a lot (high variance) get smaller diagonal entries -- meaning the Hessian is "flatter" in those directions.

### 2.5 Dynamics With Precision Pi

```
a_{t+1} = a_t  +  dt * (-pi * grad E(a_t) + u(t))
```

- `pi` is your 64-vector (element-wise multiplied with the gradient)
- `u(t) = query` for first T_in=100 steps (external input anchoring), then 0
- dt = 0.01, runs up to T_max = 3000 steps

**What pi does to convergence:** Near an attractor, linearize the dynamics:
```
a_{t+1} - a*  approx  (I - dt * diag(pi) * H) * (a_t - a*)
```

The convergence rate in direction v is controlled by the eigenvalue of `diag(pi) * H` along v, which equals the eigenvalue of `Pi^(1/2) H Pi^(1/2)` along the transformed direction.

- **Slow convergence**: direction v has small eigenvalue of Pi*H -> takes many steps to converge, can be pushed off course
- **Fast convergence**: direction v has large eigenvalue -> converges quickly (but might overshoot if too large)
- **Ideal**: all eigenvalues equal -> all directions converge at the same rate, no direction is a bottleneck

The harness clips pi to [0.1, 10.0] and **normalizes mean to 1** before applying. So absolute scale doesn't matter -- only the ratios pi_i / pi_j.

---

## Part 3: The Two Scoring Axes -- Exact Math

### 3.1 Retrieval Accuracy (70 pts)

Per seed: run all test queries, measure accuracy of your agent vs. dummy (pi=1).

```
delta = agent_accuracy - baseline_accuracy
```

Score formula:
```
if delta <= 0:             0 pts           (zero -- you hurt retrieval)
if 0 < delta < 0.05:      70 * delta/0.05  (linear scale)
if delta >= 0.05:         70 pts           (full marks)
```

**Per-seed penalty**: if ANY seed has delta < 0, the retrieval score is halved.

**Target**: delta >= 0.05 on every single seed.

With 750 queries per seed at noise levels [0.5, 0.7, 0.8]:
- Baseline might get ~78% accuracy
- You need ~83% = 37.5 more correct out of 750 to hit full marks

### 3.2 Anisotropy Spread Reduction (20 pts)

For each sampled stored pattern `x_k`, the harness:
1. Creates a slightly perturbed probe (pattern + tiny noise)
2. Calls your `predict_precision(probe)` to get pi
3. Computes `H = model.hessian(x_k)` at the stored pattern itself
4. Builds `S = diag(sqrt(pi)) @ H @ diag(sqrt(pi))` = `Pi^(1/2) H Pi^(1/2)`
5. Measures `spread = max(eigenvalues(S)) / min(eigenvalues(S))`

```
reduction_factor = baseline_spread / agent_spread
```

Score formula:
```
if reduction <= 1.0:      0 pts                      (you made it worse)
if 1.0 < reduction < 10:  20 * log(reduction)/log(10) (log-scaled)
if reduction >= 10:       20 pts                     (full marks)
```

**Per-seed penalty**: if ANY seed has reduction <= 1.0, anisotropy score is halved.

**Target**: 10x spread reduction on every seed.

The baseline `kappa(H)` (spread of H's eigenvalues) is typically 10-30 for this Hessian structure. So you need to get `kappa(Pi^(1/2) H Pi^(1/2))` down to 1.0-3.0.

**Note**: the anisotropy check evaluates your precision at lightly perturbed stored patterns. So your agent sees a near-clean probe (probe_sigma=0.05 perturbation), but the Hessian is evaluated at the exact stored pattern. This means your precision only needs to work reasonably near the attractors, not just at corrupted queries.

---

## Part 4: All Math Approaches -- From Simple to Optimal

### Approach 1: Variance-Based (Floor++, ~2x spread, ~2% delta)

**Idea**: trust dimensions where the query has signal. If a dimension is near zero (masked), don't trust it.

```python
# Simple query-magnitude version
pi = np.abs(corrupted_query) + 0.1   # high pi where query is strong

# Better: stored pattern variance (which dims are informative across patterns)
var_X = np.var(self.X, axis=0)        # (N,) variance per dim
pi = 1.0 / (var_X + 1e-6)            # amplify stable dims

# Mahalanobis: trust dims that are both stable AND have signal in query
mean_X = np.mean(self.X, axis=0)
var_X = np.var(self.X, axis=0)
pi = (corrupted_query - mean_X)**2 / (var_X + 1e-6)
```

**Math basis**: Mahalanobis distance `d^2 = (q-x)^T Sigma^{-1} (q-x)` with `Sigma = diag(var_X)` gives `pi_i = 1/var_X_i`. This weights dimensions by their discriminability across patterns.

**Why it's limited**: doesn't use local geometry (Hessian), so spread reduction is modest.

---

### Approach 2: Class-Conditional (Solid, ~3-8x spread, ~3% delta)

**Idea**: first guess which stored pattern the query came from, then set precision to match that class's characteristics.

```python
q_norm = corrupted_query / (np.linalg.norm(corrupted_query) + 1e-12)
cosines = self.X @ q_norm         # (K,) cosine similarities to each stored pattern
k_pred = np.argmax(cosines)       # predicted class
x_pred = self.X[k_pred]           # predicted pattern

# Option A: amplify dims where predicted pattern is large
pi = x_pred**2 + 0.1

# Option B: down-weight dims where query deviates from prediction
diff = corrupted_query - x_pred
pi = 1.0 / (diff**2 + 0.1)   # trust dims that match the predicted pattern
```

**Twin pair exploitation**: since patterns come in parent-twin pairs, the "confusable neighbor" of `x_pred` is approximately `x_{k_pred + K/2}` (or `x_{k_pred - K/2}`). The discriminative dimensions are those where x_pred and its twin differ the most. Amplifying those:

```python
# Find the twin/sibling of the predicted pattern
K = self.X.shape[0]
half = K // 2
twin_idx = k_pred + half if k_pred < half else k_pred - half
x_twin = self.X[twin_idx]

# Discriminative dims: where predicted and twin differ
discriminative = (x_pred - x_twin)**2
pi = discriminative + 0.1
```

**Paper reference**: Section 6.6 describes the class-conditional precision design `Pi*_class` achieving ~2.5% accuracy gain on PCA-MNIST over uniform precision.

---

### Approach 3: Jacobi Preconditioner / Hessian Diagonal (Strong, ~5-15x spread, ~5% delta)

This is the core of the paper's method. Directly minimize the condition number of `Pi^(1/2) H Pi^(1/2)`.

**Van der Sluis Theorem (1969)**: For symmetric positive definite H, the diagonal preconditioner `pi_i = 1/H_ii` minimizes the omega-condition number (geometric mean ratio) and achieves a condition number within factor N of the globally optimal.

After applying Jacobi pi, the diagonal of S = Pi^(1/2) H Pi^(1/2) becomes:
```
S_ii = pi_i * H_ii = (1/H_ii) * H_ii = 1   (for all i)
```

All diagonal entries become 1. By the Gershgorin circle theorem, all eigenvalues of S lie within radius `max_i(sum_{j!=i} |S_ij|)` of 1. If the off-diagonals are small, kappa(S) is close to 1.

```python
# Compute Hessian at the query point
eta, beta, R = self.eta, self.beta, self.R
s = np.exp(beta * (self.X @ corrupted_query))
s -= s.max(); s = np.exp(s); s /= s.sum()          # stable softmax
D = np.diag(s) - np.outer(s, s)                    # (K,K) softmax Jacobian
H = R - eta * beta * (self.X.T @ D @ self.X)       # (N,N) Hessian
H = 0.5 * (H + H.T)                               # symmetrize numerically
diag_H = np.diag(H)                               # (N,) just the diagonal
pi = 1.0 / np.maximum(diag_H, 1e-4)              # Jacobi preconditioner
```

**Better: evaluate Hessian at the predicted attractor, not the noisy query**:

The query is noisy. The attractor `x_pred` is clean. The Hessian at the attractor better represents where convergence actually happens.

```python
q_norm = corrupted_query / (np.linalg.norm(corrupted_query) + 1e-12)
k_pred = np.argmax(self.X @ q_norm)
x_att = self.X[k_pred]

s_att = np.exp(beta * (self.X @ x_att))
s_att -= s_att.max(); s_att = np.exp(s_att); s_att /= s_att.sum()
D_att = np.diag(s_att) - np.outer(s_att, s_att)
H_att = R - eta * beta * (self.X.T @ D_att @ self.X)
H_att = 0.5 * (H_att + H_att.T)

diag_H = np.diag(H_att)
pi = 1.0 / np.maximum(diag_H, 1e-4)
```

---

### Approach 4: Full Diagonal of H^{-1} (Near-Optimal, ~10-30x spread, ~7% delta)

**Key insight**: `diag(H^{-1})` is NOT the same as `1/diag(H)`!

For a general matrix:
```
[H^{-1}]_ii = sum_k q_ki^2 / lambda_k
```
where `q_ki` is the i-th component of the k-th eigenvector of H, and `lambda_k` is the k-th eigenvalue.

If `Pi = H^{-1}` (full matrix, unconstrained), then `Pi^(1/2) H Pi^(1/2) = I` (spread = 1, perfect). With diagonal constraint, `diag(H^{-1})` is the best diagonal approximation of the Newton preconditioner.

```python
H_att = ...   # computed as above
H_inv = np.linalg.inv(H_att)   # O(N^3) but N=64, this is ~0.1ms
pi = np.diag(H_inv)            # (N,) actual diagonal of H^{-1}
pi = np.maximum(pi, 1e-6)
```

For N=64, `np.linalg.inv` of a 64x64 matrix takes ~0.05ms. Completely negligible.

**Woodbury formula** (avoids full inversion, uses rank-K structure of H):

```
H = R - eta*beta * X^T D_s X         (rank-K correction to R)

H^{-1} = R^{-1} + R^{-1} X^T [ (eta*beta)^{-1} D_s^{+} + X R^{-1} X^T ]^{-1} X R^{-1}
```

where D_s^{+} is pseudoinverse. Since K=16 << N=64, inverting the K x K matrix is O(K^3) = cheap. But for this problem, just calling `np.linalg.inv` on the 64x64 H is simpler and fast enough.

---

### Approach 5: Eigendecomposition-Based Optimal (Theorem F3, ~20-30x spread)

The paper's Theorem F3 gives the theoretical optimal diagonal preconditioner:

Let `H = Q Lambda Q^T` be the full eigendecomposition where:
- `Q`: (N,N) matrix of eigenvectors (columns)
- `Lambda`: diagonal matrix of eigenvalues

The diagonal of `H^{-1}` equals:
```
[H^{-1}]_ii = sum_k Q[i,k]^2 / lambda_k = (Q^2)_i . (1/lambda)
```

where `Q^2` means element-wise square of Q and the dot is across eigenvalues.

```python
eigenvalues, eigenvectors = np.linalg.eigh(H_att)   # sorted ascending
eigenvalues = np.maximum(eigenvalues, 1e-6)           # clip negative due to numerics
# pi_i = sum_k Q[i,k]^2 / lambda_k
pi = np.sum(eigenvectors**2 / eigenvalues[np.newaxis, :], axis=1)   # (N,)
```

This is mathematically equivalent to `np.diag(np.linalg.inv(H_att))` for positive definite H, but numerically more stable (eigendecomposition is more stable than inversion for ill-conditioned matrices).

`np.linalg.eigh` for a 64x64 matrix: ~0.2ms. Still fine.

---

### Approach 6: Hybrid Maximum Performance (Recommended)

Combine class-conditional signal (for retrieval) with Hessian geometry (for spread):

```python
def predict_precision(self, corrupted_query):
    q = corrupted_query.copy()
    
    # -- Step 1: predict class --
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    cosines = self.X @ q_norm
    k_pred = np.argmax(cosines)
    x_att = self.X[k_pred]
    
    # -- Step 2: Hessian at predicted attractor --
    s = np.exp(self.beta * (self.X @ x_att))
    s -= s.max(); s = np.exp(s); s /= s.sum()
    D = np.diag(s) - np.outer(s, s)
    H = self.R - self.eta * self.beta * (self.X.T @ D @ self.X)
    H = 0.5 * (H + H.T)
    
    # -- Step 3: geometry-based pi (spread reduction) --
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    pi_geom = np.sum(eigenvectors**2 / eigenvalues[np.newaxis, :], axis=1)
    
    # -- Step 4: class-conditional boost (retrieval) --
    # Amplify dims where predicted pattern is strong
    pi_class = x_att**2 + 0.1
    
    # -- Step 5: combine --
    alpha = 0.7  # tune: 1.0 = pure geometry, 0.0 = pure class-conditional
    pi = pi_geom**alpha * pi_class**(1 - alpha)
    
    return pi
```

The alpha parameter is tunable -- run self_check with different values and pick what maximizes total score.

---

## Part 5: Implementation Blueprint

### What Is Available in `__init__`

```python
def __init__(self, stored_patterns, model_params):
    self.X = stored_patterns          # (K, N) -- all stored patterns
    self.N = stored_patterns.shape[1] # 64
    self.R = model_params['R']        # (N, N) -- the metric matrix
    self.eta = model_params['eta']    # 0.5
    self.beta = model_params['beta']  # 8.0
    
    # Precompute things that don't change per query:
    # (optional) precompute R^{-1} once
    self.R_inv = np.linalg.inv(self.R)
    
    # (optional) precompute Hessians at all stored patterns once
    self.per_pattern_pi = self._precompute_pi()
```

### Precomputing Pi Per Pattern (Best Efficiency)

Since queries almost always resolve to one of K stored patterns, precompute precision for each pattern once in `__init__`:

```python
def _precompute_pi(self):
    """Precompute optimal pi for each stored pattern as attractor."""
    pis = []
    for k in range(len(self.X)):
        x = self.X[k]
        s = np.exp(self.beta * (self.X @ x))
        s -= s.max(); s = np.exp(s); s /= s.sum()
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ D @ self.X)
        H = 0.5 * (H + H.T)
        evals, evecs = np.linalg.eigh(H)
        evals = np.maximum(evals, 1e-6)
        pi = np.sum(evecs**2 / evals[np.newaxis, :], axis=1)
        pis.append(pi)
    return np.array(pis)  # (K, N)

def predict_precision(self, corrupted_query):
    q_norm = corrupted_query / (np.linalg.norm(corrupted_query) + 1e-12)
    k_pred = np.argmax(self.X @ q_norm)
    return self.per_pattern_pi[k_pred]
```

This makes `predict_precision` O(N) -- just a dot product and lookup. All O(K*N^3) work happens once at init.

---

## Part 6: The Harness's Clip + Normalize -- What It Means

From `pcam_model.py`:
```python
def clip_and_normalise(self, pi):
    pi = np.clip(pi, self.pi_min, self.pi_max)  # clip to [0.1, 10.0]
    pi = pi / pi.mean()                           # normalize mean to 1
    return pi
```

**Implications:**
1. Absolute scale doesn't matter. `pi = [1,2,3]` and `pi = [100,200,300]` are identical after normalization.
2. Only ratios matter: `pi_i / pi_j`.
3. Any value below 0.1 becomes 0.1 (heavy clipping at the bottom).
4. Any value above 10.0 becomes 10.0 (heavy clipping at the top).
5. After normalization, values rescale around mean=1.

**Practical consequence**: if your Jacobi `pi_i = 1/H_ii` has a ratio of 100:1 between largest and smallest, the harness clips to 10:1 (ratio of pi_max/pi_min). This limits how much anisotropy you can actually impose. Design your pi so the ratio stays within 0.1 to 10 before normalization.

---

## Part 7: Anti-Gaming Explained

**L1 (seed 42)**: the "canonical" instance. Easy to hardcode for. Zero credit on its own.

**L2 (any seeds)**: for each new seed, `np.random.default_rng(seed)` generates:
- A fresh X (K random patterns + their twins)
- A fresh R (fresh Erdos-Renyi graph Laplacian)
- Fresh test queries

Your adapter is reinstantiated from scratch per seed. Hardcoded numbers from seed 42 will give wrong answers on seed 101.

**L3 (held-out)**: private seeds with larger K, N, and PCA-MNIST data swap. Only the algorithm matters.

**What does "principled" mean?** Your code should have no magic constants that come from inspecting seed 42 output. Every number in your code should be derived from the stored patterns and model params passed in at init time.

---

## Part 8: Expected Performance by Method

| Method | Retrieval delta | Spread reduction | Passes L2? |
|--------|----------------|------------------|------------|
| Dummy (pi=1) | 0 | 1x | N/A (baseline) |
| Query magnitude | +0.01-0.02 | 1-3x | Yes |
| Stored pattern variance | +0.02-0.03 | 2-5x | Yes |
| Class-conditional (cosine) | +0.03-0.05 | 3-8x | Yes |
| Jacobi (H diagonal at query) | +0.03-0.06 | 5-15x | Yes |
| Jacobi (H diagonal at attractor) | +0.04-0.07 | 8-20x | Yes |
| Full diag(H^{-1}) at attractor | +0.05-0.08 | 10-30x | Yes |
| Hybrid (geom + class) | +0.05-0.10 | 15-30x | Yes |

**Target for full 90 pts automated**: delta >= 0.05 AND spread reduction >= 10x on ALL seeds.

---

## Part 9: Key Papers and Theory

### 9.1 Modern Hopfield Networks (Ramsauer 2020) -- arXiv:2008.02217

The PCAM energy is a generalization of Modern Hopfield Networks (also called Dense Associative Memories). The connection:

Modern Hopfield update rule:
```
xi^{t+1} = X^T softmax(beta * X xi^t)
```

PCAM replaces the identity step (`xi - xi`) with `R xi` (graph-regularized step) and adds the precision operator Pi to the gradient. The softmax = attention mechanism connection is the key insight behind "Hopfield Networks is All You Need."

**Storage capacity**: exponential `~2^{N/2}` patterns for the log-sum-exp energy. For N=64, this is ~4 billion. The K=16 patterns used in the bench are nowhere near the capacity limit -- retrieval failure is purely a basin-geometry problem, not a capacity problem.

### 9.2 Van der Sluis 1969 -- Diagonal Equilibration

The theoretical foundation for the Jacobi preconditioner:

**Theorem (Van der Sluis)**: Among all diagonal scalings D > 0, the scaling `D_ii = 1/A_ii` minimizes `omega(D^{1/2} A D^{1/2})` where `omega(M) = trace(M)/N / det(M)^{1/N}` (the ratio of arithmetic to geometric mean of eigenvalues). Furthermore:

```
kappa(D_jacobi^{1/2} A D_jacobi^{1/2}) <= N * kappa_optimal
```

For structured matrices like the PCAM Hessian (strongly diagonal dominant due to R), the actual bound is much tighter -- often within 2-5x of optimal.

### 9.3 Diagonal of H^{-1} -- Why It Is Better

The eigendecomposition of H = Q Lambda Q^T gives:
```
H^{-1} = Q Lambda^{-1} Q^T
[H^{-1}]_ii = sum_k Q[i,k]^2 / lambda_k
```

If we set `pi_i = [H^{-1}]_ii`, we are approximating the full Newton preconditioner `Pi_Newton = H^{-1}` (which gives kappa=1 if not diagonal-constrained) with its diagonal. For the PCAM Hessian where R provides diagonal dominance, this approximation is very good.

### 9.4 Theorem 7 (from problem statement)

"Equilibria shift continuously with precision at a bounded rate."

This means: if you change pi slightly, the attractor moves slightly -- you don't destabilize the system. This guarantees your precision vector is safe to apply even if it's not exactly optimal. No sharp cliffs in the attractor landscape due to pi changes.

### 9.5 Section 6.6 (class-conditional design)

The paper achieves ~2.5% accuracy gain on PCA-MNIST using the class-conditional design. The bench uses synthetic twin-pairs which are designed to mimic this difficulty. The class-conditional approach (cosine-nearest pattern -> set pi based on that pattern) directly reproduces this construction.

---

## Part 10: Complete Best-Practice Implementation

```python
"""
adapters/myteam.py
Strategy: precompute optimal pi per stored pattern at init time
using the eigendecomposition-based diagonal of H^{-1}.
At inference: cosine-nearest lookup + optional query-blend.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from adapter import Adapter


class Engine(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = stored_patterns.astype(np.float64)
        self.K, self.N = self.X.shape
        self.R = np.asarray(model_params['R'], dtype=np.float64)
        self.eta = float(model_params['eta'])
        self.beta = float(model_params['beta'])

        # Precompute optimal pi for each stored pattern used as attractor.
        # Cost: K * O(N^3) = 16 * O(64^3) -- done once, negligible.
        self._pi_per_pattern = self._precompute_all()

    def _hessian_at(self, x: np.ndarray) -> np.ndarray:
        """H evaluated at state x."""
        s = self.X @ x
        s = np.exp(self.beta * (s - s.max()))
        s /= s.sum()
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ D @ self.X)
        return 0.5 * (H + H.T)

    def _optimal_pi(self, H: np.ndarray) -> np.ndarray:
        """diag(H^{-1}) via eigendecomposition -- numerically stable."""
        evals, evecs = np.linalg.eigh(H)
        evals = np.maximum(evals, 1e-7)
        # [H^{-1}]_ii = sum_k evecs[i,k]^2 / evals[k]
        pi = (evecs**2) @ (1.0 / evals)
        return np.maximum(pi, 1e-6)

    def _precompute_all(self) -> np.ndarray:
        pis = np.zeros((self.K, self.N))
        for k in range(self.K):
            H = self._hessian_at(self.X[k])
            pis[k] = self._optimal_pi(H)
        return pis

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=np.float64)

        # Cosine-nearest stored pattern = predicted attractor.
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        k_pred = int(np.argmax(self.X @ q_norm))

        # Return precomputed optimal pi for that attractor.
        return self._pi_per_pattern[k_pred]
```

---

## Part 11: Tuning and Debugging Checklist

**Run the dummy first:**
```bash
cd Anvil-P-E/bench-p04-pcam
python self_check.py --adapter adapters.dummy:DummyAgent --quick
```

Note the baseline numbers (accuracy and spread). Everything is relative to these.

**Test your agent:**
```bash
python self_check.py --adapter adapters.myteam:Engine --quick
```

**Multi-seed full eval:**
```bash
python run.py --adapter adapters.myteam:Engine \
  --seeds 7 13 31 97 211 503 1009 --K 16 --N 64
```

**Diagnosis table from README:**

| delta | spread | diagnosis |
|-------|--------|-----------|
| ~0 | ~1x | agent is effectively pi=1, no modulation working |
| ~0 | high | geometry shaped but retrieval not improved -- pulling toward wrong attractor |
| positive | ~1x | heuristic helps retrieval but not Hessian-grounded -- capped anisotropy |
| positive | high | you cracked it -- both axes firing |
| negative | any | precision hurting retrieval -- likely a clip/normalization bug or wrong sign |

**Common bugs:**
1. **Negative diagonal H entries**: H is not always PD far from attractors. Use `np.maximum(diag_H, 1e-4)` before inverting.
2. **Scale explosion**: if `diag(H^{-1})` has entries > 100x variation, the harness clips most of it away. Your effective ratio is 10:1 max. Check if your pi is getting clipped to flat.
3. **Wrong sign**: the harness clips pi to [0.1, 10.0]. If you accidentally return negative values, they all clip to 0.1 (flat), scoring like dummy.
4. **State leakage between seeds**: each seed creates a new Engine instance. Don't store anything class-level outside of `__init__`.

---

## Part 12: Score Maximization Summary

### To get 70/70 on retrieval (delta >= 0.05):
- Use class-conditional + Hessian at attractor
- The Hessian at the correct attractor tilts the basin toward it specifically
- The twin-pair structure means the "confusable neighbor" basin needs to be made harder to reach

### To get 20/20 on anisotropy (10x spread reduction):
- Use `diag(H^{-1})` (eigendecomposition method) at the predicted attractor
- Jacobi `1/H_ii` gets ~5-15x, eigendecomposition gets ~15-30x
- Evaluate H at the stored pattern (not the noisy query) for clean Hessian

### To avoid penalties:
- Never let delta < 0 on any seed (your pi is hurting retrieval)
- Never let spread <= 1x on any seed (your pi is increasing anisotropy)
- Use epsilon clips everywhere (`np.maximum(..., 1e-6)`)

### For code quality (10/10 manual):
- Clean implementation, comments explaining the theory
- README with the design choices and theory references
- No magic constants, fully general across seeds

---

## Appendix: Quick Reference Formulas

```
# Softmax
s = softmax(beta * X @ a)   -- s_i = exp(beta * x_i^T a) / Z

# Gradient
g = R @ a - eta * X.T @ s

# Hessian
D = diag(s) - outer(s, s)
H = R - eta*beta * X.T @ D @ X

# Jacobi pi (Van der Sluis)
pi = 1 / diag(H)

# Optimal diagonal pi (paper's Theorem F3)
evals, evecs = eigh(H)
pi = (evecs**2) @ (1/evals)   -- which equals diag(H^{-1})

# Spread metric (what checks.py measures)
S = diag(sqrt(pi)) @ H @ diag(sqrt(pi))
kappa = max(eigvals(S)) / min(eigvals(S))

# Score
retrieval_pts = min(70, 70 * delta / 0.05)  if delta > 0 else 0
aniso_pts     = min(20, 20 * log(reduction) / log(10))  if reduction > 1 else 0
```
