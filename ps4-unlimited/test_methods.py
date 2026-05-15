"""
Empirical test of every suggested anisotropy method.
Measures actual kappa(Pi^{1/2} H Pi^{1/2}) at stored patterns for each method.
No bias -- if it works, it works.
"""
import numpy as np
from data import make_patterns
from pcam_model import PCAMModel, build_default_R

seed = 42
X = make_patterns(K=16, N=64, seed=seed)
R = build_default_R(N=64, seed=seed)
model = PCAMModel(X, R)
N, K = 64, 16

def kappa_at_pattern(model, pi, pattern):
    """kappa(Pi^{1/2} H Pi^{1/2}) at pattern -- exactly what bench measures."""
    pi = np.clip(pi, 0.1, 10.0)
    pi = pi / pi.mean()
    H = model.hessian(pattern)
    H = 0.5 * (H + H.T)
    if np.linalg.eigvalsh(H).min() <= 0:
        return None
    pi_sqrt = np.sqrt(pi)
    S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[eigs > 1e-9]
    return float(eigs.max() / eigs.min()) if len(eigs) >= 2 else None

def mean_kappa(model, pi_fn):
    """Average kappa over all 16 stored patterns using probe = pattern + 0.05*noise."""
    rng = np.random.default_rng(seed)
    kappas = []
    for k in range(K):
        probe = X[k] + rng.standard_normal(N) * 0.05
        probe = probe / (np.linalg.norm(probe) + 1e-12)
        pi = pi_fn(probe, k)
        kappa = kappa_at_pattern(model, pi, X[k])
        if kappa is not None:
            kappas.append(kappa)
    return np.mean(kappas)

baseline_kappa = mean_kappa(model, lambda q, k: np.ones(N))
print(f"Baseline (pi=ones):         {baseline_kappa:.4f}x  (target: reduce this)")
print(f"Theoretical floor:          {baseline_kappa:.4f}x  (cannot go lower with diag Pi)")
print()

results = {}

# ---- Method 1: Global pattern covariance diagonal ----
cov_global = np.cov(X.T)              # (64,64)
diag_cov = np.diag(cov_global)        # (64,)
pi_cov = 1.0 / (diag_cov + 1e-6)
results["M1 global inv-var"] = mean_kappa(model, lambda q, k: pi_cov)

# ---- Method 2: Local covariance (8 nearest neighbors) ----
def m2_local_cov(q, k):
    sims = X @ q
    neighbors = X[np.argsort(sims)[-8:]]
    cov = np.cov(neighbors.T)
    return 1.0 / (np.diag(cov) + 1e-6)
results["M2 local cov (8-NN)"] = mean_kappa(model, m2_local_cov)

# ---- Method 3: Full eigen whitening -> extract diagonal ----
eigvals_c, eigvecs_c = np.linalg.eigh(cov_global)
eigvals_c = np.maximum(eigvals_c, 1e-8)
Pi_full = eigvecs_c @ np.diag(1.0 / eigvals_c) @ eigvecs_c.T
pi_eig = np.diag(Pi_full)
results["M3 eig-whiten diag"] = mean_kappa(model, lambda q, k: pi_eig)

# ---- Method 4: Jacobian equalization ----
def m4_jacobian(q, k):
    p = np.exp(8 * X @ q); p /= p.sum()
    J = (X * p[:, None]).T @ (X * p[:, None])   # (N,N) weighted gram
    eig_j = np.linalg.eigvalsh(J)
    # precision inversely proportional to row-norm of J
    dominance = np.diag(J)
    return 1.0 / (dominance + 1e-8)
results["M4 jacobian equalize"] = mean_kappa(model, m4_jacobian)

# ---- Method 5: SVD / PC strength ----
U, S_svd, Vt = np.linalg.svd(X, full_matrices=False)
pc_strength = np.sum(np.abs(Vt), axis=0)   # (N,)
pi_svd = 1.0 / (pc_strength + 1e-8)
results["M5 SVD pc-strength"] = mean_kappa(model, lambda q, k: pi_svd)

# ---- Method 6: Correlation decoupling ----
corr = np.corrcoef(X.T)
dominance_corr = np.sum(np.abs(corr), axis=1)
pi_corr = 1.0 / (dominance_corr + 1e-6)
results["M6 corr decoupling"] = mean_kappa(model, lambda q, k: pi_corr)

# ---- Method 7: Laplacian smoothing (correlation-based) ----
W = np.abs(corr)
D_deg = np.diag(W.sum(axis=1))
L_corr = D_deg - W
gamma = 0.1
smooth_mat = np.eye(N) + gamma * L_corr
pi_lap = np.linalg.solve(smooth_mat, pi_cov)
results["M7 lap-smooth (global cov)"] = mean_kappa(model, lambda q, k: pi_lap)

# ---- Method 8: Hessian diagonal approach (diag of H^{-1} at attractor) ----
def m8_hessian_diag(q, k):
    H = model.hessian(X[k])
    H = 0.5 * (H + H.T)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.maximum(eigvals, 1e-8)
    return (eigvecs**2 / eigvals[None, :]).sum(axis=1)
results["M8 diag(H^{-1}) exact"] = mean_kappa(model, m8_hessian_diag)

# ---- Method 9: Curvature = var across patterns ----
curvature = np.var(X, axis=0)
pi_curv = 1.0 / (np.sqrt(curvature) + 1e-8)
results["M9 sqrt-var inverse"] = mean_kappa(model, lambda q, k: pi_curv)

# ---- Method 14: Soft whitening (cov^0.5 inverse) ----
pi_soft = 1.0 / (np.sqrt(diag_cov) + 1e-6)
results["M14 soft-whiten"] = mean_kappa(model, lambda q, k: pi_soft)

# ---- Best combo: local cov -> eigen -> diag -> Laplacian smooth ----
def m_combo(q, k):
    sims = X @ q
    neighbors = X[np.argsort(sims)[-8:]]
    cov_l = np.cov(neighbors.T)
    ev, evec = np.linalg.eigh(cov_l)
    ev = np.maximum(ev, 1e-8)
    Pi_l = evec @ np.diag(1.0 / ev) @ evec.T
    pi_l = np.diag(Pi_l)
    pi_l = np.linalg.solve(smooth_mat, pi_l)
    return pi_l
results["M_COMBO best stack"] = mean_kappa(model, m_combo)

# ---- Print results ----
print(f"{'Method':<35}  {'kappa':>8}  {'vs baseline':>12}  {'better?':>8}")
print("-" * 68)
for name, kappa in sorted(results.items(), key=lambda x: x[1]):
    vs = kappa / baseline_kappa
    better = "YES" if kappa < baseline_kappa * 0.99 else ("same" if kappa < baseline_kappa * 1.01 else "WORSE")
    print(f"{name:<35}  {kappa:>8.4f}  {vs:>11.4f}x  {better:>8}")

print()
print(f"Baseline:  {baseline_kappa:.4f}x")
print(f"Best found: {min(results.values()):.4f}x  ({min(results, key=results.get)})")
print()
print("MATH PROOF CHECK:")
print(f"  alpha + delta*N = 0.5 + 0.1*64 = {0.5 + 0.1*64:.1f}  (locked max eigenvalue)")
print(f"  min eigenvalue of R = {np.linalg.eigvalsh(R).min():.4f}")
print(f"  kappa(R) = {np.linalg.eigvalsh(R).max() / np.linalg.eigvalsh(R).min():.4f}")
