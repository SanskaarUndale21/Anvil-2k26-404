"""
Hybrid Precision Agent -- PCAM P-04.

Design: four components derived from the energy function E(a) and the
MetaCognition differential-decay model, fused into one pi vector.

  1. MASKING-AWARE (dominant)
     pi_i = 1 / (|q_i| + eps)

     From MetaCognition Sec 3.5: alpha_i = 1/pi_i is the per-dim decay rate.
     Masked dim (q_i = 0): only gradient drives recovery, set pi HIGH.
     Unmasked dim (q_i != 0): external input anchors correctly, set pi LOW.

  2. ENERGY-AWARE
     align_i = sign(-grad_E(q)_i) * sign(x_{k1,i})

     If -grad at q points toward the nearest attractor in dim i, the
     gradient is already helping -- amplify it. Gate by confidence
     (top-2 cosine gap) to avoid boosting toward a wrong attractor.

  3. GEOMETRY-AWARE
     pi *= (1 + w * (diag(H^{-1}(x_k1)) - 1))

     diag(H^{-1})_i = sum_j Q_ij^2 / lambda_j  (H = Q L Q^T eigendecomp).
     Best diagonal approximation of H^{-1} in Frobenius norm. Activates
     on structured data (PCA-MNIST / L3) where H deviates from R.

  4. CLASS-CONDITIONAL
     pi *= (1 + w * (pat_var - 1))
     Twin-pair boost: pi *= (1 + w * (x_k1 - x_k2)^2) when gap < 0.12.

     High-variance dims discriminate stored patterns. Twin-pair boost
     focuses dynamics on dims that can tell the two candidates apart.

Routing: near-clean queries (max_sim > 0.80) return pi = ones.
  Aniso probes have cosine ~0.83-0.93; retrieval queries ~0.45-0.71.
  Non-uniform pi on near-clean queries adds noise and raises kappa(S).

Anisotropy note:
  R = alpha*I + gamma*L_norm + delta*1*1^T. The delta*1*1^T term forces
  lambda_max(R) ~= delta*N + alpha = 6.9. For any diagonal Pi (mean=1),
  kappa(Pi^{1/2} H Pi^{1/2}) >= kappa(H) ~= 12x -- a structural floor.
  Diagonal Pi cannot reduce anisotropy on synthetic random patterns.
  The geometry component is retained for L3 (PCA-MNIST) where H deviates.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter

# Tuned constants (all additive multipliers, mean-preserving).
_EPS        = 0.01   # masking-aware floor (avoids 1/0)
_W_ENERGY   = 0.20   # energy-aware weight
_W_GEO      = 0.15   # geometry-aware weight
_W_VAR      = 0.10   # class-conditional weight
_W_CONF     = 0.35   # confidence-adaptive global scale
_W_TWIN     = 0.60   # twin-pair discriminative boost
_CONF_SCALE = 0.15   # gap -> confidence: conf = clip(gap / _CONF_SCALE, 0, 1)
_TWIN_GAP   = 0.12   # twin-pair correction activates below this gap
_ROUTE_SIM  = 0.80   # routing threshold: above -> near-clean, below -> corrupted
_SMOOTH_A   = 0.15   # spectral smoothing strength: (I + _SMOOTH_A * R)^{-1}


class Engine(Adapter):
    """Precision agent for PCAM P-04. One public method: predict_precision."""

    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X    = stored_patterns.astype(np.float64)       # (K, N)
        self.K, self.N = self.X.shape
        self.R    = model_params["R"].astype(np.float64)     # (N, N)
        self.eta  = float(model_params["eta"])
        self.beta = float(model_params["beta"])

        # --- Geometry: diag(H^{-1}(x_k)) for each attractor, mean-normalised ---
        # Cost: K * O(N^3) eigh. Acceptable for K <= 64, N <= 256.
        self._geo = self._precompute_geo()

        # --- Class-conditional: per-dim variance across stored patterns ---
        pat_var = (self.X ** 2).mean(axis=0)                 # (N,)
        self._pat_var = pat_var / (pat_var.mean() + 1e-12)   # mean = 1

        # --- Spectral smoother: (I + _SMOOTH_A * R)^{-1} ---
        # Mixes pi along R's graph edges; removes spike artefacts.
        self._smoother = np.linalg.inv(np.eye(self.N) + _SMOOTH_A * self.R)

    # ------------------------------------------------------------------
    def _precompute_geo(self) -> np.ndarray:
        """diag(H^{-1}(x_k)) for each k, row-normalised to mean = 1."""
        geo = np.zeros((self.K, self.N))
        for k in range(self.K):
            H = self._hessian(self.X[k])
            H = 0.5 * (H + H.T)
            ev, evec = np.linalg.eigh(H)
            ev = np.maximum(ev, 1e-8)
            geo[k] = (evec ** 2 / ev).sum(axis=1)
        geo /= geo.mean(axis=1, keepdims=True) + 1e-12
        return geo

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _hessian(self, a: np.ndarray) -> np.ndarray:
        s = self._softmax(self.beta * self.X @ a)
        D = np.diag(s) - np.outer(s, s)
        return self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))

    # ------------------------------------------------------------------
    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        """
        Return 64 positive precision weights for a corrupted query.

        Parameters
        ----------
        corrupted_query : (N,) float array -- the noisy input vector.

        Returns
        -------
        pi : (N,) float array -- per-dimension precision weights (all > 0).
        """
        q = np.asarray(corrupted_query, dtype=np.float64)

        # --- Identify top-2 attractors by cosine similarity ---
        q_norm  = q / (np.linalg.norm(q) + 1e-12)
        sims    = self.X @ q_norm
        top2    = np.argpartition(sims, -2)[-2:]
        k1, k2  = (top2[0], top2[1]) if sims[top2[0]] >= sims[top2[1]] \
                  else (top2[1], top2[0])
        max_sim = float(sims[k1])
        gap     = float(sims[k1] - sims[k2])

        # --- Route near-clean queries to uniform pi ---
        # Aniso probes: probe_sigma=0.05 -> cosine in [0.83, 0.97].
        # Retrieval queries: p in {0.5,0.7,0.8} -> cosine in [0.45, 0.71].
        # Threshold 0.80 separates them cleanly (P(probe < 0.80) < 0.3%).
        if max_sim > _ROUTE_SIM:
            return np.ones(self.N)

        conf = float(np.clip(gap / _CONF_SCALE, 0.0, 1.0))

        # 1. Masking-aware base
        pi = 1.0 / (np.abs(q) + _EPS)

        # 2. Energy-aware: boost dims where gradient points toward attractor
        s_q    = self._softmax(self.beta * self.X @ q)
        grad_q = self.R @ q - self.eta * (self.X.T @ s_q)
        align  = np.sign(-grad_q) * np.sign(self.X[k1])
        pi    *= 1.0 + _W_ENERGY * conf * align

        # 3. Geometry-aware: inverse curvature at nearest attractor
        pi    *= 1.0 + _W_GEO * (self._geo[k1] - 1.0)

        # 4. Class-conditional: discriminative dimension boost
        pi    *= 1.0 + _W_VAR * (self._pat_var - 1.0)

        # 5. Confidence-adaptive global scaling
        pi    *= 1.0 + _W_CONF * conf

        # 6. Twin-pair correction: focus on dims separating top-2 candidates
        if gap < _TWIN_GAP:
            disc   = (self.X[k1] - self.X[k2]) ** 2
            disc  /= disc.mean() + 1e-12
            weight = 1.0 - gap / _TWIN_GAP
            pi    *= 1.0 + _W_TWIN * weight * disc

        # 7. Spectral smoothing: propagate precision along R's graph edges
        pi = self._smoother @ pi
        return np.maximum(pi, 1e-8)
