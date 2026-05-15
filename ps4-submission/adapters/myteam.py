"""
Masking-Aware Precision Agent -- grounded in the MetaCognition differential decay framework.

Theory (from MetaCognition ICMNAIFINAL paper, Section 2-3):
The PCAM regulatory operator R = A + gamma*L + delta*11^T is exactly the
MetaCognition decay-diffusion operator.  Precision pi is the inverse of the
dimension-specific decay rate alpha in their framework:

    pi_i  =  1 / alpha_i          (high precision = slow decay = trust this dim)

The PCAM dynamics with external input u = query:
    a_{t+1} = a_t + dt * (-Pi * grad E(a_t) + u(t))

For a MASKED dimension (u_i = query_i ~= 0):
  - External input contributes ~0 push
  - Only the gradient (Pi * grad E) drives recovery
  - HIGH pi_i => gradient dominates => pattern knowledge recovers the value fast

For an UNMASKED dimension (u_i = query_i ~= x_k_i):
  - External input correctly anchors the dim to the true value
  - HIGH pi is unnecessary; LOW pi lets input dominate safely

Optimal precision: pi_i proportional to 1/|query_i|
  => set pi_i HIGH where query has no signal (masked/noisy dims)
  => set pi_i LOW where query is reliable (unmasked dims)
  => let the MetaCognition replay term (pattern overlap) recover masked dims

This is the principled implementation of Section 3.5 of the MetaCognition preprint:
"dimensions with good query signal should have high decay alpha (low persistence),
masked dimensions should have low alpha (high persistence = high pi) so pattern
knowledge can reconstruct them via the replay operator."

Twin-pair boost (Section 3.1 of preprint -- Differentiated Decay for Memory Types):
When the two closest patterns are confusable (small cosine gap), the masked
dimensions must more aggressively distinguish them.  We amplify pi in dimensions
that discriminate the top candidate from its nearest rival by a soft correction.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter


class Engine(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X  = stored_patterns.astype(np.float64)          # (K, N)
        self.K, self.N = self.X.shape

        # Tuned epsilon: balances masked-dim boost vs numerical stability.
        # Smaller eps -> stronger pi contrast between masked and unmasked dims.
        # Chosen empirically on public seeds [42, 101, 202, 303, 404].
        self._eps = 0.01

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q   = np.asarray(corrupted_query, dtype=np.float64)
        eps = self._eps

        # --- Core: masking-aware precision ---
        # pi_i = 1 / (|q_i| + eps)
        # Masked dims: |q_i| ~= 0  -> pi large  -> gradient drives recovery
        # Unmasked dims: |q_i| > 0 -> pi small  -> external input anchors correctly
        pi = 1.0 / (np.abs(q) + eps)                          # (N,)

        # --- Twin-pair correction ---
        # Identify the top-2 candidate patterns by cosine similarity.
        # If they are close (small gap), the decision boundary is near; a small
        # discriminative boost in the right dimensions pushes dynamics clearly
        # toward the correct attractor and away from the confusable twin.
        q_norm  = q / (np.linalg.norm(q) + 1e-12)
        cosines = self.X @ q_norm                              # (K,)
        top2    = np.argpartition(cosines, -2)[-2:]
        k1, k2  = (top2[0], top2[1]) if cosines[top2[0]] >= cosines[top2[1]] \
                  else (top2[1], top2[0])

        gap = float(cosines[k1] - cosines[k2])

        if gap < 0.12:
            # Dimensions where the two candidates differ most are the most
            # discriminative. Boost pi there proportionally to how confusable
            # the pair is (weight -> 1 as gap -> 0, weight -> 0 as gap -> 0.12).
            disc   = (self.X[k1] - self.X[k2]) ** 2          # (N,) discriminability
            disc  /= disc.mean() + 1e-12                      # normalise scale
            weight = max(0.0, 1.0 - gap / 0.12)               # 0..1
            pi    *= (1.0 + 0.6 * weight * disc)

        return pi
