"""
PCAM Precision Control Adapter -- Confidence-Gated Class-Conditional Approach

Strategy:
  1. Compute softmax similarities with stored patterns (beta=0.2)
  2. Confidence = how peaked is the distribution? (0=uniform, 1=peaked)
  3. High confidence: aggressively boost dims aligned with nearest pattern
  4. Low confidence (<= threshold): return exactly uniform to avoid regression

Generalizes across seeds: all logic operates on the dynamically provided
stored_patterns and the query -- no hardcoded seed-specific values.
"""

import numpy as np

_BETA_CONF = 0.2        # softmax temperature for class identification
_CONF_THRESHOLD = 0.3   # below this, fall back to uniform (no regression guarantee)
_BOOST = 5.0            # multiplier on class_signal for high-confidence queries


class Engine:
    def __init__(self, stored_patterns: np.ndarray, **kwargs):
        self.patterns = stored_patterns.astype(np.float64)
        self.K, self.N = self.patterns.shape
        self._1_over_K = 1.0 / self.K

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = corrupted_query.astype(np.float64)

        # --- Soft class identification ---
        sims = self.patterns @ q                          # (K,) raw dot products
        logits = _BETA_CONF * sims
        exp_l = np.exp(logits - logits.max())
        soft = exp_l / exp_l.sum()                        # (K,) softmax weights

        # --- Confidence: how peaked is the softmax? ---
        max_w = soft.max()
        confidence = (max_w - self._1_over_K) / (1.0 - self._1_over_K)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Below threshold: return uniform (guaranteed no regression)
        if confidence <= _CONF_THRESHOLD:
            return np.ones(self.N)

        # Remap confidence to [0,1] above threshold
        alpha = (confidence - _CONF_THRESHOLD) / (1.0 - _CONF_THRESHOLD)

        # --- Per-dim alignment with weighted pattern mixture ---
        # class_alignment[i] in [-1, +1]: how much does q agree with weighted average
        class_alignment = (soft[:, None] * (self.patterns * q[None, :])).sum(axis=0)
        # Map to [0, 1]: +1 -> 1.0 (agree = reliable), -1 -> 0.0 (disagree = likely flipped)
        class_signal = (class_alignment + 1.0) / 2.0

        # --- Precision: boost aligned dims, suppress misaligned dims ---
        # _BOOST amplifies the contrast: high-confidence correct dims get ~BOOST precision
        # before normalization; flipped dims get ~0
        precision = alpha * (_BOOST * class_signal) + (1.0 - alpha) * 1.0 + 0.05

        precision = np.clip(precision, 0.1, 10.0)
        return precision / precision.mean()
