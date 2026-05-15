# PS4 -- PCAM Precision Control Agent

Inference-time agent that computes 64 per-dimension precision weights for a modern Hopfield network, steering retrieval toward correct attractors under noise without any retraining.

## Approach: Confidence-Gated Class-Conditional Precision

**Core insight:** At low noise (p=0.7, 0.8), the corrupted query retains enough signal to identify the target pattern. We use this to boost dimensions aligned with the target and suppress likely-corrupted ones.

At high noise (p=0.5), queries are near-random and class identification is unreliable. We fall back to uniform precision to guarantee no regression.

```
corrupted_query
  -> softmax(beta_conf * P @ q)          # soft weights over stored patterns
  -> confidence = (max_weight - 1/K) / (1 - 1/K)
  -> if confidence <= 0.3: return uniform precision (no-harm guarantee)
  -> else:
       alpha = (confidence - 0.3) / 0.7
       class_signal[i] = (weighted_alignment[i] + 1) / 2  # 1=reliable, 0=likely flipped
       precision = alpha * (BOOST * class_signal) + (1-alpha) * 1.0 + floor
       return clip(precision, 0.1, 10.0) / mean(precision)
```

**BOOST = 5.0** -- amplifies the contrast between reliable and unreliable dims.

## Local test results (5 seeds)

```
Seed   42 | Base: 0.8083 | Agent: 0.8160 | Delta: +0.0078 | OK
Seed    7 | Base: 0.8137 | Agent: 0.8221 | Delta: +0.0085 | OK
Seed   13 | Base: 0.8072 | Agent: 0.8134 | Delta: +0.0063 | OK
Seed   99 | Base: 0.8168 | Agent: 0.8187 | Delta: +0.0019 | OK
Seed  137 | Base: 0.8091 | Agent: 0.8134 | Delta: +0.0043 | OK
Mean delta: +0.0057 -- PASS, all seeds, no regressions
```

Per-query anisotropy spread (high-confidence queries): ~7x

## Setup

```bash
pip install -r requirements.txt
```

## Run official harness

```bash
git clone https://github.com/Sauhard74/Anvil-P-E
cd Anvil-P-E/bench-p04-pcam

# Copy adapter
cp ../ps4/adapters/myteam.py adapters/

# Quick test
python self_check.py --adapter adapters.myteam:Engine --quick

# Full test
python self_check.py --adapter adapters.myteam:Engine
```

## Run local test

```bash
python local_test.py          # 5 seeds, 750 queries each
python local_test.py --quick  # 2 seeds, 150 queries each
```

## Optional: Train MLP (offline neural approach)

```bash
# Generate synthetic data
python train/generate_data.py

# Train small MLP (64->256->128->64, pure NumPy, no GPU needed)
python train/train_mlp.py

# Weights saved to models/mlp_weights.npz
```

## Anti-gaming robustness

| L1 (seed 42) | Returns positive delta on all test seeds |
| L2 (any seed) | All logic operates on dynamically provided `stored_patterns` -- no seed hardcoding |
| L3 (higher K/N, MNIST swap) | Class-conditional approach is structure-agnostic; confidence gating prevents regression on novel distributions |

## Dependencies

Only numpy -- no PyTorch, no scikit-learn, no GPU required.
