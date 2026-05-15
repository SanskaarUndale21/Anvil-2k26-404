# P-04 PCAM Precision Agent -- Submission

## Setup

```bash
cd ps4-submission
pip install numpy
python self_check.py --adapter adapters.myteam:Engine --quick
```

Full multi-seed evaluation:

```bash
python run.py --adapter adapters.myteam:Engine \
  --seeds 7 13 31 97 211 503 1009 --K 16 --N 64
```

## Design

### Theoretical Grounding

The PCAM regulatory operator `R = A + gamma*L + delta*11^T` is identical to the MetaCognition differential decay operator (ICMNAIFINAL, Section 2.3). In that framework, the diagonal matrix A encodes dimension-specific decay rates `alpha_i`. Precision is the inverse decay:

```
pi_i = 1 / alpha_i
```

High precision in dimension i means slow decay -- the system "trusts" dimension i to be driven by internal pattern knowledge rather than external signal.

### Key Principle: Masking-Aware Precision

The PCAM dynamics with external input (query) are:

```
a_{t+1} = a_t + dt * (-Pi * grad E(a_t) + query)
```

Two cases at inference time:

**Masked dimension** (`query_i ~= 0`, no signal):
- External input contributes nothing
- Only the gradient term `Pi * grad E` drives recovery
- Setting `pi_i` HIGH lets the gradient (which encodes pattern knowledge through the softmax attention) recover the dimension fast
- This implements the MetaCognition replay operator: "pattern knowledge recovers what the input cannot provide"

**Unmasked dimension** (`query_i ~= x_k_i`, reliable signal):
- External input already anchors the dimension to the correct value
- Low `pi_i` lets the input dominate safely
- No need for aggressive gradient following

**The formula:**

```python
pi_i = 1 / (|query_i| + eps)
```

- Where `|query_i|` is small (masked, corrupted) -> pi large -> gradient drives
- Where `|query_i|` is large (unmasked, reliable) -> pi small -> input anchors

This is the principled choice from MetaCognition Section 3.5 (Category-Specific Decay): different memory dimensions should have different decay rates depending on their signal quality.

### Twin-Pair Correction

Stored patterns come in confusable twin pairs (parent + small perturbation). When the top-2 cosine-similarity candidates are close (gap < 0.12), the decision boundary is nearby. We amplify precision in dimensions that discriminate the two candidates:

```python
disc = (x_k1 - x_k2)^2   # discriminability per dimension
pi  *= (1 + weight * disc) # weight = 1 - gap/0.12
```

This biases the dynamics toward the more discriminative features when the query is ambiguous, reproducing the class-conditional gain described in Section 6.6 of the PCAM paper.

### Why This Works (from first principles)

The softmax attention in `grad E = Ra - eta * X^T softmax(beta Xa)` computes a weighted pull toward stored patterns. In masked dimensions, the external input is 0 and the gradient is the only recovery signal. High pi in those dimensions gives the gradient more "authority," letting pattern knowledge fill in what corruption removed.

This is effectively implementing **inference-time attention reweighting**: trust dimensions proportional to how much the input has preserved them, and let the model's memory recover everything else.

### Anisotropy Note

The anisotropy metric measures kappa of `Pi^{1/2} H Pi^{1/2}` at stored patterns. At attractors, `H ~= R = alpha*I + gamma*L + delta*11^T`. The maximum eigenvalue of R is `alpha + delta*N` in the all-ones direction (from the global inhibition term `delta*11^T`). For diagonal Pi with mean=1, the Rayleigh quotient at the all-ones direction equals `lambda_max(R)` regardless of pi, making lambda_max(S) >= lambda_max(R) always. The spread floor is thus determined by the R structure, not by pi. We focus engineering effort on retrieval where signal is recoverable.

## Files

```
adapters/myteam.py   -- the agent (Engine class)
adapter.py           -- abstract base (frozen)
pcam_model.py        -- PCAM dynamics (frozen)
data.py              -- pattern + query generation (frozen)
checks.py            -- retrieval + anisotropy metrics (frozen)
harness.py           -- multi-seed orchestration (frozen)
run.py               -- full evaluation CLI
self_check.py        -- local iteration CLI
```

## Dependencies

```
numpy >= 1.20
```

No GPU required. Full 5-seed evaluation completes in ~3-5 minutes on CPU.
