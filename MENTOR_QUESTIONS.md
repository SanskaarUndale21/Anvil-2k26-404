# Mentor Questions -- P-04 PCAM

Deep questions to understand the problem fully and clarify implementation direction.

---

## Section 1: Understanding the Energy Landscape

**Q1.** The energy has two competing terms -- the R-quadratic and the log-sum-exp. At high noise levels, the query starts far from any stored pattern. Does the R-quadratic term dominate at the start of dynamics and the log-sum-exp take over near convergence? How does beta=8 affect which term wins early vs late?

**Q2.** The Hessian simplifies to approximately R when retrieval is clean (softmax peaks on one pattern). Does that mean near an attractor the geometry is essentially just R? If so, should we be preconditioning R rather than the full Hessian?

**Q3.** R is built from an Erdos-Renyi graph Laplacian -- it changes every seed. Does the graph structure carry semantic meaning (neighboring nodes = similar dimensions), or is it purely a regularizer to make the energy well-behaved?

---

## Section 2: The Precision Operator -- What It Actually Does

**Q4.** The harness clips pi to [0.1, 10.0] and normalizes mean to 1. With this normalization, the maximum ratio between any two pi values is 100:1 before clipping, but after clipping it is at most 10:1 (pi_max/pi_min ratio). Does the clipping severely limit the theory -- specifically, if the optimal pi needs a 30:1 ratio to achieve the paper's ~30x spread reduction, are we hitting a ceiling from the harness constraints?

**Q5.** Precision is applied element-wise: `a_{t+1} = a_t - dt * pi * grad(E)`. This is equivalent to gradient descent in a re-scaled coordinate system. Is there a closed-form relationship between the precision vector and the basin of attraction -- does higher pi in dimension i actually widen or narrow the basin in that direction?

**Q6.** The paper mentions Theorem 7: "equilibria shift continuously with precision at a bounded rate." Does this mean we can use the stored pattern as a proxy for the equilibrium location when computing the Hessian, and the error in our precision is bounded proportionally to how far the actual attractor is from the stored pattern?

---

## Section 3: The Anisotropy Check -- What Gets Measured

**Q7.** The spread check evaluates `Pi^(1/2) H Pi^(1/2)` at the stored pattern itself (clean, not corrupted), but the probe passed to `predict_precision` is a slightly perturbed version (probe_sigma=0.05). Is the intent that our agent should use this near-clean probe to identify which pattern it is, then return precision optimized for that pattern's Hessian? Or should the agent also account for the fact that the actual query might be much noisier?

**Q8.** The check uses `spread = lambda_max / lambda_min` of `Pi^(1/2) H Pi^(1/2)`. Is the theoretical minimum spread achievable with a diagonal Pi anywhere close to 1, or is there a lower bound due to the off-diagonal structure of H?

**Q9.** For the anisotropy score to get 20/20, we need 10x reduction. The paper claims ~30x with an "explicitly aligned construction." Is that construction a diagonal Pi or a full matrix Pi? If it is full matrix, is 10x the realistic ceiling for diagonal-only precision?

---

## Section 4: The Hessian -- Where to Evaluate It

**Q10.** We can evaluate the Hessian at three candidate points: (a) the corrupted query, (b) the cosine-nearest stored pattern, (c) the true equilibrium (unknown until dynamics converge). Is there evidence in the paper that evaluating at the stored pattern (b) is a good proxy for the true equilibrium (c)? What is the approximation error?

**Q11.** When retrieval is hard (high noise, query near the boundary between two attractors), the softmax `s` at the query point spreads over multiple patterns. Does this mean the Hessian at the query point reflects the multi-attractor geometry correctly -- i.e., is the Hessian at a boundary point actually more informative than the Hessian at a single attractor?

---

## Section 5: Class-Conditional Design (Section 6.6 Reference)

**Q12.** Section 6.6 describes `Pi*_class` achieving ~2.5% accuracy gain on PCA-MNIST. What exactly is `Pi*_class` -- is it the Hessian inverse diagonal evaluated at the class centroid, or is it something derived from the class-conditional covariance of the training data?

**Q13.** The synthetic bench uses twin-pairs (parent + perturbed twin) to simulate confusable classes. The paper uses PCA-MNIST. Is the intent that an agent designed for twin-pairs generalizes to PCA-MNIST for L3, or do we need a different approach for the L3 swap?

**Q14.** For twin pairs: parent `x_i` and twin `x_{i+K/2}` are close in pattern space. Would a precision that emphasizes the dimensions where `x_i` and its twin differ most be a principled approach? Is that what the paper's class-conditional construction effectively does?

---

## Section 6: The R Matrix -- Can We Use It Directly?

**Q15.** R is available in `model_params['R']`. The Hessian at a clean attractor is approximately R. Would setting `pi_i = 1/R_ii` (Jacobi preconditioner for R alone) be a reasonable baseline that ignores the pattern-dependent correction? How much of the anisotropy comes from R vs the pattern correction term?

**Q16.** R is built as `alpha*I + gamma*L + delta*11^T`. The Laplacian term L is PSD. Does R being strongly diagonally dominant (alpha=0.5 dominates gamma*L for typical graph degrees) mean the Hessian is also strongly diagonally dominant, making Jacobi preconditioning nearly optimal?

---

## Section 7: Scoring Edge Cases

**Q17.** The penalty for any seed with delta < 0 halves the retrieval score. Is it better to have a conservative agent that consistently gets delta=+0.03 across all seeds (score ~42 pts), or an aggressive agent that gets delta=+0.08 on most seeds but delta=-0.01 on one seed (score ~0 pts after halving)? The answer is obvious but -- is there any grace margin, or is the penalty truly applied on any epsilon-negative delta?

**Q18.** For the anisotropy check, the harness returns `None` for a pattern when the Hessian is not positive definite (`eig_H.min() <= 0`). If several patterns hit this case (e.g. at hard seeds), they are excluded from the average. Does this mean the anisotropy score is computed on a variable number of patterns per seed? Could an adversarial agent try to force many `None` returns to game the average?

---

## Section 8: The L3 Evaluation (PCA-MNIST Swap)

**Q19.** The L3 evaluation swaps in PCA-MNIST with mask noise. The PCA transformation projects MNIST images to a lower-dimensional space -- does this mean the stored patterns are PCA components (basis vectors), or actual projected MNIST digit images? This affects whether the twin-pair intuition carries over.

**Q20.** The paper's Section 6.6 uses class-conditional precision on PCA-MNIST. In the L3 evaluation, will we have access to which class each stored pattern belongs to (the digit label), or only the raw pattern vectors? If only raw vectors, how are we expected to identify classes?

---

## Section 9: Direct Solution Direction Hints

**Q21.** The README hints at three approaches: variance-based, class-conditional, and geometry-aware (Hessian). The problem says "the best agents won't be the most complex -- they'll be the ones that actually use the per-direction control PCAM provides." Does this hint that the geometry-aware (Hessian-based) approach is expected to dominate, or is there a simpler signal in the query that is sufficient?

**Q22.** One forward pass per query means no iterative refinement. Is it intended/expected that participants pre-compute anything in `__init__` (e.g., per-pattern Hessians) and then do a fast lookup at query time? Or is computing the Hessian fresh for each query at inference time the expected pattern?

**Q23.** The harness clips and normalizes pi before applying. Is there any benefit to returning pi values near the boundaries [0.1, 10.0] -- e.g., saturating some dimensions to 10 while others drop to 0.1 -- compared to a more moderate range that does not hit the clips?

**Q24.** Theorem F3 says precision rescales convergence rates by eigenvalues of Pi*H. If we set Pi = diag(H^{-1}) (the diagonal of H inverse, not 1/diag(H)), does this approximately achieve uniform convergence in all directions? Is `diag(H^{-1})` vs `1/diag(H)` a meaningful distinction for this problem's scale (N=64, K=16)?

---

## Section 10: Practical / Logistics

**Q25.** Is there a time limit per query at inference, or just a total wall-clock limit for the full evaluation? Asking because computing a 64x64 eigendecomposition per query is ~0.2ms, which over 750 queries = 150ms -- negligible vs the dynamics runtime. Is there any reason to avoid this?

**Q26.** The benchmark says "same harness the judges run" -- does the judge's evaluation use exactly the same seed set and parameters as `run.py` default, or are the L3 seeds truly secret and run with different K/N?

**Q27.** The README says "optional: a short note tying your design back to the paper." Is the code quality score (10 pts) purely about code cleanliness, or does the design note count toward it? Should we include proofs or derivations, or just a clear explanation of the algorithm?
