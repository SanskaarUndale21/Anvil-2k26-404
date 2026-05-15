# ANVIL Hackathon -- PS3 & PS4 Deep Dive

> 24h, 1-4 ppl, real execution over simulation. No half-ass demos.

---

## Table of Contents

- [PS3 -- Multi-Agent Autonomy Pipeline](#ps3----multi-agent-autonomy-pipeline)
  - [What they actually want](#what-they-actually-want)
  - [Scoring breakdown](#ps3-scoring)
  - [Implementation approaches](#ps3-implementation-approaches)
  - [Unique angles](#ps3-unique-angles)
  - [Stack recommendations](#ps3-stack)
  - [Demo checklist](#ps3-demo-checklist)
- [PS4 -- PCAM Precision Control Agent](#ps4----pcam-precision-control-agent)
  - [What they actually want](#what-they-actually-want-1)
  - [Scoring breakdown](#ps4-scoring)
  - [Implementation approaches](#ps4-implementation-approaches)
  - [Unique angles](#ps4-unique-angles)
  - [Anti-gaming layers](#anti-gaming-layers)
  - [Quickstart](#ps4-quickstart)

---

## PS3 -- Multi-Agent Autonomy Pipeline

### What they actually want

Build a system where multiple specialized agents solve a real problem **without** you clicking next. A trigger fires, agents hand off work to each other, tools produce actual side-effects (emails sent, files written, APIs called), something runs async/long, and at the end meaningful work is done -- not just a plan.

**Kill criteria:** If a human has to intervene mid-run, you lose points. If agents just plan and never execute, you lose points.

---

### PS3 Scoring

| Axis | Weight |
|------|--------|
| Problem relevance + UX | 20% |
| Autonomous execution | 25% |
| Multi-agent workflow quality | 20% |
| Tooling + integrations | 15% |
| Demo video | 10% |
| Architecture + code | 10% |
| Omium SDK traces (bonus) | +10% |

**Grading flow:** Video is primary -> live reliability check -> code review -> 15-min Q&A on decomposition.

Take the +10% bonus (Omium SDK). It's free points if you instrument properly.

---

### PS3 Implementation Approaches

#### Approach A -- Research Automation Agent (Safe, polished)

**Use case:** User drops a topic; system autonomously researches, synthesizes, and delivers a structured report.

```
Trigger (webhook/UI)
  -> Planner Agent  (decomposes query into sub-tasks)
    -> Search Agent x3 (parallel web search per sub-task)
    -> Extractor Agent (pulls key facts from each result)
  -> Synthesizer Agent (merges + deduplicates findings)
  -> Writer Agent (formats final report)
  -> Delivery Agent (emails/Slack/saves to Drive)
```

**Why it wins:** Clear agent boundaries, visible handoffs, real side-effects (email/Slack), parallelism shows async fanout.

**Tools needed:** Tavily/Serper for web search, SMTP/Slack API for delivery, Redis/Postgres for task state.

---

#### Approach B -- Operations Triage Pipeline (High impact)

**Use case:** Incoming support tickets / GitHub issues / alerts get triaged, categorized, assigned, and partially resolved autonomously.

```
Webhook (new issue/alert fires)
  -> Classifier Agent (severity, category, owner)
  -> Context Agent (fetches related past issues, runbooks)
  -> Resolver Agent (attempts auto-fix or drafts response)
  -> Escalation Agent (if unresolved -> pings human with full context)
  -> Logger Agent (writes to Notion/Linear/JIRA)
```

**Why it wins:** Webhook-driven (real trigger), long-running (wait for fix attempt), verifiable side-effects (JIRA tickets, Slack pings), failure handling baked in via Escalation Agent.

---

#### Approach C -- Recruiting Pipeline Agent (Unique vertical)

**Use case:** Given a JD, system finds candidates, enriches profiles, drafts outreach, schedules calls.

```
Trigger (new JD posted)
  -> JD Parser Agent (extracts must-have/nice-have skills)
  -> Sourcer Agent (LinkedIn/GitHub search via APIs)
  -> Scorer Agent (parallel scoring of each candidate)
  -> Outreach Agent (drafts personalized emails per candidate)
  -> Scheduler Agent (Calendly API to propose slots)
  -> CRM Agent (writes all activity to Airtable/Notion)
```

**Real side-effects:** Emails drafted/sent, Calendly invites, CRM updated.

---

#### Approach D -- Market Intelligence Pipeline (Impressive scope)

**Use case:** Monitor competitors, track news, synthesize weekly brief, push to Slack/email.

```
Cron trigger (or webhook from RSS)
  -> Discovery Agent (finds new mentions across sources)
  -> Relevance Agent (filters noise)
  -> Analyst Agent (extracts signals, trends, threats)
  -> Comparison Agent (diffs against last week's brief)
  -> Publisher Agent (formats + delivers brief)
```

**Async component:** Long-running multi-source scrape with fanout pattern.

---

#### Approach E -- Legal Document Pipeline (Niche, memorable)

**Use case:** Upload contract -> agents extract clauses, flag risks, compare to standard templates, produce redline suggestions.

```
Upload trigger
  -> Chunker Agent (splits doc into logical sections)
  -> Extractor Agent x N (parallel clause extraction per section)
  -> Risk Assessor Agent (flags non-standard clauses)
  -> Comparator Agent (diffs against standard templates)
  -> Reporter Agent (generates redline PDF + summary)
```

**Why it wins:** Complex document -> structured output, parallel agents on chunks, verifiable output artifact.

---

### PS3 Unique Angles

1. **Failure recovery loop** -- Agent detects its own failure, retries with different strategy, logs the recovery. Judges love seeing autonomous error handling.

2. **Meta-planner** -- A planner agent that can spawn sub-agents dynamically based on task complexity. Shows genuine autonomy, not hardcoded pipelines.

3. **Reflection layer** -- After completion, a Critic Agent reviews the output quality and triggers a re-run if quality score is below threshold. Self-improving loop.

4. **Streaming observability dashboard** -- Live UI showing each agent's state (idle/running/done/failed) in real time. Huge demo value.

5. **Multi-modal trigger** -- Webhook that accepts image uploads (screenshot of a bug, photo of a document) and kicks off the pipeline from visual input.

6. **Human-in-loop escalation with timeout** -- If an agent can't decide, it pauses and pings human via Slack with a 10-min timeout. If no response, it proceeds with best-guess. Shows graceful degradation.

---

### PS3 Stack

| Layer | Options |
|-------|---------|
| Agent framework | LangGraph, CrewAI, AutoGen, custom async Python |
| LLM | Claude Sonnet 4.6, GPT-4o, Gemini 2.0 |
| Web search | Tavily, Serper, Brave Search API |
| Task queue | Celery + Redis, BullMQ (Node), Temporal |
| State store | Redis, Postgres, Supabase |
| Webhooks | FastAPI + ngrok (local), Railway/Fly.io (deployed) |
| Observability | Omium SDK (mandatory for bonus), Langfuse, LangSmith |
| Delivery | Resend (email), Slack Bolt SDK, Notion API |

**LangGraph** is the strongest choice -- built-in state management, conditional edges, human-in-loop checkpoints, async execution.

---

### PS3 Demo Checklist

- [ ] Trigger fires (webhook/button) -- show it live
- [ ] At least 2 visible agent handoffs on screen
- [ ] One tool call with verifiable side-effect (email received, file exists, API confirmed)
- [ ] Show async/parallel component (2 agents running simultaneously)
- [ ] Show failure recovery OR long-running component finishing
- [ ] Omium dashboard showing traces (bonus)
- [ ] 5-min video, 3-page arch doc

---

---

## PS4 -- PCAM Precision Control Agent

### What they actually want

Modern Hopfield networks normally have **one global temperature knob**. PCAM gives you **64 individual knobs**, one per dimension of the state vector. You write an agent that decides all 64 values at inference time (no retraining) to steer pattern retrieval toward the correct attractor even when the query is noisy.

You implement one function:

```python
def predict_precision(corrupted_query: ndarray[64]) -> ndarray[64]:
    """Return 64 positive precision weights. Mean is normalized to 1.0."""
```

The harness runs the frozen PCAM model with your weights and grades accuracy + anisotropy spread.

**Kill criteria:** Mean delta accuracy <= 0 across seeds = zero retrieval score. Any seed with regression = halved score. Must consistently beat the identity baseline (all ones).

---

### PS4 Scoring

| Component | Points | What to beat |
|-----------|--------|-------------|
| Retrieval accuracy | 70 | Beat Pi=I baseline by >=0.05 delta for full marks |
| Anisotropy spread reduction | 20 | Beat 1x baseline; paper achieves ~30x; full marks at 10x |
| Code quality | 10 | Reproducibility, README, clarity |

**Seeds:** 5 seeds in full eval, 2 in quick mode. L2 uses ANY integer seed -- your method must generalize, not memorize.

---

### PS4 Implementation Approaches

#### Approach A -- Variance-Based Precision (Baseline good approach)

**Intuition:** Dimensions where the query is noisier should get lower precision (less trust). Dimensions that look clean get higher precision.

```python
def predict_precision(corrupted_query):
    # Estimate per-dim reliability from deviation from binary poles
    deviation = np.abs(np.abs(corrupted_query) - 1.0)  # 0=clean, 1=max noise
    reliability = 1.0 - deviation
    precision = reliability + epsilon
    # Normalize to mean=1
    return precision / precision.mean()
```

**Why it works:** Noisy dims get downweighted, clean dims steer retrieval. Simple, generalizes across seeds.

**Limitation:** Doesn't use knowledge of stored patterns.

---

#### Approach B -- Class-Conditional Precision (Paper's approach, ~2.5% gain)

**Intuition:** First guess which pattern class the query belongs to, then set precision to match the typical variance profile of that class.

```python
def predict_precision(corrupted_query):
    # Step 1: rough classification via cosine sim to stored patterns
    sims = stored_patterns @ corrupted_query
    best_class = np.argmax(np.abs(sims))
    target_pattern = stored_patterns[best_class]

    # Step 2: per-dim alignment score
    alignment = corrupted_query * target_pattern  # +1 = aligned, -1 = opposed
    precision = np.clip(alignment, 0.1, None) + epsilon
    return precision / precision.mean()
```

**Why it works:** Aligned dims get boosted, misaligned (likely corrupted) dims get suppressed. Matches the paper's Section 6.6 design.

**Needs:** Access to stored patterns (available in the adapter context).

---

#### Approach C -- Geometry-Aware (Hessian curvature)

**Intuition:** At the nearest attractor, dimensions with high curvature converge fast -- they don't need high precision. Flat dimensions need boosting to not get stuck.

```python
def predict_precision(corrupted_query):
    # Approximate Hessian diagonal at query point
    # H_ii ~ sum_k w_ik^2 * (1 - tanh^2(beta * h_i))
    # For Hopfield: related to pattern outer products
    hessian_diag = np.sum(stored_patterns ** 2, axis=0)  # crude approx
    # Low curvature dims need more precision to converge
    precision = 1.0 / (hessian_diag + epsilon)
    return precision / precision.mean()
```

**Why it's unique:** Grounded in the Theorem F3 eigenvalue relationship. Judges who read the paper will recognize this.

---

#### Approach D -- Trained MLP (Neural approach, potentially strongest)

**Intuition:** Learn a mapping from corrupted query -> optimal precision vector using synthetic training data generated from the stored patterns.

```python
# Training (offline, done before submission)
# Generate: take clean pattern, add noise at levels p={0.5,0.7,0.8}
# Label: optimal precision = ones that maximize retrieval accuracy
# Train: small MLP (64 -> 128 -> 64) with ReLU + softplus output

class Engine:
    def __init__(self):
        self.mlp = load_trained_model("precision_mlp.npz")

    def predict_precision(self, corrupted_query):
        raw = self.mlp.forward(corrupted_query)
        return raw / raw.mean()
```

**Training data generation:**
```python
for pattern in stored_patterns:
    for p in [0.5, 0.7, 0.8]:
        mask = np.random.rand(64) < p
        noisy = pattern.copy()
        noisy[mask] *= -1
        # optimal target: high precision where aligned, low where flipped
        target_precision = (noisy * pattern > 0).astype(float) + 0.1
        target_precision /= target_precision.mean()
        dataset.append((noisy, target_precision))
```

**Why it might win:** Can capture non-linear relationships. But watch out -- L2/L3 use different patterns and seeds, so the MLP must generalize, not overfit to seed-42 patterns.

**Risk:** If patterns at L3 are drastically different (higher K, MNIST swap), a purely trained model may fail. Hybrid is safer.

---

#### Approach E -- Hybrid: Class-Conditional + Noise Variance (Recommended)

Combine the robustness of variance-based with the precision of class-conditional:

```python
def predict_precision(corrupted_query):
    # Variance signal
    deviation = np.abs(np.abs(corrupted_query) - 1.0)
    variance_signal = 1.0 - deviation  # high = reliable dim

    # Class signal
    sims = stored_patterns @ corrupted_query
    best_idx = np.argmax(np.abs(sims))
    target = stored_patterns[best_idx]
    class_signal = (corrupted_query * target + 1) / 2  # [0,1]

    # Combine
    precision = (0.4 * variance_signal + 0.6 * class_signal) + 0.1
    return precision / precision.mean()
```

**Why this is the recommended approach:**
- Variance signal handles unknown noise patterns (L3 robustness)
- Class signal uses stored pattern knowledge (accuracy)
- Weights tunable, no GPU needed, fast, generalizes

---

#### Approach F -- Iterative Refinement Agent

Run multiple passes of precision estimation, refining each time:

```python
def predict_precision(corrupted_query, n_iters=3):
    precision = np.ones(64)  # start flat
    for _ in range(n_iters):
        # Partial PCAM step with current precision
        h = (stored_patterns.T * precision) @ (stored_patterns @ corrupted_query)
        partial_state = np.tanh(h)
        # Recompute precision based on partial convergence
        alignment = partial_state * corrupted_query
        precision = np.clip(alignment + 1, 0.1, 2.0)
        precision /= precision.mean()
    return precision
```

**Why unique:** Treats precision estimation as an iterative optimization, not a one-shot prediction. The agent refines its own estimate. Note: check if harness allows calling stored pattern ops inside predict_precision.

---

### PS4 Unique Angles

1. **Uncertainty quantification** -- Use ensemble of precision predictors, take median. Reduces variance across seeds.

2. **Temperature annealing schedule** -- Start with high uniform precision, anneal down dims that conflict with class estimate. Simulated annealing for precision.

3. **Attention-like mechanism** -- Treat stored patterns as keys, query as the query, output precision as attention-weighted pattern features. Direct parallel to transformer attention.

4. **Dimension clustering** -- PCA the stored patterns, group correlated dims, assign shared precision per cluster. Reduces 64 independent problems to K cluster problems.

5. **Bayesian precision** -- Model each dim's true value as Gaussian, use corrupted query to compute posterior mean, set precision proportional to posterior certainty.

---

### Anti-Gaming Layers

| Layer | What it tests | How to survive |
|-------|--------------|----------------|
| L1 Canonical | Seed 42 fixed patterns | Must pass this first |
| L2 Property-based | Any integer seed, fresh patterns each run | Your method must generalize, no memorization |
| L3 Adversarial | Higher K, larger N, PCA-MNIST swap | Variance-based component ensures robustness |

**Key insight:** Any approach that hardcodes pattern-specific values will pass L1 but fail L2. The precision logic must work from first principles on the query alone (or with dynamically loaded patterns from the adapter context).

---

### PS4 Quickstart

```bash
git clone https://github.com/Sauhard74/Anvil-P-E
cd Anvil-P-E/bench-p04-pcam
pip install -r requirements.txt

# Quick test (2 seeds, ~10 sec)
python self_check.py --adapter adapters.myteam:Engine --quick

# Full test (5 seeds, ~5 min)
python self_check.py --adapter adapters.myteam:Engine
```

**Adapter template:**

```python
# adapters/myteam.py
import numpy as np

class Engine:
    def __init__(self, stored_patterns, **kwargs):
        self.patterns = stored_patterns  # shape: (K, 64)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        """
        Args:
            corrupted_query: shape (64,), values in {-1, +1} or soft
        Returns:
            precision: shape (64,), positive, will be normalized to mean=1
        """
        # YOUR LOGIC HERE
        precision = np.ones(64)
        return precision / precision.mean()
```

---

## Decision Matrix

| If you want... | Pick |
|----------------|------|
| Max PS3 score, real impact | Approach B (Ops Triage) + LangGraph + Omium |
| PS3 with unique angle | Approach E (Legal) or C (Recruiting) |
| PS4 safe and consistent | Approach E (Hybrid) |
| PS4 highest ceiling | Approach D (MLP) + Approach E fallback |
| PS4 theoretically grounded | Approach C (Hessian) for anisotropy score |

---

## Repo Structure Suggestion

```
anvil-2k26/
  ps3/
    agents/
      planner.py
      search.py
      synthesizer.py
      delivery.py
    orchestrator.py
    webhook_server.py
    requirements.txt
    README.md
  ps4/
    adapters/
      myteam.py
    train/
      generate_data.py
      train_mlp.py
    self_check.py  (from repo)
    requirements.txt
    README.md
```

---

## Key Deadlines and Constraints

| Item | PS3 | PS4 |
|------|-----|-----|
| Team size | 1-4 | 1-3 |
| Duration | 24h | 24h |
| Language | Any | Any with NumPy |
| GPU needed | No | No |
| Deliverables | Git + 5-min video + 3-page arch doc | Git + adapter file + 1-page README |
| Bonus available | +10% Omium traces | No bonus |
