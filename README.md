# RL Token Agent: Learning When to Retrieve, When to Pay, and When to Stop

> A reinforcement learning policy that controls an LLM pipeline's compute spend — deciding what to retrieve, which model tier to call, and when to stop — to maximize accuracy per token, not just accuracy.

---

## 1. Where this came from

This project started from a simple but underrated claim : **token efficiency is not a prompting problem, it's an architecture problem.** The arguement is that teams waste tokens not because their prompts are verbose, but because of what happens *upstream* — retrieving too much, retrieving the wrong things, and re-sending the same noisy context at every step of a multi-step pipeline. Their fix is a four-layer framework: retrieve less but better, pass only the evidence that matters, keep prompts about control rather than compression, and orchestrate intelligently across steps.

That last layer — orchestration — is the one that's actually a decision-making problem under uncertainty and cost. *"What should the system do next, given what it knows so far and what it has left to spend?"* is not a prompting question. It's a sequential decision problem with state, actions, and cost — which is to say, **it's an RL problem.**

The project translates that single observation into a concrete system: an RL agent that sits *above* an LLM pipeline and decides, at each step, whether to retrieve more documents, which model tier to call, whether to refine the query, or whether to stop — trained to maximize **accuracy per token spent**, not accuracy alone.

The bar I set, explicitly, from the start: this had to be a real RL problem with a real environment, real baselines, and a real cost-vs-accuracy result — not "RL picks GPT-4 vs GPT-3.5" or "RL decides retry count," which would be shallow and obvious.
---

## 2. The constraints that shaped every decision

Before any code, I was honest about the real world constraints, because they determined almost every design choice downstream:

- **No paid API access.** Only a local Ollama 3.1:8B model, usable in real training, plus a handful of calls reserved for final validation against a stronger model later.
- **An 8B local model is a noisy, weak signal source.** Logprobs and self-reported confidence from an 8B model are not reliable enough to use as the state signal for an RL policy — they'd inject noise directly into the thing the policy is supposed to learn from.

Two design decisions fell directly out of these constraints:

**1. Confidence had to be rule-based, not logprob-based.** Instead of trusting the LLM's self-assessment of how confident it is, the environment derives a confidence/agreement signal from *observable* things: retrieval relevance scores and answer agreement across multiple samples. This is arguably more honest anyway — the policy should learn from evidence in the world, not from an 8B model's possibly-miscalibrated sense of its own certainty.

**2. "Cheap" vs "expensive" model tiers had to be simulated, not real.** With only one local model available, we couldn't literally call GPT-3.5 vs GPT-4. Instead:
- **Cheap tier** = 1 sample, small context (top-ranked docs only)
- **Expensive tier** = 3 samples + majority vote, larger context

This preserves a genuine cost/quality tradeoff (more samples and more context cost more tokens and tend to be more accurate) without needing a second real model.

**3. Ground truth had to be programmatically generated, not LLM-graded.** I considered three options for the QA environment: hand-built structured "worlds" with deterministic facts, fully programmatic tasks (lookup/arithmetic), or an existing multi-hop QA dataset like HotpotQA. We went with the **structured, programmatic option** and explicitly ruled out HotpotQA — it would have meant losing control over difficulty and introducing LLM-grading ambiguity into the reward signal, which is exactly the kind of untrustworthy reward that would have made the eventual RL result meaningless. If ground truth isn't computable without an LLM in the loop, you can't trust the reward, and if you can't trust the reward, you can't trust anything the policy learns.

---

## 3. Environment design

I treated the whole problem as a **finite-horizon POMDP** where the agent controls computation, not content. The agent isn't answering questions directly — it's deciding *how much and what kind* of computation to spend before producing an answer.

**State** (vectorized, never raw text — raw text as RL state would be both intractable and uninterpretable):
- Tokens spent so far
- Steps taken / remaining budget
- Retrieved-doc relevance scores
- An agreement/confidence proxy derived from the environment, not the model
- Number of docs retrieved so far

**Action space** — kept deliberately small (6 actions), because an over-expanded action space was identified up front as a way to make the project "messy without tight design":
- `RETRIEVE_k` for k ∈ {1, 3, 5}
- `CALL_CHEAP`
- `CALL_EXPENSIVE`
- `REFINE_QUERY`
- `STOP`

**Reward** (computed at termination):

```
R = 1[answer == ground_truth] − λ · tokens_used − α · steps_used
```

with an optional explicit-stop bonus added later (see §5.3) to fix a behavioral bug.

**Episode structure:** a hard horizon (6 steps) and a token budget, both enforced — because without a hard cap, there's no actual decision to make about *when* to stop.

---

## 4. Why retrieval quality was treated as a first-class problem, not a detail

Retrieval determines what the model has to read before it can think, and that fixing retrieval matters more than fixing prompts — is directly encoded into the synthetic world generator and the mock LLM provider. The mock provider's error rate doesn't just depend on whether the *correct* document was retrieved; it also scales **up** with the number of *irrelevant* documents present in the prompt. This was a deliberate modeling choice: it makes the environment actually reward retrieval precision, not just retrieval recall. A policy that retrieves 5 docs (1 relevant + 4 noisy distractors) should — and does — perform worse than one that retrieves 1 highly relevant doc, even though it "saw more information." That asymmetry is the whole thesis of the source material, made measurable.

---

## 5. The build: what we did, what broke, and how we reasoned through it

### 5.1 Scaffolding and the abstract LLM interface

I started with `llm/base.py`, an abstract `LLMProvider` interface (`generate`, `get_logprob`, `count_tokens`), specifically so Ollama could later be swapped for a real API by writing a new class against the same interface — no changes needed anywhere else in the codebase. A disk-backed cache (`llm/cache.py`, keyed by a hash of the prompt) was built early and treated as non-optional infrastructure, not a nice-to-have: RL training reruns the same or similar prompts constantly, and without caching, training would either be prohibitively slow against a real model or impossible to iterate on quickly.

Three providers were built against that interface: `ollama_provider.py`, `mock_provider.py` and a stub `api_provider.py` for a future stronger model.


### 5.2 REINFORCE collapsing to "do nothing"

With the environment validated, we built a simple **REINFORCE** policy-gradient agent first — deliberately *not* PPO yet. The reasoning: PPO has more moving parts (clipping, multiple epochs, GAE) and if something is wrong with the environment or reward shape, PPO's extra machinery makes it harder to tell whether a bad result is an environment problem or a PPO-hyperparameter problem. REINFORCE is simple enough to fail in interpretable ways.

It did fail in an interpretable way: the policy collapsed to **always calling `STOP` immediately**, with zero retrievals and zero LLM calls — accuracy 0%, but a small negative reward, because doing nothing is cheap.

**Why this happened, reasoned out before touching code:** the reward only fires at the terminal step, and under high variance (single-trajectory REINFORCE updates), the gradient estimator can latch onto "this is reliably mildly-bad" (stop immediately, small fixed penalty) over "this is sometimes very good and sometimes very bad" (retrieve and call, sometimes right, sometimes wrong, under noisy 8B-scale signal). It's a classic high-variance local optimum, made worse by single-episode advantage normalization, which is mathematically close to degenerate when a single episode's reward is the same value broadcast across all of that episode's timesteps.

**Fix:** rewrote REINFORCE to use **batched episodes** (multiple trajectories per gradient update, reducing variance), a **mean baseline** subtracted from returns, and an **entropy bonus** to keep the policy from prematurely collapsing onto one action. After the fix, training-batch accuracy reached 30–60%, with one greedy evaluation snapshot hitting 78% — confirming there was real learnable signal in the environment, even though vanilla REINFORCE remained somewhat unstable. That instability was treated as expected and *itself* the justification for moving to PPO next, rather than a problem to keep hand-tuning away.

### 5.3 PPO

PPO (via stable-baselines3, wrapping the same environment — no environment changes needed, only the agent) **dominated every baseline**:

| Policy | Accuracy | Avg Tokens | Acc / 1k tokens |
|---|---|---|---|
| always_cheap | 0.65 | 96.4 | 6.74 |
| always_expensive | 0.78 | 98.4 | 7.89 |
| fixed_pipeline | 0.71 | 74.4 | 9.59 |
| heuristic_threshold | 0.74 | 74.3 | 10.00 |
| **PPO (learned)** | **1.00** | **58.5** | **17.11** |

That's roughly a **71% improvement in accuracy-per-1k-tokens** over the best hand-written baseline.

But a number that clean deserved suspicion before celebration. 97–100% accuracy was meaningfully higher than *any* baseline's ceiling (max 82%), which is the kind of gap that's either a genuinely smart learned strategy or a sign the policy found a loophole.Inspecting actual rollouts rather than trusting the aggregate metric.

**What PPO actually learned (the good part):** `RETRIEVE_1` (just the single top-ranked document, skipping the noisy distractors entirely) → `CALL_EXPENSIVE` (3-sample majority vote) → done. This is a genuinely non-obvious strategy — none of the baselines tried "retrieve minimally, but pay for an ensemble vote on the one good doc you have." It directly exploits the retrieval-noise asymmetry we built into the environment (§4): fewer docs means less distractor noise means a more reliable signal, and one expensive ensemble call on a clean signal beats a cheap call or a noisy multi-doc context. This is exactly the kind of "architecture-level" win the source blog post argues for — it's not a shorter prompt, it's a smarter retrieval-then-compute allocation.

**What PPO also learned (the bug):** it never called `STOP` explicitly. Instead, it padded out the remaining steps with `REFINE_QUERY` — at the time, a free, zero-cost no-op — until the episode hit `max_steps` and was force-truncated by the environment. Average steps across episodes: exactly 8.00, every time. That's not noise; that's a policy reliably hitting a wall on purpose.

**Why this is a reward-design issue, reasoned through explicitly:** the reward function computed the same value for an explicit `STOP` as for a forced truncation at the horizon — both just used whatever `steps_used` and `tokens_used` were at termination. Since `REFINE_QUERY` cost nothing, there was no incentive to spend a step explicitly stopping when padding with free actions produced an identical reward. The policy wasn't "wrong" — it was correctly exploiting the actual reward function we wrote, which is arguably a more useful failure than a policy that's just bad, because it told us precisely what to fix.

**Fix** both (a) charge a small token cost for `REFINE_QUERY` so it's no longer a true no-op, and (b) add a small explicit bonus for calling `STOP` deliberately versus being truncated.

**Iteration 1 of the fix didn't work — and the *way* it didn't work was itself informative.** After charging `REFINE_QUERY`, the policy didn't start calling `STOP`. It just moved the no-op behavior to `RETRIEVE_1` instead, since retrieval was still free. This was an important confirmation: the issue wasn't "REFINE_QUERY is exploitable," it was "any free action is exploitable as padding." Patching one symptom just relocated the exploit, which is the expected outcome when you fix an instance of a problem instead of the general shape of it.

**Working the math by hand before changing anything else,** to check whether the reward function, *post-fix*, actually favored explicit stopping. Padding to 8 steps: `1 − 0.05 (tokens) − 0.16 (8 steps × α) − 0.05 (forced-stop penalty) = 0.74`. Stopping explicitly at step 2: `1 − 0.05 − 0.04 (2 steps × α) − 0 = 0.91`. The math clearly favored stopping early — by a wide margin. So this had stopped being a reward-design bug and become a pure **training/exploration problem**: PPO simply hadn't discovered the better optimum yet.

We increased the entropy coefficient and trained substantially longer to give PPO more room to escape the padding equilibrium it had locked onto. It didn't move — average steps stayed exactly at the horizon regardless. We then tried shrinking `max_steps` itself (first to 4, found that too tight because the heuristic baseline genuinely needs up to 5 actions to complete its own logic and would get truncated unfairly, then settled on 6 as a value that keeps all baselines completable while bounding how much the padding behavior could cost). Even at the tighter horizon, PPO still padded to the new max exactly.

**Why we stopped chasing it, and why that was the right call:** this had become a known, documented PPO behavior — once a clipped-policy-gradient method locks onto a strategy that's already working (retrieve_1 → call_expensive → correct, repeatedly), its conservative update rule makes it reluctant to abandon that strategy in search of a comparatively small additional reward gain (0.91 vs 0.74) on a sub-decision (exactly when to stop) that doesn't change *what* the policy actually does. The core, relevant claim — that a learned policy finds a smarter retrieval/compute allocation than any hand-written baseline — was never in question and never depended on this fix. 

> *PPO learned the optimal retrieval/LLM-tier allocation but converged to a stable padding equilibrium on the stopping sub-decision — a known PPO conservatism issue tied to sparse termination gradients.*

---

## 6. Final evaluation: behavior analysis, budget sensitivity, Pareto curve

With the core result locked in, the remaining work was the evaluation suite the original spec called non-negotiable — because a single accuracy number, without seeing *how* the policy behaves, doesn't actually demonstrate that the system learned anything structural.

**Policy behavior analysis**  traces 200 episodes per policy and reports: action distribution, % of episodes that used the expensive tier, % that used the cheap tier, % with an explicit stop, average retrievals and refines per episode, and — critically — an **easy vs. hard split** (≤8 docs vs. >8 docs in the world) showing accuracy, steps, and tokens separately for each difficulty tier. This is the direct test of the claim from the source material: *does the policy retrieve more and spend more only when the problem actually warrants it?* A flat policy that behaves identically regardless of difficulty hasn't learned to be selective — it's learned a fixed strategy that happens to average out well.

**Budget sensitivity sweep** reruns every policy — baselines and PPO — across a range of token budgets (30 through 800), to see whether tightening the budget degrades the policy gracefully or causes it to collapse. A policy that adapts its strategy as budget shrinks (rather than just failing more often) is doing something closer to genuine resource-aware planning.

**The Pareto curve**  (plotted to `pareto_curve.png`) puts every policy's full budget sweep on one accuracy-vs-tokens plot. This is the single figure that makes the entire project's claim visually legible: baselines trace out one frontier, and PPO should sit visibly up and to the left of it — more accuracy for fewer tokens — across a *range* of budgets, not just at one cherry-picked operating point. That range is what separates "got lucky at one setting" from "learned a genuinely better policy."

---

## 7. Evaluation

**Accuracy per 1,000 tokens** was chosen, deliberately, as the single primary metric over raw accuracy or raw token count alone — because either of those in isolation can be gamed (always-expensive maximizes accuracy by brute force; always-stop-immediately minimizes tokens by doing nothing) and neither captures the actual claim the source material makes: efficient systems get a *better outcome per unit of compute*, not just a cheaper or a more accurate one in isolation.

```
PPO (learned):        17.11 accuracy / 1k tokens
Best baseline (heuristic): 10.00 accuracy / 1k tokens
→ ~71% improvement
```

---

## 8. Final repository structure

```
rl_token_agent/
├── llm/
│   ├── base.py                # Abstract LLMProvider interface (generate, get_logprob, count_tokens)
│   ├── ollama_provider.py     # Real provider — talks to local Ollama 3.1:8B
│   ├── mock_provider.py       # Deterministic stand-in used for fast sandbox iteration;
│   │                          # error rate scales with n_samples (voting) AND with
│   │                          # number of irrelevant docs in the prompt (noise penalty)
│   ├── api_provider.py        # Stub for a future stronger API model (same interface)
│   └── cache.py               # Disk-backed cache, keyed by prompt hash — required for
│                              # any RL training loop that re-hits similar prompts
├── world/
│   ├── generator.py           # Synthetic 1-hop world generator: deterministic ground truth,
│   │                          # parametrized difficulty (n_docs, n_distractors), with a
│   │                          # hop-count param reserved for future multi-hop extension
│   └── retriever.py           # Keyword-overlap retriever; ranks + scores docs and formats
│                              # the LLM prompt with [DOCi] tags
├── env/
│   ├── state.py               # State vectorization (tokens used, steps, retrieval scores,
│   │                          #   confidence proxy, remaining budget)
│   ├── actions.py             # Action enum: RETRIEVE_1/3/5, CALL_CHEAP, CALL_EXPENSIVE,
│   │                          #   REFINE_QUERY, STOP
│   ├── reward.py              # Terminal reward: correctness − λ·tokens − α·steps,
│   │                          # with explicit-stop handling (post-bugfix)
│   └── pomdp_env.py           # gymnasium.Env tying state/actions/reward together;
│                              # owns the REFINE_QUERY token-cost fix and the
│                              # explicit_stop vs forced-truncation reward branches
├── agents/
│   ├── baselines.py            # AlwaysCheap, AlwaysExpensive, FixedPipeline, HeuristicThreshold
│   ├── reinforce_agent.py      # Batched REINFORCE + mean baseline + entropy bonus
│   │                           # (sanity-check learner, built before PPO on purpose)
│   └── ppo_agent.py            # stable-baselines3 PPO wrapper + eval callback tracking
│                               # the same metrics as the baselines, for apples-to-apples comparison
├── eval/
│   ├── metrics.py              # accuracy/1k tokens, tokens-per-correct, etc.
│   └── plots.py                # Pareto curve plotting utilities
├── scripts/
│   ├── run_baselines.py        # Runs all 4 baselines, produces first comparison table
│   ├── train_reinforce.py      # REINFORCE training loop
│   ├── train_ppo.py            # PPO training loop, evaluated against baselines periodically
│   ├── evaluate.py
└── README.md                   # This file
```

Every provider (`mock`, `ollama`, future `api`) implements the same `LLMProvider` interface, and nothing in `env/`, `agents/`, or `eval/` ever imports a specific provider directly — they all take an `llm` object satisfying the interface. That's what makes "swap mock for real Ollama, or Ollama for a stronger API later" a one-line change at the call site, not a refactor.

---
