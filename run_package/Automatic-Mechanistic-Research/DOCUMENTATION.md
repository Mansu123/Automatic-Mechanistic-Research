# AutoMechInterp Handbook

**Method & Review Documentation** · Rev. 2026-09-02 · research prototype, runs on CPU

A hierarchical multi-agent system that performs mechanistic interpretability on a
neural network automatically — and adversarially verifies every claim before
trusting it. This document covers what it does, how the method works, the S-EAP
mathematical layer, and how a human expert should review its output.

> Web version (theme-aware, with a table of contents):
> https://claude.ai/code/artifact/27319adc-1c12-4f21-96ed-180b1cc36844

**Contents**

1. [What this project does](#1-what-this-project-does)
2. [The multi-agent system](#2-the-multi-agent-system)
3. [Input & output](#3-input--output)
4. [Method & pipeline stages](#4-method--pipeline-stages)
5. [The S-EAP mathematical layer](#5-the-s-eap-mathematical-layer)
6. [Mathematical terms](#6-mathematical-terms)
7. [What each file does](#7-what-each-file-does)
8. [The technique toolkit](#8-the-technique-toolkit)
9. [How a human reviewer should review it](#9-how-a-human-reviewer-should-review-it)
10. [Limitations & future work](#10-limitations--future-work)

---

## 1. What this project does

**Automated mechanistic interpretability: point it at a model and a behaviour,
and it returns the circuit responsible, verified.**

Automated interpretability today is either **one flat agent asked to do too much**
(MAIA 2024 — documented confirmation bias, correlational findings) or **a fixed
statistical pipeline** that cannot decide what to investigate next (Dalvi et al.
2019). AutoMechInterp addresses the systems-level version of that problem.

### The research question

Can a hierarchical multi-agent LLM system — whose agent structure mirrors the
target network's architecture — autonomously analyse a model layer by layer,
discover and causally verify the circuits within it, and produce a verified
mechanistic account, without a human driving each step?

### Two contributions, kept separate

1. **The architecture (engineering).** A Network Analyst over per-layer Layer
   Agents over per-head Component Agents, plus a *structurally separate*
   Skeptic/Judge pair whose only job is to falsify claims — not an agent grading
   its own work.
2. **S-EAP (mathematical).** A second-order extension to circuit discovery that
   targets a documented blind spot: *backup heads* whose individual ablation
   effect is near zero because another component compensates. Covered in full in
   §5.

A large model-agnostic **technique toolkit** (§8) sits alongside both, covering
the standard interpretability methods — lenses, patching, probing, steering,
SAEs, concept erasure.

> **Scope of "model".** The target being interpreted must be an **open-weights**
> model loadable locally (`AutoModelForCausalLM.from_pretrained`). Every method
> here needs internal activations, gradients, attention tensors, or weights via
> PyTorch hooks. An API-served model (GPT-4, Gemini, DeepSeek) exposes only text
> and cannot be analysed mechanistically — an API model can only serve as the
> *agent brain* driving analysis of a local target.

---

## 2. The multi-agent system

Twelve agent types. The structure mirrors the network: whole-model → per-layer →
per-head, with a separate adversarial arm.

```
Orchestrator  (coordinates the run, merges claims into one ledger)
 ├─ Network Analyst        profiles the model, flags interesting layers
 │   ├─ Weight Agent          SVD / rank / norms / E-U tying   (whole model)   [new]
 │   └─ Safety Agent          refusal dir, anomaly monitor, copy-suppression   [new]
 ├─ Layer Agent × N         one per flagged layer  (runs in parallel)
 │   ├─ Lens Agent            logit / Jacobian lens, Patchscopes               [new]
 │   ├─ Probe Agent           linear / sparse / MDL probes                     [new]
 │   └─ Feature Agent         toy SAE, feature geometry, dimensionality        [new]
 ├─ Component Agent × M     one per attention-driven layer
 │      EAP → ACDC → S-EAP second-order pass
 │   └─ Steering Agent        CAA / ablation / LEACE constructive validation   [new]
 └─ Skeptic   tries to FALSIFY the claimed circuit (4 tests)
     └─ Judge   reads claim + Skeptic evidence → verdict
```

`[new]` agents (Weight, Safety, Lens, Probe, Feature, Steering) were added on top
of the original six (Orchestrator, Network Analyst, Layer, Component, Skeptic,
Judge). Report scoring is not an agent: humans (`human_review/`) and the GLM
judge (`llm_review/`) score finished reports against the shared rubric in
`automechinterp/eval/matrix_rubric.py`.

### How the agents coordinate

- **Same interface for every agent.** `LLMBackend.decide(system, evidence, tools)
  -> next_action`. The "brain" is swappable with zero changes to agent logic: a
  hand-written rule table (`heuristic`, no LLM needed), a local open-source LLM
  (`hf_local`, e.g. Qwen2.5), or an API model.
- **Scoped context.** Each agent sees only its own evidence log — a Layer Agent
  never holds the whole network in context.
- **Shared budget.** A global `ToolCallBudget` plus a per-agent cap; agents draw
  from one pool so a run terminates predictably (~110 tool calls, ~40 agent
  instances on a GPT-2 run).
- **Adversarial separation.** The Skeptic receives only the *claimed circuit* and
  the behaviour — never the Component Agent's reasoning — and tries four
  falsifications: `ablate_component`, `exclusion_ablation`, `minimality_check`,
  `counterexample_search`. Only a circuit that survives all four is labelled
  **Confirmed**.
- **Parallelism.** Layer Agents (and each layer's Lens/Probe/Feature agents) run
  in threads; a global `MODEL_LOCK` serialises the actual forward passes against
  the shared model instance.

---

## 3. Input & output

### Input

- **Target model** — any local HF causal LM: `gpt2`,
  `Qwen/Qwen2.5-0.5B-Instruct`, `EleutherAI/pythia-160m`, Llama, Gemma…
- **Task** — a *clean* prompt, a *corrupted* (counterfactual) prompt, and the two
  answer tokens. Example (IOI):
  clean `"When Bob and Tom went to the store, Bob gave a drink to"` → expect
  `Tom` not `Bob`; corrupted = same with the names' roles swapped.
- **Agent backend** — `heuristic` / `hf_local` / `openai`.
- ~200 built-in tasks ship in `behaviors.py` (syntax, factual recall, arithmetic,
  induction, bias…).

### Output

- **Circuit** — the causally load-bearing `(layer, head)` list, with a **Judge
  verdict**: Confirmed / Probable / Speculative / Refuted.
- **Faithfulness & minimality** scores — does the circuit alone reproduce the
  behaviour; does every head matter.
- **Per load-bearing layer** — logit-lens decode + emergence layer, probe
  accuracy & MDL, toy-SAE (FVU, L0, top features), effective dimensionality.
- **Causal validation** — self-repair map, steering effect, LEACE erasure effect.
- **Whole-model** — parameter rank/norm profile, E/U tying, refusal direction,
  anomaly monitor, copy-suppression heads.
- **Negative controls** — fabricated claims the Skeptic must refute
  (false-confirmation rate).
- A **Markdown + JSON report** (`human_review/reports/`, `output/eval/`,
  `experiments/techniques/`).

### Real numbers — GPT-2, IOI task

| Metric | Per-layer ACDC only | + S-EAP pass |
|---|---|---|
| Claimed circuit | `[(9,9),(10,10)]` | `[(9,6),(9,9),(10,7),(10,10),(11,10)]` |
| Circuit recall vs ground truth | 11% (1/9) | 44% (4/9) |
| Negative controls refuted | 4/4 | 4/4 |
| False-confirmation rate | 0% | 0% |
| Tool calls / agent runs | ~52 / ~15 | ~110 / ~40 |

`python3 main.py --target-model gpt2` runs Stage A end to end.
`python3 main.py --techniques` runs the full toolkit and writes a standalone
report.

---

## 4. Method & pipeline stages

### Stages

| Stage | What it does | Command |
|---|---|---|
| **A — Ground-truth validation** | Full pipeline on GPT-2, IOI task, scored against Wang et al. 2022's published circuit, plus negative controls. | `main.py` |
| **B — Layer Atlas** | Runs the pipeline once per behaviour across ~200 tasks, assembles a cross-behaviour map of what each layer does. | `main.py --stage-b` |
| **C — Cross-model diff** | Diffs a fine-tuned model against its base layer by layer, with causal transplant confirmation. | `main.py --stage-c` |
| **D — Evaluation** | Writes one report per behaviour to `human_review/reports/` for scoring against the shared 8-metric rubric. | `main.py --stage-d` |

### Tool tiers

| Tier | File | Used by | Key tools |
|---|---|---|---|
| **N** — Network | `tools/tier_n.py` | Network Analyst | `profile_network`, `layerwise_cka_scan`, `redundancy_scan` |
| **L** — Layer | `tools/tier_l.py` | Layer Agent | `patch_layer`, `attn_mlp_attribution`, `logit_lens`, `sae_layer_profile` |
| **C** — Component | `tools/tier_c.py` | Component Agent | `run_eap`, `run_acdc`, `run_synergy_eap`, `direct_logit_attribution`, `get_attention_pattern` |
| **V** — Verification | `tools/tier_v.py` | Skeptic | `ablate_component`, `exclusion_ablation`, `minimality_check`, `counterexample_search`, `interchange_intervention` |

### Circuit discovery (first-order)

**EAP — Edge Attribution Patching.** Gradient of the metric w.r.t. each head's
out-projection input, times the (clean − corrupted) activation difference. One
forward + one backward pass scores every head. Cheap, approximate.

**ACDC — Automatic Circuit Discovery.** Patch each `(layer, head)` from the clean
run into the corrupted run, one at a time; keep heads whose patch recovers more
than a threshold of the clean–corrupted metric gap. Exact, but O(layers×heads)
forward passes.

### Verification — Prover / Skeptic / Judge

Nothing is trusted because an agent found it. The Skeptic runs four falsification
tests; the Judge reads the claim *and* the Skeptic's evidence and assigns the
verdict. Negative controls (deliberately wrong claims) measure how often a wrong
claim slips through — the false-confirmation rate.

---

## 5. The S-EAP mathematical layer

**Synergy-Aware Edge Attribution Patching** — a second-order pass that catches the
backup heads first-order discovery is structurally blind to.

### 5.1 The gap it targets

ACDC and EAP both measure a component's **marginal effect**: ablate it alone, see
what breaks. That is a first-order test. It is *structurally* blind to a component
whose effect only appears jointly — a **backup head** that compensates when the
primary one is removed will show a near-zero effect when tested alone, so greedy
single-edge pruning removes it even though it is a real part of the circuit.

This is not hypothetical. Wang et al. 2022's GPT-2 IOI circuit names backup
name-mover heads and negative name-mover heads that exist for exactly this
reason. This repo's plain `run_acdc` recovers only **1 of 9** ground-truth heads
(11% recall) and misses both negative name movers entirely.

### 5.2 S-ACDC — the formal definition

Cooperative game theory has the right object: a component's Shapley value
decomposes into first-order (marginal) and higher-order (interaction) terms. For
a candidate set of components:

```
m(S)     = task metric with only components in S active (rest ablated)   [coalition value]

phi1(i)  = m({i}) - m({})                                                [marginal effect;
                                                                         what ACDC/EAP compute]

Syn(i,j) = m({i,j}) - m({}) - phi1(i) - phi1(j)                          [synergy score;
                                                                         2nd-order Harsanyi dividend]
```

A true backup pair reads `phi1(i) ≈ 0`, `phi1(j) ≈ 0`, `|Syn(i,j)| >> 0` — the
exact shape single-edge ACDC cannot see.

### 5.3 S-EAP — the tractable approximation

Computing `m(S)` for all pairs is an O(2^N) / O(M²) combinatorial explosion.
S-EAP estimates the off-diagonal interaction with a finite difference of
gradients: ablate candidate head *j*, re-measure every other head *i*'s
first-order attribution.

```
Syn(i,j) ≈ ( ∇_{A_i} L_clean  -  ∇_{A_i} L_ablate(j) ) · ( A_i^clean - A_i^corrupt )
```

One backward pass per candidate *j* yields the interaction row for **all** *i* at
once — cost is O(M) backward passes, not O(M²). If head *j* is a backup for head
*i*, ablating *j* makes the network route more heavily through *i*, so `∇_{A_i}`
spikes and the gradient difference isolates the interaction magnitude.

A multi-step generalisation (PI-HS) interpolates *j* along an ablation path in K
steps; S-EAP is the K=1 case.

### 5.4 What it does empirically

Benchmarked against *exact* pairwise coalition patching (`ablate_head_set`) on
GPT-2 IOI and Qwen2.5-0.5B. Full reproduction in `experiments/seap/`.

| Check | GPT-2 IOI | Qwen2.5-0.5B IOI |
|---|---|---|
| ACDC first-order recall vs GT | 1/9 (blind spot reproduced) | no ground truth |
| Spearman( \|S-EAP Syn\|, \|exact Syn\| ) | **+0.65**, p = 9e-11 | +0.20, p = 0.30 |
| Spearman(signed, signed) | **−0.12**, p = 0.30 | +0.14, p = 0.49 |
| Circuit faithfulness, ACDC → + synergy heads | 0.31 → 0.36 | 0.53 → 0.58 |

**Honest verdict.** S-EAP is a valid interaction-**magnitude** screen on GPT-2 —
but it **loses sign** (cannot tell cooperation from cancellation), so it can only
shortlist pairs for exact `ablate_head_set` confirmation; it is not a standalone
discovery method. It did not replicate on the 0.5B model (underpowered candidate
set). With few prompts, nothing survives FDR correction. This is a
**workshop-scale mixed result, not a main-track contribution.** Every named
"novel" formulation attempted before this one (LGAP, PI-HS, HH-CD…) reduced on
inspection to *[known method] + [asserted interpretability metaphor]*.

### 5.5 Integration into the pipeline

The Component Agent now runs, after ACDC: `direct_logit_attribution`
(write-direction seed — sees name-mover heads even at phi1 ≈ 0) → `run_synergy_eap`
over the layer + 2 neighbours → merge any head with `|Syn| ≥ 0.05` to a circuit
head. On GPT-2 IOI this lifts recall **11% → 44%** and recovers the negative name
movers.

> **Known downstream issue — the Skeptic has the same blind spot.** The recovered
> circuit is then **Refuted** by the Judge, because the Skeptic's
> `ablate_component` test ablates each claimed head *individually* and requires
> all to show an effect. Backup heads show ≈0 individual effect *by definition*,
> so the first-order Skeptic cannot distinguish "real backup head" from "padded
> claim". Fix (not yet done): the Skeptic must test synergy-recovered heads as
> *coalitions* with their partner via `ablate_head_set`, not as singletons — the
> same first-order → second-order upgrade applied to verification.

---

## 6. Mathematical terms

| Term | Meaning in this project |
|---|---|
| **Clean / corrupted prompt** | A minimal pair: same structure, differing only in the variable the task turns on. The corrupted run is the counterfactual baseline for every patch. |
| **Logit difference** | The metric: `logit(correct token) − logit(contrast token)` at the final position. All patching effects are fractions of the clean–corrupted gap in this quantity. |
| **Activation patching** | Replace one component's activation with the value it took on a different run; measure the effect on the metric. Establishes causation, not correlation. |
| **Ablation** | Remove a component by replacing it with its mean (or zero, or a resampled value). *Mean ablation* is used throughout; it is mildly off-distribution. |
| **Marginal effect phi1(i)** | How much the metric moves when component *i* alone is ablated. First-order. What ACDC / EAP measure. |
| **Coalition value m(S)** | The metric with only the set *S* of components active (the rest ablated). The primitive `ablate_head_set` computes it. |
| **Synergy / Harsanyi dividend** | `Syn(i,j) = m({i,j}) − m({i}) − m({j}) + m({})`. The part of a pair's joint effect not explained by the two parts separately. Second-order interaction term. |
| **Shapley value** | Game-theoretic fair attribution of a coalition's value to its members, averaged over all orderings. Full computation is exponential; S-ACDC targets only the pairwise interaction terms. |
| **Hessian (off-diagonal)** | ∂²L / ∂A_i ∂A_j — the second derivative of the metric w.r.t. two components. S-EAP's gradient difference is a finite-difference estimate of this. |
| **Integrated gradients** | Attribution by integrating the gradient along a path from a baseline to the input. The multi-step (PI-HS) variant of S-EAP is a path integral of this kind. |
| **Direct logit attribution (DLA)** | Project a component's write onto `W_U[:,correct] − W_U[:,contrast]`. A *write-direction* signal, orthogonal to ablation effect. |
| **Logit lens** | Apply the final norm + unembedding to an intermediate layer to read what it "would predict" there. The *emergence layer* is where the final answer first becomes the top token. |
| **Faithfulness / completeness / minimality** | Faithfulness: the circuit alone reproduces the behaviour. Completeness: nothing outside it matters. Minimality: every head in it matters. |
| **Self-repair / Hydra effect** | Downstream components changing their behaviour to compensate when an upstream one is ablated. The reason backup heads exist and first-order tests mislead. |
| **LEACE** | Least-squares concept erasure: a closed-form projection that removes all *linear* predictability of a concept from an activation while changing it minimally. |
| **Participation ratio** | (Σλ)² / Σλ² over the activation covariance eigenvalues — the effective number of dimensions a layer actually uses. |
| **FVU / L0** | SAE quality: fraction of variance unexplained by the reconstruction; average number of active features per token. |
| **CKA** | Centered Kernel Alignment — similarity between two activation matrices; used to find representation shifts across depth. |
| **FDR / permutation test** | Multiple-comparison control. A paired sign-flip permutation test asks whether a mean synergy is distinguishable from zero; Benjamini–Hochberg corrects across all pairs tested. |

---

## 7. What each file does

Everything lives in `automechinterp/`. Entry point is `main.py`.

### Orchestration & stages

| File | Responsibility |
|---|---|
| `main.py` | CLI. Flags: `--stage-b/c/d`, `--ablations`, `--techniques`, `--target-model`, `--backend`, `--gap-matrix`. |
| `run_eval_matrix.py` | Repo-root runner: Stage D across a size ladder / tier of target models (≤7B) from `eval/model_matrix.py`. **Strictly sequential** — one model downloaded, run, freed, and (with `--purge-downloads`) deleted from the HF cache before the next. Resumable (`--skip-done`), `--check` for a fast wiring-only pass, one failing model doesn't sink the sweep. For handing to a teammate with compute. |
| `automechinterp/eval/model_matrix.py` | The evaluation model matrix as data: ~25 HF causal-LM ids (GPT-2 / Pythia / Qwen2.5 / OPT / … ladders), each with params/layers/attention/tier, plus `select()` filtering and named `LADDERS`. |
| `config.py` | All tunables via env vars: target model, agent backend, budgets, thresholds, `DEEP_TECHNIQUES`. |
| `hierarchy.py` | Wires the whole agent flow together — the waves in §2. Returns one result dict. |
| `stage_a.py` | Builds the GPT-2 IOI task, runs the hierarchy, scores against ground truth, runs negative controls. Holds the GT head lists. |
| `behaviors.py` | ~200 task definitions across 25 cognitive angles. Each returns a standard task dict (clean/corrupted/tokens/eval prompts/contrastive sets). |
| `stage_b.py` / `stage_c.py` / `stage_d_eval.py` | Layer Atlas / cross-model diff / evaluation-report pipeline. |
| `ablations.py` | Ablation studies (hierarchy vs flat, verification on/off, flagging strategy). |
| `llm_backends.py` | The swappable agent brain: `HeuristicBackend`, `HFLocalBackend`, `OpenAIBackend`, `AnthropicBackend`. |

### Tools (`automechinterp/tools/`)

| File | Responsibility |
|---|---|
| `adapter.py` | Architecture-agnostic hook layer — `register_model`, `capture_activations`, `patch_layer`, `run_with_head_patch`, `ablate_head`, **new** `ablate_head_set` (coalition value `m(S)`). No TransformerLens dependency. |
| `tier_n.py` / `tier_l.py` / `tier_c.py` / `tier_v.py` | The four tool tiers (§4). `tier_c` holds `run_acdc`, `run_eap`, **new** `run_synergy_eap`, **new** `direct_logit_attribution`. |
| `sae.py` | Public pretrained-SAE tools via `sae_lens` — GPT-2 only (needs a TransformerLens model + a matching release). |
| `digest.py` / `cache.py` | Tensor-to-text summarisation (agents never see raw tensors); content-addressed result cache. |

### Agents (`automechinterp/agents/`)

| File | Agent |
|---|---|
| `base.py` | Shared ReAct loop, `ToolCallBudget`, `AgentLog`, `digest_dict` helper. |
| `orchestrator.py` | `HypothesisLedger` (dedupes claims), `decide_layers_to_spawn`. |
| `network_analyst.py` / `layer_agent.py` / `component_agent.py` | The whole-model → layer → head discovery chain. `component_agent` now runs the S-EAP second-order pass. |
| `skeptic.py` / `judge.py` | Falsification and the Confirmed/Probable/Speculative/Refuted verdict. |
| `lens_agent.py` **new** | Per layer: logit / Jacobian lens, logit-lens trajectory, Patchscopes. |
| `probe_agent.py` **new** | Per layer: linear / sparse / MDL probes, probe-direction causal test. |
| `feature_agent.py` **new** | Per layer: toy SAE, feature-direction geometry, activation dimensionality. |
| `steering_agent.py` **new** | Per circuit layer: activation addition, ablation steering, LEACE — constructive validation feeding the Judge. |
| `weight_agent.py` **new** | Whole model: parameter SVD / effective rank, weight-norm profile, E/U tying. |
| `safety_agent.py` **new** | Whole model: refusal direction, activation-anomaly monitor, copy-suppression scan. |

### Technique toolkit (`automechinterp/techniques/`) — all new

| File | Category |
|---|---|
| `basic.py` | logit lens, tuned lens, Jacobian lens, DLA, attention-pattern classification |
| `causal.py` | activation / attribution / path patching, self-repair, interchange intervention, causal-mediator selection, integrated-gradients (refined) attribution |
| `probing.py` | linear / k-sparse / MDL probes, probe-direction causal test, geometry-of-truth, attention probes, LAT reading vectors |
| `steering.py` | activation addition (CAA), ablation / affine / multi-layer steering, function vectors, unsupervised (MELBO) steering vectors |
| `editing.py` | LEACE, inference-time concept ablation, hook-based localized fact edit; `machine_unlearning` / `rome` = documented stubs |
| `superposition.py` | toy SAE trainer, gated SAE + SAE evaluation, feature dashboards, feature steering, temporal features, public SAE decompose; transcoder / crosscoder = stubs |
| `feature_geometry.py` | participation ratio, feature-direction angles / cliques, principal-component manifold, on- vs off-manifold steering |
| `model_diffing.py` | logit-diff amplification, per-layer activation drift, fine-tuning traces (GPT-2 auto-pairs with the bio stand-in); `feature_level_model_diffing` = stub |
| `hidden_state.py` | logit-lens trajectory, Patchscopes, SelfIE read-back, concept-injection introspection; `activation_oracle` = stub |
| `circuits.py` | ACDC, EAP, S-EAP, QK/OV decomposition, faithfulness / completeness / minimality, copy-suppression, attribution graph, entity binding; `universality_across_models` = stub |
| `blackbox.py` | counterfactual resampling, minimal-pair contrast |
| `weight_space.py` | parameter SVD / effective rank, weight-norm profile, E/U alignment, parameter-space head grouping; `interpretable_training_note` = stub |
| `safety.py` | refusal direction, Mahalanobis anomaly monitor, evaluation-awareness probe; deception / sleeper-agent = stubs |
| `runner.py` | `run_all(model_id)` — executes every applicable technique, writes Markdown + JSON. |
| `_common.py` | shared helpers: logit-diff metric, contrastive directions, DLA scoring, steering hooks. |

---

## 8. The technique toolkit

~70 model-agnostic techniques across 13 curriculum categories
(learnmechinterp.com). Forward/backward-pass only; runs on any open-weights
model unchanged. Every function takes an `adapter.ModelHandle` and returns a
dict.

```
python3 main.py --techniques                                      # target from config
python3 main.py --techniques --target-model Qwen/Qwen2.5-0.5B-Instruct
# or:  from automechinterp.techniques.runner import run_all;  run_all("gpt2")
```

On GPT-2 a full run executes ~63 techniques with 0 errors. Nine are honest
stubs, reported as `needs-more` with the exact requirement: `rome` /
`machine_unlearning` (weight edits), `transcoder` / `crosscoder` /
`feature_level_model_diffing` (own training run), `deception_detection` /
`sleeper_agent_scan` (labelled dataset), `activation_oracle` (trained
decoder), `universality_across_models` (run the module per model on a size
ladder), `interpretable_training_note` (training-time intervention). Nothing
is faked. `model_diffing.*` auto-pairs GPT-2 with the biomedical fine-tuned
stand-in (`finetune_stand_in.py`); for other models pass `other_model_id=`.

**Cross-checks that landed.** On GPT-2 IOI the Safety Agent's `copy_suppression`
scan independently flags `L10H7` and `L11H10` — the negative name movers. The
`self_repair` probe shows their DLA spiking (`+0.74`, `+0.48`) when `L9H9` is
ablated. Two independent methods converging on the same heads is the kind of
evidence a reviewer should look for.

---

## 9. How the reports are scored

Two arms, **one rubric**: eight metrics, integer 1–10, identical for the human
reviewers (`human_review/`) and the GLM 5.2 judge (`llm_review/`), scoring the
same exported evidence. Source of truth: `automechinterp/eval/matrix_rubric.py`
(`mir_shared8_1to10_v2`); full text in `human_review/rubrics.md`.

### 9.1 The process

1. Run the sweep (`run_colab.py`) and export the review material:
   `python run_review.py --run-dir <run> --export`. This writes
   `<run>/review/review_items.jsonl` (every report / layer / agent / tool /
   S-EAP item) and the `human_*_BLANK_TEMPLATE.csv` files.
2. **Calibrate** on the 100-report calibration set, sharpen any metric wording
   in `matrix_rubric.py` that reviewers keep splitting on, then freeze.
3. **Double-score** the disjoint 550-report validation set — two reviewers
   independently (distinct `reviewer_id`), before seeing judge output — then
   file an explicit `reviewer_id=consensus` reference. Keep `item_id`,
   `rubric_version` and `job_sha256` unchanged.
4. `python human_review/aggregate_scores.py --run-dir <run>` →
   `human_health.json` + inter-reviewer agreement.
   `python -m llm_review.run_judge --run-dir <run>` then
   `python llm_review/aggregate_scores.py --run-dir <run>` → `judge_health.json`.
5. `python run_review.py --run-dir <run> --human-scores <run>/review/completed_human_scores/*.csv`
   → `judge_validation.json` (macro quadratic weighted kappa + the gate).

### 9.2 The eight metrics (1–10, every report and element)

| # | Metric | What to check |
|---|---|---|
| 1 | **localization** | Are the investigated layers / components / positions evidenced, or a carefully bounded null? |
| 2 | **causal_validity** *(critical)* | Correct intervention definitions, signed effects, baselines, uncertainty; decodability is not causation. |
| 3 | **verification** *(critical)* | Necessity / sufficiency / minimality / joint controls / counterexamples — not just an asserted verdict. |
| 4 | **generalization** | Tested beyond the discovery example using this angle's controls and independent data. |
| 5 | **faithfulness** *(critical)* | Every material claim traceable to actual tool evidence, including errors and nulls. |
| 6 | **explanation** | Explains the computation (or precisely diagnoses why it is unresolved), not just a coordinate. |
| 7 | **calibration** | Confidence matches evidence; scientific null vs low capability vs failed tools vs thin coverage all distinguished. |
| 8 | **reproducibility** | Pinned model, prompts, intervention scope, agent/tool coverage, metrics and run config. |

Each metric has metric-specific 2-point bands plus the shared 1–10 anchors in
`matrix_rubric.py`. The 25 angle-specific generalization checks (feeding metric
4) are in `human_review/rubrics.md`. **Report pass:** mean ≥ 7.5, every metric
≥ 5, each critical metric ≥ 7, completed execution.

### 9.3 Reviewing the S-EAP layer specifically

When a report includes an S-EAP / synergy-recovered circuit, additionally check:

| Question | Where to look |
|---|---|
| Do the synergy-recovered heads actually have `phi1 ≈ 0`? (If they have a large marginal effect, ordinary ACDC should have found them — S-EAP added nothing.) | The `run_synergy_eap` digest prints `phi1=(…)` per pair. |
| Is the `|Syn|` ranking consistent with exact coalition patching? | `experiments/seap/` — the Spearman validation. Signed correlation is expected to be near zero; that is a known limitation, not a bug. |
| Did adding the synergy heads actually improve *faithfulness* (circuit-alone metric recovery), or just recall against a hand-labelled list? | The `circuit_faithfulness` result. Recall improvement without faithfulness improvement is weak evidence. |
| If the Judge Refuted the claim, is the reason the *Skeptic's first-order blind spot* (§5.5) rather than the circuit being wrong? | The Skeptic transcript — look for backup heads failing individual `ablate_component` while the full set matters. |

### Red flags — score low wherever you see these

- A verdict of Confirmed with a partial or empty Skeptic evidence transcript.
- A "bias circuit" or "X circuit" generalised from a single prompt pair / template.
- A mechanism story that sounds plausible but has no tool-call result behind it.
- S-EAP synergy heads presented as a discovery without exact-coalition confirmation.
- Any use of the word "novel" for the S-EAP layer — it is a useful engineering
  combination of published methods (integrated Hessians, AtP*, self-repair
  analysis), not a new one.
- A null result reported as just "nothing found" with no substantive reasoning
  (common and *expected* for MLP-heavy arithmetic at small scale — the reasoning
  still has to be there).

---

## 10. Limitations & future work

### Method limitations

- **S-EAP loses sign.** It ranks interaction magnitude, not direction. Usable
  only as a shortlist for exact confirmation.
- **Statistical power.** With a handful of eval prompts, the sign-flip
  permutation test floor is `1/2^N`; nothing survives FDR across ~66 pairs until
  ~20–30 prompts.
- **The Skeptic is still first-order.** It refutes valid backup-containing
  circuits because it tests heads individually (§5.5).
- **Candidate selection.** True backup heads are invisible to first-order EAP *by
  construction*, so "EAP-ranked but sub-threshold" does not reliably surface
  them; DLA write-direction is the better seed but not a complete solution.
- **Mean ablation throughout** is mildly off-distribution; resampling ablation is
  the assumption-free alternative and is not yet the default.
- **Ground truth exists only for GPT-2 IOI.** On every other model / task, recall
  and precision are reported as N/A — only faithfulness and internal consistency
  can be checked.
- **Toy SAE / tuned lens** are deliberately tiny stand-ins (seconds, few tokens);
  they overfit and are not research-grade. Public SAEs exist only for GPT-2 (and
  are currently blocked by an `sae_lens`/pydantic version conflict in this
  environment).
- **Heuristic backend by default.** Much of what "the agents" do is a rule table.
  Claims about agentic confirmation bias need a real LLM backend and a fair
  Prover-only baseline — the current `verification_ablation` hard-codes the
  baseline at 100% acceptance, which is circular.

### Compute limitations

- CPU only in this environment. GPT-2 and sub-1B models are comfortable; 1.5–3B
  need fp16 and run the full suite in 15–40 min; 7B is impractical.
- The full technique suite per behaviour is too slow for Stage B/D — they set
  `deep_techniques=False`.

### Where it can be improved

1. **Second-order Skeptic.** Make `ablate_component` / `minimality_check` test
   synergy-recovered heads as coalitions. This is the single highest-value fix —
   it would let the improved 44%-recall circuit reach a Confirmed verdict.
2. **Paired permutation test + BH-FDR + split-half replication** across all
   `eval_prompts`, as the S-ACDC spec describes. Turns S-EAP from a heuristic
   score into a controlled test.
3. **Resampling ablation** as a first-class option everywhere, with a
   mean-vs-resample comparison arm — this also addresses the open "is self-repair
   partly an ablation artefact?" question.
4. **A second circuit with published ground truth** (Greater-Than on GPT-2,
   docstring) to test whether the S-EAP recall gain generalises beyond IOI.
5. **Real LLM backend + fair baseline** for the verification ablation, via a
   local Qwen or an API model, so the multi-agent contribution is measured rather
   than assumed.
6. **Integrated-gradients EAP** (multi-step, ~20 lines) as a more faithful
   first-order pass under the S-EAP second-order layer.
7. **AtP\* gradient corrections** (QK-fix / GradDrop) so the second-order
   estimate is not built on saturated-softmax-attenuated gradients.

### On venue

Realistic fit for the S-EAP layer as it stands: an **ICML MI workshop /
BlackboxNLP / ATTRIB** contribution, framed as "a synergy-aware extension to
attribution patching that operationalises self-repair detection" — not "a novel
mathematical layer". Main-track would require the improvements above plus
multi-circuit, multi-model faithfulness curves and a direct comparison to AtP\*.

---

*Source of truth is always the code: `automechinterp/eval/matrix_rubric.py` for
the rubric and `automechinterp/eval/agreement.py` for the judge-validation gate,
`experiments/seap/FINDINGS.md` for the S-EAP benchmark, the module docstrings for
everything else.*
