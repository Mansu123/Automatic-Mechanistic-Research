# Comprehensive Guide: Latest Commit (`2f11634`) & S-EAP Deep Dive

> **Summary Document**: Overview of the architectural upgrades, Stage D evaluation pipeline, the ~70-technique mechanistic toolkit, evaluation model matrix, human review apparatus, and an in-depth breakdown of **S-EAP** (Synergy-Aware Edge Attribution Patching).

---

## Table of Contents
1. [Overview of Commit `2f11634`](#1-overview-of-commit-2f11634)
2. [S-EAP Deep Dive (Synergy-Aware EAP)](#2-s-eap-deep-dive-synergy-aware-eap)
   - [The Problem S-EAP Solves (Backup Heads & Negative Movers)](#the-problem-s-eap-solves)
   - [Mathematical Formulation & Hessian Approximation](#mathematical-formulation--hessian-approximation)
   - [The $O(M)$ Computational Advantage](#the-om-computational-advantage)
   - [Empirical Benchmark Results (GPT-2 & Qwen2.5)](#empirical-benchmark-results)
   - [Honest Read: Strengths, Blind Spots, and Next Steps](#honest-read-strengths-and-limitations)
3. [The ~70-Technique Mechanistic Toolkit](#3-the-70-technique-mechanistic-toolkit)
4. [6 New Specialized Agents in the Hierarchy](#4-6-new-specialized-agents-in-the-hierarchy)
5. [Stage D Evaluation & Automated Report Generator](#5-stage-d-evaluation--automated-report-generator)
6. [Evaluation Model Matrix & Automated Sweep Runner](#6-evaluation-model-matrix--automated-sweep-runner)
7. [Human Review Apparatus & Inter-Rater Reliability](#7-human-review-apparatus--inter-rater-reliability)
8. [Quick Command Reference](#8-quick-command-reference)

---

## 1. Overview of Commit `2f11634`

Commit `2f11634` (`Add Stage D evaluation, ~70-technique toolkit, eval model matrix, and human-review apparatus`) is a major expansion (+17,672 lines across 87 files) that transforms the repository from a basic multi-agent circuit explorer into an **end-to-end automated scientific research suite**.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                               COMMIT 2f11634 ARCHITECTURE                                │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  [ Stage D Pipeline ]          [ Technique Toolkit ]         [ Evaluation Matrix ]       │
│  • stage_d_eval.py             • ~70 techniques (13 modules) • ~25 HF model ladders      │
│  • Automated Report Writer     • Model-agnostic hooks        • Disk-safe sequential runs │
│  • RubricJudge (5 criteria)    • MD + JSON export            • --purge-downloads support │
│                                                                                          │
│  [ Deep-Technique Agents ]     [ S-EAP 2nd-Order Screen ]    [ Human Review Suite ]      │
│  • Lens, Probe, Feature        • Pairwise Hessian term       • rubrics.md & templates    │
│  • Weight, Safety, Steering    • O(M) backward passes        • Inter-rater kappa scorer  │
│  • Integrated into Hierarchy   • Recovers backup coalitions  • Agent x Layer matrix      │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. S-EAP Deep Dive (Synergy-Aware EAP)

### The Problem S-EAP Solves
In standard mechanistic interpretability workflows, circuits are discovered using **first-order attribution** (single-edge ACDC or first-order EAP). These methods calculate the marginal effect $\phi_1(i)$ of ablating a single component $i$ in isolation.

**The Fundamental Blind Spot:**
Language models frequently contain **backup heads** and **inhibitory/negative name-movers**:
1. When primary head $j$ is active, backup head $i$ remains quiet ($\phi_1(i) \approx 0$).
2. When backup head $i$ is ablated alone, primary head $j$ handles the load ($\phi_1(i) \approx 0$).
3. First-order methods conclude that *neither* head is important and discard both!

In GPT-2 Indirect Object Identification (IOI), first-order ACDC achieves only **11% recall** (1 of 9 ground-truth heads), completely missing negative name-movers `L10H7` and `L11H10`.

---

### Mathematical Formulation & Hessian Approximation
S-EAP approximates the mixed second-order partial derivative (the pairwise Hessian cross-term $\frac{\partial^2 \mathcal{L}}{\partial A_i \partial A_j}$) using finite differences across clean and corrupt activations:

$$\text{Syn}(i, j) \approx \Big( \nabla_{A_i} \mathcal{L}_{\text{clean}} - \nabla_{A_i} \mathcal{L}_{\text{ablate}(j)} \Big) \cdot \Big( A_i^{\text{clean}} - A_i^{\text{corrupt}} \Big)$$

Where:
- $\nabla_{A_i} \mathcal{L}_{\text{clean}}$ is the gradient of task loss with respect to head $i$'s activations on the uncorrupted prompt.
- $\nabla_{A_i} \mathcal{L}_{\text{ablate}(j)}$ is the gradient at head $i$ when candidate head $j$ is mean-ablated.
- $A_i^{\text{clean}} - A_i^{\text{corrupt}}$ is the activation patch difference.

A **backup / synergy pair** is formally identified when:
$$\phi_1(i) \approx 0, \quad \phi_1(j) \approx 0, \quad \text{and} \quad |\text{Syn}(i, j)| \gg 0$$

---

### The $O(M)$ Computational Advantage
- **Exact Coalition Patching ($O(M^2)$):** Measuring all pairs among $M$ candidate heads requires $M(M-1)/2$ forward intervention passes with combinatorial overhead.
- **S-EAP ($O(M)$):** By ablating candidate head $j$ and computing a single backward pass, **the entire interaction row $\text{Syn}(*, j)$ for all other heads $i$ is computed simultaneously**.

---

### Empirical Benchmark Results
*Benchmark run on GPT-2 Small (IOI ground truth) and Qwen2.5-0.5B (`experiments/seap/`):*

| Metric | First-Order ACDC | + S-EAP Second-Order Pass |
| :--- | :---: | :---: |
| **GPT-2 IOI Circuit Recall** | **11%** (1/9 heads) | **44%** (4/9 heads) |
| **Negative Name-Movers (`L10H7`, `L11H10`)** | ❌ Missed completely | ✅ **Recovered** |
| **Magnitude Correlation ($\text{Spearman}(|\text{S-EAP}|, |\text{Exact}|)$)** | — | **$\rho = +0.653$ ($p = 8.9 \times 10^{-11}$)** |
| **Sign Correlation ($\text{Spearman}(\text{signed}, \text{signed})$)** | — | $\rho = -0.118$ ($p = 0.30$, not significant) |
| **Circuit Faithfulness** | 0.314 | **0.355** |

---

### Honest Read: Strengths and Limitations
1. **Strengths:** S-EAP is an effective, computationally cheap **magnitude screening filter** that uncovers multi-head backup coalitions that first-order methods miss.
2. **Limitation (Sign Loss):** First-order gradient differences lose sign information (cannot distinguish whether head $i$ helps or cancels head $j$). Therefore, S-EAP should be used to **shortlist candidate pairs**, which are then verified using exact coalition ablation (`ablate_head_set`).
3. **The Skeptic Blind Spot:** Because synergy-recovered heads have $\phi_1 \approx 0$ by definition, a single-head Skeptic will refute them individually. The Skeptic must be upgraded to perform coalition-level ablations on partner pairs.

---

## 3. The ~70-Technique Mechanistic Toolkit

Located in [`automechinterp/techniques/`](file:///home/dlcv/Desktop/research/cohort2/Automatic-Mechanistic-Research/automechinterp/techniques/), this library implements ~70 techniques from the *learnmechinterp* curriculum across 13 modular categories:

| Module | Category | Key Implemented Techniques |
| :--- | :--- | :--- |
| `basic.py` | Basic Probing & Lens | Logit lens, tuned lens, direct logit attribution (DLA), attention entropy, rank-1 projections |
| `causal.py` | Causal Interventions | Activation patching, path patching, causal scrubbing, noise/resample ablation, knockouts |
| `circuits.py` | Circuit Discovery | ACDC, EAP, S-EAP, QK/OV decomposition, copy-suppression detection, attribution graphs |
| `probing.py` | Linear & Non-Linear Probing | Mass-mean probing, logistic probes, probe generalization, contrastive probing |
| `steering.py` | Activation Steering | CAA (Contrastive Activation Addition), steering vectors, concept suppression |
| `editing.py` | Knowledge Editing | ROME (Rank-One Model Editing), MEMIT, gradient-based factual patching |
| `superposition.py` | SAEs & Dictionary Learning | Top-K SAE decomposition, feature sparsity, cross-coder attribution, dead latent detection |
| `feature_geometry.py` | Representation Geometry | Intrinsic dimensionality, representation drift, simplex volume, polysemanticity metrics |
| `model_diffing.py` | Model Comparison | Weight diffing, activation CKA, functional similarity, layerwise representation shifts |
| `hidden_state.py` | Hidden State Dynamics | Residual stream trajectory, token clustering, subspace angles, PCA dynamics |
| `blackbox.py` | Behavioral Auditing | Consistency checks, counterfactual sensitivity, adversarial prompt generation |
| `weight_space.py` | Weight Spectral Analysis | Singular Value Decomposition (SVD), spectral norms, stable rank, low-rank factorization |
| `safety.py` | Safety & Alignment | Toxicity representation probes, refusal directions, jailbreak vector monitoring |

*All techniques operate on generic forward/backward hooks on `adapter.ModelHandle` without external dependencies.*

---

## 4. 6 New Specialized Agents in the Hierarchy

Commit `2f11634` wires 6 deep-technique agents into [`automechinterp/hierarchy.py`](file:///home/dlcv/Desktop/research/cohort2/Automatic-Mechanistic-Research/automechinterp/hierarchy.py):

```
                                [ Orchestrator ]
                                       │
         ┌───────────────┬─────────────┴─────────────┬───────────────┐
         ▼               ▼                           ▼               ▼
  [Network Analyst] [Weight Agent]            [Safety Agent]   [Layer Agents]
   (CKA / Skips)    (SVD / Spectra)           (Refusal Dirs)    (Patch / Lens)
                                                                     │
                                                                     ▼
                                                             [Component Agents]
                                                             (ACDC / S-EAP / SAE)
                                                                     │
                                                                     ▼
                                                             [Steering Agent]
                                                             (CAA Intervention)
                                                                     │
                                                                     ▼
                                                              [Skeptic Agent]
                                                             (Adversarial Test)
                                                                     │
                                                                     ▼
                                                               [Rubric Judge]
                                                             (5-Dimension Score)
```

1. **Lens Agent (`lens_agent.py`)**: Runs logit lens, tuned lens, and unembedding projections on active layers.
2. **Probe Agent (`probe_agent.py`)**: Trains mass-mean linear probes on residual streams to test concept linearly separable states.
3. **Feature Agent (`feature_agent.py`)**: Performs sparse autoencoder (SAE) feature decomposition and dictionary extraction.
4. **Weight Agent (`weight_agent.py`)**: Analyzes whole-model weight matrices (SVD spectra, low-rank structure).
5. **Safety Agent (`safety_agent.py`)**: Identifies safety vectors, refusal representations, and alignment directions.
6. **Steering Agent (`steering_agent.py`)**: Performs constructive activation steering (CAA) to test if adding extracted directions induces behavior before final adjudication.
7. **Rubric Judge (`rubric_judge.py`)**: Evaluates evidence against a strict 5-dimension rubric (1–5 scale).

---

## 5. Stage D Evaluation & Automated Report Generator

Stage D ([`stage_d_eval.py`](file:///home/dlcv/Desktop/research/cohort2/Automatic-Mechanistic-Research/automechinterp/stage_d_eval.py)) runs the hierarchy across behaviors and produces standardized, human-readable research reports stored in `human_review/reports/`:

### Rubric Dimensions (Scored 1 to 5)
1. **Soundness (1–5):** Are causal claims supported by specific ablation and counterexample testing?
2. **Completeness (1–5):** Does the discovered circuit explain the full behavior gap without unexplained heads?
3. **Minimality (1–5):** Is every component in the circuit necessary (no redundant elements)?
4. **Constructive Validation (1–5):** Does activation steering or patching with the circuit faithfully reproduce behavior?
5. **Readability & Structure (1–5):** Is the finding clearly documented with attention patterns and logit projections?

---

## 6. Evaluation Model Matrix & Automated Sweep Runner

[`run_eval_matrix.py`](file:///home/dlcv/Desktop/research/cohort2/Automatic-Mechanistic-Research/run_eval_matrix.py) manages multi-model sweeps across ~25 models defined in [`eval/model_matrix.py`](file:///home/dlcv/Desktop/research/cohort2/Automatic-Mechanistic-Research/automechinterp/eval/model_matrix.py):

- **Model Ladders:**
  - `gpt2_ladder`: `gpt2` (124M), `gpt2-medium` (355M), `gpt2-large` (774M), `gpt2-xl` (1.5B)
  - `pythia_ladder`: `EleutherAI/pythia-70m`, `160m`, `410m`, `1b`, `1.4b`, `2.8b`
  - `qwen25_ladder`: `Qwen/Qwen2.5-0.5B`, `1.5B`, `3B`, `7B`
  - `opt_ladder`: `facebook/opt-125m`, `350m`, `1.3b`, `2.7b`
  - `llama_ladder`: `meta-llama/Llama-3.2-1B`, `3B`
  - `gemma_ladder`: `google/gemma-2-2b`, `9b`
- **Memory & Disk Safe:** Sequential pipeline: Downloads weights $\rightarrow$ Runs evaluation $\rightarrow$ Frees GPU/RAM $\rightarrow$ (Optional `--purge-downloads`) deletes HuggingFace cache to prevent disk exhaustion.

---

## 7. Human Review Apparatus & Inter-Rater Reliability

The [`human_review/`](file:///home/dlcv/Desktop/research/cohort2/Automatic-Mechanistic-Research/human_review/) directory provides tools for validating AI-generated reports against human evaluations:
- **`rubrics.md`**: Official human-reviewer grading handbook.
- **`scoring_template.csv`**: Structured template for recording human review scores.
- **`aggregate_scores.py`**: Calculates **Cohen’s $\kappa$** and **Fleiss’ $\kappa$** to measure inter-rater reliability between human judges and the automated AI `RubricJudge`.
- **`agent_layer_matrix.py`**: Visualizes which agents inspected which layers across all reports.

---

## 8. Quick Command Reference

```bash
# 1. Run S-EAP benchmark validation
python3 experiments/seap/seap_benchmark_v2.py

# 2. Run the ~70 technique toolkit on GPT-2
python3 -m automechinterp.techniques.runner --model gpt2 --output-dir experiments/techniques/

# 3. Run Stage D evaluation on a single model and generate human-readable reports
python3 main.py --target-model gpt2 --stage-d --ai-judge

# 4. Check wiring across all models in the evaluation matrix (fast dry-run)
python3 run_eval_matrix.py --check

# 5. Run sequential evaluation matrix with automatic disk cache purging
python3 run_eval_matrix.py --ladder gpt2_ladder --purge-downloads --skip-done

# 6. Compute human vs. AI inter-rater agreement scores
python3 human_review/aggregate_scores.py
```

