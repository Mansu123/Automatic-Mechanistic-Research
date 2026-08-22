# AutoMechInterp

A runnable implementation of the AutoMechInterp research proposal: a hierarchical
multi-agent system that autonomously performs mechanistic interpretability on a
neural network — profiling it layer by layer, discovering circuits, decomposing
them into sparse features, diffing fine-tuned models against their base, and
adversarially verifying every claim before trusting it.

## Research question this solves

> Can a hierarchical multi-agent LLM system — whose agent structure mirrors the
> target network's architecture — autonomously analyze a model layer by layer,
> discover and causally verify the circuits within it, and produce a verified
> mechanistic account, without a human researcher driving each step?

Prior automated-interpretability systems fall short in specific, documented ways:

| Prior work | What it couldn't do |
|---|---|
| Dalvi et al. 2019 | Fixed statistical pipeline — can't decide what to test next |
| SASC (2023) | One-pass, single-shot — never follows up on a weak explanation |
| FIND / AIA (2023) | Only tested on synthetic functions, no real network internals |
| MAIA (2024) | Single agent, documented confirmation bias, correlational only |

Run `python3 main.py --gap-matrix` to see exactly which class/function in this
codebase closes each of these gaps.

## How it works

The system mirrors the target network's own structure instead of using one flat
agent with every tool in its context at once:

```
Orchestrator
  -> Network Analyst        profiles the whole model, flags "interesting" layers
       -> Layer Agent (x N)     one per flagged layer, runs in parallel
            -> Component Agent   spawned only if a layer's effect is attention-driven;
                                  finds the specific head(s) responsible, then
                                  decomposes them into sparse (SAE) features
       -> Skeptic                tries to FALSIFY the claimed circuit
            -> Judge               reads Prover's claim + Skeptic's evidence,
                                    assigns Confirmed / Probable / Speculative / Refuted
```

Nothing is trusted just because an agent found it. The **Skeptic** is a
structurally separate agent that never sees the Component Agent's reasoning —
only the claimed circuit — and tries to break it four ways:

1. **`ablate_component`** — does removing it actually hurt the behavior?
2. **`exclusion_ablation`** — does ablating everything *outside* the claim also
   hurt it? (if so, the claim is incomplete)
3. **`minimality_check`** — does every claimed head matter on its own? (if not,
   the claim is padded)
4. **`counterexample_search`** — is there an input where the claimed circuit is
   active but doesn't explain the behavior?

Only a circuit that survives all four gets labeled **Confirmed** by the Judge.

Every agent (Network Analyst, Layer Agent, Component Agent, Skeptic, Judge) is
driven by the same interface — `LLMBackend.decide(system, evidence, tools) ->
next_action` — so the "brain" behind the decisions is swappable with zero
changes to agent logic: a hand-written rule table by default, or a real
open-source LLM (Qwen2.5-7B-Instruct, or a smaller Qwen model — this machine
comfortably handles up to ~3-4B) when you have the hardware for it.

## Project layout

```
main.py                              entrypoint / CLI
requirements.txt                     pip dependencies

automechinterp/
  config.py                          all tunables: which model is being interpreted,
                                      which LLM drives the agents, budgets, thresholds
  gap_work.py                        the gap table above, as data — each entry names
                                      the prior system, its limitation, and the class/
                                      function in this repo that closes it
  llm_backends.py                    the "brain" behind every agent:
                                        HeuristicBackend  - rule-based, no LLM needed
                                        HFLocalBackend    - runs a real open-source LLM
                                                            (Qwen2.5-7B-Instruct by default)
                                        OpenAIBackend / AnthropicBackend - optional, need
                                                            an API key, off by default
  hierarchy.py                       wires the whole pipeline together: Network Analyst
                                      -> Layer Agents -> Component Agents -> Skeptic ->
                                      Judge, with a shared tool-call budget and a
                                      hypothesis ledger that dedupes claims
  stage_a.py                         Stage A: builds the IOI task on GPT-2, runs the
                                      pipeline, scores it against the published
                                      ground-truth circuit, runs negative controls
  behaviors.py                       4 distinct task definitions (coreference, syntax,
                                      factual recall, lexical semantics) used by Stage B
                                      -- every one reuses the SAME hierarchy machinery,
                                      since it was never IOI-specific to begin with
  stage_b.py                         Stage B: runs the full pipeline once per behavior
                                      and assembles a cross-behavior Layer Atlas
  finetune_stand_in.py               builds the base/fine-tuned model pair Stage C
                                      needs: briefly fine-tunes GPT-2 on a small
                                      biomedical corpus (a fast, local stand-in for the
                                      proposal's Mistral-7B/BioMistral-7B pair)
  stage_c.py                         Stage C: CKA layer-alignment between base and
                                      fine-tuned model, then causal layer-transplant
                                      confirmation
  ablations.py                       Sec. 4.7 ablation studies: hierarchy (flat vs
                                      scoped context), verification (self-confirmation
                                      vs Prover-Skeptic-Judge), Network Analyst
                                      (guided vs uniform vs random layer flagging)
  visualize.py                       every plotting function (Stage A/B/C, ablations) --
                                      saves a PNG + JSON pair to output/
  ablations.py, stage_b.py, stage_c.py all reuse hierarchy.py / the Tier N-V tools
                                      directly; none of them needed new plumbing

  agents/
    base.py                          the shared ReAct loop every agent runs (ask the
                                      backend what to do next, call the tool, log the
                                      result, repeat until budget runs out)
    network_analyst.py                Stage-0 whole-model profiling; decides which
                                      layers are worth a deeper look
    layer_agent.py                   diagnoses one layer: is it causally load-bearing,
                                      attention- or MLP-driven, and (if SAEs are
                                      available for this model) how superposed?
    component_agent.py               finds which specific attention head(s) inside a
                                      layer are responsible (EAP then exact ACDC), then
                                      decomposes the finding into sparse SAE features
    skeptic.py                       the adversarial verifier described above
    judge.py                         reads the Prover's claim + Skeptic's evidence,
                                      assigns the final verdict
    orchestrator.py                  the hypothesis ledger (dedupes claims found by
                                      multiple agents) + budget-aware layer selection

  tools/
    adapter.py                       the architecture-agnostic core: works on any
                                      HuggingFace causal LM by discovering its layer
                                      stack via named_modules() — no assumption that
                                      it's a GPT-2-shaped model. register_model,
                                      capture_activations, layer_similarity,
                                      ablate_layer, swap_layer, transplant_layer
                                      (weight-level, for Stage C), freeze_and_probe,
                                      patch_layer, plus head-level patching primitives
    tier_n.py                        Network-Level tools (profile_network,
                                      layerwise_cka_scan, redundancy_scan,
                                      depth_probe_sweep)
    tier_l.py                        Layer-Level tools (logit_lens, patch_layer,
                                      attn_mlp_attribution, sae_layer_profile,
                                      layer_role_card)
    tier_c.py                        Component-Level tools, transformer-specific
                                      (run_acdc, run_eap, get_attention_pattern,
                                      get_max_activating_examples, activation_patch,
                                      run_sae_decompose)
    tier_v.py                        Verification tools, used only by the Skeptic
                                      (ablate_component, exclusion_ablation,
                                      minimality_check, counterexample_search,
                                      interchange_intervention, steer_with_vector)
    sae.py                           SAE tool internals: loads pretrained SAEs via
                                      SAELens and sources activations via
                                      TransformerLens (see "SAE tools" section below
                                      for why that specific combination is required)
    digest.py                        turns raw tensors into short text summaries
                                      before they ever reach an agent's context
                                      (e.g. "L12H5: 83% mass units->tens digit")
    cache.py                         content-addressed cache so identical tool calls
                                      (e.g. the same CKA scan) aren't recomputed
```

## What research problem each piece is actually solving

- **`adapter.py`** solves *"most interpretability tooling assumes a transformer"* —
  it locates a model's decoder-block list generically (`transformer.h` for GPT-2,
  `model.layers` for Qwen2/Llama-family) so the same tool works on either.
- **`tier_c.py`'s `run_acdc`/`run_eap`** solve *"how do you find which specific
  component matters, not just which layer"* — exact per-head causal patching
  (ACDC) and a fast gradient-based approximation (EAP), the same two methods
  the proposal names.
- **`tools/sae.py`** solves *"a causally-important head has an effect size, but
  what does it actually represent?"* — sparse feature decomposition gives a
  finding semantic content, not just a number.
- **`skeptic.py` + `judge.py`** solve *"how do you stop an agent from convincing
  itself"* — the MAIA confirmation-bias problem — by making falsification a
  separate agent's entire job.
- **`llm_backends.py`** solves *"does this depend on a specific paid API"* — the
  agent logic is identical whether the backend is a free rule table or a local
  open-source model.
- **`ablations.py`** solves *"why should anyone believe the hierarchy/verification
  design actually helps"* — measures it directly instead of asserting it.

## Which models this runs, and how

| Role | Model | Why |
|---|---|---|
| **Target model** (the network being interpreted) | GPT-2 small (124M) | Has a published, well-studied ground-truth circuit (IOI) to score against |
| **Reasoning backend** (the agents' "brain") | Heuristic rule table by default, or a Qwen model (`hf_local`) | Open-source, no API key, swappable via one flag |

Both are configurable via `--target-model` / `--backend` (or the equivalent env
vars in `automechinterp/config.py`) — nothing is hardcoded to GPT-2 or Qwen
specifically; `adapter.py` would work the same way against, say,
`mistralai/Mistral-7B-v0.1` as the target model. On this development machine,
`hf_local` has been verified up through a 3B Qwen model comfortably; a 7B
model works but expect a long first-time download and a hot laptop on a
fanless machine — see the CLI section below for the practical guidance this
was learned from.

## SAE tools

`sae_layer_profile` (Tier L) and `run_sae_decompose` (Tier C) were the two
tools originally left unimplemented. They use SAELens' public pretrained
GPT-2 residual-stream SAE release (`gpt2-small-res-jb`, Joseph Bloom) — the
same public-artifact role GemmaScope plays for Gemma 2 in the original
proposal.

**One real gotcha worth documenting**: this SAE was trained on activations
from a TransformerLens `HookedTransformer`, which applies weight processing
(LayerNorm folding, weight centering) that plain `transformers.GPT2Model`
does not. Feeding it activations captured the normal way (via `adapter.py`'s
plain-HF hooks) produces badly wrong numbers — verified: FVU (fraction of
variance unexplained) ≈ 16 and L0 (active features/token) ≈ 2000, i.e. the
SAE barely reconstructing noise. Sourcing the same activations from a
TransformerLens model instead fixes it completely — FVU ≈ 0.0000, L0 ≈ 50-60,
matching the SAE's published dashboard numbers. `tools/sae.py` is the only
module in this codebase that loads a `TransformerLens` model for this reason;
everything else stays on the architecture-agnostic HF path. This is also why
SAE support is currently GPT-2-only: it needs both a TransformerLens
implementation of the target model AND a matching public SAE release.

In the hierarchy, a Layer Agent runs `sae_layer_profile` on any causally
load-bearing layer to check for superposition, and a Component Agent runs
`run_sae_decompose` on a confirmed head to report which sparse features it
correlates with — turning "L9H9 matters" into "L9H9's output correlates with
SAE features {24120, 1721, 20971, ...}".

## Ablation studies (Sec. 4.7)

`python3 main.py --ablations --graph` runs three real comparisons against the
same GPT-2 IOI task, not a table of predicted numbers:

- **Hierarchy ablation**: a flat single agent's evidence log grows linearly as
  it checks more layers (122 → 1665 characters over 12 layers); the
  hierarchical version's largest single Layer Agent never exceeds ~482
  characters, because each agent only ever sees its own scope. **3.45x context
  reduction**, and — more importantly — a qualitatively different growth
  curve (linear vs. flat), which is the actual claim Sec. 3.1 makes.
- **Verification ablation**: a self-confirmation baseline (no Skeptic) accepts
  100% of 4 deliberately fabricated claims by construction; the real
  Prover-Skeptic-Judge pipeline, run on the exact same claims, refutes all 4
  (**0% false-confirmation rate**).
- **Network Analyst ablation**: CKA+redundancy-guided layer flagging vs.
  uniform-all-layers vs. random-subset (averaged over 5 draws). Honest
  finding: on GPT-2's shallow 12-layer stack, all three converge (100% /
  100% / 96% recall) — guided flagging only excludes 1 of 12 layers here, so
  there's little room to show an advantage. The proposal's own framing
  predicts this trade-off should widen on deeper models; this codebase's
  `network_analyst.py` already switches strategy by depth (exhaustive scan
  under ~16 layers, CKA-guided above that) for exactly this reason.

## Stage B — Layer Atlas (Sec. 4.4)

`python3 main.py --stage-b --graph` runs the identical hierarchy across 4
distinct behaviors (scaled down from the proposal's 24, one per named
category — coreference, syntactic agreement, factual recall, lexical
semantics) and assembles a **Layer Atlas**: for every layer, which behaviors'
Component Agents found a *confirmed circuit* there (not just "was
investigated" — Network Analyst flagging is deliberately permissive, so the
atlas is built from the much more selective circuit-level evidence).

A real, coherent finding from an actual run: layers 0-6 have no confirmed
circuit for any tested behavior (early layers build general representations,
not task-specific circuits); layer 9 is a cross-behavior **hub** (circuits
found for 3 of 4 behaviors — coreference, factual recall, lexical semantics);
layers 7, 8, and 10 each have exactly one behavior-specific circuit. This
matches the broader interpretability literature's picture of GPT-2's
mid-to-late layers doing general-purpose work before output-specific layers
take over.

## Stage C — Cross-Model Layer Diff (Sec. 4.5)

`python3 main.py --stage-c --graph` runs the proposal's Mistral-7B vs
BioMistral-7B methodology at a scale this machine can actually run: GPT-2
base vs. `finetune_stand_in.py`'s output, a GPT-2 briefly fine-tuned locally
on a small biomedical corpus (same architecture before/after by construction,
same "domain adaptation without changing the base model" relationship
BioMistral has to Mistral).

1. **Alignment**: per-layer CKA between base and fine-tuned activations on a
   shared general+biomedical probe set.
2. **Causal confirmation**: `adapter.transplant_layer()` — a new primitive,
   distinct from the existing `swap_layer()` (which swaps *activations*
   within one model) — substitutes a layer's actual *weights* from the
   fine-tuned model into the base model and re-measures a biomedical-capability
   metric (logit preference for domain-appropriate completions).

A real result from an actual run: CKA drift concentrates sharply in the last
3 layers (L11 drops to 0.756 vs. >0.97 everywhere else — representational
change is NOT spread evenly). Transplanting those three layers'
weights transfers 10-12% of the base/fine-tuned capability gap each, vs. only
1% for a control transplant of the least-changed layer (L0) — a genuine
causal confirmation that the CKA-flagged layers are the ones that matter, not
just correlated with the diff.

## How the evaluation works (Stage A)

`stage_a.py` runs the **Indirect Object Identification (IOI)** task — "When John
and Mary went to the store, John gave a drink to ___" should complete with the
un-repeated name — because Wang et al. (2022) already published GPT-2's exact
circuit for it (which heads, which layers), giving us ground truth to check
the pipeline's output against instead of just trusting it.

For every run it reports:

- **Layer-localization recall** — did the Network Analyst flag the real
  ground-truth layers?
- **Circuit recall / precision** — of the heads the pipeline claimed, how many
  are real, and how many of the real ones did it find?
- **Judge verdict** — Confirmed / Probable / Speculative / Refuted on the final
  merged claim
- **Negative-control refutation rate** — 4 deliberately wrong claims are
  injected straight into the Skeptic (bypassing discovery); a correct Skeptic
  refutes all 4. This measures the false-confirmation rate directly.
- **Tool-call budget spent** and **cache hit rate** — the cost-efficiency
  metrics the proposal asks for.

## What a real run actually found (GPT-2 walkthrough)

This is a real result from this codebase, not a mockup — the saved
[`output/`](output/) folder in this repo has the full log and the plot from
the run described below.

Running `python3 main.py --graph` with the heuristic backend against GPT-2
small produces, step by step:

1. **Network Analyst** runs `profile_network`, `layerwise_cka_scan`, then an
   exhaustive `redundancy_scan` (GPT-2's 12 layers are cheap enough to check
   all of them rather than trust the coarse CKA signal alone) and flags 11 of
   12 layers as worth investigating — **100% layer-localization recall**
   against the published ground-truth layers `[7, 8, 9, 10, 11]`.
2. **11 Layer Agents run in parallel.** Most immediately report near-zero
   causal effect from `patch_layer` and stop (no wasted deeper analysis).
   Layers 9, 10, and 11 show real effect — the plot's left panel shows this
   ramping up cleanly through the ground-truth layers (0.45, 0.94, 1.00
   fraction of the clean/corrupted gap recovered).
3. **Component Agents spawn** on those three layers. `run_eap` (cheap
   gradient approximation) flags `L9H9` with a score of `+2.22` — an order of
   magnitude above every other head in that layer. `run_acdc` (exact
   per-head patching) confirms it: `L9H9` alone recovers `+0.29` of the
   metric gap. **`L9H9` is a real entry in Wang et al.'s published GPT-2 IOI
   circuit** (a name-mover head) — the pipeline found it without being told
   what to look for. A Component Agent then runs `run_sae_decompose` on the
   finding, reporting which SAE features it correlates with.
4. **The Skeptic tries to break the claim** on 6 fresh held-out prompts the
   discovery agents never saw. Verdicts vary run to run since IOI prompts are
   resampled each time — sometimes Refuted (a real signal that didn't
   generalize past a stricter check — e.g. `L9H9` at `+0.192`, just under the
   `0.2` cutoff), sometimes **Confirmed** (Stage B's antonym-prediction
   behavior, for instance, survived the full falsification suite on `L9H7`).
   Both outcomes are the verification step working as intended.
5. **4 fabricated claims are injected straight into the Skeptic**, bypassing
   discovery entirely. All 4 are correctly refuted — **0% false-confirmation
   rate**.

|  | Value |
|---|---|
| Layer-localization recall | 100% |
| Circuit recall / precision | 11% / 50% (found `L9H9`, one real head, out of 9 ground-truth heads; 1 of 2 claimed heads was real) |
| Negative controls refuted | 4/4 (0% false-confirmation rate) |
| Tool calls spent | 45-50 / 120 budget |

## Output files

`--graph` (or a plain redirect/`tee`) saves results into `output/`:

- **`stage_a_<model>_<timestamp>.png/.json`** — per-layer causal effect,
  headline metrics, run summary
- **`ablations_<model>_<timestamp>.png/.json`** — the three Sec. 4.7
  comparisons above
- **`stage_b_atlas_<model>_<timestamp>.png/.json`** — the cross-behavior Layer
  Atlas heatmap
- **`stage_c_diff_<model>_<timestamp>.png/.json`** — the CKA alignment +
  causal transplant confirmation
- **`*.log`** — full terminal transcripts from example runs (every agent
  decision, tool call, and verdict), saved via `tee`

This repo includes real examples of all of these from actual runs — open them
directly to see what a run looks like without installing anything.

## Running it

No API key needed for the default setup — everything downloads and runs locally.

```bash
cd "/Users/mansuba/Desktop/MI MAIN RESEARCH/Automatemechanistic code "
pip install -r requirements.txt

python3 main.py                 # Stage A, heuristic backend, GPT-2 target
python3 main.py --graph         # same, plus saves a PNG + JSON to output/
python3 main.py --ablations --graph   # Sec. 4.7 ablation studies
python3 main.py --stage-b --graph     # multi-behavior Layer Atlas
python3 main.py --stage-c --graph     # cross-model layer diff (base vs fine-tuned)
python3 main.py --gap-matrix    # just print the related-work gap table
python3 main.py --smoke-test-qwen   # proves the open-source-LLM backend works,
                                     # using a small Qwen model so it's fast
```

**CLI flags** (all optional — most have an equivalent env var in
`automechinterp/config.py`, the flags just make one-off overrides easier):

| Flag | What it changes | Default |
|---|---|---|
| `--target-model <id>` | which model is **interpreted** (any HF causal LM id) | `gpt2` |
| `--backend <kind>` | which brain drives the **agents**: `heuristic`, `hf_local`, `openai`, `anthropic` | `heuristic` |
| `--heavy-model <id>` | which open-source model to load when `--backend hf_local` | `Qwen/Qwen2.5-7B-Instruct` |
| `--graph` | save a PNG + JSON of the results | off |
| `--output-dir <dir>` | where `--graph` saves to | `output` |
| `--ablations` | run Sec. 4.7 ablations instead of Stage A | off |
| `--stage-b` | run the multi-behavior Layer Atlas instead of Stage A | off |
| `--stage-c` | run the cross-model layer diff instead of Stage A | off |

**Switch the target model** (the network being interpreted) — note: the
published IOI ground truth only exists for GPT-2, so recall/precision are
reported as `N/A` for anything else, honestly, not guessed:

```bash
python3 main.py --target-model gpt2-medium --graph
```

**Use a real open-source LLM as the agents' brain** instead of the rule table
— still no API key. This machine has been verified comfortable up through a
3B-parameter Qwen model; a 7B model works but expect a genuinely long
first-time download (multi-GB) and a hot laptop if you're on a fanless
machine — that's normal, not a bug, and only affects the one-time download +
inference speed, not correctness:

```bash
pip install accelerate
python3 main.py --backend hf_local --heavy-model "Qwen/Qwen2.5-3B-Instruct" --graph
```

**Save the full terminal transcript alongside the graph:**

```bash
python3 main.py --graph 2>&1 | tee "output/stage_a_$(date +%Y%m%d_%H%M%S).log"
```

Only needed if you want a paid API model driving the agents instead:

```bash
pip install openai   # or anthropic
export OPENAI_API_KEY=sk-...
python3 main.py --backend openai
```

## What's implemented vs. not (from the full research proposal)

**Implemented:** the full agent hierarchy, all Tier N/L/C/V tools including
SAE decomposition, Prover-Skeptic-Judge verification, cost tracking,
content-addressed caching, Stage A (GPT-2 ground-truth validation), Stage B
(multi-behavior Layer Atlas), Stage C (cross-model layer diff with causal
transplant confirmation), and the Sec. 4.7 ablation studies (hierarchy,
verification, Network Analyst flagging strategy).

**Not yet implemented:** Stage B/C at the proposal's original scale (Gemma 2
9B, Mistral-7B/BioMistral-7B — the methodology is proven identical on small
stand-ins instead, deliberately, after a 3B-model download/thermal test on
this machine made the full-scale versions impractical here); vision-model
support (ResNet/AlexNet-style CNN classifiers, and eventually object
detectors like YOLO — a fundamentally different output space that the
Skeptic/Judge causal-patching metric isn't built for yet); the
cost-mechanism ablation (with/without caching — already visible via
`cache_hit_rate` in every run, not yet a standalone comparison) and a formal
backbone-comparison ablation (heuristic vs. `hf_local` — demonstrated
informally, not yet run as a scored comparison).
