# AutoMechInterp

A runnable implementation of the AutoMechInterp research proposal: a hierarchical
multi-agent system that autonomously performs mechanistic interpretability on a
neural network — profiling it layer by layer, discovering circuits, and
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
                                  finds the specific head(s) responsible
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
open-source LLM (Qwen2.5-7B-Instruct) when you have the hardware for it.

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
  stage_a.py                         the experiment: builds the IOI task on GPT-2, runs
                                      the pipeline, scores it against the published
                                      ground-truth circuit, and runs negative controls

  agents/
    base.py                          the shared ReAct loop every agent runs (ask the
                                      backend what to do next, call the tool, log the
                                      result, repeat until budget runs out)
    network_analyst.py                Stage-0 whole-model profiling; decides which
                                      layers are worth a deeper look
    layer_agent.py                   diagnoses one layer: is it causally load-bearing,
                                      and is the effect attention- or MLP-driven?
    component_agent.py               finds which specific attention head(s) inside a
                                      layer are responsible, via EAP then exact ACDC
    skeptic.py                       the adversarial verifier described above
    judge.py                         reads the Prover's claim + Skeptic's evidence,
                                      assigns the final verdict
    orchestrator.py                  the hypothesis ledger (dedupes claims found by
                                      multiple agents) + budget-aware layer selection

  tools/
    adapter.py                       the architecture-agnostic core: works on any
                                      HuggingFace causal LM by discovering its layer
                                      stack via named_modules() — no assumption that
                                      it's a GPT-2-shaped model. This is where every
                                      actual PyTorch hook, forward pass, and gradient
                                      computation lives register_model,
                                      capture_activations, layer_similarity,
                                      ablate_layer, swap_layer, freeze_and_probe,
                                      patch_layer, plus the head-level patching
                                      primitives used by tier_c.py
    tier_n.py                        Network-Level tools (profile_network,
                                      layerwise_cka_scan, redundancy_scan,
                                      depth_probe_sweep)
    tier_l.py                        Layer-Level tools (logit_lens, patch_layer,
                                      attn_mlp_attribution, layer_role_card)
    tier_c.py                        Component-Level tools, transformer-specific
                                      (run_acdc, run_eap, get_attention_pattern,
                                      get_max_activating_examples, activation_patch)
    tier_v.py                        Verification tools, used only by the Skeptic
                                      (ablate_component, exclusion_ablation,
                                      minimality_check, counterexample_search,
                                      interchange_intervention, steer_with_vector)
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
- **`skeptic.py` + `judge.py`** solve *"how do you stop an agent from convincing
  itself"* — the MAIA confirmation-bias problem — by making falsification a
  separate agent's entire job.
- **`llm_backends.py`** solves *"does this depend on a specific paid API"* — the
  agent logic is identical whether the backend is a free rule table or a local
  open-source model.

## Which models this runs, and how

| Role | Model | Why |
|---|---|---|
| **Target model** (the network being interpreted) | GPT-2 small (124M) | Has a published, well-studied ground-truth circuit (IOI) to score against |
| **Reasoning backend** (the agents' "brain") | Heuristic rule table by default, or `Qwen/Qwen2.5-7B-Instruct` | Open-source, no API key, swappable via one env var |

Both are configurable in `automechinterp/config.py` or via environment variables
— nothing is hardcoded to GPT-2 or Qwen specifically; `adapter.py` would work
the same way against, say, `mistralai/Mistral-7B-v0.1` as the target model.

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
   what to look for.
4. **The Skeptic tries to break the claim** (`[(9,9), (10,10)]`) on 6 fresh
   held-out prompts the discovery agents never saw. `L9H9` averages `+0.192`
   effect — just under the `0.2` "specific effect" cutoff — so the **Judge
   returns Refuted**. This is the verification step working as intended: a
   real signal that didn't generalize past a stricter, independent check is
   correctly not trusted, rather than rubber-stamped.
5. **4 fabricated claims are injected straight into the Skeptic**, bypassing
   discovery entirely. All 4 are correctly refuted — **0% false-confirmation
   rate**.

|  | Value |
|---|---|
| Layer-localization recall | 100% |
| Circuit recall / precision | 11% / 50% (found `L9H9`, one real head, out of 9 ground-truth heads; 1 of 2 claimed heads was real) |
| Judge verdict | Refuted (borderline — see above) |
| Negative controls refuted | 4/4 (0% false-confirmation rate) |
| Tool calls spent | 45 / 120 budget |

## Output files

`--graph` (or a plain redirect/`tee`) saves results into `output/`:

- **`stage_a_<model>_<timestamp>.png`** — 3-panel plot: per-layer causal
  effect (ground-truth layers highlighted green), headline metrics as bars,
  and a text run summary
- **`stage_a_<model>_<timestamp>.json`** — the same numbers, machine-readable,
  for scripting comparisons across runs/models
- **`*.log`** — full terminal transcripts from example runs (every agent
  decision, tool call, and verdict), saved via `tee`

This repo includes real examples of all three from an actual GPT-2 run — open
them directly to see what a run looks like without installing anything.

## Running it

No API key needed for the default setup — everything downloads and runs locally.

```bash
cd "/Users/mansuba/Desktop/MI MAIN RESEARCH/Automatemechanistic code "
pip install -r requirements.txt

python3 main.py                 # full pipeline, heuristic backend, GPT-2 target
python3 main.py --graph         # same, plus saves a PNG + JSON to output/
python3 main.py --gap-matrix    # just print the related-work gap table
python3 main.py --smoke-test-qwen   # proves the open-source-LLM backend works,
                                     # using a small Qwen model so it's fast
```

**CLI flags** (all optional — every one has an equivalent env var in
`automechinterp/config.py`, the flags just make one-off overrides easier):

| Flag | What it changes | Default |
|---|---|---|
| `--target-model <id>` | which model is **interpreted** (any HF causal LM id) | `gpt2` |
| `--backend <kind>` | which brain drives the **agents**: `heuristic`, `hf_local`, `openai`, `anthropic` | `heuristic` |
| `--heavy-model <id>` | which open-source model to load when `--backend hf_local` | `Qwen/Qwen2.5-7B-Instruct` |
| `--graph` | save a PNG + JSON of the results | off |
| `--output-dir <dir>` | where `--graph` saves to | `output` |

**Switch the target model** (the network being interpreted) — note: the
published IOI ground truth only exists for GPT-2, so recall/precision are
reported as `N/A` for anything else, honestly, not guessed:

```bash
python3 main.py --target-model gpt2-medium --graph
```

**Use a real open-source LLM (Qwen2.5-7B-Instruct) as the agents' brain**
instead of the rule table — needs real hardware (GPU or a lot of RAM), still
no API key:

```bash
pip install accelerate
python3 main.py --backend hf_local --heavy-model "Qwen/Qwen2.5-7B-Instruct" --graph
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

**Implemented:** the full agent hierarchy, all Tier N/L/V tools, the ACDC/EAP
circuit-discovery tools, Prover-Skeptic-Judge verification, cost tracking,
content-addressed caching, and Stage A (GPT-2 ground-truth validation).

**Not yet implemented:** SAE-based tools (`sae_layer_profile`,
`run_sae_decompose` — need SAELens + pretrained SAEs), Stage B (Gemma 2 9B
Layer Atlas), Stage C (Mistral vs BioMistral cross-model diff — though
`swap_layer` already exists as the building block for it), the ResNet-50
non-transformer demo, and the formal ablation-study experiments from Section
4.7 of the proposal.
