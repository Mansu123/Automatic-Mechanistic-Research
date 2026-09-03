"""Stage A -- Ground-Truth Validation on GPT-2 (Sec. 4.3).

Task: Indirect Object Identification (IOI, Wang et al. 2022, "Interpretability
in the Wild", ICLR 2023) -- "When A and B went to the store, S gave a drink
to ___" should complete with the other name (the Indirect Object), not the
subject S. GT_HEADS below is the paper's simplified name-mover / negative
name-mover / S-inhibition head set for GPT-2 small, used here only as a
recall/precision reference for H1 (Sec. 5.1), not re-derived by us.

This is also the "few for test at last" step: it runs the full
Network Analyst -> Layer Agents -> Component Agents -> Skeptic -> Judge
pipeline end to end on a real 124M-parameter model, plus Sec. 4.3's negative
controls (deliberately wrong claims the Skeptic must refute).
"""
from __future__ import annotations

import random

import torch

from . import config
from .hierarchy import run_hierarchy
from .tools import adapter
from .agents.base import ToolCallBudget, LOG
from .agents.skeptic import build_skeptic
from .agents.judge import adjudicate

GT_NAME_MOVERS = [(9, 9), (9, 6), (10, 0)]
GT_NEGATIVE_NAME_MOVERS = [(10, 7), (11, 10)]
GT_S_INHIBITION = [(7, 3), (7, 9), (8, 6), (8, 10)]
# Wang et al. 2022, Fig. 2 -- "backup name mover" heads: near-zero marginal
# effect until a primary name mover is ablated, at which point they take over.
# The canonical target for S-EAP / synergy-aware discovery.
GT_BACKUP_NAME_MOVERS = [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)]
GT_HEADS = GT_NAME_MOVERS + GT_NEGATIVE_NAME_MOVERS + GT_S_INHIBITION
GT_LAYERS = sorted({l for l, _ in GT_HEADS})

_TEMPLATE = "When {A} and {B} went to the store, {S} gave a drink to"
_NAMES = ["John", "Mary", "Alice", "Bob", "Sarah", "Tom"]


def _make_ioi_pair(rng: random.Random):
    a, b = rng.sample(_NAMES, 2)
    s_name = a
    io_name = b
    clean_prompt = _TEMPLATE.format(A=a, B=b, S=s_name)
    # corrupted run: swap which name plays subject vs object-of-repetition,
    # matching the standard ABC-ABB IOI corruption (name identities kept,
    # role assignment flipped, so the model's easiest shortcut -- surface
    # token identity -- is controlled for).
    corrupted_prompt = _TEMPLATE.format(A=b, B=a, S=io_name)
    return clean_prompt, corrupted_prompt, io_name, s_name


def build_ioi_task(handle: adapter.ModelHandle, n_eval: int = 6, seed: int = 0) -> dict:
    rng = random.Random(seed)
    clean_prompt, corrupted_prompt, io_token, s_token = _make_ioi_pair(rng)
    eval_prompts = []
    for _ in range(n_eval):
        cp, _, io_t, s_t = _make_ioi_pair(rng)
        eval_prompts.append((cp, io_t, s_t))

    def task_metric_fn() -> float:
        total = 0.0
        for prompt, io_t, s_t in eval_prompts:
            batch = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
            # [-1] not [0] -- see behaviors.py's _make_task for why (cross-tokenizer safety).
            io_id = handle.tokenizer.encode(" " + io_t)[-1]
            s_id = handle.tokenizer.encode(" " + s_t)[-1]
            with torch.no_grad():
                logits = handle.model(**batch).logits[0, -1]
            total += (logits[io_id] - logits[s_id]).item()
        return total / len(eval_prompts)

    probe_texts = [p for p, _, _ in eval_prompts] + [clean_prompt]
    n_heads = handle.model.config.num_attention_heads if hasattr(handle.model.config, "num_attention_heads") \
        else handle.model.config.n_head

    # contrastive sets for the Probe / Feature / Steering agents: clean vs
    # role-swapped (corrupted) IOI prompts differ exactly in the task variable.
    cpos, cneg = [clean_prompt], [corrupted_prompt]
    for _ in range(5):
        c, x, _, _ = _make_ioi_pair(rng)
        cpos.append(c)
        cneg.append(x)

    return {
        "behavior": "Indirect Object Identification (IOI): predict the un-repeated name",
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "io_token": io_token,
        "s_token": s_token,
        "eval_prompts": eval_prompts,
        "probe_texts": probe_texts,
        "contrastive_pos": cpos,
        "contrastive_neg": cneg,
        "task_metric_fn": task_metric_fn,
        "n_heads": n_heads,
    }


def score_against_ground_truth(result: dict, n_layers: int, model_id: str = "gpt2") -> dict:
    """GT_HEADS/GT_LAYERS are Wang et al.'s published GPT-2-small IOI circuit --
    they are not valid ground truth for any other model. If `model_id` isn't
    GPT-2, recall/precision are reported as None (not a fabricated number)."""
    ground_truth_valid = model_id in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")

    flagged = set(result["flagged_layers"])
    claimed = set(result["claimed_heads"])

    if not ground_truth_valid:
        return {
            "ground_truth_valid": False,
            "layer_localization_recall": None,
            "flagged_layers": sorted(flagged),
            "circuit_recall": None,
            "circuit_precision": None,
            "true_positive_heads": [],
            "claimed_heads": sorted(claimed),
        }

    layer_recall = len(flagged & set(GT_LAYERS)) / len(GT_LAYERS) if GT_LAYERS else 0.0
    true_positives = claimed & set(GT_HEADS)
    circuit_recall = len(true_positives) / len(GT_HEADS) if GT_HEADS else 0.0
    circuit_precision = len(true_positives) / len(claimed) if claimed else 0.0

    return {
        "ground_truth_valid": True,
        "layer_localization_recall": layer_recall,
        "flagged_layers": sorted(flagged),
        "circuit_recall": circuit_recall,
        "circuit_precision": circuit_precision,
        "true_positive_heads": sorted(true_positives),
        "claimed_heads": sorted(claimed),
    }


def run_negative_controls(handle: adapter.ModelHandle, task: dict, backend_kind: str,
                           backend_kwargs: dict | None = None, n_controls: int = 4, seed: int = 0) -> dict:
    """Sec. 4.3 point 4: inject deliberately wrong claims; the Skeptic must
    refute them. Bypasses discovery entirely -- these heads are picked to be
    causally irrelevant to IOI (very early layers, arbitrary head index)."""
    rng = random.Random(seed)
    candidate_wrong_heads = [(l, h) for l in range(min(3, handle.n_layers))
                              for h in range(task["n_heads"])]
    rng.shuffle(candidate_wrong_heads)
    wrong_claims = candidate_wrong_heads[:n_controls]

    all_heads = [(l, h) for l in range(handle.n_layers) for h in range(task["n_heads"])]
    refuted = 0
    details = []
    for claim in wrong_claims:
        budget = ToolCallBudget(global_remaining=[20], per_agent_limit=20)
        LOG.emit("Orchestrator", f"[negative control] injecting deliberately wrong claim {[claim]}")
        sk_agent, sk_state = build_skeptic(handle, [claim], all_heads, task["eval_prompts"],
                                            task["behavior"], backend_kind, budget, backend_kwargs)
        sk_agent.run()
        verdict = adjudicate([claim], sk_state, backend_kind, backend_kwargs)
        LOG.emit("Judge", f"[negative control] Verdict [{verdict['verdict']}] for {claim}: {verdict['reasoning']}")
        was_refused = verdict["verdict"] in ("Refuted", "Speculative")
        refuted += int(was_refused)
        details.append({"claim": claim, "verdict": verdict["verdict"], "correctly_refuted": was_refused})

    return {"n_controls": len(wrong_claims), "n_refuted": refuted,
            "false_confirmation_rate": 1 - (refuted / len(wrong_claims)) if wrong_claims else 0.0,
            "details": details}


def run_stage_a(backend_kind: str | None = None, backend_kwargs: dict | None = None,
                 target_model_id: str | None = None) -> dict:
    backend_kind = backend_kind or config.LLM_BACKEND
    target_model_id = target_model_id or config.TARGET_MODEL_ID
    print("=" * 78)
    print(f"STAGE A -- Ground-Truth Validation ({target_model_id}, IOI task)")
    print("=" * 78)

    handle = adapter.register_model(target_model_id, device=config.DEVICE)
    LOG.emit("System", f"registered {target_model_id}: {handle.n_layers} layers "
                         f"via '{handle.layer_stack_path}'")

    task = build_ioi_task(handle)
    LOG.emit("System", f"IOI probe: '{task['clean_prompt']}' -> expect '{task['io_token']}' "
                         f"over '{task['s_token']}'")

    result = run_hierarchy(handle, task, backend_kind=backend_kind, backend_kwargs=backend_kwargs)
    scored = score_against_ground_truth(result, handle.n_layers, model_id=target_model_id)
    negative = run_negative_controls(handle, task, backend_kind, backend_kwargs)

    print("\n" + "-" * 78)
    print("STAGE A RESULTS")
    print("-" * 78)
    if scored["ground_truth_valid"]:
        print(f"Flagged layers            : {scored['flagged_layers']}  (ground truth layers: {GT_LAYERS})")
        print(f"Layer-localization recall : {scored['layer_localization_recall']:.0%}")
        print(f"Claimed circuit           : {scored['claimed_heads']}")
        print(f"Ground-truth circuit      : {sorted(GT_HEADS)}")
        print(f"Circuit recall            : {scored['circuit_recall']:.0%}")
        print(f"Circuit precision         : {scored['circuit_precision']:.0%}")
    else:
        print(f"Flagged layers            : {scored['flagged_layers']}")
        print(f"Claimed circuit           : {scored['claimed_heads']}")
        print("Ground-truth circuit      : N/A -- published IOI ground truth only exists for GPT-2 small; "
              f"recall/precision cannot be computed for '{target_model_id}'")
    if result["verdict"]:
        print(f"Judge verdict on claim    : {result['verdict']['verdict']} -- {result['verdict']['reasoning']}")
    print(f"Negative controls refuted : {negative['n_refuted']}/{negative['n_controls']} "
          f"(false-confirmation rate {negative['false_confirmation_rate']:.0%})")
    print(f"Tool calls spent          : {result['tool_calls_spent']}")
    print(f"Cache hit rate            : {result['cache_stats']['hit_rate']:.0%} "
          f"({result['cache_stats']['hits']} hits / {result['cache_stats']['misses']} misses)")

    if result.get("lens_findings") or result.get("weight_findings"):
        print("\n" + "-" * 78)
        print("TECHNIQUE AGENTS")
        print("-" * 78)
        w = result.get("weight_findings", {})
        if w.get("tie"):
            print(f"Weight Agent    : E/U tied={w['tie'].get('tied')}, "
                  f"attn-out eff-rank {[r[2] for r in w.get('attn_svd', {}).get('per_layer_(top_sv, effective_rank, full_rank)', [])][:4]}")
        sf = result.get("safety_findings", {})
        if sf.get("refusal"):
            print(f"Safety Agent    : refusal-dir tokens {sf['refusal'].get('direction_top_tokens', [])[:3]}; "
                  f"copy-suppression heads {sf.get('copy_suppression', {}).get('copy_suppression_heads (head, suppressed_token, attn, DLA)', [])[:3]}")
        for li in sorted(result.get("lens_findings", {})):
            ll = result["lens_findings"][li]
            pr = result.get("probe_findings", {}).get(li, {})
            fe = result.get("feature_findings", {}).get(li, {})
            print(f"L{li}: lens_emergence={ll.get('logit_lens', {}).get('emergence_layer')}  "
                  f"probe_acc={pr.get('linear_probe', {}).get('cv_accuracy')}  "
                  f"toy_SAE_L0={fe.get('toy_sae', {}).get('L0')}  "
                  f"FVU={fe.get('toy_sae', {}).get('FVU')}")
        for li in sorted(result.get("steering_findings", {})):
            st = result["steering_findings"][li]
            print(f"L{li}: steering add->{st.get('addition', {}).get('next_token_after')}  "
                  f"LEACE erased_acc={st.get('leace', {}).get('concept_probe_acc_after_erasure')}")

    return {"result": result, "scored": scored, "negative_controls": negative,
            "target_model_id": target_model_id, "n_layers": handle.n_layers}
