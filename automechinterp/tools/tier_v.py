"""Tier V -- Verification tools (Skeptic, Sec. 3.4).

This module is what structurally separates AutoMechInterp from MAIA's
documented confirmation bias and from Dalvi et al.'s ablation-only (but
non-adversarial) validation: every function here is designed to try to break
a claim, not support it. The Skeptic is the only agent allowed to call these
(agents/skeptic.py) -- see gap_work.py:gap "maia-confirmation-bias".
"""
from __future__ import annotations

import random

import torch

from . import adapter
from .digest import metric_delta_digest


def ablate_component(handle: adapter.ModelHandle, layer_idx: int, head_idx: int,
                      prompts: list[tuple[str, str, str]], mode: str = "mean") -> str:
    """Tool: ablate_component(). Zero/mean-ablate the claimed head across a
    prompt batch; performance drop when the claimed component is removed."""
    from .tier_c import _ioi_metric_fn
    deltas = []
    for clean_prompt, io_token, s_token in prompts:
        metric_fn = _ioi_metric_fn(handle, io_token, s_token)
        batch = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
        with adapter.MODEL_LOCK, torch.no_grad():
            baseline = metric_fn(handle.model(**batch).logits)
        ablated = adapter.ablate_head(handle, layer_idx, head_idx, batch["input_ids"],
                                       batch["attention_mask"], metric_fn, mode=mode)
        deltas.append(ablated - baseline)
    avg_delta = sum(deltas) / len(deltas)
    return (f"ablate_component L{layer_idx}H{head_idx} ({mode}): "
            f"avg_delta={avg_delta:+.3f} over {len(prompts)} prompts -> "
            f"{'SPECIFIC EFFECT' if abs(avg_delta) > 0.2 else 'weak/no effect'}")


def exclusion_ablation(handle: adapter.ModelHandle, claimed_heads: list[tuple[int, int]],
                        all_heads: list[tuple[int, int]],
                        prompts: list[tuple[str, str, str]], mode: str = "mean") -> str:
    """Tool: exclusion_ablation(). Ablates every head NOT in the claim; the
    metric must survive close to baseline, or the claim is incomplete (the
    real circuit includes components the Prover missed)."""
    from .tier_c import _ioi_metric_fn
    complement = [h for h in all_heads if h not in claimed_heads]
    max_drop = 0.0
    for clean_prompt, io_token, s_token in prompts:
        metric_fn = _ioi_metric_fn(handle, io_token, s_token)
        batch = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
        with adapter.MODEL_LOCK, torch.no_grad():
            baseline = metric_fn(handle.model(**batch).logits)
        for layer_idx, head_idx in complement:
            ablated = adapter.ablate_head(handle, layer_idx, head_idx, batch["input_ids"],
                                           batch["attention_mask"], metric_fn, mode=mode)
            max_drop = max(max_drop, abs(ablated - baseline))
    verdict = "COMPLETE (claim survives exclusion)" if max_drop < 0.10 else "INCOMPLETE (unexplained heads found)"
    return f"exclusion_ablation: max single-head drop outside claim = {max_drop:.3f} -> {verdict}"


def minimality_check(handle: adapter.ModelHandle, claimed_heads: list[tuple[int, int]],
                      prompts: list[tuple[str, str, str]], mode: str = "mean") -> str:
    """Tool: minimality_check(). Removes claimed elements one at a time; each
    must matter on its own, or the claim is inflated (padded with
    non-load-bearing components)."""
    from .tier_c import _ioi_metric_fn
    results = []
    for layer_idx, head_idx in claimed_heads:
        drops = []
        for clean_prompt, io_token, s_token in prompts:
            metric_fn = _ioi_metric_fn(handle, io_token, s_token)
            batch = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
            with adapter.MODEL_LOCK, torch.no_grad():
                baseline = metric_fn(handle.model(**batch).logits)
            ablated = adapter.ablate_head(handle, layer_idx, head_idx, batch["input_ids"],
                                           batch["attention_mask"], metric_fn, mode=mode)
            drops.append(abs(ablated - baseline))
        avg = sum(drops) / len(drops)
        results.append((f"L{layer_idx}H{head_idx}", avg, avg > 0.05))
    inflated = [r[0] for r in results if not r[2]]
    verdict = "MINIMAL" if not inflated else f"INFLATED (non-load-bearing: {inflated})"
    detail = "; ".join(f"{n}:{v:.3f}" for n, v, _ in results)
    return f"minimality_check: [{detail}] -> {verdict}"


_NAME_BANK = ["John", "Mary", "Alice", "Bob", "Sarah", "Tom", "Emma", "James",
              "Laura", "David", "Anna", "Mike"]
_TEMPLATES = [
    "When {A} and {B} went to the store, {S} gave a drink to",
    "Then {A} and {B} went to the park, and {S} handed the ball to",
    "After {A} and {B} left the office, {S} passed the file to",
]


def counterexample_search(handle: adapter.ModelHandle, claimed_heads: list[tuple[int, int]],
                           behavior: str, budget: int = 20, seed: int = 0) -> str:
    """Tool: counterexample_search(). Template-based search over fresh
    name/template combinations (a concrete, runnable stand-in for the
    proposal's 'LLM-guided search') for an input where ablating the claimed
    circuit does NOT hurt the behavior it is supposed to explain -- i.e. a
    case that would falsify the claim."""
    from .tier_c import _ioi_metric_fn
    rng = random.Random(seed)
    n_falsified = 0
    for _ in range(budget):
        a, b = rng.sample(_NAME_BANK, 2)
        template = rng.choice(_TEMPLATES)
        s_name = rng.choice([a, b])
        io_name = b if s_name == a else a
        prompt = template.format(A=a, B=b, S=s_name)
        metric_fn = _ioi_metric_fn(handle, io_name, s_name)
        batch = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
        with adapter.MODEL_LOCK, torch.no_grad():
            baseline = metric_fn(handle.model(**batch).logits)
        total_drop = 0.0
        for layer_idx, head_idx in claimed_heads:
            ablated = adapter.ablate_head(handle, layer_idx, head_idx, batch["input_ids"],
                                           batch["attention_mask"], metric_fn, mode="mean")
            total_drop += abs(ablated - baseline)
        if total_drop < 0.10 and abs(baseline) > 0.5:
            n_falsified += 1  # circuit was supposedly load-bearing here but ablating it did nothing
    verdict = "NO COUNTEREXAMPLE" if n_falsified == 0 else f"{n_falsified}/{budget} COUNTEREXAMPLES FOUND"
    return f"counterexample_search('{behavior}', budget={budget}): {verdict}"


def interchange_intervention(handle: adapter.ModelHandle, layer_idx: int, head_idx: int,
                              prompt_a: tuple[str, str, str], prompt_b: tuple[str, str, str]) -> str:
    """Tool: interchange_intervention(). IIT-style causal swap: inject this
    head's activation from run B into run A and check whether A's metric
    moves toward B's -- confirms the node carries the causally relevant
    variable rather than merely correlating with it."""
    from .tier_c import _ioi_metric_fn
    prompt_a_text, io_a, s_a = prompt_a
    prompt_b_text, io_b, s_b = prompt_b
    metric_fn_a = _ioi_metric_fn(handle, io_a, s_a)
    ba = handle.tokenizer([prompt_a_text], return_tensors="pt").to(handle.device)
    bb = handle.tokenizer([prompt_b_text], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK, torch.no_grad():
        metric_a = metric_fn_a(handle.model(**ba).logits)
    swapped_metric = adapter.run_with_head_patch(
        handle, layer_idx, head_idx,
        bb["input_ids"], bb["attention_mask"], ba["input_ids"], ba["attention_mask"], metric_fn_a)
    return metric_delta_digest(f"interchange_intervention L{layer_idx}H{head_idx} (inject B into A)",
                                metric_a, swapped_metric)


def steer_with_vector(handle: adapter.ModelHandle, layer_idx: int,
                       positive_texts: list[str], negative_texts: list[str],
                       prompt: str, strength: float = 4.0) -> str:
    """Tool: steer_with_vector(). Builds a diff-of-means direction from two
    contrastive text sets at this layer, adds it to the residual stream at
    generation time, and reports whether the next-token distribution shifts
    toward the positive concept -- confirms a claimed feature direction is
    causally sufficient, not just correlated with activation."""
    pos = adapter.capture_activations(handle, [layer_idx], positive_texts)[layer_idx]
    neg = adapter.capture_activations(handle, [layer_idx], negative_texts)[layer_idx]
    direction = torch.tensor(pos.mean(0) - neg.mean(0), dtype=torch.float32, device=handle.device)

    batch = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)

    def steer_hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        hs = hs + strength * direction
        return (hs,) + output[1:] if is_tuple else hs

    lm_head = handle.model.get_output_embeddings()

    def get_top_token(add_hook: bool):
        with adapter.MODEL_LOCK:
            h = handle.layers[layer_idx].register_forward_hook(steer_hook) if add_hook else None
            try:
                with torch.no_grad():
                    logits = handle.model(**batch).logits[0, -1]
            finally:
                if h:
                    h.remove()
        top_id = int(torch.argmax(logits).item())
        return handle.tokenizer.decode([top_id])

    before = get_top_token(False)
    after = get_top_token(True)
    return f"steer_with_vector L{layer_idx} (strength={strength}): top token '{before}' -> '{after}'"
