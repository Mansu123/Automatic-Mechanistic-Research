"""Tier L -- Layer-Level tools (Layer Agents, Sec. 3.4).

Scoped to one layer plus its immediate neighbors, matching the proposal's
context-scoping design (Sec. 3.1): a Layer Agent only ever calls these with
its own layer index.
"""
from __future__ import annotations

import torch

from . import adapter, sae as _sae
from .digest import metric_delta_digest


def sae_layer_profile(handle: adapter.ModelHandle, layer_idx: int, texts: list[str]) -> str:
    """Tool: sae_layer_profile(). Per-layer SAE reconstruction loss & feature
    density -- maps where superposition is worst, so the Layer Agent knows
    whether a Component Agent will need run_sae_decompose (Tier C) to make
    sense of this layer, or whether individual heads are already
    monosemantic enough that circuit discovery alone will be interpretable."""
    return _sae.sae_layer_profile(layer_idx, texts, model_id=handle.model_id, device=handle.device)


def logit_lens(handle: adapter.ModelHandle, layer_idx: int, prompt: str, top_k: int = 5) -> str:
    """Project this layer's residual-stream output through the unembedding to
    see what the model would predict if it stopped here."""
    batch = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    store = {}

    def hook(module, inputs, output):
        store["hs"] = (output[0] if isinstance(output, tuple) else output).detach()

    with adapter.MODEL_LOCK:
        h = handle.layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                handle.model(**batch)
        finally:
            h.remove()

    hs_last = store["hs"][:, -1, :]
    lm_head = handle.model.get_output_embeddings()
    ln_f = (getattr(getattr(handle.model, "transformer", None), "ln_f", None)
            or getattr(getattr(handle.model, "model", None), "norm", None))
    if ln_f is not None:
        hs_last = ln_f(hs_last)
    logits = lm_head(hs_last)[0]
    top = torch.topk(logits, top_k)
    toks = [handle.tokenizer.decode([t]) for t in top.indices.tolist()]
    return f"logit_lens L{layer_idx}: top-{top_k} = {list(zip(toks, [round(v, 2) for v in top.values.tolist()]))}"


def patch_layer(handle: adapter.ModelHandle, layer_idx: int,
                 clean_prompt: str, corrupted_prompt: str,
                 clean_token: str, corrupted_token: str) -> str:
    """Does patching this whole layer flip the output toward the clean
    answer? Coarse causal localization before any component-level work."""
    def metric_fn(logits: torch.Tensor) -> float:
        clean_id = handle.tokenizer.encode(" " + clean_token.strip())[0]
        corr_id = handle.tokenizer.encode(" " + corrupted_token.strip())[0]
        last = logits[0, -1]
        return float((last[clean_id] - last[corr_id]).item())

    ci = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([corrupted_prompt], return_tensors="pt").to(handle.device)
    r = adapter.patch_layer(handle, layer_idx, ci["input_ids"], ci["attention_mask"],
                             xi["input_ids"], xi["attention_mask"], metric_fn)
    return (f"patch_layer L{layer_idx}: fraction_recovered={r['fraction_recovered']:.2f} "
            f"(clean_metric={r['clean_metric']:.2f}, corrupted_metric={r['corrupted_metric']:.2f}, "
            f"patched_metric={r['patched_metric']:.2f})")


def attn_mlp_attribution(handle: adapter.ModelHandle, layer_idx: int, prompt: str) -> str:
    """Decomposes this layer's residual contribution into attention vs MLP by
    norm of each sublayer's additive contribution -- decides whether to spawn
    a Component Agent on heads (attention-heavy) or the MLP."""
    batch = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    layer = handle.layers[layer_idx]
    attn_mod = getattr(layer, "attn", None) or getattr(layer, "self_attn", None)
    mlp_mod = getattr(layer, "mlp", None)

    norms = {}

    def make_hook(name):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            norms[name] = hs.norm().item()
        return hook

    hooks = []
    with adapter.MODEL_LOCK:
        if attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(make_hook("attn")))
        if mlp_mod is not None:
            hooks.append(mlp_mod.register_forward_hook(make_hook("mlp")))
        try:
            with torch.no_grad():
                handle.model(**batch)
        finally:
            for h in hooks:
                h.remove()

    total = sum(norms.values()) or 1.0
    share = {k: v / total for k, v in norms.items()}
    return f"attn_mlp_attribution L{layer_idx}: " + ", ".join(f"{k}={v*100:.0f}%" for k, v in share.items())


def layer_role_card(layer_idx: int, hypothesized_role: str, evidence: list[str],
                     confidence: str, open_questions: str = "") -> str:
    """Structured template the Layer Agent fills in (Tier L output object)."""
    ev = "; ".join(evidence)
    return (f"[Layer Role Card L{layer_idx}] role='{hypothesized_role}' "
            f"confidence={confidence} evidence=[{ev}] open_questions='{open_questions}'")
