"""SAE-based tools (Sec. 2.1 "Sparse Autoencoders"; Tier L `sae_layer_profile`,
Tier C `run_sae_decompose` -- the two tools flagged "not yet implemented" in
the original scope).

Uses SAELens pretrained SAEs (Sec. 4.8 stack lists "SAELens / GemmaScope").
For GPT-2 small, Joseph Bloom's public residual-stream release
(`gpt2-small-res-jb`, one SAE per layer, d_in=768, d_sae=24576) is the
reference set -- the same public-artifact role GemmaScope plays for Gemma.

Important: this SAE (like most public releases) was trained on activations
from a TransformerLens `HookedTransformer`, which applies weight processing
(LayerNorm folding, weight centering) that a plain `transformers.GPT2Model`
does not. Feeding it activations captured via adapter.py's plain-HF hooks
produces badly wrong numbers (verified: FVU ~16 and L0 ~2000, i.e. the SAE
barely reconstructing noise) instead of the correct FVU ~0 and L0 ~52 you get
from TransformerLens-sourced activations. So, unlike every other tool in this
codebase, this module deliberately loads its own `HookedTransformer` (adding
`transformer_lens` as a dependency scoped to this file only) rather than
reusing the architecture-agnostic `ModelHandle` from adapter.py. This is also
why SAE support is currently GPT-2-only: it needs both a TransformerLens
implementation of the target model AND a matching public SAE release
(GemmaScope for Gemma would extend this the same way).
"""
from __future__ import annotations

import threading

import torch

DEFAULT_SAE_RELEASE = "gpt2-small-res-jb"

# Which target models this module can currently serve: needs both a
# TransformerLens implementation of the model AND a public SAE release for
# it (GemmaScope would extend this the same way for Gemma).
SUPPORTED_MODELS = {"gpt2"}


def supports_sae(model_id: str) -> bool:
    import os
    return model_id in SUPPORTED_MODELS and os.environ.get("AMI_PUBLIC_SAE", "0") == "1"

_SAE_CACHE: dict[tuple, object] = {}
_TL_MODEL_CACHE: dict[str, object] = {}
_LOCK = threading.Lock()


def _sae_id_for_layer(layer_idx: int) -> str:
    return f"blocks.{layer_idx}.hook_resid_pre"


def load_sae(layer_idx: int, device: str = "cpu", release: str = DEFAULT_SAE_RELEASE):
    key = (release, layer_idx, device)
    with _LOCK:
        if key not in _SAE_CACHE:
            from sae_lens import SAE
            _SAE_CACHE[key] = SAE.from_pretrained(release, _sae_id_for_layer(layer_idx), device=device)
    return _SAE_CACHE[key]


def _load_tl_model(model_id: str, device: str = "cpu"):
    with _LOCK:
        if model_id not in _TL_MODEL_CACHE:
            from transformer_lens import HookedTransformer
            _TL_MODEL_CACHE[model_id] = HookedTransformer.from_pretrained(model_id, device=device)
    return _TL_MODEL_CACHE[model_id]


def capture_resid_pre(model_id: str, layer_idx: int, texts: list[str], device: str = "cpu"):
    """The residual-stream value flowing INTO layer_idx -- i.e. exactly
    `hook_resid_pre`, sourced from TransformerLens so it matches the
    distribution the public SAE was trained on. Returns (acts, per-row
    valid-length list) since TransformerLens right-pads variable-length text."""
    tl_model = _load_tl_model(model_id, device)
    hook_name = _sae_id_for_layer(layer_idx)
    with torch.no_grad():
        tokens = tl_model.to_tokens(texts)
        _, cache = tl_model.run_with_cache(tokens, names_filter=hook_name)
    acts = cache[hook_name]  # [batch, seq, d_model]
    lengths = [len(tl_model.to_tokens([t])[0]) for t in texts]
    return acts, lengths


def sae_layer_profile(layer_idx: int, texts: list[str], model_id: str = "gpt2",
                       release: str = DEFAULT_SAE_RELEASE, device: str = "cpu") -> str:
    """Tool: sae_layer_profile(). Per-layer SAE reconstruction loss & feature
    density -- maps where superposition is worst (Tier L)."""
    sae = load_sae(layer_idx, device, release)
    acts, lengths = capture_resid_pre(model_id, layer_idx, texts, device)
    rows = [acts[i, :lengths[i]] for i in range(len(texts))]
    flat = torch.cat(rows, dim=0)
    with torch.no_grad():
        feats = sae.encode(flat)
        recon = sae.decode(feats)
    mse = (recon - flat).pow(2).mean().item()
    var = flat.var(dim=0).sum().item()
    fvu = mse / var if var > 1e-8 else float("nan")  # fraction of variance unexplained
    l0 = (feats > 0).float().sum(dim=-1).mean().item()  # avg active features per token
    verdict = "high superposition (dense)" if l0 > 100 else "sparse / interpretable"
    return (f"sae_layer_profile L{layer_idx} ({release}): FVU={fvu:.4f} (lower=better "
            f"reconstruction), L0={l0:.1f} active features/token of {feats.shape[-1]} -> {verdict}")


def run_sae_decompose(layer_idx: int, text: str, token_idx: int = -1, k: int = 5,
                       model_id: str = "gpt2", release: str = DEFAULT_SAE_RELEASE,
                       device: str = "cpu") -> str:
    """Tool: run_sae_decompose(). Sparse interpretable feature directions for
    one flagged token's activation -- the top-k SAE features actually driving
    that specific activation, not just an aggregate statistic (Tier C)."""
    sae = load_sae(layer_idx, device, release)
    acts, lengths = capture_resid_pre(model_id, layer_idx, [text], device)
    seq_len = lengths[0]
    pos = token_idx if token_idx >= 0 else seq_len + token_idx
    x = acts[0, pos]
    with torch.no_grad():
        feats = sae.encode(x.unsqueeze(0))[0]
    top = torch.topk(feats, min(k, feats.shape[0]))
    pairs = [(int(i), round(float(v), 3)) for i, v in zip(top.indices, top.values) if v > 0]
    return f"run_sae_decompose L{layer_idx} token[{pos}]: top active features (id, activation) = {pairs}"
