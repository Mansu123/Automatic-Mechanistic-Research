"""Shared helpers for the techniques package."""
from __future__ import annotations

import torch

from ..tools import adapter


def last_token_logit_diff(handle: adapter.ModelHandle, prompt: str, pos_token: str, neg_token: str) -> float:
    io = handle.tokenizer.encode(" " + pos_token.strip())[-1]
    s = handle.tokenizer.encode(" " + neg_token.strip())[-1]
    b = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK, torch.no_grad():
        lg = handle.model(**b).logits[0, -1]
    return float(lg[io] - lg[s])


def logit_diff_metric_fn(handle: adapter.ModelHandle, pos_token: str, neg_token: str):
    io = handle.tokenizer.encode(" " + pos_token.strip())[-1]
    s = handle.tokenizer.encode(" " + neg_token.strip())[-1]
    return lambda logits: float((logits[0, -1, io] - logits[0, -1, s]).item())


def final_norm(handle: adapter.ModelHandle):
    for path in ("transformer.ln_f", "gpt_neox.final_layer_norm", "model.norm",
                 "model.final_layernorm", "model.decoder.final_layer_norm"):
        obj = handle.model
        for name in path.split("."):
            obj = getattr(obj, name, None)
        if obj is not None:
            return obj
    return None


def unembed(handle: adapter.ModelHandle) -> torch.Tensor:
    return handle.model.get_output_embeddings().weight  # [vocab, d_model]


def capture_resid(handle: adapter.ModelHandle, layer_idx: int, texts: list[str]) -> torch.Tensor:
    """Per-text last-token residual-stream output of `layer_idx`. [n_texts, d_model]."""
    import numpy as np
    acts = adapter.capture_activations(handle, [layer_idx], texts)[layer_idx]
    return torch.tensor(np.asarray(acts), dtype=torch.float32, device=handle.device)


def diff_of_means_direction(handle: adapter.ModelHandle, layer_idx: int,
                             pos_texts: list[str], neg_texts: list[str],
                             normalize: bool = True) -> torch.Tensor:
    p = capture_resid(handle, layer_idx, pos_texts).mean(0)
    n = capture_resid(handle, layer_idx, neg_texts).mean(0)
    d = p - n
    return d / d.norm().clamp_min(1e-8) if normalize else d


def top_tokens(handle: adapter.ModelHandle, vec: torch.Tensor, k: int = 8) -> list[tuple[str, float]]:
    """Project a d_model vector through the unembedding, return top-k tokens."""
    weight = unembed(handle)
    with torch.no_grad():
        logits = vec.to(device=weight.device, dtype=weight.dtype) @ weight.T
    t = torch.topk(logits, min(k, logits.numel()))
    return [(handle.tokenizer.decode([i]).strip(), round(float(v), 3))
            for i, v in zip(t.indices.tolist(), t.values.tolist())]


def add_direction_hook(direction: torch.Tensor, strength: float):
    def hook(module, inputs, output):
        is_t = isinstance(output, tuple)
        hs = output[0] if is_t else output
        hs = hs + strength * direction.to(device=hs.device, dtype=hs.dtype)
        return (hs,) + output[1:] if is_t else hs
    return hook


def dla_scores(handle: adapter.ModelHandle, prompt: str, pos_token: str, neg_token: str,
               extra_pre_hooks: list | None = None) -> dict:
    """{(layer,head): direct logit-diff attribution} at the last position.
    `extra_pre_hooks` = list of (module, fn) forward_pre_hooks (with_kwargs)
    installed for the whole computation -- lets a caller ablate a head and see
    how the DLA of the others shifts, without nested MODEL_LOCK."""
    io = handle.tokenizer.encode(" " + pos_token.strip())[-1]
    s = handle.tokenizer.encode(" " + neg_token.strip())[-1]
    W = unembed(handle)
    dir_vec = (W[io] - W[s]).detach()
    n_heads = getattr(handle.model.config, "num_attention_heads", None) or handle.model.config.n_head
    head_dim = adapter.get_head_dim(handle)
    b = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    ln = final_norm(handle)
    out: dict = {}
    with adapter.MODEL_LOCK, torch.no_grad():
        extra = []
        for (mod, fn) in (extra_pre_hooks or []):
            extra.append(mod.register_forward_pre_hook(fn, with_kwargs=True))
        try:
            fin = {}
            hln = ln.register_forward_hook(
                lambda m, i, o: fin.__setitem__("h", (i[0] if isinstance(i, tuple) else i).detach())
            ) if ln is not None else None
            handle.model(**b)
            if hln:
                hln.remove()
            rms = fin["h"][0, -1].pow(2).mean().sqrt().clamp_min(1e-6) if "h" in fin else torch.tensor(1.0)
            for l in range(handle.n_layers):
                op = adapter._find_attn_out_proj(handle.layers[l])
                z = {}
                hz = op.register_forward_pre_hook(
                    lambda m, a, k: z.__setitem__("v", (a[0] if a else k["input"]).detach().clone()),
                    with_kwargs=True)
                handle.model(**b)
                hz.remove()
                zc = z["v"]
                bias = op(torch.zeros_like(zc[:, -1:, :]))[0, -1]
                for hd in range(n_heads):
                    lo, hi = hd * head_dim, (hd + 1) * head_dim
                    masked = torch.zeros_like(zc[:, -1:, :])
                    masked[:, :, lo:hi] = zc[:, -1:, lo:hi]
                    r_h = op(masked)[0, -1] - bias
                    out[(l, hd)] = float((r_h / rms) @ dir_vec)
        finally:
            for e in extra:
                e.remove()
    return out


def project_out_hook(direction: torch.Tensor):
    u = direction / direction.norm().clamp_min(1e-8)
    def hook(module, inputs, output):
        is_t = isinstance(output, tuple)
        hs = output[0] if is_t else output
        # Compute the projection in FP32, then preserve the model's residual dtype.
        # FP32 coefficients previously promoted BF16 residuals and broke the next matmul.
        unit = u.to(device=hs.device, dtype=torch.float32)
        coef = (hs.float() * unit).sum(-1, keepdim=True)
        hs = hs - (coef * unit).to(dtype=hs.dtype)
        return (hs,) + output[1:] if is_t else hs
    return hook
