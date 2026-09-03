"""Basic techniques -- logit lens, tuned lens, Jacobian lens, direct logit
attribution, attention-pattern reading.  (learnmechinterp.com / Basic Techniques)
"""
from __future__ import annotations

import torch

from ..tools import adapter, tier_c, tier_l
from . import _common as C


def logit_lens(handle: adapter.ModelHandle, prompt: str, top_k: int = 5) -> dict:
    """nostalgebraist 2020. Apply final norm + unembedding to every layer's
    residual output; report the top predicted token per layer and the layer
    at which the model's final top token first appears."""
    b = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    store: dict[int, torch.Tensor] = {}
    hooks = []
    with adapter.MODEL_LOCK:
        for i in range(handle.n_layers):
            def mk(i):
                def h(m, inp, out):
                    store[i] = (out[0] if isinstance(out, tuple) else out).detach()[:, -1, :]
                return h
            hooks.append(handle.layers[i].register_forward_hook(mk(i)))
        try:
            with torch.no_grad():
                final = handle.model(**b).logits[0, -1]
        finally:
            for h in hooks:
                h.remove()
    ln, W = C.final_norm(handle), C.unembed(handle)
    final_top = handle.tokenizer.decode([int(final.argmax())]).strip()
    per_layer = []
    emergence = None
    for i in range(handle.n_layers):
        h = store[i]
        h = ln(h) if ln is not None else h
        lg = (h @ W.T)[0]
        tok = handle.tokenizer.decode([int(lg.argmax())]).strip()
        per_layer.append((i, tok, round(float(lg.softmax(-1).max()), 3)))
        if emergence is None and tok == final_top:
            emergence = i
    return {"final_top_token": final_top, "emergence_layer": emergence,
            "per_layer_top": per_layer}


def tuned_lens(handle: adapter.ModelHandle, corpus: list[str], top_k: int = 5,
               probe_prompt: str | None = None) -> dict:
    """Belrose et al. 2023, tiny closed-form fit. For each layer solve the
    least-squares affine map (A_l, b_l) that best sends h_l -> h_final over
    `corpus` (last-token states), then read that translated state through the
    unembedding. Not the paper's full SGD/KL fit -- a fast ridge-regression
    stand-in so the technique runs on any model in seconds."""
    layers = list(range(handle.n_layers))
    b = handle.tokenizer(corpus, return_tensors="pt", padding=True).to(handle.device)
    last = b["attention_mask"].sum(1) - 1
    store: dict[int, torch.Tensor] = {}
    hooks = []
    with adapter.MODEL_LOCK:
        for i in layers:
            def mk(i):
                def h(m, inp, out):
                    hs = (out[0] if isinstance(out, tuple) else out).detach()
                    store[i] = hs[torch.arange(hs.shape[0]), last]
                return h
            hooks.append(handle.layers[i].register_forward_hook(mk(i)))
        try:
            with torch.no_grad():
                handle.model(**b)
        finally:
            for h in hooks:
                h.remove()
    H_final = store[layers[-1]].float()                       # [N, d]
    d = H_final.shape[1]
    lam = 1e-2 * torch.eye(d + 1)
    fits = {}
    for i in layers[:-1]:
        X = torch.cat([store[i].float(), torch.ones(store[i].shape[0], 1)], 1)  # [N, d+1]
        W_aff = torch.linalg.solve(X.T @ X + lam, X.T @ H_final)  # [d+1, d]
        fits[i] = W_aff
    result = {"n_corpus": len(corpus), "note": "ridge-regression stand-in, not the paper's KL-SGD fit"}
    if probe_prompt:
        ll = logit_lens(handle, probe_prompt, top_k)
        pb = handle.tokenizer([probe_prompt], return_tensors="pt").to(handle.device)
        pstore = {}
        hks = []
        with adapter.MODEL_LOCK:
            for i in layers:
                def mk(i):
                    def h(m, inp, out):
                        pstore[i] = (out[0] if isinstance(out, tuple) else out).detach()[:, -1, :]
                    return h
                hks.append(handle.layers[i].register_forward_hook(mk(i)))
            try:
                with torch.no_grad():
                    handle.model(**pb)
            finally:
                for h in hks:
                    h.remove()
        ln, W = C.final_norm(handle), C.unembed(handle)
        rows = []
        for i in layers[:-1]:
            x = torch.cat([pstore[i][0].float(), torch.ones(1)])
            h_tr = (x @ fits[i]).unsqueeze(0)
            h_tr = ln(h_tr) if ln is not None else h_tr
            lg = (h_tr @ W.T)[0]
            rows.append((i, handle.tokenizer.decode([int(lg.argmax())]).strip()))
        result["tuned_per_layer_top"] = rows
        result["raw_logit_lens_per_layer_top"] = ll["per_layer_top"]
    return result


def jacobian_lens(handle: adapter.ModelHandle, prompt: str, layer_idx: int,
                   pos_token: str, neg_token: str, top_k: int = 8) -> dict:
    """Gurnee, Lindsey et al. 2026 -- single-prompt, single-target
    approximation. The full J-lens averages the d_model x d_model Jacobian
    d h_final / d h_l over ~1000 prompts (expensive). Here we take the one
    row that matters for a chosen readout: d(logit_diff)/d h_l at this prompt
    (one backward pass) and report which layer-l directions most move the
    output -- i.e. the J-lens vector for the (pos,neg) logit-diff readout."""
    io = handle.tokenizer.encode(" " + pos_token.strip())[-1]
    s = handle.tokenizer.encode(" " + neg_token.strip())[-1]
    b = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    leaf = {}

    def pre(m, args, kwargs):
        x = (args[0] if args else kwargs["input"]).detach().clone().requires_grad_(True)
        leaf["x"] = x
        if args:
            return (x,) + args[1:], kwargs
        kwargs["input"] = x
        return args, kwargs

    tgt = handle.layers[layer_idx]
    with adapter.MODEL_LOCK:
        h = tgt.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            logits = handle.model(**b).logits
            md = logits[0, -1, io] - logits[0, -1, s]
            (g,) = torch.autograd.grad(md, leaf["x"])
        finally:
            h.remove()
    g_last = g[0, -1].detach()                     # d(logit_diff)/d h_l  [d_model]
    W = C.unembed(handle)
    # compose with unembedding: the readout gradient already lives in resid space;
    # top tokens whose unembed direction aligns with it are what h_l is poised to move
    aligned = C.top_tokens(handle, g_last, top_k)
    return {"layer": layer_idx, "readout": f"logit({pos_token}) - logit({neg_token})",
            "grad_norm": round(float(g_last.norm()), 4),
            "tokens_this_layer_is_poised_to_move": aligned,
            "note": "single-prompt/single-readout approximation of the averaged J-lens"}


def direct_logit_attribution(handle: adapter.ModelHandle, prompt: str,
                              pos_token: str, neg_token: str, k: int = 8) -> dict:
    """Elhage et al. 2021 -- see tier_c.direct_logit_attribution."""
    digest, top = tier_c.direct_logit_attribution(
        handle, prompt, pos_token, neg_token, range(handle.n_layers),
        handle.model.config.num_attention_heads if hasattr(handle.model.config, "num_attention_heads")
        else handle.model.config.n_head, k=k)
    return {"digest": digest, "top_write_direction_heads": [list(t) for t in top]}


def _classify_pattern(attn: torch.Tensor) -> str:
    """attn: [seq, seq] lower-triangular. Coarse label."""
    seq = attn.shape[0]
    if seq < 3:
        return "too-short"
    diag_prev = torch.diagonal(attn, offset=-1).mean().item()
    col_mass = attn.mean(0)                       # avg attention received per source
    to_bos = col_mass[0].item()
    uniform_ref = 1.0 / torch.arange(1, seq + 1).float()
    unif_err = (attn.sum(0) / torch.arange(1, seq + 1).float().flip(0).clamp_min(1)
                ).std().item()
    if diag_prev > 0.5:
        return "previous-token (diagonal)"
    if to_bos > 0.5:
        return "anchor / BOS column"
    if col_mass.max().item() > 0.4 and col_mass.argmax().item() not in (0, seq - 1):
        return "content column"
    if attn.max().item() < 2.0 / seq:
        return "approx uniform"
    return "mixed / content-based"


def attention_pattern_readout(handle: adapter.ModelHandle, prompt: str,
                               heads: list[tuple[int, int]] | None = None) -> dict:
    """Read + classify attention maps (learnmechinterp / Reading the Attention
    Patterns). Labels: previous-token, induction-like, anchor column, uniform,
    content-based."""
    b = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK, torch.no_grad():
        out = handle.model(**b, output_attentions=True)
    if not out.attentions:
        return {"error": "model returned no attentions (need attn_implementation='eager')"}
    toks = [handle.tokenizer.decode([t]) for t in b["input_ids"][0].tolist()]
    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]
    if heads is None:
        heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    rows = []
    for (l, h) in heads:
        a = out.attentions[l][0, h]
        rows.append({"head": f"L{l}H{h}", "pattern": _classify_pattern(a),
                     "peak_dest": toks[int(a.max(0).values.argmax())].strip(),
                     "self_attn_last": round(float(a[-1].max()), 3)})
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["pattern"]] = counts.get(r["pattern"], 0) + 1
    return {"tokens": toks, "pattern_counts": counts,
            "heads": rows if len(rows) <= 40 else rows[:40]}


ALL = {
    "logit_lens": logit_lens,
    "tuned_lens": tuned_lens,
    "jacobian_lens": jacobian_lens,
    "direct_logit_attribution": direct_logit_attribution,
    "attention_pattern_readout": attention_pattern_readout,
}
