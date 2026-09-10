"""Circuit finding -- ACDC, EAP, S-EAP, QK/OV decomposition, faithfulness /
completeness / minimality, copy-suppression detection.
(learnmechinterp / Circuit Finding)
"""
from __future__ import annotations

import torch

from ..tools import adapter, tier_c
from . import _common as C


def _nh(handle):
    return getattr(handle.model.config, "num_attention_heads", None) or handle.model.config.n_head


def acdc(handle, clean_prompt, corrupted_prompt, pos_token, neg_token, threshold=0.10) -> dict:
    """Conmy et al. 2023 -- exhaustive per-head patch sweep (tools/tier_c)."""
    digest, circuit = tier_c.run_acdc(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                                      range(handle.n_layers), _nh(handle), threshold=threshold)
    return {"digest": digest, "circuit": [list(c) for c in circuit]}


def eap(handle, clean_prompt, corrupted_prompt, pos_token, neg_token) -> dict:
    """Syed et al. 2023 -- gradient edge attribution patching."""
    return {"digest": tier_c.run_eap(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                                     range(handle.n_layers), _nh(handle))}


def synergy_eap(handle, clean_prompt, corrupted_prompt, pos_token, neg_token) -> dict:
    """S-EAP (this repo) -- second-order pass for backup heads ACDC's
    first-order threshold drops. Seeded with direct-logit-attribution heads."""
    _, dla_top = tier_c.direct_logit_attribution(handle, clean_prompt, pos_token, neg_token,
                                                 range(handle.n_layers), _nh(handle), k=10)
    _, circ = tier_c.run_acdc(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                              range(handle.n_layers), _nh(handle), threshold=0.10)
    cands = sorted(set(dla_top) | set(circ))
    digest, rows = tier_c.run_synergy_eap(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                                          range(handle.n_layers), _nh(handle),
                                          ablate_candidates=cands, k=10)
    circ_set = set(circ)
    recovered = sorted({i for score, i, j in rows if abs(score) >= 0.05
                        and i not in circ_set and j in circ_set})
    return {"acdc_circuit": [list(c) for c in circ], "digest": digest,
            "synergy_recovered_heads": [list(r) for r in recovered]}


def qk_ov_decomposition(handle, layer_idx, head_idx) -> dict:
    """Elhage et al. 2021 -- separate a head into its QK circuit (who it
    attends to) and OV circuit (what it copies). OV eigen/vocab structure
    tells copy vs anti-copy; only wired for GPT-2-style c_attn packing."""
    layer = handle.layers[layer_idx]
    attn = getattr(layer, "attn", None) or getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    d = handle.model.config.hidden_size if hasattr(handle.model.config, "hidden_size") else handle.model.config.n_embd
    nh = _nh(handle)
    hd = d // nh
    if hasattr(attn, "c_attn"):                       # GPT-2: packed QKV
        Wq, Wk, Wv = attn.c_attn.weight.detach().T.chunk(3, dim=0)  # each [d, d] (in x out)... c_attn is Conv1D
        Wq, Wk, Wv = [w.T for w in (Wq, Wk, Wv)]
        Wo = attn.c_proj.weight.detach()             # [d, d]
    elif hasattr(attn, "q_proj"):                     # Llama/Qwen
        Wq, Wk, Wv = attn.q_proj.weight.detach(), attn.k_proj.weight.detach(), attn.v_proj.weight.detach()
        Wo = attn.o_proj.weight.detach().T
        if Wk.shape[0] != d:                          # GQA -- skip, needs head grouping
            return {"skipped": "grouped-query attention; QK/OV per-head decomposition needs KV-head mapping"}
    else:
        return {"skipped": f"unrecognised attention module {type(attn).__name__}"}
    sl = slice(head_idx * hd, (head_idx + 1) * hd)
    W_OV = (Wv[sl, :].T @ Wo[:, sl].T)                # [d, d] full OV map (approx)
    E = handle.model.get_input_embeddings().weight.detach()
    U = C.unembed(handle)
    ov_vocab = E @ W_OV @ U.T                         # token -> token copy scores
    diag = ov_vocab.diagonal()
    copy_score = float((diag > 0).float().mean())
    return {"head": f"L{layer_idx}H{head_idx}",
            "OV_copy_score": round(copy_score, 3),
            "interpretation": "OV_copy_score >> 0.5 => copying head; << 0.5 => anti-copy / suppression"}


def circuit_faithfulness(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                          circuit: list[tuple[int, int]]) -> dict:
    """Wang et al. / Hanna et al. -- faithfulness (circuit alone recovers the
    metric), completeness (complement doesn't), minimality (each head matters)."""
    mfn = C.logit_diff_metric_fn(handle, pos_token, neg_token)
    ci = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
    clean_m = C.last_token_logit_diff(handle, clean_prompt, pos_token, neg_token)
    corr_m = C.last_token_logit_diff(handle, corrupted_prompt, pos_token, neg_token)
    denom = clean_m - corr_m or 1e-6
    allh = [(l, h) for l in range(handle.n_layers) for h in range(_nh(handle))]
    circuit = [tuple(c) for c in circuit]
    complement = [h for h in allh if h not in set(circuit)]
    m_circ = adapter.ablate_head_set(handle, complement, ci["input_ids"], ci["attention_mask"], mfn)
    m_compl = adapter.ablate_head_set(handle, circuit, ci["input_ids"], ci["attention_mask"], mfn) if circuit else clean_m
    minimality = []
    for h in circuit:
        m_wo = adapter.ablate_head_set(handle, [h], ci["input_ids"], ci["attention_mask"], mfn)
        minimality.append((f"L{h[0]}H{h[1]}", round((clean_m - m_wo) / denom, 3)))
    return {"circuit_size": len(circuit),
            "faithfulness": round((m_circ - corr_m) / denom, 3),
            "completeness_gap": round((m_compl - corr_m) / denom, 3),
            "per_head_minimality": minimality}


def copy_suppression(handle, prompt, top_k_heads: int = 6) -> dict:
    """McDougall et al. 2023 -- heads that attend to a token and write
    *against* its logit (Negative Name Movers). Detect by: high attention to a
    prior token AND negative DLA for that token."""
    b = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK, torch.no_grad():
        out = handle.model(**b, output_attentions=True)
    if not out.attentions:
        return {"error": "no attentions"}
    ids = b["input_ids"][0].tolist()
    W = C.unembed(handle)
    nh = _nh(handle)
    hd = adapter.get_head_dim(handle)
    flags = []
    for l in range(handle.n_layers):
        op = adapter._find_attn_out_proj(handle.layers[l])
        z = {}
        with adapter.MODEL_LOCK:
            hz = op.register_forward_pre_hook(
                lambda m, a, k: z.__setitem__("v", (a[0] if a else k["input"]).detach().clone()),
                with_kwargs=True)
            with torch.no_grad():
                handle.model(**b)
            hz.remove()
        for h in range(nh):
            a = out.attentions[l][0, h, -1]           # last pos attention over sources
            src = int(a[:-1].argmax()) if a.shape[0] > 1 else 0
            if a[src] < 0.3:
                continue
            lo, hi = h * hd, (h + 1) * hd
            masked = torch.zeros_like(z["v"][:, -1:, :])
            masked[:, :, lo:hi] = z["v"][:, -1:, lo:hi]
            r_h = op(masked)[0, -1] - op(torch.zeros_like(masked))[0, -1]
            dla_src = float(r_h @ W[ids[src]])
            if dla_src < -0.5:
                flags.append((f"L{l}H{h}", handle.tokenizer.decode([ids[src]]).strip(),
                              round(float(a[src]), 2), round(dla_src, 2)))
    flags.sort(key=lambda t: t[3])
    return {"copy_suppression_heads (head, suppressed_token, attn, DLA)": flags[:top_k_heads]}


def attribution_graph(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                       top_n: int = 12) -> dict:
    """Ameisen et al. 2025 / circuit-tracing -- turn per-head attributions
    into a small graph. Uses DLA (heads -> output) and EAP head scores as
    node weights, then links a head to the earlier heads whose ablation most
    changes its DLA (edges), giving an adjacency list of the dominant flow."""
    nh = _nh(handle)
    _, dla_top = tier_c.direct_logit_attribution(handle, clean_prompt, pos_token, neg_token,
                                                 range(handle.n_layers), nh, k=top_n)
    nodes = [tuple(h) for h in dla_top]
    base = C.dla_scores(handle, clean_prompt, pos_token, neg_token)
    edges = []
    for (rl, rh) in nodes:
        hd = adapter.get_head_dim(handle)
        for (sl, sh) in nodes:
            if sl >= rl:
                continue
            op = adapter._find_attn_out_proj(handle.layers[sl])
            lo, hi = sh * hd, (sh + 1) * hd

            def abl(m, a, k):
                x = (a[0] if a else k["input"]).clone()
                x[:, :, lo:hi] = x[:, :, lo:hi].mean(dim=(0, 1), keepdim=True)
                if a:
                    return (x,) + a[1:], k
                k["input"] = x
                return a, k
            after = C.dla_scores(handle, clean_prompt, pos_token, neg_token,
                                 extra_pre_hooks=[(op, abl)])
            delta = abs(after[(rl, rh)] - base[(rl, rh)])
            if delta > 0.05:
                edges.append((f"L{sl}H{sh}", f"L{rl}H{rh}", round(delta, 3)))
    edges.sort(key=lambda t: -t[2])
    return {"nodes": [f"L{l}H{h}" for l, h in nodes],
            "edges_(sender, receiver, |dDLA|)": edges[:top_n],
            "note": "output nodes ranked by DLA; edges = upstream head whose ablation moves that node"}


def entity_binding(handle, prompt, pos_token, neg_token, bind_pos: int, other_pos: int) -> dict:
    """Feng & Steinhardt 2023 / Prakash et al. -- test whether the model
    binds an attribute to the right entity. Patch the residual stream at the
    binding position vs a control position (from a run where the answer is
    the contrast token) and see which patch flips the decision."""
    mfn = C.logit_diff_metric_fn(handle, pos_token, neg_token)
    ci = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    mid = handle.n_layers // 2
    store = {}
    with adapter.MODEL_LOCK:
        h = handle.layers[mid].register_forward_hook(
            lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            with torch.no_grad():
                base = mfn(handle.model(**ci).logits)
        finally:
            h.remove()

    def patch_zeroing(pos):
        def hook(m, i, o):
            is_t = isinstance(o, tuple)
            hs = (o[0] if is_t else o).clone()
            p = pos if pos >= 0 else hs.shape[1] + pos
            hs[:, p, :] = store["h"][:, p, :].mean() * 0        # zero that position's state
            return (hs,) + o[1:] if is_t else hs
        with adapter.MODEL_LOCK:
            hh = handle.layers[mid].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    return mfn(handle.model(**ci).logits)
            finally:
                hh.remove()

    return {"layer": mid, "baseline_metric": round(base, 3),
            "metric_after_zeroing_binding_pos": round(patch_zeroing(bind_pos), 3),
            "metric_after_zeroing_control_pos": round(patch_zeroing(other_pos), 3),
            "note": "large drop at the binding position but not the control => attribute is bound there"}


def universality_across_models(*a, **k) -> dict:
    """Chughtai et al. / Gould et al. -- do the same circuits form across
    model sizes/seeds? NOT IMPLEMENTED here as a single call: run this whole
    `circuits` module (or `acdc`) on each model of a size ladder via
    `run_eval_matrix.py` / `techniques.runner.run_all`, then compare the
    recovered head sets. This module has no second model in scope."""
    raise NotImplementedError(universality_across_models.__doc__)


ALL = {
    "acdc": acdc, "eap": eap, "synergy_eap": synergy_eap,
    "qk_ov_decomposition": qk_ov_decomposition,
    "circuit_faithfulness": circuit_faithfulness,
    "copy_suppression": copy_suppression,
    "attribution_graph": attribution_graph,
    "entity_binding": entity_binding,
    "universality_across_models": universality_across_models,
}
