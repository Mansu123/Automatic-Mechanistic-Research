"""Hidden state decoding -- logit-lens trajectory, Patchscopes, SelfIE-style
readback.  (learnmechinterp / Hidden State Decoding)
"""
from __future__ import annotations

import torch

from ..tools import adapter
from . import _common as C
from .basic import logit_lens


def logit_lens_trajectory(handle, prompt, target_token: str) -> dict:
    """Track one token's logit-lens rank/prob across every layer -- the
    'emergence' curve (Geva et al. / nostalgebraist)."""
    tid = handle.tokenizer.encode(" " + target_token.strip())[-1]
    b = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    store = {}
    hooks = []
    with adapter.MODEL_LOCK:
        for i in range(handle.n_layers):
            hooks.append(handle.layers[i].register_forward_hook(
                (lambda i: (lambda m, inp, o: store.__setitem__(
                    i, (o[0] if isinstance(o, tuple) else o).detach()[:, -1, :])))(i)))
        try:
            with torch.no_grad():
                handle.model(**b)
        finally:
            for h in hooks:
                h.remove()
    ln, W = C.final_norm(handle), C.unembed(handle)
    rows = []
    for i in range(handle.n_layers):
        h = store[i]
        h = ln(h) if ln is not None else h
        lg = (h @ W.T)[0]
        prob = float(lg.softmax(-1)[tid])
        rank = int((lg > lg[tid]).sum())
        rows.append((i, rank, round(prob, 4)))
    return {"target": target_token, "per_layer_(rank,prob)": rows}


def patchscopes(handle, source_prompt, source_pos: int, id_prompt: str | None = None,
                 layer_idx: int | None = None) -> dict:
    """Ghandeharioun et al. 2024 -- lift the residual state at (source_prompt,
    source_pos, layer) and inject it at the last position of an identity
    prompt ('cat->cat; 1->1; ?'), then decode what the model verbalises. Reads
    a hidden state through the model's own vocabulary, no probe."""
    layer_idx = layer_idx if layer_idx is not None else handle.n_layers // 2
    id_prompt = id_prompt or "apple -> apple\n1234 -> 1234\nhello -> hello\n? -> "
    sb = handle.tokenizer([source_prompt], return_tensors="pt").to(handle.device)
    store = {}
    with adapter.MODEL_LOCK:
        h = handle.layers[layer_idx].register_forward_hook(
            lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            with torch.no_grad():
                handle.model(**sb)
        finally:
            h.remove()
    pos = source_pos if source_pos >= 0 else store["h"].shape[1] + source_pos
    vec = store["h"][0, pos]

    tb = handle.tokenizer([id_prompt], return_tensors="pt").to(handle.device)

    def inject(m, i, o):
        is_t = isinstance(o, tuple)
        hs = o[0] if is_t else o
        hs = hs.clone()
        hs[:, -1, :] = vec.to(hs.dtype)
        return (hs,) + o[1:] if is_t else hs
    with adapter.MODEL_LOCK:
        hh = handle.layers[layer_idx].register_forward_hook(inject)
        try:
            with torch.no_grad():
                lg = handle.model(**tb).logits[0, -1]
        finally:
            hh.remove()
    t = torch.topk(lg, 8)
    src_toks = [handle.tokenizer.decode([x]) for x in sb["input_ids"][0].tolist()]
    return {"layer": layer_idx, "source_token": src_toks[pos].strip(),
            "decoded_from_hidden_state":
                [(handle.tokenizer.decode([i]).strip(), round(float(v), 2))
                 for i, v in zip(t.indices.tolist(), t.values.tolist())]}


def selfie_readback(handle, source_prompt, source_pos, layer_idx: int | None = None) -> dict:
    """Chen et al. 2024 SelfIE -- same lift-and-inject idea with a natural
    'the concept here is' frame instead of the identity frame."""
    return patchscopes(handle, source_prompt, source_pos,
                       id_prompt="In one word, the meaning of this is: ",
                       layer_idx=layer_idx)


def concept_injection_introspection(handle, concept_pos_texts, concept_neg_texts,
                                     question_prompt: str | None = None,
                                     layer_idx: int | None = None, strength: float = 8.0) -> dict:
    """Anthropic 2025, 'Emergent Introspective Awareness' -- inject a concept
    vector (diff-of-means) into the residual stream while the model answers a
    question about its own state, and check whether the answer shifts toward
    the injected concept. A crude introspection test: does the model 'notice'
    the injected thought."""
    layer_idx = layer_idx if layer_idx is not None else 2 * handle.n_layers // 3
    question_prompt = question_prompt or "I am currently thinking about the topic of"
    d = C.diff_of_means_direction(handle, layer_idx, concept_pos_texts, concept_neg_texts)
    rms = float(C.capture_resid(handle, layer_idx, concept_pos_texts).norm(dim=1).mean()
                / (d.numel() ** 0.5))
    b = handle.tokenizer([question_prompt], return_tensors="pt").to(handle.device)

    def top(hook):
        with adapter.MODEL_LOCK:
            hs = [handle.layers[layer_idx].register_forward_hook(hook)] if hook else []
            try:
                with torch.no_grad():
                    lg = handle.model(**b).logits[0, -1]
            finally:
                for h in hs:
                    h.remove()
        t = torch.topk(lg, 6)
        return [(handle.tokenizer.decode([i]).strip(), round(float(v), 2))
                for i, v in zip(t.indices.tolist(), t.values.tolist())]

    return {"layer": layer_idx, "injected_concept_tokens": C.top_tokens(handle, d, 6),
            "answer_without_injection": top(None),
            "answer_with_injection": top(C.add_direction_hook(d, strength * rms)),
            "note": "overlap between injected-concept tokens and the with-injection answer "
                    "= the model surfaced the injected thought"}


def activation_oracle(*a, **k) -> dict:
    """Chen et al. / 'activation oracle' line -- a learned decoder that maps
    an arbitrary hidden state to a natural-language description. NOT
    IMPLEMENTED: needs a trained oracle/decoder model (or an API model doing
    the read-out) plus a paired (hidden state, description) corpus. patchscopes
    and selfie_readback are the training-free stand-ins in this module."""
    raise NotImplementedError(activation_oracle.__doc__)


ALL = {
    "logit_lens_trajectory": logit_lens_trajectory,
    "patchscopes": patchscopes,
    "selfie_readback": selfie_readback,
    "concept_injection_introspection": concept_injection_introspection,
    "activation_oracle": activation_oracle,
}
