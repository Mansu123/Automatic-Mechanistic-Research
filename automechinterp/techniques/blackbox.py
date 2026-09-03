"""Black-box interpretability -- counterfactual resampling / input-ablation
sensitivity, minimal-pair contrast.  (learnmechinterp / Black-Box Interpretability)
"""
from __future__ import annotations

import torch

from ..tools import adapter
from . import _common as C


def counterfactual_resampling(handle, prompt, pos_token, neg_token, n_variants: int = 12,
                               seed: int = 0) -> dict:
    """Modify the input, not the weights -- per-token importance by replacing
    each token with the pad/unk token and measuring the logit-diff drop."""
    b = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    ids = b["input_ids"][0]
    base = C.last_token_logit_diff(handle, prompt, pos_token, neg_token)
    io = handle.tokenizer.encode(" " + pos_token.strip())[-1]
    s = handle.tokenizer.encode(" " + neg_token.strip())[-1]
    pad = handle.tokenizer.pad_token_id or handle.tokenizer.eos_token_id
    rows = []
    for i in range(len(ids)):
        alt = ids.clone()
        alt[i] = pad
        with adapter.MODEL_LOCK, torch.no_grad():
            lg = handle.model(input_ids=alt.unsqueeze(0),
                              attention_mask=b["attention_mask"]).logits[0, -1]
        rows.append((handle.tokenizer.decode([ids[i]]).strip(),
                     round(base - float(lg[io] - lg[s]), 3)))
    rows_sorted = sorted(enumerate(rows), key=lambda t: -abs(t[1][1]))
    return {"baseline_logit_diff": round(base, 3),
            "per_token_importance": [r for _, r in rows_sorted[:10]]}


def minimal_pair_contrast(handle, prompt_a, prompt_b, pos_token, neg_token) -> dict:
    """The clean/corrupt contrast, as a black-box behavioural probe: how much
    does the single controlled edit move the model's decision."""
    a = C.last_token_logit_diff(handle, prompt_a, pos_token, neg_token)
    b = C.last_token_logit_diff(handle, prompt_b, pos_token, neg_token)
    return {"logit_diff_A": round(a, 3), "logit_diff_B": round(b, 3),
            "behavioural_effect_of_edit": round(a - b, 3)}


ALL = {
    "counterfactual_resampling": counterfactual_resampling,
    "minimal_pair_contrast": minimal_pair_contrast,
}
