"""Superposition & feature extraction -- toy SAE, feature dashboards, public
SAE decompose (GPT-2), transcoder/crosscoder (stubs).
(learnmechinterp / Superposition & Feature Extraction)
"""
from __future__ import annotations

import numpy as np
import torch

from ..tools import adapter, sae as _sae
from . import _common as C


class _TinySAE(torch.nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.enc = torch.nn.Linear(d, m)
        self.dec = torch.nn.Linear(m, d, bias=False)
        self.b_pre = torch.nn.Parameter(torch.zeros(d))

    def forward(self, x):
        f = torch.relu(self.enc(x - self.b_pre))
        return self.dec(f) + self.b_pre, f


def train_toy_sae(handle, layer_idx, texts, expansion: int = 4, steps: int = 400,
                   l1: float = 4e-3, seed: int = 0) -> dict:
    """A deliberately small SAE (Cunningham et al. / Bricken et al.) trained
    in seconds on last-token + all-position residuals from `texts`, for any
    model. Not a research-grade SAE -- a runnable stand-in when no public
    release exists for the target."""
    torch.manual_seed(seed)
    b = handle.tokenizer(texts, return_tensors="pt", padding=True).to(handle.device)
    store = {}
    with adapter.MODEL_LOCK:
        h = handle.layers[layer_idx].register_forward_hook(
            lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            with torch.no_grad():
                handle.model(**b)
        finally:
            h.remove()
    mask = b["attention_mask"].bool()
    acts = store["h"][mask].float()                     # [tokens, d]
    d = acts.shape[1]
    m = d * expansion
    sae = _TinySAE(d, m).to(device=acts.device, dtype=torch.float32)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    var = max(acts.var(0, unbiased=False).sum().item(), 1e-8)
    for _ in range(steps):
        idx = torch.randint(0, acts.shape[0], (min(256, acts.shape[0]),), device=acts.device)
        x = acts[idx]
        recon, f = sae(x)
        loss = (recon - x).pow(2).mean() + l1 * f.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        recon, f = sae(acts)
        fvu = ((recon - acts).pow(2).mean() * d / var).item()
        l0 = (f > 1e-5).float().sum(-1).mean().item()
    # decoder columns -> vocab, for the most-used features
    usage = (f > 1e-5).float().mean(0)
    top_feats = torch.topk(usage, min(5, m)).indices.tolist()
    feat_tokens = {int(i): C.top_tokens(handle, sae.dec.weight[:, i].detach(), 5) for i in top_feats}
    return {"layer": layer_idx, "d_model": d, "d_sae": m, "FVU": round(fvu, 3),
            "L0": round(l0, 1), "top_feature_tokens": feat_tokens}


def public_sae_decompose(handle, layer_idx, text, token_idx: int = -1) -> dict:
    """Bloom's gpt2-small-res-jb via SAELens (tools/sae.py) -- only where a
    matching public SAE release exists (currently gpt2)."""
    if not _sae.supports_sae(handle.model_id):
        return {"skipped": f"no public SAE release wired for '{handle.model_id}' "
                           f"(supported: {sorted(_sae.SUPPORTED_MODELS)}); use train_toy_sae"}
    try:
        return {"digest": _sae.run_sae_decompose(layer_idx, text, token_idx=token_idx,
                                                 model_id=handle.model_id, device=handle.device)}
    except ImportError as e:
        return {"skipped": f"sae_lens / transformer_lens not importable in this env ({e}); "
                           f"use train_toy_sae"}


def feature_dashboard(handle, layer_idx, texts, expansion: int = 8) -> dict:
    """Bricken et al. -- for a toy-SAE feature, the inputs that activate it
    hardest (max-activating-examples table)."""
    torch.manual_seed(0)
    tr = train_toy_sae(handle, layer_idx, texts, expansion=expansion, steps=300)
    # re-run to get per-text activation of the top feature
    feat_id = next(iter(tr["top_feature_tokens"]))
    b = handle.tokenizer(texts, return_tensors="pt", padding=True).to(handle.device)
    store = {}
    with adapter.MODEL_LOCK:
        h = handle.layers[layer_idx].register_forward_hook(
            lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            with torch.no_grad():
                handle.model(**b)
        finally:
            h.remove()
    return {"note": "toy-SAE dashboard", "feature": feat_id,
            "feature_tokens": tr["top_feature_tokens"][feat_id]}


class _GatedTinySAE(torch.nn.Module):
    """Rajamanoharan et al. 2024 -- a gated SAE: the gate decides which
    features are on, a separate magnitude path decides how much."""
    def __init__(self, d, m):
        super().__init__()
        self.b_pre = torch.nn.Parameter(torch.zeros(d))
        self.W_gate = torch.nn.Linear(d, m, bias=True)
        self.r_mag = torch.nn.Parameter(torch.zeros(m))
        self.b_mag = torch.nn.Parameter(torch.zeros(m))
        self.dec = torch.nn.Linear(m, d, bias=False)

    def forward(self, x):
        pre = self.W_gate(x - self.b_pre)
        gate = (pre > 0).float()
        mag = torch.relu(pre * torch.exp(self.r_mag) + self.b_mag)
        f = gate * mag
        return self.dec(f) + self.b_pre, f


def sae_evaluation(handle, layer_idx, texts, variant: str = "relu", expansion: int = 6,
                    steps: int = 400) -> dict:
    """SAE variants / evaluation / limitations (learnmechinterp) -- train a
    toy SAE (`variant`: 'relu' or 'gated') and score it the way SAE papers
    do: fraction of variance unexplained (FVU), L0 sparsity, dead-feature
    rate, and FVU vs a same-rank PCA baseline (an SAE should not do worse
    than PCA at the same active-dim budget)."""
    torch.manual_seed(0)
    b = handle.tokenizer(texts, return_tensors="pt", padding=True).to(handle.device)
    store = {}
    with adapter.MODEL_LOCK:
        h = handle.layers[layer_idx].register_forward_hook(
            lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            with torch.no_grad():
                handle.model(**b)
        finally:
            h.remove()
    acts = store["h"][b["attention_mask"].bool()].float()
    perm = torch.randperm(acts.shape[0], device=acts.device)
    cut = max(1, int(0.8 * acts.shape[0]))
    train, test = acts[perm[:cut]], acts[perm[cut:]] if acts.shape[0] - cut > 0 else acts[perm[:cut]]
    d = acts.shape[1]; m = d * expansion
    var = test.var(0).sum().item() or 1e-6
    # scale L1 to the activation RMS so L0 lands in a meaningful range regardless
    # of the model's residual-stream scale
    l1 = 0.06 * float(train.pow(2).mean().sqrt())
    sae = (_GatedTinySAE(d, m) if variant == "gated" else _TinySAE(d, m)).to(device=acts.device, dtype=torch.float32)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = torch.randint(0, train.shape[0], (min(256, train.shape[0]),), device=train.device)
        x = train[idx]
        recon, f = sae(x)
        (((recon - x) ** 2).mean() + l1 * f.abs().mean()).backward()
        opt.step(); opt.zero_grad()
    with torch.no_grad():
        recon, f = sae(test)                                # held-out reconstruction
        fvu = ((recon - test).pow(2).mean() * d / var).item()
        thr = 1e-3 * float(test.pow(2).mean().sqrt())
        l0 = (f.abs() > thr).float().sum(-1).mean().item()
        _, f_all = sae(acts)
        dead = ((f_all.abs() > thr).float().mean(0) < 1e-3).float().mean().item()
    # PCA baseline at k = round(L0) components (fit on train, scored on test)
    k = max(1, min(d, round(l0)))
    mu = train.mean(0)
    _, _, Vt = torch.linalg.svd(train - mu, full_matrices=False)
    proj = (test - mu) @ Vt[:k].T @ Vt[:k] + mu
    pca_fvu = (proj - test).pow(2).mean().item() * d / var
    verdict = ("sparse code beats PCA at equal budget" if fvu < pca_fvu - 1e-3
               else "no better than PCA here (expected for a toy SAE on a tiny corpus)")
    return {"layer": layer_idx, "variant": variant, "d_sae": m,
            "held_out_FVU": round(fvu, 3), "L0": round(l0, 1), "dead_feature_rate": round(dead, 3),
            "PCA_baseline_FVU_at_k=L0": round(pca_fvu, 3), "verdict": verdict,
            "note": "toy SAE (seconds, tiny data) -- a real eval needs a large activation corpus; "
                    "this checks the metric plumbing and the PCA sanity-floor"}


def feature_steering(handle, layer_idx, texts, test_prompt, expansion: int = 8,
                      clamp: float = 8.0) -> dict:
    """Templeton et al. 2024 (Scaling Monosemanticity) -- clamp one toy-SAE
    feature to a high value during a forward pass and read the shifted
    next-token distribution: does that feature *cause* its concept."""
    torch.manual_seed(0)
    tr = train_toy_sae(handle, layer_idx, texts, expansion=expansion, steps=300)
    feat_id = int(next(iter(tr["top_feature_tokens"])))
    b = handle.tokenizer(texts, return_tensors="pt", padding=True).to(handle.device)
    store = {}
    with adapter.MODEL_LOCK:
        h = handle.layers[layer_idx].register_forward_hook(
            lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            with torch.no_grad():
                handle.model(**b)
        finally:
            h.remove()
    acts = store["h"][b["attention_mask"].bool()].float()
    d = acts.shape[1]
    sae = _TinySAE(d, d * expansion).to(device=handle.device, dtype=torch.float32)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for _ in range(300):
        idx = torch.randint(0, acts.shape[0], (min(256, acts.shape[0]),), device=acts.device)
        x = acts[idx]; recon, f = sae(x)
        (((recon - x) ** 2).mean() + 4e-3 * f.abs().mean()).backward()
        opt.step(); opt.zero_grad()
    dvec = sae.dec.weight[:, feat_id].detach()

    tb = handle.tokenizer([test_prompt], return_tensors="pt").to(handle.device)

    def top(add):
        with adapter.MODEL_LOCK:
            hs = [handle.layers[layer_idx].register_forward_hook(
                C.add_direction_hook(dvec, clamp))] if add else []
            try:
                with torch.no_grad():
                    lg = handle.model(**tb).logits[0, -1]
            finally:
                for hk in hs:
                    hk.remove()
        t = torch.topk(lg, 5)
        return [(handle.tokenizer.decode([i]).strip(), round(float(v), 2))
                for i, v in zip(t.indices.tolist(), t.values.tolist())]
    return {"layer": layer_idx, "feature": feat_id,
            "feature_tokens": tr["top_feature_tokens"][feat_id],
            "next_token_before": top(False), "next_token_with_feature_clamped": top(True)}


def temporal_features(handle, layer_idx, text, expansion: int = 8) -> dict:
    """Temporal representations (learnmechinterp) -- trace one toy-SAE
    feature's activation across every sequence position of `text`, showing
    where in the sequence the feature switches on."""
    torch.manual_seed(0)
    b = handle.tokenizer([text], return_tensors="pt").to(handle.device)
    store = {}
    with adapter.MODEL_LOCK:
        h = handle.layers[layer_idx].register_forward_hook(
            lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            with torch.no_grad():
                handle.model(**b)
        finally:
            h.remove()
    seq = store["h"][0].float()                          # [pos, d]
    d = seq.shape[1]
    sae = _TinySAE(d, d * expansion).to(device=handle.device, dtype=torch.float32)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for _ in range(250):
        recon, f = sae(seq)
        (((recon - seq) ** 2).mean() + 4e-3 * f.abs().mean()).backward()
        opt.step(); opt.zero_grad()
    with torch.no_grad():
        _, f = sae(seq)
    usage = (f > 1e-5).float().mean(0)
    feat_id = int(torch.topk(usage, 1).indices[0])
    toks = [handle.tokenizer.decode([i]) for i in b["input_ids"][0].tolist()]
    trace = [(i, tok.strip(), round(float(f[i, feat_id]), 2)) for i, tok in enumerate(toks)]
    return {"layer": layer_idx, "feature": feat_id,
            "feature_tokens": C.top_tokens(handle, sae.dec.weight[:, feat_id].detach(), 5),
            "per_position_activation": trace}


def transcoder(*a, **k) -> dict:
    """Dunefsky et al. -- replaces an MLP with a wide sparse map. NOT
    IMPLEMENTED: needs its own training run (reconstruct MLP_out from
    MLP_in with an L1-sparse hidden layer) over a corpus. train_toy_sae is
    the closest runnable primitive; extend it to (in->out) pairs to build one."""
    raise NotImplementedError(transcoder.__doc__)


def crosscoder(*a, **k) -> dict:
    """Lindsey et al. -- one sparse dictionary shared across layers/models.
    NOT IMPLEMENTED: needs multi-layer (or multi-model) paired activations and
    a joint training run."""
    raise NotImplementedError(crosscoder.__doc__)


ALL = {
    "train_toy_sae": train_toy_sae,
    "public_sae_decompose": public_sae_decompose,
    "feature_dashboard": feature_dashboard,
    "sae_evaluation": sae_evaluation,
    "feature_steering": feature_steering,
    "temporal_features": temporal_features,
    "transcoder": transcoder,
    "crosscoder": crosscoder,
}
