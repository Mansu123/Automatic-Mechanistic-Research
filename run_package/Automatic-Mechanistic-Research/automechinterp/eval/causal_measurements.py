"""Explicit continuation metrics and scoped interventions for reproducible runs.

The unit of observation is a prompt/answer contrast. No final-token shortcut
is used for multi-token answers, and unavailable evidence is never zero-filled.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from ..tools import adapter


def _candidate_batch(handle, prompt, answer):
    prompt = prompt.rstrip()
    continuation = " " + answer.strip()
    prefix = handle.tokenizer.encode(prompt, add_special_tokens=True)
    full = handle.tokenizer.encode(prompt + continuation, add_special_tokens=True)
    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Answer boundary retokenizes the prompt; revise the template explicitly")
    ids = torch.tensor([full], device=handle.device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, len(prefix)


def continuation_log_probability(handle, prompt, answer, intervention=None):
    batch, start = _candidate_batch(handle, prompt, answer)
    with adapter.MODEL_LOCK, (intervention() if intervention else nullcontext()), torch.no_grad():
        logits = handle.model(**batch, use_cache=False).logits.float()
    selected = logits[0, start - 1:-1]
    labels = batch["input_ids"][0, start:]
    if not torch.isfinite(selected).all():
        raise FloatingPointError("Nonfinite continuation logits")
    values = selected.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    return float(values.sum().item()), len(labels)


def score_contrast(handle, prompt, positive, negative, intervention=None):
    """One batched forward scores both full continuations, including shared prefixes."""
    if positive.strip() == negative.strip():
        raise ValueError("Identical candidate answers cannot define a contrast")
    rows = [_candidate_batch(handle, prompt, x) for x in (positive, negative)]
    width = max(r[0]["input_ids"].shape[1] for r in rows)
    pad = getattr(handle.tokenizer, "pad_token_id", None)
    pad = pad if pad is not None else 0
    ids = torch.full((2, width), pad, dtype=torch.long, device=handle.device)
    mask = torch.zeros_like(ids)
    for i, (row, start) in enumerate(rows):
        n = row["input_ids"].shape[1]
        ids[i, :n] = row["input_ids"][0]; mask[i, :n] = 1
    with adapter.MODEL_LOCK, (intervention() if intervention else nullcontext()), torch.no_grad():
        logits = handle.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    scores, counts = [], []
    for i, (row, start) in enumerate(rows):
        end = row["input_ids"].shape[1]
        selected = logits[i, start-1:end-1].float()
        if not torch.isfinite(selected).all():
            raise FloatingPointError("Nonfinite continuation logits")
        labels = ids[i, start:end]
        values = selected.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        scores.append(float(values.sum())); counts.append(end-start)
    pos, neg = scores; np_, nn_ = counts
    return {"positive_logp": pos, "negative_logp": neg, "margin": pos-neg,
            "positive_tokens": np_, "negative_tokens": nn_,
            "length_normalized_margin": pos/np_ - neg/nn_,
            "pairwise_accuracy": float(pos > neg)}


def summarize(values, seed=1729, draws=2000):
    """Prompt-level percentile CI; callers must pass independent cluster means."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return {"n": len(values), "mean": None, "ci95": None, "status": "invalid_or_empty"}
    result = {"n": len(values), "mean": float(values.mean()), "ci95": None, "status": "ok"}
    if len(values) > 1:
        rng = np.random.default_rng(seed)
        boot = rng.choice(values, (draws, len(values)), replace=True).mean(1)
        result["ci95"] = np.quantile(boot, [.025, .975]).tolist()
    return result


def proportion(successes, n):
    """Wilson 95% interval stays informative at zero/all successes in small pilots."""
    if n<=0 or not 0<=successes<=n:
        return {"n":n,"successes":successes,"mean":None,"ci95":None,"status":"invalid_or_empty"}
    z=1.959963984540054
    p=successes/n
    scale=1+z*z/n
    center=(p+z*z/(2*n))/scale
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/scale
    return {"n":n,"successes":successes,"mean":p,"ci95":[max(0.,center-radius),min(1.,center+radius)],
            "status":"ok","interval_method":"Wilson 95%; independent Bernoulli observations"}


def normalized_effect(numerators, denominators, seed=1729, draws=2000, tolerance=1e-6):
    num, den = np.asarray(numerators, float), np.asarray(denominators, float)
    if num.shape != den.shape or num.ndim != 1 or not len(num):
        raise ValueError("Paired nonempty numerator/denominator arrays required")
    if not np.isfinite(num).all() or not np.isfinite(den).all():
        raise FloatingPointError("Nonfinite causal effects")
    result = {"n": len(num), "numerator_mean": float(num.mean()),
              "denominator_mean": float(den.mean()), "estimate": None, "ci95": None,
              "status": "unresolved_denominator"}
    if den.mean() <= tolerance:
        return result
    if len(num) == 1:
        result.update(estimate=float(num.mean()/den.mean()), status="descriptive_single_item")
        return result
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(num), size=(draws, len(num)))
    db = den[indices].mean(1)
    if np.quantile(db, .025) <= tolerance:
        return result
    ratios = num[indices].mean(1) / db
    result.update(estimate=float(num.mean()/den.mean()), ci95=np.quantile(ratios, [.025, .975]).tolist(), status="ok")
    return result


def _hidden(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


def _mlp(layer):
    for name in ("mlp", "feed_forward"):
        if hasattr(layer, name):
            return getattr(layer, name)
    # OPT's fc2 is the block's MLP output projection.
    if hasattr(layer, "fc2"):
        return layer.fc2
    raise ValueError(f"No MLP output module on {type(layer).__name__}")


@dataclass
class ActivationBank:
    """Last prompt-position states; scope is explicit and never suffix-aligned."""
    handle: adapter.ModelHandle
    prompt: str
    residual: dict
    update: dict
    head_input: dict
    mlp_output: dict
    prompt_length: int

    @classmethod
    def capture(cls, handle, prompt):
        batch = handle.tokenizer([prompt.rstrip()], return_tensors="pt").to(handle.device)
        residual, update, heads, mlps, inputs = {}, {}, {}, {}, {}
        hooks = []
        def pre(layer):
            def callback(module, args, kwargs):
                h = _hidden(args, kwargs)
                inputs[layer] = (h[:, -1, :] if h.dim() == 3 else h[-1, :].unsqueeze(0)).detach().clone()
            return callback
        def out(layer):
            def callback(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                hs_last = (hs[:, -1, :] if hs.dim() == 3 else hs[-1, :].unsqueeze(0))
                residual[layer] = hs_last.detach().clone()
                update[layer] = (residual[layer].float() - inputs[layer].float())
            return callback
        def head(layer):
            def callback(module, args, kwargs):
                x = args[0] if args else kwargs["input"]
                heads[layer] = (x[:, -1, :] if x.dim() == 3 else x[-1, :].unsqueeze(0)).detach().clone()
            return callback
        def mlp(layer):
            def callback(module, args, output):
                x = output[0] if isinstance(output, tuple) else output
                mlps[layer] = (x[:, -1, :] if x.dim() == 3 else x[-1, :].unsqueeze(0)).detach().clone()
            return callback
        with adapter.MODEL_LOCK:
            try:
                for i, layer in enumerate(handle.layers):
                    hooks.append(layer.register_forward_pre_hook(pre(i), with_kwargs=True))
                    hooks.append(layer.register_forward_hook(out(i)))
                    hooks.append(adapter._find_attn_out_proj(layer).register_forward_pre_hook(head(i), with_kwargs=True))
                    hooks.append(_mlp(layer).register_forward_hook(mlp(i)))
                with torch.no_grad():
                    logits = handle.model(**batch, use_cache=False).logits
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite activation-bank forward")
            finally:
                for h in hooks:
                    h.remove()
        return cls(handle, prompt, residual, update, heads, mlps, batch["input_ids"].shape[1])

    @contextmanager
    def patch(self, kind, layer, destination_position, head=None):
        handle = self.handle
        hooks, incoming = [], {}
        if not 0 <= layer < handle.n_layers or destination_position < 0:
            raise ValueError("Invalid patch coordinate")
        def pre(module, args, kwargs):
            h = _hidden(args, kwargs)
            if h.dim() == 2:
                incoming["x"] = h[destination_position, :].unsqueeze(0).clone()
            else:
                incoming["x"] = h[:, destination_position, :].clone()
        def output_hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            changed = hs.clone()
            if kind == "block_update":
                value = (incoming["x"].float() + self.update[layer].float()).to(changed.dtype)
            elif kind == "mlp":
                value = self.mlp_output[layer]
            else:
                value = self.residual[layer]
            if changed.dim() == 2:
                changed[destination_position, :] = value.view(-1).to(changed)
            else:
                changed[:, destination_position, :] = value.to(changed).expand_as(changed[:, destination_position, :])
            return (changed,) + output[1:] if isinstance(output, tuple) else changed
        def head_hook(module, args, kwargs):
            x = (args[0] if args else kwargs["input"]).clone()
            dim = adapter.get_head_dim(handle)
            if head is None or not 0 <= head < adapter.get_num_heads(handle):
                raise ValueError("Invalid query-head coordinate")
            lo, hi = head * dim, (head + 1) * dim
            if x.dim() == 2:
                x[destination_position, lo:hi] = self.head_input[layer].view(-1)[lo:hi].to(x)
            else:
                x[:, destination_position, lo:hi] = self.head_input[layer].reshape(1, -1).expand(x.shape[0], -1)[:, lo:hi].to(x)
            if args:
                return (x,) + args[1:], kwargs
            return args, {**kwargs, "input": x}
        with adapter.MODEL_LOCK:
            try:
                if kind == "head":
                    hooks.append(adapter._find_attn_out_proj(handle.layers[layer]).register_forward_pre_hook(head_hook, with_kwargs=True))
                elif kind in ("residual", "block_update", "mlp"):
                    module = _mlp(handle.layers[layer]) if kind == "mlp" else handle.layers[layer]
                    if kind == "block_update":
                        hooks.append(module.register_forward_pre_hook(pre, with_kwargs=True))
                    hooks.append(module.register_forward_hook(output_hook))
                else:
                    raise ValueError(f"Unsupported intervention kind: {kind}")
                yield
            finally:
                for h in hooks:
                    h.remove()


@contextmanager
def ablate_heads(handle, heads, *, positions=None, reference=None):
    """Joint zero ablation, or an explicitly supplied independent reference.

    positions=None means all positions. A provided mapping holds per-layer
    reference vectors and must come from a separate reference population.
    """
    grouped, hooks = {}, []
    dim = adapter.get_head_dim(handle)
    for layer, head in set(map(tuple, heads)):
        if not 0 <= layer < handle.n_layers or not 0 <= head < adapter.get_num_heads(handle):
            raise ValueError("Invalid coalition coordinate")
        grouped.setdefault(layer, []).append(head)
    def make_hook(layer, selected):
        def callback(module, args, kwargs):
            x = (args[0] if args else kwargs["input"]).clone()
            where = slice(None) if positions is None else list(positions)
            for head in selected:
                lo, hi = head * dim, (head + 1) * dim
                x[:, where, lo:hi] = 0 if reference is None else reference[layer][lo:hi].to(x)
            if args:
                return (x,) + args[1:], kwargs
            return args, {**kwargs, "input": x}
        return callback
    with adapter.MODEL_LOCK:
        try:
            for layer, selected in grouped.items():
                hooks.append(adapter._find_attn_out_proj(handle.layers[layer]).register_forward_pre_hook(make_hook(layer, selected), with_kwargs=True))
            yield
        finally:
            for h in hooks:
                h.remove()


class ForwardCounter:
    def __init__(self, model):
        self.model, self.calls, self.tokens = model, 0, 0
    def __enter__(self):
        self.started = time.perf_counter()
        def count(module, args, kwargs):
            self.calls += 1
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None:
                self.tokens += ids.numel()
        self.hook = self.model.register_forward_pre_hook(count, with_kwargs=True)
        return self
    def __exit__(self, *exc):
        self.hook.remove()
        self.seconds = time.perf_counter() - self.started
    def as_dict(self):
        return {"model_forward_calls": self.calls, "model_input_tokens": self.tokens,
                "wall_seconds": getattr(self, "seconds", time.perf_counter()-self.started)}
