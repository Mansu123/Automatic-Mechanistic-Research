"""Architecture-agnostic tool layer (Sec. 3.3, Key Contribution #2).

Built entirely on torch.nn.Module.named_modules() + forward hooks -- no
TransformerLens import anywhere in this file. The same functions run against
GPT-2 (`transformer.h`) and any Llama-family decoder stack such as Qwen2
(`model.layers`), which is exactly the "works identically on LLM decoder
blocks" claim in the proposal. Circuit-discovery tools that need attention
weights (tools/tier_c.py) are transformer-specific by nature, but this file
-- register_model, capture_activations, layer_similarity, ablate_layer,
swap_layer, freeze_and_probe, patch_layer -- has no such assumption.
"""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

_LAYER_STACK_PATHS = ["transformer.h", "model.layers", "gpt_neox.layers", "model.decoder.layers"]

# Layer Agents and Component Agents run concurrently (Sec. 3.1: "Layer Agents
# run in parallel"), but PyTorch's hook registration bookkeeping (handle-id
# allocation, the with_kwargs tracking set) is not safe under concurrent
# register/remove from multiple threads on a shared model -- it manifests as
# spurious "missing 1 required positional argument: 'kwargs'" TypeErrors deep
# in nn.Module.__call__. Every function that registers a hook, runs a forward
# pass, and removes the hook acquires this lock, so agent *orchestration*
# stays concurrent while the actual tensor ops against the shared model
# instance are serialized -- the same trade-off a real single-GPU inference
# server makes.
MODEL_LOCK = threading.Lock()


@dataclass
class ModelHandle:
    model: nn.Module
    tokenizer: object
    model_id: str
    device: str
    layer_stack_path: str
    layers: list[nn.Module] = field(repr=False)
    # "eager" gives per-head attention weights (output_attentions=True) but some
    # architectures (GPT-NeoX / Pythia on transformers >= 5.x) go numerically
    # unstable in the eager attention path and return NaN logits. register_model
    # detects that and falls back; anything below "eager" means Tier C's
    # get_attention_pattern is unavailable for this model (see that function).
    attn_impl: str = "eager"

    @property
    def n_layers(self) -> int:
        return len(self.layers)


def _discover_layers(model: nn.Module) -> tuple[str, list[nn.Module]]:
    for path in _LAYER_STACK_PATHS:
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)) and len(obj) > 1:
            return path, list(obj)
    candidates = [(n, m) for n, m in model.named_modules()
                  if isinstance(m, nn.ModuleList) and len(m) > 1]
    if candidates:
        name, m = max(candidates, key=lambda nm: len(nm[1]))
        return name, list(m)
    raise ValueError(f"Could not auto-discover a layer stack on {type(model).__name__}. "
                      f"Tried: {_LAYER_STACK_PATHS}")


def _logits_are_finite(model: nn.Module, tok, device: str) -> bool:
    """One tiny forward pass -- catches architectures whose eager attention
    path is numerically broken on the installed transformers (GPT-NeoX /
    Pythia return all-NaN logits from a mid-stack layer)."""
    try:
        batch = tok(["The quick brown fox jumps over the lazy dog."],
                    return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**batch).logits
        return bool(torch.isfinite(logits).all())
    except Exception:
        return False


def register_model(model_id: str, device: str = "cpu", torch_dtype=None) -> ModelHandle:
    """Tool: register_model(). Introspects the module graph and returns a
    structural map the Network Analyst reasons over (Tier N)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _load(attn_impl: str | None):
        kwargs = {} if attn_impl is None else {"attn_implementation": attn_impl}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        m = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        m.to(device)
        m.eval()
        return m

    # "eager" is required for output_attentions=True (Tier C's
    # get_attention_pattern) to return per-head weights -- SDPA/flash skip
    # materializing them. But if eager is numerically broken for this
    # architecture (NaN logits), a model that runs is worth more than one
    # attention tool: fall back, and let get_attention_pattern report itself
    # unavailable.
    model = _load("eager")
    attn_impl = "eager"
    if not _logits_are_finite(model, tok, device):
        del model
        for fallback in ("sdpa", None):
            try:
                cand = _load(fallback)
            except (ValueError, ImportError):
                continue
            if _logits_are_finite(cand, tok, device):
                model, attn_impl = cand, (fallback or "default")
                print(f"[adapter] {model_id}: eager attention gives non-finite logits on this "
                      f"transformers build -- using {attn_impl!r} instead (get_attention_pattern "
                      f"will be unavailable for this model)")
                break
            del cand
        else:
            print(f"[adapter] WARNING: {model_id} produced non-finite logits with every "
                  f"attention backend tried; proceeding with eager anyway")
            model = _load("eager")

    path, layers = _discover_layers(model)
    return ModelHandle(model=model, tokenizer=tok, model_id=model_id, device=device,
                        layer_stack_path=path, layers=layers, attn_impl=attn_impl)


def profile_network(handle: ModelHandle) -> dict:
    """Tool: profile_network(). Depth, widths, parameter distribution, module
    taxonomy (Tier N)."""
    n_params = sum(p.numel() for p in handle.model.parameters())
    taxonomy: dict[str, int] = {}
    for _, m in handle.model.named_modules():
        cls = type(m).__name__
        taxonomy[cls] = taxonomy.get(cls, 0) + 1
    cfg = getattr(handle.model, "config", None)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    return {
        "model_id": handle.model_id,
        "n_layers": handle.n_layers,
        "hidden_size": hidden_size,
        "n_heads": n_heads,
        "n_params": sum(p.numel() for p in handle.model.parameters()),
        "layer_stack_path": handle.layer_stack_path,
    }


def _tokenize_batch(handle: ModelHandle, texts: list[str]):
    return handle.tokenizer(texts, return_tensors="pt", padding=True).to(handle.device)


def capture_activations(handle: ModelHandle, layer_indices: list[int], texts: list[str]) -> dict[int, np.ndarray]:
    """Tool: capture_activations(). Last-token hidden state per requested
    layer, one row per input text. Identical interface regardless of block
    type (GPT2Block vs Qwen2DecoderLayer) because it only touches the tuple
    the forward hook receives."""
    batch = _tokenize_batch(handle, texts)
    last_idx = batch["attention_mask"].sum(dim=1) - 1  # last real (non-pad) token per row

    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(idx):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[idx] = hs.detach()
        return hook

    with MODEL_LOCK:
        for idx in layer_indices:
            hooks.append(handle.layers[idx].register_forward_hook(make_hook(idx)))
        try:
            with torch.no_grad():
                handle.model(**batch)
        finally:
            for h in hooks:
                h.remove()

    out = {}
    for idx in layer_indices:
        hs = captured[idx]  # [batch, seq, d_model]
        rows = hs[torch.arange(hs.shape[0]), last_idx]  # [batch, d_model] pooled at last real token
        out[idx] = rows.float().cpu().numpy()
    return out


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    normx = np.linalg.norm(X.T @ X, "fro")
    normy = np.linalg.norm(Y.T @ Y, "fro")
    denom = normx * normy
    return float(hsic / denom) if denom > 1e-12 else 0.0


def layer_similarity(acts_a: np.ndarray, acts_b: np.ndarray, method: str = "cka") -> float:
    """Tool: layer_similarity(). CKA between two activation matrices -- used
    both within one model (consecutive-layer shift detection) and across two
    models (fine-tuning diff, Sec. 4.5)."""
    if method != "cka":
        raise NotImplementedError("only linear CKA is implemented; SVCCA/Procrustes are drop-in extensions")
    return _linear_cka(acts_a, acts_b)


def ablate_layer(handle: ModelHandle, layer_idx: int, mode: str,
                  run_fn: Callable[[], float]) -> dict:
    """Tool: ablate_layer(). Zero / mean / identity-skip ablation of an entire
    layer; returns the behavioral delta on whatever scalar `run_fn` computes."""
    assert mode in {"zero", "mean", "skip"}
    layer = handle.layers[layer_idx]
    baseline = run_fn()

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if mode == "zero":
            hs = torch.zeros_like(hs)
        elif mode == "mean":
            hs = hs.mean(dim=(0, 1), keepdim=True).expand_as(hs).contiguous()
        elif mode == "skip":
            hs = inputs[0]
        return (hs,) + output[1:] if is_tuple else hs

    with MODEL_LOCK:
        h = layer.register_forward_hook(hook)
        try:
            ablated = run_fn()
        finally:
            h.remove()
    return {"layer": layer_idx, "mode": mode, "baseline": baseline, "ablated": ablated,
            "delta": ablated - baseline}


def swap_layer(handle: ModelHandle, layer_i: int, layer_j: int, run_fn: Callable[[], float]) -> dict:
    """Tool: swap_layer(). Permutes two layers' outputs for one forward pass
    and measures the behavioral delta -- redundancy / order-sensitivity probe."""
    baseline = run_fn()
    captured = {}

    def capture_hook(idx):
        def hook(module, inputs, output):
            captured[idx] = (output[0] if isinstance(output, tuple) else output).detach().clone()
        return hook

    with MODEL_LOCK:
        h1 = handle.layers[layer_i].register_forward_hook(capture_hook(layer_i))
        h2 = handle.layers[layer_j].register_forward_hook(capture_hook(layer_j))
        try:
            run_fn()
        finally:
            h1.remove()
            h2.remove()

        def swap_hook(idx, other_idx):
            def hook(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                replacement = captured[other_idx]
                return (replacement,) + output[1:] if is_tuple else replacement
            return hook

        h1 = handle.layers[layer_i].register_forward_hook(swap_hook(layer_i, layer_j))
        h2 = handle.layers[layer_j].register_forward_hook(swap_hook(layer_j, layer_i))
        try:
            swapped = run_fn()
        finally:
            h1.remove()
            h2.remove()
    return {"layer_i": layer_i, "layer_j": layer_j, "baseline": baseline, "swapped": swapped,
            "delta": swapped - baseline}


def _get_layer_container(handle: ModelHandle):
    """Walks layer_stack_path to the actual nn.ModuleList living inside
    handle.model (e.g. model.transformer.h) -- distinct from handle.layers,
    which is a plain Python list snapshot. transplant_layer() has to mutate
    the real container, not the snapshot, or the forward pass won't see it."""
    obj = handle.model
    for part in handle.layer_stack_path.split("."):
        obj = getattr(obj, part)
    return obj


def transplant_layer(target_handle: ModelHandle, source_handle: ModelHandle, layer_idx: int,
                      run_fn: Callable[[], float]) -> dict:
    """Tool: transplant_layer() (Sec. 4.5 point 3, Stage C). Temporarily
    substitutes layer `layer_idx`'s actual WEIGHTS (the whole nn.Module, not
    just its runtime activation -- distinct from swap_layer(), which swaps
    activations within one model) from source_handle into target_handle,
    runs run_fn(), then restores the original. Requires both models to share
    the same layer-stack shape (same architecture) -- true for a base/
    fine-tuned pair by construction, since fine-tuning doesn't change
    architecture."""
    assert target_handle.n_layers == source_handle.n_layers, \
        "transplant_layer requires both models to have the same layer count (same architecture)"
    baseline = run_fn()

    container = _get_layer_container(target_handle)
    original_module = container[layer_idx]
    source_module = source_handle.layers[layer_idx]

    with MODEL_LOCK:
        container[layer_idx] = source_module
        target_handle.layers[layer_idx] = source_module
        try:
            transplanted = run_fn()
        finally:
            container[layer_idx] = original_module
            target_handle.layers[layer_idx] = original_module

    return {"layer": layer_idx, "baseline": baseline, "transplanted": transplanted,
            "delta": transplanted - baseline}


def freeze_and_probe(acts: np.ndarray, labels: np.ndarray, test_size: float = 0.3, seed: int = 0) -> dict:
    """Tool: freeze_and_probe(). Linear probe (logistic regression) on frozen
    activations; returns concept decodability at that depth."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(acts, labels, test_size=test_size, random_state=seed,
                                           stratify=labels if len(set(labels)) > 1 else None)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    return {"probe_accuracy": float(acc), "n_train": len(Xtr), "n_test": len(Xte)}


def capture_layer_output(handle: ModelHandle, layer_idx: int, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
    """Primitive used by patch_layer(): run a forward pass and cache one
    layer's raw output tensor (not pooled) for later injection."""
    store = {}

    def hook(module, inputs, output):
        store["val"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

    with MODEL_LOCK:
        h = handle.layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.no_grad():
                handle.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            h.remove()
    return store["val"]


def patch_layer(handle: ModelHandle, layer_idx: int,
                 clean_input_ids: torch.Tensor, clean_attention_mask: torch.Tensor,
                 corrupted_input_ids: torch.Tensor, corrupted_attention_mask: torch.Tensor,
                 metric_fn: Callable[[torch.Tensor], float]) -> dict:
    """Tool: patch_layer(). Coarse activation patching at layer granularity --
    cache this layer's clean-run output, splice it into the corrupted run at
    the final token position only, and see how much of `metric_fn` gets
    restored. This is both the Layer Agent's primary causal-localization tool
    and the Skeptic's primary falsification tool (same function, different
    caller).

    Patching only the last position (not the whole sequence) matters: the
    hidden state at a decoder layer is the *entire* accumulated residual
    stream, so overwriting it at every position would make every downstream
    layer simply replay the clean run for the rest of the forward pass --
    every layer would trivially show 100% recovery regardless of whether it
    actually matters. Patching just the position the metric reads keeps this
    a real per-layer localization test."""
    clean_cache = capture_layer_output(handle, layer_idx, clean_input_ids, clean_attention_mask)

    def patch_hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        patched = hs.clone()
        patched[:, -1, :] = clean_cache[:, -1, :]
        return (patched,) + output[1:] if is_tuple else patched

    with MODEL_LOCK:
        h = handle.layers[layer_idx].register_forward_hook(patch_hook)
        try:
            with torch.no_grad():
                logits = handle.model(input_ids=corrupted_input_ids,
                                       attention_mask=corrupted_attention_mask).logits
            patched_metric = metric_fn(logits)
        finally:
            h.remove()

    with MODEL_LOCK, torch.no_grad():
        clean_metric = metric_fn(handle.model(input_ids=clean_input_ids,
                                               attention_mask=clean_attention_mask).logits)
        corrupted_metric = metric_fn(handle.model(input_ids=corrupted_input_ids,
                                                    attention_mask=corrupted_attention_mask).logits)

    denom = (clean_metric - corrupted_metric)
    recovered = (patched_metric - corrupted_metric) / denom if abs(denom) > 1e-8 else float("nan")
    return {"layer": layer_idx, "clean_metric": clean_metric, "corrupted_metric": corrupted_metric,
            "patched_metric": patched_metric, "fraction_recovered": recovered}


def _find_attn_out_proj(layer: nn.Module) -> nn.Module:
    """Locates the attention output-projection submodule generically: its
    *input* is the per-head concatenated attention output, laid out
    contiguously as [.., n_head * head_dim], which is what makes head-level
    patching architecture-agnostic across GPT-2 (`attn.c_proj`) and
    Llama-family models incl. Qwen2 (`self_attn.o_proj`)."""
    for path in ["attn.c_proj", "self_attn.o_proj", "attention.dense"]:
        obj = layer
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.Module):
            return obj
    raise ValueError(f"Could not locate attention output projection on {type(layer).__name__}")


def get_head_dim(handle: ModelHandle) -> int:
    cfg = handle.model.config
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    n_head = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    return hidden // n_head


def run_with_head_patch(handle: ModelHandle, layer_idx: int, head_idx: int,
                         clean_input_ids: torch.Tensor, clean_attention_mask: torch.Tensor,
                         corrupted_input_ids: torch.Tensor, corrupted_attention_mask: torch.Tensor,
                         metric_fn: Callable[[torch.Tensor], float]) -> float:
    """Primitive behind Tier-C `activation_patch()` / `run_acdc()` /
    `run_eap()`: patches a single attention head's contribution from the
    clean run into the corrupted run by slicing the out-projection's input
    tensor along the head_dim axis (see _find_attn_out_proj docstring)."""
    out_proj = _find_attn_out_proj(handle.layers[layer_idx])
    head_dim = get_head_dim(handle)
    lo, hi = head_idx * head_dim, (head_idx + 1) * head_dim

    clean_slice = {}

    def capture_pre_hook(module, args, kwargs):
        x = args[0] if args else kwargs["input"]
        clean_slice["val"] = x[:, :, lo:hi].detach().clone()
        return None

    def patch_pre_hook(module, args, kwargs):
        x = (args[0] if args else kwargs["input"]).clone()
        seq_len = x.shape[1]
        x[:, :, lo:hi] = clean_slice["val"][:, -seq_len:, :]
        if args:
            return (x,) + args[1:], kwargs
        kwargs["input"] = x
        return args, kwargs

    with MODEL_LOCK:
        h = out_proj.register_forward_pre_hook(capture_pre_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                handle.model(input_ids=clean_input_ids, attention_mask=clean_attention_mask)
        finally:
            h.remove()

        h = out_proj.register_forward_pre_hook(patch_pre_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                logits = handle.model(input_ids=corrupted_input_ids,
                                       attention_mask=corrupted_attention_mask).logits
        finally:
            h.remove()
    return metric_fn(logits)


def ablate_head(handle: ModelHandle, layer_idx: int, head_idx: int,
                 input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 metric_fn: Callable[[torch.Tensor], float], mode: str = "mean") -> float:
    """Primitive behind Tier-V `ablate_component()` at head granularity:
    zero/mean the out-projection's input slice belonging to one head."""
    out_proj = _find_attn_out_proj(handle.layers[layer_idx])
    head_dim = get_head_dim(handle)
    lo, hi = head_idx * head_dim, (head_idx + 1) * head_dim

    def pre_hook(module, args, kwargs):
        x = (args[0] if args else kwargs["input"]).clone()
        if mode == "zero":
            x[:, :, lo:hi] = 0.0
        else:
            x[:, :, lo:hi] = x[:, :, lo:hi].mean(dim=(0, 1), keepdim=True)
        if args:
            return (x,) + args[1:], kwargs
        kwargs["input"] = x
        return args, kwargs

    with MODEL_LOCK:
        h = out_proj.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                logits = handle.model(input_ids=input_ids, attention_mask=attention_mask).logits
            return metric_fn(logits)
        finally:
            h.remove()


def ablate_head_set(handle: ModelHandle, heads: list[tuple[int, int]],
                     input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     metric_fn: Callable[[torch.Tensor], float], mode: str = "mean") -> float:
    """`ablate_head` generalized to a SET of (layer, head) pairs ablated in a
    single forward pass -- the coalition-value primitive m(S) that S-EAP's
    exact second-order ground truth needs (Syn(i,j) = m({i,j}) - m({i}) -
    m({j}) + m({})). One pre-hook per distinct layer."""
    head_dim = get_head_dim(handle)
    by_layer: dict[int, list[int]] = {}
    for (l, hd) in heads:
        by_layer.setdefault(l, []).append(hd)

    def make_hook(layer_heads: list[int]):
        def pre_hook(module, args, kwargs):
            x = (args[0] if args else kwargs["input"]).clone()
            for hd in layer_heads:
                lo, hi = hd * head_dim, (hd + 1) * head_dim
                if mode == "zero":
                    x[:, :, lo:hi] = 0.0
                else:
                    x[:, :, lo:hi] = x[:, :, lo:hi].mean(dim=(0, 1), keepdim=True)
            if args:
                return (x,) + args[1:], kwargs
            kwargs["input"] = x
            return args, kwargs
        return pre_hook

    with MODEL_LOCK:
        handles = [_find_attn_out_proj(handle.layers[l]).register_forward_pre_hook(
                       make_hook(hds), with_kwargs=True) for l, hds in by_layer.items()]
        try:
            with torch.no_grad():
                logits = handle.model(input_ids=input_ids, attention_mask=attention_mask).logits
            return metric_fn(logits)
        finally:
            for h in handles:
                h.remove()
