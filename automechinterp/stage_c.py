"""Stage C -- Cross-Model Layer Diff (Sec. 4.5), run on GPT-2 vs a locally
fine-tuned biomedical stand-in (finetune_stand_in.py) instead of the
proposal's Mistral-7B vs BioMistral-7B pair -- same methodology (CKA
alignment to find change loci, then a causal layer-transplant confirmation),
scaled to fit this machine.
"""
from __future__ import annotations

import torch

from . import config
from .finetune_stand_in import build_finetuned_stand_in
from .tools import adapter
from .agents.base import LOG

# Shared probe set (general + biomedical), per Sec. 4.5 point 1
_GENERAL_PROBES = [
    "The weather today is bright and sunny.",
    "She walked to the store to buy some bread.",
    "The football match ended in a draw.",
    "He enjoys reading novels on the weekend.",
]
_BIOMEDICAL_PROBES = [
    "The patient was diagnosed with acute bronchitis.",
    "The physician prescribed an antibiotic for the infection.",
    "Blood tests revealed elevated liver enzymes.",
    "The surgeon performed a laparoscopic procedure.",
]
SHARED_PROBE_SET = _GENERAL_PROBES + _BIOMEDICAL_PROBES

# "Biomedical capability" contrastive prompts: domain-appropriate completion
# vs a generic, clearly-wrong-in-context one -- the metric the transplant
# step checks for causal transfer (Sec. 4.5 point 3).
_CAPABILITY_PROMPTS = [
    ("The patient was diagnosed with", "pneumonia", "happiness"),
    ("The doctor ordered a", "biopsy", "vacation"),
    ("Symptoms included fever and", "cough", "laughter"),
    ("The treatment plan included daily", "medication", "shopping"),
]


def _capability_metric_fn(handle: adapter.ModelHandle):
    def metric_fn() -> float:
        total = 0.0
        for prompt, domain_word, generic_word in _CAPABILITY_PROMPTS:
            batch = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
            domain_id = handle.tokenizer.encode(" " + domain_word)[0]
            generic_id = handle.tokenizer.encode(" " + generic_word)[0]
            with torch.no_grad():
                logits = handle.model(**batch).logits[0, -1]
            total += (logits[domain_id] - logits[generic_id]).item()
        return total / len(_CAPABILITY_PROMPTS)
    return metric_fn


def run_stage_c(n_change_loci: int = 3) -> dict:
    print("=" * 78)
    print("STAGE C -- Cross-Model Layer Diff (GPT-2 base vs biomedical fine-tuned stand-in)")
    print("=" * 78)

    base_handle = adapter.register_model(config.TARGET_MODEL_ID, device=config.DEVICE)
    finetuned_path = build_finetuned_stand_in(config.TARGET_MODEL_ID, device=config.DEVICE)
    ft_handle = adapter.register_model(finetuned_path, device=config.DEVICE)
    LOG.emit("System", f"registered base ({config.TARGET_MODEL_ID}) and fine-tuned stand-in "
                        f"({finetuned_path}): {base_handle.n_layers} layers each")

    # 1. Alignment: per-layer CKA between base and fine-tuned on the shared probe set
    layer_ckas = []
    for layer_idx in range(base_handle.n_layers):
        base_acts = adapter.capture_activations(base_handle, [layer_idx], SHARED_PROBE_SET)[layer_idx]
        ft_acts = adapter.capture_activations(ft_handle, [layer_idx], SHARED_PROBE_SET)[layer_idx]
        cka = adapter.layer_similarity(base_acts, ft_acts)
        layer_ckas.append(cka)
        LOG.emit("System", f"L{layer_idx}: CKA(base, fine-tuned) = {cka:.4f}")

    # candidate change loci = biggest representational shift (lowest CKA)
    ranked = sorted(range(base_handle.n_layers), key=lambda l: layer_ckas[l])
    change_loci = ranked[:n_change_loci]
    LOG.emit("System", f"candidate change loci (lowest CKA): {change_loci}")

    # 2. Characterization: does the base model's own capability-metric predict
    # domain-appropriate words at these layers, or only the fine-tuned model?
    base_metric_fn = _capability_metric_fn(base_handle)
    ft_metric_fn = _capability_metric_fn(ft_handle)
    with torch.no_grad():
        base_capability = base_metric_fn()
        ft_capability = ft_metric_fn()
    LOG.emit("System", f"biomedical-capability metric: base={base_capability:.3f}, "
                        f"fine-tuned={ft_capability:.3f} (gap={ft_capability - base_capability:.3f})")

    # 3. Causal confirmation: transplant each candidate layer's WEIGHTS from
    # fine-tuned into base, measure what fraction of the capability gap
    # transfers.
    transplant_results = []
    capability_gap = ft_capability - base_capability
    for layer_idx in change_loci:
        r = adapter.transplant_layer(base_handle, ft_handle, layer_idx, base_metric_fn)
        frac_transferred = r["delta"] / capability_gap if abs(capability_gap) > 1e-6 else float("nan")
        transplant_results.append({"layer": layer_idx, "cka": layer_ckas[layer_idx],
                                    "baseline": r["baseline"], "transplanted": r["transplanted"],
                                    "delta": r["delta"], "fraction_of_gap_transferred": frac_transferred})
        LOG.emit("System", f"transplant L{layer_idx}: delta={r['delta']:+.3f} "
                            f"({frac_transferred:.0%} of capability gap)")

    # Also transplant every OTHER layer as a control -- change loci should
    # transfer more of the gap than a random non-change layer would.
    control_layer = max(range(base_handle.n_layers), key=lambda l: layer_ckas[l])  # least-changed layer
    r_control = adapter.transplant_layer(base_handle, ft_handle, control_layer, base_metric_fn)
    control_frac = r_control["delta"] / capability_gap if abs(capability_gap) > 1e-6 else float("nan")
    LOG.emit("System", f"control transplant L{control_layer} (least-changed layer, CKA="
                        f"{layer_ckas[control_layer]:.3f}): {control_frac:.0%} of gap transferred")

    print("\n" + "-" * 78)
    print("STAGE C RESULTS")
    print("-" * 78)
    print(f"Capability gap (fine-tuned - base): {capability_gap:+.3f}")
    print(f"Change loci (lowest CKA, top {n_change_loci}): {change_loci}")
    for tr in transplant_results:
        print(f"  L{tr['layer']}: CKA={tr['cka']:.3f}, transplant delta={tr['delta']:+.3f} "
              f"({tr['fraction_of_gap_transferred']:.0%} of gap)")
    print(f"Control (least-changed layer L{control_layer}): {control_frac:.0%} of gap transferred")

    return {
        "base_model_id": config.TARGET_MODEL_ID,
        "finetuned_model_path": finetuned_path,
        "n_layers": base_handle.n_layers,
        "layer_ckas": layer_ckas,
        "change_loci": change_loci,
        "base_capability": base_capability,
        "finetuned_capability": ft_capability,
        "capability_gap": capability_gap,
        "transplant_results": transplant_results,
        "control_layer": control_layer,
        "control_fraction_transferred": control_frac,
    }
