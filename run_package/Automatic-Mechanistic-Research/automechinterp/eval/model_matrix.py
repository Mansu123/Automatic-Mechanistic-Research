"""The evaluation model matrix: which target models Stage D reports cover.

Stage D (`stage_d_eval.py`) interprets exactly one target model per call
(`--target-model`). The scaling question ("how does the finding change
across model scale?") needs the *same* battery of behaviors run against a
size ladder of models and scored against the same shared rubric
(`matrix_rubric.py`), so the `generalization` metric and the cross-size
comparison have something to bite on.

This module is that ladder, as data. Every entry is a plain Hugging Face
`AutoModelForCausalLM` id whose decoder stack `tools/adapter.py` already
auto-discovers (`transformer.h` / `model.layers` / `gpt_neox.layers` /
`model.decoder.layers`) and whose attention supports
`attn_implementation="eager"` + `output_attentions=True` (Tier C needs
per-head weights). Nothing here is > 7B parameters.

`run_eval_matrix.py` (repo root) is the runner: it walks `select(...)` and
calls `run_stage_d(...)` once per model.

    python3 run_eval_matrix.py --list                 # print this table
    python3 run_eval_matrix.py --ladder pythia        # one clean size ladder
    python3 run_eval_matrix.py --tier laptop          # everything a CPU can do
    python3 run_eval_matrix.py --max-params 3B        # <= 3B, any family
    python3 run_eval_matrix.py --models gpt2-large,EleutherAI/pythia-1.4b
"""
from __future__ import annotations

from dataclasses import dataclass

# Rough CPU guidance for the "tier" field, assuming the heuristic agent
# backend (no second LLM) and float32:
#   laptop      -- <= ~1.5B params: minutes per behavior on a modern laptop CPU
#   workstation -- ~2B-7B params: wants a GPU, or a patient desktop + lots of RAM
LAPTOP = "laptop"
WORKSTATION = "workstation"


@dataclass(frozen=True)
class EvalModel:
    model_id: str          # Hugging Face id passed straight to --target-model
    family: str
    params: str            # human-readable, approximate
    params_millions: int   # for --max-params filtering
    n_layers: int
    d_model: int
    attention: str         # "MHA" (every head has its own K/V) or "GQA"
    tier: str              # LAPTOP | WORKSTATION
    gated: bool = False     # needs `huggingface-cli login` + license acceptance
    notes: str = ""


# ---------------------------------------------------------------------------
# The matrix. Grouped by family so the size ladders are obvious. Ungated,
# interp-friendly families first (GPT-2, Pythia, Qwen2.5); gated / non-standard
# ones last and clearly marked.
# ---------------------------------------------------------------------------
MATRIX: list[EvalModel] = [
    # -- GPT-2 family: MHA, ungated, the reference ladder (GPT-2 small is the
    #    only model with published IOI ground truth -- see stage_a.py).
    EvalModel("gpt2",         "gpt2", "124M",  124,  12,  768, "MHA", LAPTOP),
    EvalModel("gpt2-medium",  "gpt2", "355M",  355,  24, 1024, "MHA", LAPTOP),
    EvalModel("gpt2-large",   "gpt2", "774M",  774,  36, 1280, "MHA", LAPTOP),
    EvalModel("gpt2-xl",      "gpt2", "1.5B", 1558,  48, 1600, "MHA", LAPTOP),

    # -- Pythia (GPT-NeoX): MHA, ungated, *same data + tokenizer at every size*
    #    -- the cleanest scaling study available. Recommended default ladder.
    #    Note: on transformers >= 5.x the eager attention path is numerically
    #    unstable for GPT-NeoX, so adapter.register_model falls back to SDPA
    #    (metrics/patching/probing all fine; get_attention_pattern unavailable).
    EvalModel("EleutherAI/pythia-70m",  "pythia", "70M",    70,  6,  512, "MHA", LAPTOP),
    EvalModel("EleutherAI/pythia-160m", "pythia", "160M",  162, 12,  768, "MHA", LAPTOP),
    EvalModel("EleutherAI/pythia-410m", "pythia", "410M",  405, 24, 1024, "MHA", LAPTOP),
    EvalModel("EleutherAI/pythia-1b",   "pythia", "1.0B", 1011, 16, 2048, "MHA", LAPTOP),
    EvalModel("EleutherAI/pythia-1.4b", "pythia", "1.4B", 1414, 24, 2048, "MHA", LAPTOP),
    EvalModel("EleutherAI/pythia-2.8b", "pythia", "2.8B", 2775, 32, 2560, "MHA", WORKSTATION),
    EvalModel("EleutherAI/pythia-6.9b", "pythia", "6.9B", 6857, 32, 4096, "MHA", WORKSTATION,
              notes="approximately 27.4 GB FP32 weights or 13.7 GB BF16 weights, excluding runtime memory"),

    # -- Qwen2.5 Instruct: GQA, ungated. Already partly covered by the existing
    #    human_review reports (0.5B, 3B); fill in the rest of the ladder.
    EvalModel("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5", "0.5B",  494, 24,  896, "GQA", LAPTOP),
    EvalModel("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5", "1.5B", 1544, 28, 1536, "GQA", LAPTOP),
    EvalModel("Qwen/Qwen2.5-3B-Instruct",   "qwen2.5", "3.1B", 3086, 36, 2048, "GQA", WORKSTATION),
    EvalModel("Qwen/Qwen2.5-7B-Instruct",   "qwen2.5", "7.6B", 7615, 28, 3584, "GQA", WORKSTATION),

    # -- Other ungated single points (not full ladders, but useful spread).
    EvalModel("EleutherAI/gpt-neo-1.3B", "gpt-neo", "1.3B", 1316, 24, 2048, "MHA", LAPTOP),
    EvalModel("EleutherAI/gpt-neo-2.7B", "gpt-neo", "2.7B", 2652, 32, 2560, "MHA", WORKSTATION),
    EvalModel("facebook/opt-1.3b", "opt", "1.3B", 1316, 24, 2048, "MHA", LAPTOP),
    EvalModel("facebook/opt-2.7b", "opt", "2.7B", 2652, 32, 2560, "MHA", WORKSTATION),
    EvalModel("facebook/opt-6.7b", "opt", "6.7B", 6658, 32, 4096, "MHA", WORKSTATION),
    EvalModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", "1.1B", 1100, 22, 2048, "GQA", LAPTOP),
    EvalModel("HuggingFaceTB/SmolLM2-1.7B-Instruct", "llama", "1.7B", 1711, 24, 2048, "MHA", LAPTOP),
    EvalModel("microsoft/phi-2", "phi", "2.7B", 2779, 32, 2560, "MHA", WORKSTATION),

    # -- Gated: run `huggingface-cli login` and accept the license first, or
    #    these 404 at download time. Standard decoder stacks, still fine for the tools.
    EvalModel("meta-llama/Llama-3.2-1B", "llama-3", "1.2B", 1236, 16, 2048, "GQA", LAPTOP, gated=True),
    EvalModel("meta-llama/Llama-3.2-3B", "llama-3", "3.2B", 3213, 28, 3072, "GQA", WORKSTATION, gated=True),
    EvalModel("google/gemma-2-2b", "gemma-2", "2.6B", 2614, 26, 2304, "GQA", WORKSTATION, gated=True,
              notes="attention logit soft-capping; eager attention handles it"),
    EvalModel("mistralai/Mistral-7B-v0.1", "mistral", "7.2B", 7242, 32, 4096, "GQA", WORKSTATION,
              notes="just over 7B by the strict count; commonly grouped as '7B'"),
]

BY_ID: dict[str, EvalModel] = {m.model_id: m for m in MATRIX}

# Named size ladders -- a within-family sweep is what the cross-scale
# comparison actually wants (same architecture + training recipe, only scale
# changes).
LADDERS: dict[str, list[str]] = {
    "gpt2":    ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    "pythia":  ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
                "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b",
                "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b"],
    "qwen2.5": ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
    "opt":     ["facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"],
}

_SIZE_SUFFIX = {"m": 1, "b": 1000, "": 1}


def parse_params(text: str) -> int:
    """'3B' / '350M' / '1.4b' -> millions of parameters."""
    t = text.strip().lower()
    for suffix, mult in (("m", 1), ("b", 1000)):
        if t.endswith(suffix):
            return int(float(t[:-1]) * mult)
    return int(float(t))


def select(*, models: list[str] | None = None, ladder: str | None = None,
           tier: str | None = None, family: str | None = None,
           max_params: str | None = None, include_gated: bool = False) -> list[EvalModel]:
    """Resolve CLI filters to an ordered, de-duplicated list of EvalModel.

    Filters combine with AND. `models` and `ladder` name entries explicitly and
    (unless the entry is gated and include_gated is False) bypass tier/family
    filtering; `tier` / `family` / `max_params` scan the whole matrix.
    """
    picked: list[EvalModel] = []

    def _add(m: EvalModel, *, explicit: bool) -> None:
        if m in picked:
            return
        if m.gated and not include_gated and not explicit:
            return
        picked.append(m)

    if models:
        for mid in models:
            m = BY_ID.get(mid)
            if m is None:
                raise KeyError(f"{mid!r} is not in the eval matrix (see `--list`)")
            _add(m, explicit=True)
    if ladder:
        if ladder not in LADDERS:
            raise KeyError(f"unknown ladder {ladder!r}; known: {sorted(LADDERS)}")
        for mid in LADDERS[ladder]:
            _add(BY_ID[mid], explicit=True)

    if not models and not ladder:
        cap = parse_params(max_params) if max_params else None
        for m in MATRIX:
            if tier and m.tier != tier:
                continue
            if family and m.family != family:
                continue
            if cap is not None and m.params_millions > cap:
                continue
            _add(m, explicit=False)

    return picked


def format_table(rows: list[EvalModel] | None = None) -> str:
    rows = rows if rows is not None else MATRIX
    head = f"{'model_id':40s} {'family':9s} {'params':7s} {'L':>3s} {'d_model':>7s} {'attn':4s} {'tier':11s} gated"
    lines = [head, "-" * len(head)]
    for m in rows:
        lines.append(f"{m.model_id:40s} {m.family:9s} {m.params:7s} {m.n_layers:3d} "
                     f"{m.d_model:7d} {m.attention:4s} {m.tier:11s} {'yes' if m.gated else ''}"
                     + (f"   # {m.notes}" if m.notes else ""))
    return "\n".join(lines)
