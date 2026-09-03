# Eval-matrix runbook — commands to run

Copy-paste guide for running the cross-model / scaling sweep (Stage D across a
size ladder of target models, ≤ 7B) on a teammate's machine. Runs **one model
at a time**; with `--purge-downloads` only one model's weights are ever on disk.

- Code: [`run_eval_matrix.py`](run_eval_matrix.py) · model list: [`automechinterp/eval/model_matrix.py`](automechinterp/eval/model_matrix.py)
- Background / design notes: README → "Evaluation model matrix (cross-model / scaling sweep)"

---

## 0. One-time setup

```bash
cd "/path/to/Automatic-Mechanistic-Research-main"
python3 -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

No API key needed — the agent brain stays on the deterministic `heuristic`
backend; only the target models download from Hugging Face.

Check disk: the biggest single model (`pythia-6.9b`) is ~14 GB in float32.
With `--purge-downloads` you only need room for the largest one model at a
time, plus the text reports (tiny).

Optional — a HF token silences the rate-limit warning and speeds downloads:

```bash
export HF_TOKEN=hf_xxx        # from https://huggingface.co/settings/tokens
```

Gated models (`meta-llama/*`, `google/gemma-2-2b`) additionally need:

```bash
huggingface-cli login        # then accept each model's license on its HF page
```

---

## 1. See the menu

```bash
python3 run_eval_matrix.py --list
```

Prints every model (id, params, layers, attention type, tier, gated?) and the
named ladders. `tier`: `laptop` ≈ ≤1.5B (CPU fine), `workstation` ≈ 2–7B
(GPU or a patient desktop with lots of RAM).

---

## 2. Validate wiring first (fast, cheap)

Downloads each model, checks the pipeline can discover its decoder stack, then
deletes it. No Stage D. Do this before a long run.

```bash
python3 run_eval_matrix.py --tier laptop --check --purge-downloads
python3 run_eval_matrix.py --tier workstation --check --purge-downloads   # if you'll run these too
```

Each line should say `OK: <stack> -> N layers (expected N)`. A `WARN` means the
discovered layer count disagrees with the matrix — tell the maintainer, don't
run Stage D on that one yet.

---

## 3. Quick correctness check (a few behaviours only)

```bash
python3 run_eval_matrix.py --tier laptop --max-params 200M --behaviors 3
```

Runs 3 behaviours on the two smallest models (~minutes). Open one of the
generated files in `human_review/reports/` and sanity-check it has real numbers
(not `nan`) in the "Layer Agent findings" table.

---

## 4. Full runs — pick the ladders you want

Each command runs its models **in order**, writes a report per behaviour to
`human_review/reports/<model>__<behavior>.md`, writes
`output/eval/stage_d_<model>_*.json` per model, and deletes each model's
weights before starting the next.

```bash
# GPT-2 family (all ungated, CPU-friendly)
python3 run_eval_matrix.py --ladder gpt2 --purge-downloads

# Pythia family — cleanest scaling study (same data + tokenizer at every size)
python3 run_eval_matrix.py --ladder pythia --purge-downloads

# Qwen2.5 family (0.5B → 7B; the 3B/7B want a GPU or patience)
python3 run_eval_matrix.py --ladder qwen2.5 --purge-downloads

# OPT family
python3 run_eval_matrix.py --ladder opt --purge-downloads
```

On a CUDA box, add half precision for the 2–7B entries:

```bash
python3 run_eval_matrix.py --tier workstation --dtype float16 --purge-downloads --skip-done
```

Save the console log alongside:

```bash
python3 run_eval_matrix.py --ladder pythia --purge-downloads 2>&1 | tee "eval_matrix_$(date +%Y%m%d_%H%M%S).log"
```

---

## 5. Resume / re-run

`--skip-done` skips any model that already has an `output/eval/stage_d_<model>_*.json`.
Safe to re-run the same command after an interruption:

```bash
python3 run_eval_matrix.py --ladder pythia --purge-downloads --skip-done
```

To force a model to re-run, delete its `output/eval/stage_d_<model>_*.json` first.

---

## 6. Other selection flags

```bash
# explicit picks (ids exactly as in --list)
python3 run_eval_matrix.py --models gpt2-large,EleutherAI/pythia-1.4b --purge-downloads

# whole tier
python3 run_eval_matrix.py --tier laptop --purge-downloads

# size cap (strict "<": 1.5B excludes gpt2-xl at 1558M — use 1.6B or name it)
python3 run_eval_matrix.py --family pythia --max-params 1.5B --purge-downloads

# gated models (after huggingface-cli login)
python3 run_eval_matrix.py --models meta-llama/Llama-3.2-1B --include-gated --purge-downloads

# see the plan without running
python3 run_eval_matrix.py --ladder pythia --purge-downloads --dry-run
```

| Flag | Meaning |
|---|---|
| `--purge-downloads` | delete each model from the HF cache after its result is saved (reports total GB freed) |
| `--purge-preexisting` | also delete models that were already cached before the run (default: leave those) |
| `--skip-done` | skip models that already have a Stage D json |
| `--check` | wiring check only, no Stage D |
| `--behaviors N` | only the first N behaviours (smoke test) |
| `--dtype float16\|bfloat16\|float32` | target-model dtype (use `float16` on GPU for 3B–7B) |
| `--reports-dir DIR` | write reports somewhere other than `human_review/reports/` |
| `--dry-run` | print the plan and exit |

---

## 7. What you get / what to send back

After a sweep:

```
human_review/reports/<model>__<behavior>.md      one report per (model, behaviour)
output/eval/stage_d_<model>_*.json               per-model result
output/eval/eval_matrix_<timestamp>.json         rolling summary (verdict tallies, GB freed)
```

Regenerate the agent×layer matrices over the new reports:

```bash
python3 human_review/agent_layer_matrix.py        # -> output/eval/agent_layer_matrix.md
```

**Send back:** the whole `human_review/reports/` folder + `output/eval/*.json`.
That's everything the scoring side needs.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `[adapter] <model>: eager attention gives non-finite logits ... using 'sdpa'` | Expected for Pythia / GPT-NeoX on transformers ≥ 5.x. Not an error — everything works except `get_attention_pattern` for that model. |
| `run_sae_decompose -> ERROR: ImportError ... AliasChoices` | Pre-existing `sae-lens` / `pydantic` version clash. Harmless — that one tool is skipped, the run continues. |
| `... is not in the eval matrix` | Use an id exactly as printed by `--list`, or add it to `automechinterp/eval/model_matrix.py`. |
| Gated model 404 / auth error | `huggingface-cli login` and accept the license on the model's HF page, then pass `--include-gated`. |
| Out of RAM on a 2–7B model | Use a machine with more RAM, or `--dtype float16` on a GPU, or skip that model. |
| Out of disk | Make sure `--purge-downloads` is set; check nothing else filled the HF cache. |
| All-`nan` in a report's Layer Agent table | Run `--check` on that model and report it — the metric degenerated for that model/task. |
