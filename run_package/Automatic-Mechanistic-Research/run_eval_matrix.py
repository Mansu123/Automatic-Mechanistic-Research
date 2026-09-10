#!/usr/bin/env python3
"""Run Stage D across the evaluation model matrix (up to 7B), one model at a time.

Stage D (`python3 main.py --stage-d`) interprets ONE target model. This
walks a whole size ladder / tier of models -- defined in
`automechinterp/eval/model_matrix.py` -- STRICTLY SEQUENTIALLY:

    for each selected model, in order:
        1. download it (transformers pulls weights on first use)
        2. run Stage D -> one report per behavior in human_review/reports/
        3. write output/eval/stage_d_<model>_*.json  (its eval-matrix result)
        4. free RAM + clear the tool cache
        5. with --purge-downloads: delete the weights from the HF cache
        6. move on to the next model

Never in parallel: only one model's weights are on disk / in RAM at a time
(modulo whatever was already cached before the run -- see --purge-downloads).

No API key needed: the agent brain stays on the deterministic `heuristic`
backend; only the target models download. Meant to be handed to a teammate
with spare compute.

    # see the menu
    python3 run_eval_matrix.py --list

    # cheap wiring check for a whole tier: download -> introspect -> delete, no Stage D
    python3 run_eval_matrix.py --tier laptop --check --purge-downloads

    # quick correctness check: 3 behaviors on the two smallest models
    python3 run_eval_matrix.py --tier laptop --max-params 500M --behaviors 3

    # a clean within-family scaling ladder (recommended), deleting each model after
    python3 run_eval_matrix.py --ladder pythia --purge-downloads

    # 2B-7B tier on a CUDA box, half precision, resumable, disk-frugal
    python3 run_eval_matrix.py --tier workstation --dtype float16 --skip-done --purge-downloads

    # explicit picks
    python3 run_eval_matrix.py --models gpt2-large,EleutherAI/pythia-1.4b

Resume a partial sweep with --skip-done (skips a model that already has a
stage_d_*.json in output/eval/). Gated models (Llama-3, Gemma) need
`huggingface-cli login` + license acceptance and are only included with
--include-gated or by naming them in --models.

--purge-downloads deletes each model from the shared Hugging Face cache
(~/.cache/huggingface/hub) once its result is written. Models that were
ALREADY cached before this run starts are left alone unless you also pass
--purge-preexisting.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from automechinterp import config
from automechinterp.behaviors import ALL_BEHAVIORS, CANONICAL_16_BEHAVIORS
from automechinterp.eval import model_matrix
from automechinterp.eval.report_writer import _slug
from automechinterp.stage_d_eval import HUMAN_REVIEW_REPORTS_DIR, run_stage_d, save_stage_d_json

OUT_DIR = Path("output/eval")


def _already_done(model_id: str) -> bool:
    return any(OUT_DIR.glob(f"stage_d_{_slug(model_id)}_*.json"))


def _dtype(name: str | None):
    if not name:
        return None
    import torch
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


# --- Hugging Face cache management --------------------------------------------
def _cached_model_repos() -> set[str]:
    """repo_ids of every model currently in the local HF cache."""
    try:
        from huggingface_hub import scan_cache_dir
        return {r.repo_id for r in scan_cache_dir().repos if r.repo_type == "model"}
    except Exception as e:  # huggingface_hub too old / no cache yet
        print(f"  (could not scan HF cache: {e})", file=sys.stderr)
        return set()


def _purge_model(model_id: str) -> dict:
    """Delete every revision of `model_id` from the HF cache. Returns a dict
    describing what happened (for the run summary)."""
    try:
        from huggingface_hub import scan_cache_dir
    except Exception as e:
        return {"purged": False, "reason": f"huggingface_hub unavailable: {e}"}
    info = scan_cache_dir()
    hashes = [rev.commit_hash
              for repo in info.repos
              if repo.repo_type == "model" and repo.repo_id == model_id
              for rev in repo.revisions]
    if not hashes:
        return {"purged": False, "reason": "not in cache"}
    strategy = info.delete_revisions(*hashes)
    freed_gb = round(strategy.expected_freed_size / 1e9, 2)
    strategy.execute()
    return {"purged": True, "freed_gb": freed_gb}


def _free_memory() -> None:
    """Drop the shared tool cache and reclaim RAM/VRAM between models. The
    content cache is keyed by tool+args, not by model, so clearing it between
    targets is also a correctness safeguard, not just a memory one."""
    try:
        from automechinterp.tools.cache import CACHE
        CACHE._store.clear()
        CACHE.hits = CACHE.misses = 0
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --- wiring check ------------------------------------------------------------
def _check_wiring(model_id: str, expected_layers: int, dtype) -> dict:
    """Download + register + introspect only -- confirms adapter.py discovers
    this model's decoder stack and Tier N can profile it. No Stage D."""
    from automechinterp.tools import adapter
    handle = adapter.register_model(model_id, device=config.DEVICE, torch_dtype=dtype)
    prof = adapter.profile_network(handle)
    found = prof["n_layers"]
    ok = found == expected_layers
    out = {
        "layer_stack_path": prof["layer_stack_path"],
        "layers_found": found, "layers_expected": expected_layers,
        "hidden_size": prof["hidden_size"], "n_heads": prof["n_heads"],
        "n_params_millions": round(prof["n_params"] / 1e6),
        "status": "ok" if ok else "warn",
    }
    del handle, prof
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the model matrix and exit")
    ap.add_argument("--models", help="comma-separated model ids from the matrix")
    ap.add_argument("--ladder", help=f"named size ladder: {sorted(model_matrix.LADDERS)}")
    ap.add_argument("--tier", choices=[model_matrix.LAPTOP, model_matrix.WORKSTATION])
    ap.add_argument("--family", help="e.g. gpt2, pythia, qwen2.5, opt")
    ap.add_argument("--max-params", help="upper bound, e.g. 1.5B or 500M")
    ap.add_argument("--include-gated", action="store_true",
                    help="also run license-gated models (needs huggingface-cli login)")
    ap.add_argument("--behaviors", type=int, default=None,
                    help="run only the first N behaviors (smoke test); default: all "
                         f"{len(ALL_BEHAVIORS)}")
    ap.add_argument("--canonical-16", action="store_true",
                    help="evaluate the 16 canonical behaviors across all 6 cognitive angles")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None,
                    help="target-model dtype (float16 on a GPU for the 3B-7B entries)")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip a model that already has a stage_d_*.json in output/eval/")
    ap.add_argument("--check", action="store_true",
                    help="wiring check only: download + introspect each model, no Stage D "
                         "(pair with --purge-downloads to validate the whole matrix cheaply)")
    ap.add_argument("--purge-downloads", action="store_true",
                    help="after each model is processed, delete its weights from the HF cache "
                         "before starting the next one")
    ap.add_argument("--purge-preexisting", action="store_true",
                    help="with --purge-downloads, also delete models that were already cached "
                         "before this run started (default: leave those alone)")
    ap.add_argument("--reports-dir", default=None,
                    help="where per-report .md files go (default: human_review/reports/)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, don't run")
    args = ap.parse_args()

    if args.list:
        print(model_matrix.format_table())
        print(f"\nladders: {json.dumps(model_matrix.LADDERS, indent=2)}")
        return 0

    try:
        selected = model_matrix.select(
            models=[m.strip() for m in args.models.split(",")] if args.models else None,
            ladder=args.ladder, tier=args.tier, family=args.family,
            max_params=args.max_params, include_gated=args.include_gated)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not selected:
        print("no models matched those filters -- try `--list`", file=sys.stderr)
        return 2

    if args.canonical_16:
        behaviors = CANONICAL_16_BEHAVIORS
    elif args.behaviors:
        behaviors = ALL_BEHAVIORS[: args.behaviors]
    else:
        behaviors = None
    n_beh = len(behaviors) if behaviors is not None else len(ALL_BEHAVIORS)
    reports_dir = Path(args.reports_dir) if args.reports_dir else HUMAN_REVIEW_REPORTS_DIR
    dtype = _dtype(args.dtype)

    # Snapshot what's already cached so --purge-downloads doesn't nuke models the
    # user had before this run (unless they asked for that with --purge-preexisting).
    preexisting = _cached_model_repos() if (args.purge_downloads and not args.purge_preexisting) else set()

    mode = "wiring check" if args.check else f"Stage D x {n_beh} behavior(s)"
    print(f"Eval matrix run (SEQUENTIAL) -- {len(selected)} model(s), {mode}, "
          f"backend={config.LLM_BACKEND}, device={config.DEVICE}, dtype={args.dtype or 'default'}, "
          f"purge={'on' if args.purge_downloads else 'off'}")
    for i, m in enumerate(selected, 1):
        done = "  [done, will skip]" if args.skip_done and _already_done(m.model_id) else ""
        keep = "  [pre-cached, kept]" if m.model_id in preexisting else ""
        print(f"  {i:2d}/{len(selected)}  {m.model_id:40s} {m.params:>6s}  {m.tier}{done}{keep}")
    if args.dry_run:
        return 0
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = OUT_DIR / f"eval_matrix_{ts}.json"
    results: list[dict] = []
    total_freed_gb = 0.0

    def _flush_summary() -> None:
        summary_path.write_text(json.dumps(
            {"started": ts, "mode": "check" if args.check else "stage_d",
             "backend": config.LLM_BACKEND, "device": config.DEVICE,
             "n_behaviors": None if args.check else n_beh,
             "purge_downloads": args.purge_downloads,
             "total_freed_gb": round(total_freed_gb, 2), "models": results}, indent=2))

    for i, m in enumerate(selected, 1):
        tag = f"[{i}/{len(selected)}] {m.model_id}"
        if args.skip_done and _already_done(m.model_id):
            print(f"== skip {tag} (already has a stage_d_*.json)")
            results.append({"model_id": m.model_id, "status": "skipped"})
            _flush_summary()
            continue

        print(f"\n{'#' * 78}\n# {tag}  ({m.params}, {m.n_layers}L, {m.attention}, {m.tier})\n{'#' * 78}")
        entry: dict = {"model_id": m.model_id, "params": m.params, "tier": m.tier}
        try:
            if args.check:
                entry.update(_check_wiring(m.model_id, m.n_layers, dtype))
                print(f"  {entry['status'].upper()}: {entry['layer_stack_path']} -> "
                      f"{entry['layers_found']} layers (expected {entry['layers_expected']}), "
                      f"d_model={entry['hidden_size']}, heads={entry['n_heads']}")
            else:
                result = run_stage_d(
                    backend_kind=config.LLM_BACKEND,
                    target_model_id=m.model_id,
                    reports_dir=reports_dir,
                    behaviors=behaviors,
                    target_dtype=dtype,
                )
                json_path = save_stage_d_json(result)
                verdicts: dict[str, int] = {}
                for r in result["reports"]:
                    verdicts[str(r["verdict"])] = verdicts.get(str(r["verdict"]), 0) + 1
                entry.update(status="ok", stage_d_json=str(json_path),
                             n_reports=len(result["reports"]), verdicts=verdicts)
                print(f"  -> {json_path}   verdicts={verdicts}")
                del result
        except Exception as e:  # one bad model must not sink the sweep
            entry.update(status="error", error=f"{type(e).__name__}: {e}")
            print(f"  !! FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()

        _free_memory()

        if args.purge_downloads:
            if m.model_id in preexisting:
                entry["purge"] = {"purged": False, "reason": "pre-existing, kept"}
            else:
                entry["purge"] = _purge_model(m.model_id)
                if entry["purge"].get("purged"):
                    total_freed_gb += entry["purge"]["freed_gb"]
                    print(f"  purged {m.model_id} from HF cache (freed ~{entry['purge']['freed_gb']} GB)")
                else:
                    print(f"  not purged: {entry['purge']['reason']}")

        results.append(entry)
        _flush_summary()

    print(f"\n{'=' * 78}\nEval matrix summary -> {summary_path}")
    if args.purge_downloads:
        print(f"Total disk freed by purge: ~{round(total_freed_gb, 2)} GB")
    for e in results:
        line = f"  {e['model_id']:40s} {e['status']}"
        if e.get("verdicts"):
            line += f"   {e['verdicts']}"
        if e.get("layers_found") is not None and "verdicts" not in e:
            line += f"   {e.get('layers_found')}L (exp {e.get('layers_expected')})"
        if e.get("purge", {}).get("purged"):
            line += f"   [-{e['purge']['freed_gb']}GB]"
        if e.get("error"):
            line += f"   {e['error']}"
        print(line)
    return 0 if all(e["status"] not in ("error", "warn") for e in results) else 1


if __name__ == "__main__":
    sys.exit(main())
