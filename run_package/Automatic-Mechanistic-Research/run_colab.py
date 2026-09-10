#!/usr/bin/env python3
"""Resumable open-22 × all-203 Colab runner. No paid API or gated models.

The parent supervises one isolated model process at a time. Each successful
behavior becomes a checksummed, atomic, independently resumable checkpoint;
legacy Stage-D files can never satisfy this run's completion contract.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parent
SCHEMA = 3
from automechinterp.eval.catalog import BUILDERS, COUNTS as ANGLE_COUNTS, metadata


def canonical(value):
    import math
    def safe(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None  # Undefined ratios remain unavailable, never zero-filled.
        if isinstance(item, dict):
            return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [safe(v) for v in item]
        if hasattr(item, "tolist"):
            return safe(item.tolist())
        return item
    return json.dumps(safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def slug(value):
    import re
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).lower()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    text = canonical(value)
    with temp.open("w") as f:
        f.write(text + "\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def load_config(path):
    config = json.loads(Path(path).read_text())
    ids = [m["model_id"] for m in config["models"]]
    if len(ids) != 22 or len(set(ids)) != 22 or len(set(config["behavior_ids"])) != 203 or set(config["behavior_ids"]) != set(BUILDERS):
        raise ValueError("The Colab contract requires exactly 22 unique models and every one of the 203 registered behaviors")
    if any(m.startswith(("meta-llama/", "google/")) for m in ids):
        raise ValueError("Gated checkpoints are outside the open-22 contract")
    if any(len(m.get("revision", "")) != 40 for m in config["models"]):
        raise ValueError("Every model must have an explicit immutable revision")
    if config.get("backend") != "heuristic":
        raise ValueError("This no-API suite explicitly uses the heuristic backend")
    if config["behavior_ids"] != list(BUILDERS):
        raise ValueError("Behavior IDs must follow the authoritative angle-ordered catalog")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Provide distinct seeds")
    return config


def source_hash():
    files = list((REPO / "automechinterp").rglob("*.py")) + sorted(REPO.glob("run_*.py")) + [REPO / "requirements-colab.txt"]
    return digest({str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})


def configure(config, device):
    os.environ.update({"AMI_DEVICE": device, "AMI_LAYER_SCOPE": config.get("layer_scope", "all"), "AMI_LLM_BACKEND": "heuristic",
        "AMI_PUBLIC_SAE": "1" if config["public_sae"] else "0",
        "AMI_CAPTURE_BATCH_SIZE": str(config["capture_batch_size"]),
        "AMI_SAE_EXPANSION": str(config["sae_expansion"]), "AMI_SAE_STEPS": str(config["sae_steps"]),
        "AMI_TOOL_BUDGET": str(config["tool_budget"]), "AMI_DEEP_TECHNIQUES": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg", "HF_HUB_DISABLE_XET": "1"})


def environment(device):
    import torch
    # Stable numerical controls; no global inference_mode (EAP needs gradients).
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support is required; select A100 in Colab")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {"device": device, "python": sys.version.split()[0],
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "numpy", "scipy", "scikit-learn")},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "tf32": False}


def identity(run_id, model, behavior, seed):
    return {"schema": SCHEMA, "run_id": run_id, "model": model["model_id"],
            "revision": model["revision"], "behavior_id": behavior, "seed": seed}


def job_path(root, expected):
    return Path(root) / "jobs" / f"{slug(expected['model'])}__{expected['behavior_id']}__seed{expected['seed']}.json"


def read_checkpoint(path, expected):
    try:
        envelope = json.loads(Path(path).read_text())
        payload = envelope["payload"]
        if envelope["sha256"] != digest(payload) or payload["identity"] != expected:
            return None
        if payload["status"] != "completed" or payload["entry"]["execution_status"] != "completed":
            return None
        raw = payload["entry"]["raw_hierarchy_result"]
        if raw["execution_status"] != "completed" or any(a["errors"] for a in raw["agent_runs"]):
            return None
        if not raw["agent_runs"] or not payload.get("report_markdown"):
            return None
        return payload
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_checkpoint(root, persist, payload):
    envelope = {"payload": payload, "sha256": digest(payload)}
    local = job_path(root, payload["identity"])
    atomic_json(local, envelope)
    if persist:
        atomic_json(job_path(persist, payload["identity"]), envelope)


def restore(root, persist, expected):
    for directory in (root, persist):
        if not directory:
            continue
        result = read_checkpoint(job_path(directory, expected), expected)
        if result:
            if directory != root:
                atomic_json(job_path(root, expected), {"payload": result, "sha256": digest(result)})
            report = Path(root)/"reports"/(job_path(root, expected).stem + ".md")
            report.parent.mkdir(parents=True, exist_ok=True)
            if not report.exists():
                report.write_text(result["report_markdown"])
            return result
    return None


def coverage(config, root, run_id, persist=None):
    rows = []
    for model in config["models"]:
        for behavior in config["behavior_ids"]:
            for seed in config["seeds"]:
                expected = identity(run_id, model, behavior, seed)
                result = restore(root, persist, expected)
                state = "completed" if result else "missing_or_failed"
                if not result:
                    try:
                        failed = json.loads(job_path(root, expected).read_text())["payload"]
                        if failed.get("identity") == expected:
                            state = failed.get("status", state)
                    except (OSError, ValueError, KeyError):
                        pass
                rows.append({**expected, "status": state,
                    "verdict": result["entry"].get("verdict") or "No claim" if result else "",
                    "angle": metadata(behavior)["angle"], "angle_name": metadata(behavior)["angle_name"]})
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root/"coverage.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    completed = sum(r["status"] == "completed" for r in rows)
    out = {"run_id": run_id, "expected": len(rows), "completed": completed,
           "remaining": len(rows)-completed, "all_22_models_203_behaviors_complete": completed == len(rows),
           "scientific_scope": "engineering coverage; incomplete verification is not a Confirmed discovery"}
    atomic_json(root/"coverage.json", out)
    return out


def worker(args, config, runtime):
    import torch
    from automechinterp import behaviors
    from automechinterp.tools import adapter
    from automechinterp.eval.acceptance import run_acceptance
    from automechinterp.eval.causal_measurements import ForwardCounter, score_contrast, summarize
    from automechinterp.stage_d_eval import run_stage_d
    from automechinterp.agents.base import LOG
    from automechinterp.tools.cache import CACHE
    root, persist = Path(args.worker_root), Path(args.worker_persist) if args.worker_persist else None
    model = next(m for m in config["models"] if m["model_id"] == args.worker_model)
    requested = args.behavior_ids.split(",") if args.behavior_ids else config["behavior_ids"]
    pending = [(b, s) for b in requested for s in config["seeds"]
               if not restore(root, persist, identity(args.run_id, model, b, s))]
    progress = root/"progress.json"
    def announce(phase, **extra):
        atomic_json(progress, {"model": model["model_id"], "phase": phase, **extra})
    if not pending:
        announce("already complete"); return 0
    announce("loading pinned checkpoint")
    dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
    handle = adapter.register_model(model["model_id"], device=args.device, torch_dtype=dtype,
        revision=model["revision"], cache_dir=args.cache_dir, local_files_only=args.local_files_only,
        attn_implementation=config["attention_backend"])
    if handle.revision != model["revision"]:
        raise ValueError("Loaded checkpoint revision differs from the frozen model manifest")
    announce("real-checkpoint acceptance")
    # BF16 self-patching compares separately shaped forwards; expose the tolerance.
    acceptance = run_acceptance(handle, tolerance=.15 if dtype == torch.bfloat16 else 1e-4)
    acceptance.update(runtime=runtime, run_id=args.run_id)
    for destination in (root, persist):
        if destination:
            atomic_json(destination/"acceptance"/(slug(model["model_id"])+".json"), acceptance)
    if acceptance["status"] != "passed":
        announce("acceptance failed")
        return 3
    from automechinterp.eval.task_audit import audit_task
    announce("auditing every requested behavior on the loaded checkpoint tokenizer")
    audits=[]
    for behavior,seed in pending:
        try:
            record=audit_task(handle,BUILDERS[behavior](handle,seed=seed))
        except Exception as exc:
            record={"status":"failed","error":f"{type(exc).__name__}: {exc}"}
        audits.append({"behavior_id":behavior,"seed":seed,**record})
    for destination in (root,persist):
        if destination:
            atomic_json(destination/"task_audits"/(slug(model["model_id"])+".json"),
                        {"model":model,"run_id":args.run_id,"audits":audits})
    if any(a["status"]!="passed" for a in audits):
        announce("task audit failed; inspect task_audits before GPU interpretation")
        return 3
    failures = 0
    for number, (behavior, seed) in enumerate(pending, 1):
        announce("evaluating", behavior=behavior, seed=seed, remaining=len(pending)-number+1)
        expected = identity(args.run_id, model, behavior, seed)
        handle._causal_reference_cache = {}
        LOG.entries.clear(); CACHE._store.clear(); CACHE.hits = CACHE.misses = 0
        torch.manual_seed(seed)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            builder = BUILDERS[behavior]
            with ForwardCounter(handle.model) as counter:
                stage = run_stage_d("heuristic", model["model_id"], behaviors=[builder],
                    reports_dir=root/"reports", target_handle=handle, target_dtype=dtype,
                    seed=seed, deep_techniques=True, tool_budget=config["tool_budget"],
                    max_component_layers=config["max_component_layers"],
                    max_specialist_layers=config["max_specialist_layers"])
                entry = stage["reports"][0]
                entry.update(metadata(behavior))
                # Capability evaluation is separate from discovery and uses saved held-out pairs.
                task = builder(handle, seed=seed)
                entry["task_definition"] = {k:v for k,v in task.items() if k != "task_metric_fn"}
                from automechinterp.eval.seap_evaluation import evaluate_seap
                entry["seap_evaluation"] = evaluate_seap(handle, task, **config.get("seap", {}))
                if entry["seap_evaluation"]["status"] == "error":
                    entry["execution_status"] = "tool_errors"
                measurements = []
                for cp, xp, pos, neg in task["eval_pairs"]:
                    clean = score_contrast(handle, cp, pos, neg)
                    corrupt = score_contrast(handle, xp, pos, neg)
                    measurements.append({"clean": clean, "corrupted": corrupt})
            capability = {"examples": measurements,
                "clean_margin": summarize([r["clean"]["margin"] for r in measurements]),
                "corruption_gap": summarize([r["clean"]["margin"]-r["corrupted"]["margin"] for r in measurements]),
                "interpretation": "unique-prompt descriptive statistics; repeated templates/entities remain correlated"}
            if task["category"] != "Angle 5: Social Bias":
                capability["pairwise_accuracy"] = summarize([r["clean"]["pairwise_accuracy"] for r in measurements])
            entry["capability"] = capability
            from automechinterp.eval.report_writer import write_report
            report = write_report(task, entry["raw_hierarchy_result"], model["model_id"],
                                  seap=entry["seap_evaluation"], capability=capability)
            Path(entry["report_path"]).write_text(report)
            status = "completed" if entry["execution_status"] == "completed" else "tool_errors"
            payload = {"identity": expected, "status": status, "entry": entry,
                "capability": capability, "report_markdown": Path(entry["report_path"]).read_text(),
                "telemetry": counter.as_dict(), "wall_seconds": time.perf_counter()-start,
                "runtime": runtime, "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9 if args.device == "cuda" else None}
            failures += status != "completed"
        except Exception as exc:
            failures += 1
            payload = {"identity": expected, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc(), "wall_seconds": time.perf_counter()-start}
            traceback.print_exc()
        # A persistent write failure stops the worker; it is never reported as successful backup.
        save_checkpoint(root, persist, payload)
        announce(payload["status"], behavior=behavior, seconds=round(time.perf_counter()-start, 1))
        handle._causal_reference_cache = {}
        if args.device == "cuda":
            torch.cuda.empty_cache()
    announce("model complete" if not failures else "model has failures", failures=failures)
    return 0 if failures == 0 else 2


def sync_metadata(root, persist):
    if not persist:
        return
    persist.mkdir(parents=True, exist_ok=True)
    for name in ("run_manifest.json", "coverage.csv", "coverage.json", "progress.json"):
        source = root/name
        if source.exists():
            shutil.copy2(source, persist/name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO/"colab_config.json"))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--results-root", default=str(REPO.parent/"colab_results"))
    ap.add_argument("--persist-root")
    ap.add_argument("--cache-dir")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--models", help="Optional explicit subset for an engineering smoke run")
    ap.add_argument("--behavior-ids", help="Optional explicit subset for an engineering smoke run")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--evict-model-cache", action="store_true", help="Delete only this runner-owned per-model weight cache after each worker")
    ap.add_argument("--worker-model", help=argparse.SUPPRESS)
    ap.add_argument("--worker-root", help=argparse.SUPPRESS)
    ap.add_argument("--worker-persist", help=argparse.SUPPRESS)
    ap.add_argument("--run-id", help=argparse.SUPPRESS)
    args = ap.parse_args()
    config = load_config(args.config)
    expected = len(config["models"])*len(config["behavior_ids"])*len(config["seeds"])
    selected = args.models.split(",") if args.models else [m["model_id"] for m in config["models"]]
    selected_behaviors = args.behavior_ids.split(",") if args.behavior_ids else config["behavior_ids"]
    if not set(selected).issubset({m["model_id"] for m in config["models"]}) or not set(selected_behaviors).issubset(config["behavior_ids"]):
        ap.error("Requested model/behavior is outside the frozen suite")
    if args.plan:
        print(f"{len(config['models'])} models × {len(config['behavior_ids'])} behaviors × {len(config['seeds'])} seed(s) = {expected} jobs")
        print(f"25-angle counts: {ANGLE_COUNTS}; backend=heuristic; dtype={config['dtype']}")
        for m in config["models"]: print(f"  {m['model_id']} @ {m['revision'][:12]}")
        for b in config["behavior_ids"]: print("  "+b)
        print("Planning only: no downloads, model loads or evaluations.")
        return 0
    configure(config, args.device)
    runtime = environment(args.device)
    if args.worker_model:
        return worker(args, config, runtime)
    if args.evict_model_cache and (not args.cache_dir or args.local_files_only):
        ap.error("Cache eviction needs an explicit dedicated --cache-dir and online downloads")
    run_id = digest({"config": config, "source": source_hash(), "runtime": runtime})[:20]
    root = Path(args.results_root).resolve()/run_id
    persist = Path(args.persist_root).resolve()/run_id if args.persist_root else None
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "run_id": run_id, "config": config,
                "source_sha256": source_hash(), "runtime": runtime, "expected_jobs": expected}
    atomic_json(root/"run_manifest.json", manifest)
    before = coverage(config, root, run_id, persist)
    sync_metadata(root, persist)
    print(f"Run {run_id}: {before['completed']}/{expected} completed; results={root}", flush=True)
    if args.summarize:
        print(json.dumps(before, indent=2)); return 0
    failures = []
    for index, model_id in enumerate(selected, 1):
        model = next(m for m in config["models"] if m["model_id"] == model_id)
        if all(restore(root, persist, identity(run_id, model, b, s))
               for b in selected_behaviors for s in config["seeds"]):
            print(f"[{index}/{len(selected)}] resume: {model_id} already complete for requested behaviors", flush=True)
            continue
        print(f"[{index}/{len(selected)}] {model_id}", flush=True)
        log = root/"logs"/(slug(model_id)+".log"); log.parent.mkdir(exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--config", str(Path(args.config).resolve()),
               "--worker-model", model_id, "--worker-root", str(root), "--run-id", run_id,
               "--device", args.device, "--behavior-ids", ",".join(selected_behaviors)]
        if persist: cmd += ["--worker-persist", str(persist)]
        model_cache = None
        if args.cache_dir:
            model_cache = Path(args.cache_dir).resolve()
            if args.evict_model_cache:
                model_cache = model_cache / "mir_owned_model_caches" / slug(model_id)
                model_cache.mkdir(parents=True, exist_ok=True)
                (model_cache/".mir-owned").write_text("disposable model weights\n")
            cmd += ["--cache-dir", str(model_cache)]
        if args.local_files_only: cmd += ["--local-files-only"]
        previous = None
        with log.open("a") as output:
            process = subprocess.Popen(cmd, cwd=REPO, stdout=output, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    try:
                        progress = json.loads((root/"progress.json").read_text())
                        if progress != previous and progress.get("model") == model_id:
                            print("  "+json.dumps(progress), flush=True); previous = progress
                    except (OSError, ValueError):
                        pass
                    time.sleep(2)
            except BaseException:
                process.terminate()
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                coverage(config, root, run_id, persist); sync_metadata(root, persist)
                raise
        if args.evict_model_cache and model_cache and (model_cache/".mir-owned").is_file():
            shutil.rmtree(model_cache)  # Exactly the child allocated above; never a shared HF cache.
        if process.returncode:
            failures.append(model_id)
            print(f"  FAILED/INCOMPLETE: exit {process.returncode}; inspect {log}", flush=True)
        if persist:
            (persist/"logs").mkdir(exist_ok=True)
            shutil.copy2(log, persist/"logs"/log.name)
        current = coverage(config, root, run_id, persist); sync_metadata(root, persist)
        print(f"  Overall coverage: {current['completed']}/{expected}", flush=True)
    result = coverage(config, root, run_id, persist); sync_metadata(root, persist)
    atomic_json(Path(args.results_root)/"latest_run.json", {"run_id": run_id, "path": str(root)})
    from automechinterp.eval.review_export import export_run
    export_run(root, persist=persist)
    print(json.dumps(result, indent=2), flush=True)
    if failures: return 2
    if len(selected) == 22 and len(selected_behaviors) == 203 and not result["all_22_models_203_behaviors_complete"]: return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
