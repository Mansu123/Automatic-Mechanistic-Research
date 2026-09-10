#!/usr/bin/env python3
"""Audit all 22 pinned tokenizers × 203 tasks without downloading model weights."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoConfig, AutoTokenizer
from automechinterp.eval.catalog import BUILDERS, metadata
from automechinterp.eval.task_audit import audit_task
from run_colab import REPO, load_config, atomic_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",type=Path,default=REPO/"colab_config.json")
    ap.add_argument("--output",type=Path,default=REPO/"tokenizer_audit.json")
    ap.add_argument("--cache-dir",type=Path)
    ap.add_argument("--local-files-only",action="store_true")
    args=ap.parse_args();config=load_config(args.config);rows=[]
    for model in config["models"]:
        options={"revision":model["revision"],"cache_dir":args.cache_dir,
                 "local_files_only":args.local_files_only,"trust_remote_code":False}
        try:
            cfg=AutoConfig.from_pretrained(model["model_id"],**options)
            tokenizer=AutoTokenizer.from_pretrained(model["model_id"],**options)
            handle=SimpleNamespace(model=SimpleNamespace(config=cfg),tokenizer=tokenizer)
            for name,builder in BUILDERS.items():
                for seed in config["seeds"]:
                    try: result=audit_task(handle,builder(handle,seed=seed))
                    except Exception as exc: result={"status":"failed","error":f"{type(exc).__name__}: {exc}"}
                    rows.append({**model,**metadata(name),"seed":seed,**result})
        except Exception as exc:
            rows.append({**model,"status":"load_failed","error":f"{type(exc).__name__}: {exc}"})
        failures=sum(r["status"]!="passed" for r in rows)
        atomic_json(args.output,{"expected":22*203*len(config["seeds"]),"tested":len(rows),
                    "failures":failures,"rows":rows,"scope":"Real pinned tokenizers/configs; no pretrained model forward passes"})
        print(f"{model['model_id']}: {len(rows)} cumulative rows, {failures} failures",flush=True)
    return 0 if len(rows)==22*203*len(config["seeds"]) and all(r["status"]=="passed" for r in rows) else 2


if __name__=="__main__":raise SystemExit(main())
