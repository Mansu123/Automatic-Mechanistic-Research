#!/usr/bin/env python3
"""Full BF16 fixture regression: 203 behaviors on nine small architecture configs.

No downloads or API calls. These are randomly initialized engineering fixtures,
not evaluation results for the 22 pretrained checkpoints.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir",type=Path,default=Path("architecture_validation"))
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parent/"tests"))
    os.environ.update(AMI_SAE_STEPS="2",AMI_SAE_EXPANSION="1",AMI_PUBLIC_SAE="0",AMI_LAYER_SCOPE="all",MPLBACKEND="Agg")
    import torch
    torch.set_num_threads(2)
    from test_colab_bundle import fixture
    from test_evaluation_science import small_configs
    from automechinterp.eval.catalog import BUILDERS,metadata
    from automechinterp.hierarchy import run_hierarchy
    from automechinterp.eval.seap_evaluation import evaluate_seap
    from automechinterp.agents.base import LOG
    from automechinterp.tools.cache import CACHE
    from run_colab import atomic_json
    rows=[]
    with (args.output_dir/"console.log").open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        for cfg in small_configs():
            handle=fixture(cfg,dtype=torch.bfloat16)
            for name,builder in BUILDERS.items():
                handle._causal_reference_cache={};LOG.entries.clear();CACHE._store.clear()
                try:
                    task=builder(handle)
                    result=run_hierarchy(handle,task,backend_kind="heuristic",deep_techniques=True,
                                         tool_budget=640,max_component_layers=2,max_specialist_layers=2)
                    seap=evaluate_seap(handle,task,candidate_heads=3,evaluation_pairs=1)
                    row={"status":result["execution_status"],"layers":len(result["layer_states"]),
                         "errors":[a for a in result["agent_runs"] if a["errors"]],"seap_status":seap["status"],"seap_error":seap.get("error")}
                except Exception as exc:row={"status":"exception","error":f"{type(exc).__name__}: {exc}"}
                rows.append({"architecture":cfg.model_type,**metadata(name),**row})
                atomic_json(args.output_dir/"results.json",rows)
    failures=[r for r in rows if r["status"]!="completed" or r.get("seap_status")!="completed"]
    summary={"tested":len(rows),"expected":1827,"failures":failures,"scope":"BF16 random-weight engineering fixtures"}
    atomic_json(args.output_dir/"summary.json",summary);print(json.dumps(summary,indent=2))
    return 0 if len(rows)==1827 and not failures else 2


if __name__=="__main__":raise SystemExit(main())
