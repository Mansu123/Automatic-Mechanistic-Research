#!/usr/bin/env python3
"""Load a pinned checkpoint and save its actual adapter/metric acceptance results."""
import argparse
import json
from pathlib import Path
import torch
from automechinterp.tools.adapter import register_model
from automechinterp.eval.acceptance import run_acceptance


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--revision',default=None)
    p.add_argument('--tolerance',type=float,default=None)
    p.add_argument('--cache-dir')
    p.add_argument('--device',default='cpu')
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    p.add_argument('--output',required=True)
    p.add_argument('--local-files-only',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2)
    rev = args.revision if (args.revision and args.revision.lower() not in ('none', 'main', 'latest')) else None
    handle=register_model(args.model,args.device,getattr(torch,args.dtype),revision=rev,
                          cache_dir=args.cache_dir,local_files_only=args.local_files_only)
    if args.tolerance is not None:
        tol = args.tolerance
    elif args.dtype == 'float32':
        tol = 1e-4
    elif args.dtype == 'float16':
        tol = 0.05
    else:
        tol = 0.6
    result=run_acceptance(handle,tolerance=tol)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'checks':len(result['checks']),
                      'failed':[r for r in result['checks'] if not r['passed']],
                      'telemetry':result['telemetry'],'output':str(out)},indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
