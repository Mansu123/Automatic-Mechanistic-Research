#!/usr/bin/env python3
"""Fail-closed handoff check. Does not launch models or call paid APIs."""
from pathlib import Path
import argparse
import csv
import json
import os

import torch
from huggingface_hub import get_token

from automechinterp.eval.datasets import validate_dataset


def inspect(config_path):
    config_path=Path(config_path).resolve();base=config_path.parent
    cfg=json.loads(config_path.read_text())
    required=(base/cfg['required_models']).resolve().read_text().splitlines()
    inventory=base.parent/'evaluation_plan'/'behavior_inventory.csv'
    behaviors=list(csv.DictReader(inventory.open()))
    blockers=[]
    if len(required)!=25 or len(set(required))!=25: blockers.append('required model roster must contain exactly 25 unique IDs')
    if cfg['target_device']=='cuda' and not torch.cuda.is_available():blockers.append('configured CUDA execution environment unavailable on this host')
    credentials={}
    for role in ('discovery_api','independent_judge_api'):
        spec=cfg[role];provider=spec.get('provider')
        env={'openai':'OPENAI_API_KEY','anthropic':'ANTHROPIC_API_KEY'}.get(provider)
        credentials[role]=bool(env and os.environ.get(env))
        if not spec.get('model') or not env:blockers.append(f'{role}: explicit supported provider/model not configured')
        elif not credentials[role]:blockers.append(f'{role}: environment credential unavailable')
    if not get_token():blockers.append('authenticated gated-model access not configured; server must verify Llama/Gemma access')
    datasets=[]
    for behavior in behaviors:
        path=base/cfg['formal_dataset_directory']/(behavior['factory']+'.json')
        if not path.exists():
            datasets.append({'behavior_id':behavior['factory'],'status':'missing_formal_dataset'})
        else:
            validation=validate_dataset(json.loads(path.read_text()))
            datasets.append({'behavior_id':behavior['factory'],'status':'valid' if validation['valid'] else 'invalid','validation':validation})
    missing=sum(x['status']!='valid' for x in datasets)
    if missing:blockers.append(f'{missing}/203 formal datasets are missing or not validated')
    repair_path=base/'repair_status.json'
    repairs=json.loads(repair_path.read_text()) if repair_path.exists() else {'formal_launch_ready':False}
    if not repairs.get('formal_launch_ready'):blockers.append('remaining code/ground-truth/method readiness gates are not closed')
    return {'ready':not blockers,'blockers':blockers,'required_models':required,'formal_dataset_status':datasets,
            'credentials_available':credentials,'cuda_available':torch.cuda.is_available(),
            'completed_formal_model_reports':0,'action':'continue independent repairs; do not launch unvalidated sweep'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    result=inspect(args.config);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'ready':result['ready'],'blockers':result['blockers']},indent=2))
    raise SystemExit(0 if result['ready'] else 2)
