#!/usr/bin/env python3
"""Bounded real-model engineering pilot. It never counts as a formal matrix run."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from automechinterp.eval.datasets import validate_dataset, canonical_hash
from automechinterp.eval.causal_measurements import (ActivationBank, score_contrast,
                                                   normalized_effect, summarize, proportion, ForwardCounter)
from automechinterp.tools import adapter


def source_hash():
    root=Path(__file__).resolve().parent
    h=hashlib.sha256()
    for p in sorted((root/'automechinterp').rglob('*.py')):
        h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--revision',default=None)
    ap.add_argument('--dataset',required=True);ap.add_argument('--output-dir',required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    ap.add_argument('--cache-dir');ap.add_argument('--examples',type=int,default=8)
    ap.add_argument('--local-files-only',action='store_true')
    args=p=ap.parse_args()
    if args.examples<2:raise ValueError('At least two independent calibration examples required')
    dataset=json.loads(Path(args.dataset).read_text())
    valid=validate_dataset(dataset)
    if not valid['valid']:raise ValueError(valid)
    selected=[];seen=set()
    for row in dataset['splits']['validation']:
        if row['cluster_id'] not in seen:
            selected.append(row);seen.add(row['cluster_id'])
        if len(selected)>=args.examples:break
    rev = args.revision if (args.revision and args.revision.lower() not in ('none', 'main', 'latest')) else None
    config={'scope':'engineering_calibration_not_formal_evaluation','model':args.model,'revision':rev or 'main',
            'dataset_hash':valid['dataset_hash'],'source_hash':source_hash(),
            'split':'validation','device':args.device,'dtype':args.dtype,
            'selected_clusters':[r['cluster_id'] for r in selected],
            'intervention_scope':'last prompt position; residual/block-update/MLP/head readout sites',
            'metric':'full-continuation summed log-probability contrast','seed':11}
    run_id=canonical_hash(config)
    out=Path(args.output_dir)/run_id[:16];out.mkdir(parents=True,exist_ok=True)
    result_path=out/'result.json'
    if result_path.exists():
        previous=json.loads(result_path.read_text())
        if previous['run_id']==run_id and previous['status']=='complete':
            print(f'Compatible completed engineering pilot: {result_path}');return
        raise RuntimeError('Incompatible output collision')
    torch.set_num_threads(2);torch.manual_seed(11)
    handle=adapter.register_model(args.model,args.device,getattr(torch,args.dtype),revision=rev,
                                  cache_dir=args.cache_dir,local_files_only=args.local_files_only)
    records=[]
    with ForwardCounter(handle.model) as counter:
        for i,row in enumerate(selected):
            started=time.perf_counter()
            cp,xp,a,b=[row[k] for k in ('clean_prompt','corrupted_prompt','positive_answer','negative_answer')]
            clean=score_contrast(handle,cp,a,b);corrupt=score_contrast(handle,xp,a,b)
            bank=ActivationBank.capture(handle,cp)
            destination=len(handle.tokenizer.encode(xp,add_special_tokens=True))-1
            effects=[]
            for layer in range(handle.n_layers):
                for kind in ('residual','block_update','mlp'):
                    m=score_contrast(handle,xp,a,b,lambda:bank.patch(kind,layer,destination))
                    effects.append({'kind':kind,'layer':layer,'head':None,'patched_margin':m['margin']})
                for head in range(adapter.get_num_heads(handle)):
                    m=score_contrast(handle,xp,a,b,lambda:bank.patch('head',layer,destination,head=head))
                    effects.append({'kind':'head','layer':layer,'head':head,'patched_margin':m['margin']})
            record={'item':row,'clean':clean,'corrupt':corrupt,'effects':effects,'seconds':time.perf_counter()-started}
            records.append(record)
            # Individual prompt records survive interruption; never marked complete on partial output.
            (out/f'prompt_{i:03d}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            print(json.dumps({'completed_calibration_prompts':i+1,'total':len(selected),'seconds':record['seconds']}),flush=True)
    den=[r['clean']['margin']-r['corrupt']['margin'] for r in records]
    aggregate=[]
    for effect_index,effect in enumerate(records[0]['effects']):
        num=[r['effects'][effect_index]['patched_margin']-r['corrupt']['margin'] for r in records]
        aggregate.append({k:effect[k] for k in ('kind','layer','head')} | {'recovery':normalized_effect(num,den)})
    result={'status':'complete','run_id':run_id,'configuration':config,'model':adapter.profile_network(handle),
            'records':records,'aggregate_effects':aggregate,'telemetry':counter.as_dict(),
            'clean_pairwise_accuracy':proportion(sum(r['clean']['pairwise_accuracy'] for r in records),len(records)),
            'clean_margin':summarize([r['clean']['margin'] for r in records]),'clean_corrupt_gap':summarize(den),
            'scientific_status':'pilot_only_not_eligible_for_main_results',
            'formal_behaviors_completed':0,'autonomous_discovery_executed':False,
            'notes':['Sealed test and stress splits untouched.','No ground-truth edge recall or Confirmed mechanism claim is made.',
                     'Every head is measured at the last prompt position; these are not whole-circuit interventions.']}
    tmp=result_path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(result_path)
    print(json.dumps({'result':str(result_path),'telemetry':counter.as_dict(),'accuracy':result['clean_pairwise_accuracy']}),flush=True)


if __name__=='__main__':main()
