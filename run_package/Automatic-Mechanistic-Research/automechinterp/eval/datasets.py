"""Dataset provenance and split validation. Upstream examples remain unvalidated."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from unittest.mock import patch

from .. import behaviors


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def export_upstream_catalog(destination, attempts=1000):
    catalog=[]
    def capture(handle,behavior,category,pair_generator,n_eval=4,seed=0):
        return behavior,category,pair_generator
    with patch.object(behaviors,'_make_task',capture):
        for factory in behaviors.ALL_BEHAVIORS:
            name,category,generator=factory(None)
            seen={}
            for seed in range(attempts):
                pair=tuple(generator(random.Random(seed)))
                seen.setdefault(pair,seed)
            rows=[{'item_id':canonical_hash(pair)[:20], 'clean_prompt':pair[0],
                   'corrupted_prompt':pair[1], 'positive_answer':pair[2],
                   'negative_answer':pair[3], 'first_generator_seed':seed}
                  for pair,seed in sorted(seen.items())]
            catalog.append({'behavior_id':factory.__name__,'behavior':name,'category':category,
                            'generator_attempts':attempts,'observed_unique_pairs':len(rows),
                            'semantic_review':'pending','split_status':'not_independent','items':rows})
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(catalog,indent=2,ensure_ascii=False)+'\n')
    return catalog


def validate_dataset(dataset, require_formal=True):
    required=('discovery','validation','test','stress')
    issues=[]
    if require_formal and dataset.get('review_status')!='validated':
        issues.append('semantic review not validated')
    if require_formal and not dataset.get('review_evidence'):
        issues.append('semantic review evidence missing')
    global_ids=set()
    global_prompts=set()
    clusters={}
    for split in required:
        rows=dataset.get('splits',{}).get(split,[])
        if not rows:
            issues.append(f'{split}: empty')
        for row in rows:
            key=canonical_hash([row.get(k) for k in ('clean_prompt','corrupted_prompt','positive_answer','negative_answer')])
            if key in global_ids:
                issues.append(f'{split}: duplicate pair')
            global_ids.add(key)
            pair_prompts={row.get('clean_prompt'),row.get('corrupted_prompt')}
            if pair_prompts & global_prompts:
                issues.append(f'{split}: prompt leakage/duplicate')
            global_prompts.update(pair_prompts)
            if not all(isinstance(row.get(k),str) and row[k].strip() for k in ('clean_prompt','corrupted_prompt','positive_answer','negative_answer','cluster_id')):
                issues.append(f'{split}: missing fields')
            if row.get('positive_answer')==row.get('negative_answer'):
                issues.append(f'{split}: identical candidates')
            cid=row.get('cluster_id')
            if cid in clusters and clusters[cid]!=split:
                issues.append(f'{split}: cluster crosses split boundary')
            clusters[cid]=split
    return {'valid':not issues,'issues':sorted(set(issues)), 'dataset_hash':canonical_hash(dataset),
            'split_sizes':{s:len(dataset.get('splits',{}).get(s,[])) for s in required}}


def build_ioi_calibration_dataset():
    """Executable label contract for a separate engineering pilot, not H1 gold."""
    banks={
        'discovery':['John','Mary','Alice','Bob','Sarah','Tom','Emma','James','Laura','David'],
        'validation':['Anna','Mike','Julia','Peter','Clara','Paul','Helen','Daniel','Sophie','George'],
        'test':['Rachel','Robert','Lucy','Charles','Diana','Edward','Grace','Henry','Irene','Jack'],
        'stress':['Karen','Louis','Martha','Nathan','Olivia','Patrick','Rose','Simon','Teresa','Victor']}
    templates={
        'discovery':['When {A} and {B} went to the store, {A} gave a drink to',
                     'After {A} and {B} arrived at the park, {A} handed a ball to'],
        'validation':['While {A} and {B} were in the library, {A} passed a book to',
                      'Once {A} and {B} reached the office, {A} gave a folder to'],
        'test':['When {A} and {B} visited the garden, {A} handed a flower to',
                'After {A} and {B} entered the kitchen, {A} passed a plate to',
                'While {A} and {B} were at the station, {A} gave a ticket to',
                'Once {A} and {B} reached the classroom, {A} handed a pencil to'],
        'stress':['After {A} and {B} finished a long walk through the forest, {A} gave a bottle of water to',
                  'When {A} and {B} returned from a busy afternoon at the market, {A} passed a shopping bag to']}
    counts={'discovery':100,'validation':50,'test':200,'stress':100}
    splits={}
    for split,names in banks.items():
        rows=[]
        for ti,template in enumerate(templates[split]):
            # Unordered name pairs avoid clean/corrupted prompt collisions across rows.
            for ai,a in enumerate(names):
                for b in names[ai+1:]:
                    for role in (0,1):
                        a_,b_=(a,b) if role==0 else (b,a)
                        rows.append({'clean_prompt':template.format(A=a_,B=b_),
                                     'corrupted_prompt':template.format(A=b_,B=a_),
                                     'positive_answer':b_,'negative_answer':a_,
                                     'cluster_id':f'{split}:template{ti}:{a}:{b}'})
        # One role direction per unordered pair/template prevents pair reversal leakage.
        rows=rows[::2]
        # Add a distinct second sentence family if finite templates are insufficient.
        expanded=[]
        for row in rows:
            expanded.append(row)
            expanded.append({**row,'clean_prompt':'Yesterday, '+row['clean_prompt'][0].lower()+row['clean_prompt'][1:],
                             'corrupted_prompt':'Yesterday, '+row['corrupted_prompt'][0].lower()+row['corrupted_prompt'][1:]})
        random.Random(20260905).shuffle(expanded)
        splits[split]=expanded[:counts[split]]
    dataset={'dataset_id':'ioi_engineering_calibration_v1','behavior_id':'ioi_behavior',
             'purpose':'engineering_calibration_not_formal_published_graph_evaluation',
             'review_status':'validated','review_evidence':'Executable coreference role-swap contract; distinct template and name banks by split; clustering retains paraphrase dependence.',
             'splits':splits}
    validation=validate_dataset(dataset)
    if not validation['valid']:
        raise ValueError(validation)
    return dataset
