#!/usr/bin/env python3
"""Export review material, render one review item, and compare human/judge scores."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from automechinterp.eval.review_export import (export_run,item_evidence,write_csv,_plot_matrix,
    summarize_source,load_review_items,load_human_ratings,load_judge_scores)
from automechinterp.eval.matrix_rubric import IDS,VERSION,validate_scores,validate_judgment,health,rubric_text
from automechinterp.eval.agreement import compare
from run_colab import digest


def analyze(root,files):
    review=root/"review";items=load_review_items(root)
    ratings=load_human_ratings(root,files,items)
    judge,versions=load_judge_scores(root,items)
    sample=json.loads((review/"judge_validation_sample.json").read_text());valid=set(sample["validation_items"])
    reports={i:m for i,m in items.items() if m["item_type"]=="report" and i in valid}
    consensus=ratings.get("consensus",{})
    result=compare(consensus,judge,reports)
    # A consensus column alone does not prove an independent human study occurred.
    independent={name:scores for name,scores in ratings.items() if name!="consensus"}
    human_counts={i:sum(i in scores for scores in independent.values()) for i in reports}
    if not sample["frozen"] or any(human_counts[i]<2 for i in reports) or len(reports)!=550:
        result["status"]="unvalidated";result["human_study_gate"]="Need frozen 550-report sample and two independent human scores plus explicit consensus per report"
    result["judge_versions"]=list(versions)
    result["human_raters"]=list(independent)
    result["human_human_agreement"]={}
    names=sorted(independent)
    for i,a in enumerate(names):
        for b in names[i+1:]:result["human_human_agreement"][a+" vs "+b]=compare(independent[a],independent[b],reports,bootstrap=200)
    (review/"judge_validation.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    element_scores=[]
    for source, scores in [("judge",judge)]+[("human:"+name,values) for name,values in ratings.items()]:
        for i,values in scores.items():
            item=items[i];h=health(values,item["execution_status"]=="completed")
            trusted=source!="judge" or (result["status"]=="validated" and item["item_type"]=="report")
            element_scores.append({"item_id":i,"source":source,"item_type":item["item_type"],"model":item["model"],
                                   "angle":item["angle"],"behavior_id":item["behavior_id"],**values,
                                   "element":item["element"],"execution_status":item["execution_status"],
                                   "mean":h["mean_1to10"],"health":h["status"] if trusted else "provisional_"+h["status"]})
    write_csv(review/"all_element_scores_and_health.csv",element_scores)
    pairs=[]
    for i in sorted(set(consensus)&set(judge)):
        row={"item_id":i,"item_type":items[i]["item_type"],"model":items[i]["model"],"angle":items[i]["angle"]}
        for k in IDS:row.update({"human_"+k:consensus[i][k],"judge_"+k:judge[i][k]})
        pairs.append(row)
    write_csv(review/"human_vs_judge_matrix.csv",pairs)
    models=[m["model_id"] for m in json.loads((root/"run_manifest.json").read_text())["config"]["models"]]
    # Each arm's per-element and per-report graphs + health verdict. A report-level
    # judge calibration cannot establish trust for layer/agent/tool/S-EAP items,
    # so the judge arm stays "provisional" until run_review validates it here.
    judge_validated=result["status"]=="validated"
    summarize_source(root,"human",consensus,items,models,judge_validated=True)
    summarize_source(root,"judge",judge,items,models,judge_validated=judge_validated)
    if pairs:
        human=np.array([[r["human_"+k] for k in IDS] for r in pairs]);llm=np.array([[r["judge_"+k] for k in IDS] for r in pairs])
        _plot_matrix(np.stack([human.mean(0),llm.mean(0)]),["Human consensus","GLM judge"],IDS,"Same-item rubric comparison",review/"graphs/06_human_judge_metrics.png",1,10)
        confusion=np.zeros((10,10))
        np.add.at(confusion,(human.ravel()-1,llm.ravel()-1),1)
        _plot_matrix(confusion,[str(x) for x in range(1,11)],[str(x) for x in range(1,11)],"Human (rows) versus judge (columns) score counts",review/"graphs/07_agreement_confusion.png")
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--run-dir",type=Path,required=True)
    ap.add_argument("--export",action="store_true");ap.add_argument("--human-scores",type=Path,nargs="*",default=[])
    ap.add_argument("--item-id");args=ap.parse_args()
    if args.export:print(json.dumps(export_run(args.run_dir),indent=2))
    if args.item_id:
        items=map(json.loads,(args.run_dir/"review/review_items.jsonl").read_text().splitlines())
        item=next(i for i in items if i["item_id"]==args.item_id)
        path=args.run_dir/"review/selected_item.md"
        path.write_text(rubric_text(item["angle"])+"\n\n```json\n"+json.dumps(item_evidence(args.run_dir,item),indent=2)+"\n```\n")
        print(path)
    if not args.item_id:print(json.dumps(analyze(args.run_dir,args.human_scores),indent=2))


if __name__=="__main__":main()
