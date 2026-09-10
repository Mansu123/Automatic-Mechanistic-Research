"""Export full evidence, human forms and graphs from checksummed run results."""
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .catalog import metadata, ANGLE_NAMES
from .matrix_rubric import IDS, VERSION, health, validate_scores, validate_judgment
from .report_complete import coverage_records, evidence_bundle, safe, _slug


def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=iter(rows)
    if fields is None:
        first=next(rows,None)
        if first is None: path.write_text("");return
        fields=list(first)
    else: first=None
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        if first is not None: w.writerow(first)
        w.writerows(rows)


def read_payload(path):
    envelope=json.loads(Path(path).read_text())
    from run_colab import digest
    if envelope.get("sha256")!=digest(envelope.get("payload")):
        raise ValueError(f"Corrupt checkpoint: {Path(path).name}")
    return envelope["payload"]


def item_evidence(root,item):
    payload=read_payload(Path(root)/item["job_file"])
    entry=payload.get("entry",{})
    if not entry:
        return {"E000":{"identity":payload["identity"],"status":payload["status"],"error":payload.get("error")}}
    evidence=evidence_bundle(entry["task_definition"],entry["raw_hierarchy_result"],entry["model"],
                             entry.get("seap_evaluation"),payload.get("capability"))
    if item["item_type"]=="report": return evidence
    selected=["E000","E001","E003","E005","E006"]+item["evidence_selectors"]
    result={k:v for k,v in evidence.items() if k in selected}
    rows=result["E006"]["agent_layer_tool_coverage"]
    kind=item["item_type"];element=item["element"]
    if kind=="layer":rows=[r for r in rows if r["layer"]==int(element[1:])]
    elif kind=="agent":rows=[r for r in rows if r["agent"]==element]
    elif kind=="tool":
        agent,tool=element.split("/",1)
        rows=[r for r in rows if r["agent"]==agent and r["tool"]==tool]
    else:rows=[]
    result["E006"]={"agent_layer_tool_coverage":rows,"scope":"Coverage for this review element; complete inventory is in the whole report"}
    return result


def _plot_matrix(data,labels,columns,title,path,vmin=None,vmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr=np.asarray(data,dtype=float)
    fig,ax=plt.subplots(figsize=(max(11,len(columns)*.38),max(4,len(labels)*.28)))
    cmap=plt.colormaps["viridis"].copy();cmap.set_bad("#d9d9d9")
    im=ax.imshow(np.ma.masked_invalid(arr),aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_yticks(range(len(labels)),labels,fontsize=7)
    ax.set_xticks(range(len(columns)),columns,rotation=90,fontsize=7)
    ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7);fig.tight_layout()
    fig.savefig(path,dpi=145);plt.close(fig)


def load_review_items(root):
    """{item_id: item record} from review/review_items.jsonl."""
    path = Path(root)/"review"/"review_items.jsonl"
    return {i["item_id"]: i for i in map(json.loads, path.read_text().splitlines())}


def load_human_ratings(root, files, items):
    """{reviewer_id: {item_id: validated 8-metric scores}} from completed human
    CSVs. Raises (after writing review/invalid_human_scores.json) on any stale,
    malformed or duplicated row so a bad sheet never scores silently."""
    root = Path(root); review = root/"review"
    ratings = defaultdict(dict); invalid = []
    for file in files:
        with Path(file).open(newline="") as f:
            for row in csv.DictReader(f):
                if not any(row.get(k,"").strip() for k in IDS): continue
                try:
                    if row["item_id"] not in items or not row.get("reviewer_id","").strip():
                        raise ValueError("Unknown item or missing reviewer")
                    item = items[row["item_id"]]
                    if row.get("rubric_version") != VERSION or row.get("job_sha256") != item["job_sha256"]:
                        raise ValueError("Human scores refer to stale evidence or a different rubric; use the current template")
                    scores = validate_scores({k:int(row[k]) for k in IDS})
                    if row["item_id"] in ratings[row["reviewer_id"]]:
                        raise ValueError("Duplicate reviewer/item")
                    ratings[row["reviewer_id"]][row["item_id"]] = scores
                except (ValueError,KeyError,TypeError) as exc:
                    invalid.append({"file":str(file),"item_id":row.get("item_id"),"error":str(exc)})
    if invalid:
        (review/"invalid_human_scores.json").write_text(json.dumps(invalid,indent=2))
        raise ValueError("Invalid/duplicate human scores; fix invalid_human_scores.json before analysis")
    return ratings


def load_judge_scores(root, items):
    """({item_id: validated scores}, {(rubric_version, judge_model, prompt_version)})
    from review/judge_scores/*.json, keeping only rows whose stored scores still
    match the evidence-backed judgment for the current rubric version."""
    from run_colab import digest
    root = Path(root); judge = {}; versions = set()
    for file in sorted((root/"review"/"judge_scores").glob("*.json")):
        row = json.loads(file.read_text())
        if row.get("status") != "scored": continue
        identity = row["identity"]; i = identity["item_id"]
        if i not in items or identity["evidence_sha256"] != digest(item_evidence(root, items[i])): continue
        if identity["rubric_version"] != VERSION: continue
        checked = validate_judgment(row["judgment"], set(item_evidence(root, items[i])))
        if checked != row["scores"]:
            raise ValueError("Stored judge scores differ from the evidence-backed judgment")
        if i in judge:
            raise ValueError("Multiple judge versions for one item; analyze one frozen version at a time")
        judge[i] = row["scores"]
        versions.add((identity["rubric_version"], identity["judge_model"], identity["prompt_version"]))
    if len(versions) > 1:
        raise ValueError("Do not pool different judge/rubric versions")
    return judge, versions


def summarize_source(root, source, scores, items, models=None, judge_validated=True):
    """One evaluation arm's aggregation: per-element (layer/agent/tool/S-EAP)
    and per-report rubric-mean graphs plus a health verdict written to
    ``review/<source>_health.json``. Shared by run_review.py (which passes the
    real judge-validation state) and the ``*/aggregate_scores.py`` helpers in
    human_review/ and llm_review/. Returns the health dict.

    scores: {item_id: {metric: 1..10}}  (already validated by the caller)
    items:  {item_id: item record from review_items.jsonl}
    source: label used in filenames/titles, e.g. "human" or "judge".
    """
    root = Path(root); review = root/"review"; (review/"graphs").mkdir(parents=True, exist_ok=True)
    if models is None:
        models = [m["model_id"] for m in json.loads((root/"run_manifest.json").read_text())["config"]["models"]]
    provisional = source == "judge" and not judge_validated
    for kind in ("layer","agent","tool","seap"):
        by_element=defaultdict(list)
        for i,s in scores.items():
            item=items[i]
            if item["item_type"]==kind:
                by_element[(item["model"],item["element"])].append(health(s,item["execution_status"]=="completed")["mean_1to10"])
        if not by_element:continue
        labels=sorted({e for m,e in by_element})
        for page,start in enumerate(range(0,len(labels),32),1):
            columns=labels[start:start+32]
            _plot_matrix([[np.mean(by_element[(m,e)]) if by_element[(m,e)] else np.nan for e in columns] for m in models],
                models,columns,f"{source} {kind} rubric mean — "+("provisional judge" if provisional else source),
                review/"graphs"/f"09_{source}_{kind}_scores_{page:02d}.png",1,10)
    values=defaultdict(list);passing=0;scored=0
    for i,s in scores.items():
        item=items[i]
        if item["item_type"]!="report":continue
        h=health(s,item["execution_status"]=="completed");scored+=1;passing+=h["status"]=="pass"
        values[(item["model"],item["angle"])].append(h["mean_1to10"])
    _plot_matrix([[np.mean(values[(m,a)]) if values[(m,a)] else np.nan for a in range(1,26)] for m in models],models,
                 [str(a) for a in range(1,26)],source+" rubric mean (1–10); grey = unscored",
                 review/"graphs"/("05_"+source+"_rubric_matrix.png"),1,10)
    manifest=json.loads((root/"run_manifest.json").read_text());expected=manifest["expected_jobs"]
    try: completed=json.loads((root/"coverage.json").read_text())["completed"]
    except (OSError,ValueError,KeyError): completed=None
    pass_rate=passing/scored if scored else None
    validated=source!="judge" or judge_validated
    status=("pass" if scored==expected and completed==expected and pass_rate is not None
            and pass_rate>=.8 and validated else "incomplete_or_needs_review")
    out={"status":status,"source":source,"scored_reports":scored,"expected_reports":expected,
         "completed_jobs":completed,"passing_reports":passing,"pass_rate":pass_rate,
         "judge_validated":(judge_validated if source=="judge" else None),
         "rule":"Full execution and scoring coverage; >=80% of reports pass the shared 8-metric rubric; LLM-only health also requires a validated judge."}
    (review/(source+"_health.json")).write_text(json.dumps(out,indent=2))
    return out


def export_run(root,persist=None):
    root=Path(root);out=root/"review";out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/"run_manifest.json").read_text())
    models=[m["model_id"] for m in manifest["config"]["models"]]
    items=[];summaries=[];tool_sums=defaultdict(lambda:Counter());seap_rows=[];errors=[]
    tool_fields=["report_id","model","behavior_id","angle","agent","role","layer","tool","status","calls","seconds","reason"]
    with (out/"agent_layer_tool_matrix.csv").open("w",newline="") as tf:
        tw=csv.DictWriter(tf,fieldnames=tool_fields);tw.writeheader()
        for path in sorted((root/"jobs").glob("*.json")):
            try: payload=read_payload(path)
            except (ValueError,KeyError) as exc: errors.append(str(exc));continue
            if payload["identity"].get("run_id")!=manifest["run_id"]: continue
            identity=payload["identity"];entry=payload.get("entry",{});b=metadata(identity["behavior_id"])
            rid=path.stem
            if payload.get("report_markdown"):
                reports=root/"reports";reports.mkdir(exist_ok=True)
                (reports/(rid+".md")).write_text(payload["report_markdown"])
            raw=entry.get("raw_hierarchy_result",{})
            common={"report_id":rid,"model":identity["model"],"behavior_id":b["behavior_id"],
                    "behavior":entry.get("behavior",b["behavior_id"]),"angle":b["angle"],"angle_name":b["angle_name"],
                    "seed":identity["seed"],"execution_status":payload["status"],"rubric_version":VERSION,
                    "job_file":str(path.relative_to(root))}
            from run_colab import digest
            common["job_sha256"]=digest(payload)
            def add(kind,selector,label,status=None):
                item_id=rid+"::"+kind+"::"+label
                items.append({**common,"item_id":item_id,"item_type":kind,"element":label,"evidence_selectors":selector,
                              "execution_status":status or common["execution_status"]})
            add("report",[],"whole_report")
            for l in range(raw.get("n_layers",0)): add("layer",[f"L{l:03d}"],f"L{l}")
            records=coverage_records(raw)
            agent_rows=defaultdict(list)
            for row in records: agent_rows[row["agent"]].append(row)
            evidence_ids={a["agent"]:f"A{i:03d}" for i,a in enumerate(raw.get("agent_runs",[]))}
            if "SkepticAgent" in evidence_ids: evidence_ids["Skeptic"]=evidence_ids["SkepticAgent"]
            for name,rows in agent_rows.items():
                selectors=[evidence_ids[name]] if name in evidence_ids else ["E006"]
                if name=="Orchestrator": selectors.append("E007")
                status="tool_errors" if any(r["status"]=="error" for r in rows) else "completed" if any(r["status"]=="ok" for r in rows) else "not_run"
                add("agent",selectors,name,status)
                for row in rows:
                    add("tool",selectors,name+"/"+row["tool"],"completed" if row["status"]=="ok" else row["status"])
            seap_status=entry.get("seap_evaluation",{}).get("status","not_run")
            add("seap",["E004"],"signed_interaction_validation",seap_status)
            for row in records:
                tw.writerow({"report_id":rid,"model":identity["model"],"behavior_id":b["behavior_id"],"angle":b["angle"],**row})
                key=(identity["model"],row["layer"],row["role"],row["tool"])
                tool_sums[key]["cells"]+=1;tool_sums[key][row["status"]]+=1;tool_sums[key]["calls"]+=row["calls"]
            seap=entry.get("seap_evaluation",{});metrics=seap.get("metrics",{})
            for row in seap.get("rows",[]): seap_rows.append({"report_id":rid,"model":identity["model"],"angle":b["angle"],**row})
            summaries.append({"report_id":rid,"model":identity["model"],"behavior_id":b["behavior_id"],"angle":b["angle"],
                "seed":identity["seed"],"status":payload["status"],"verdict":entry.get("verdict") or "No claim",
                "layers_total":raw.get("n_layers",0),"layers_measured":len(raw.get("layer_states",{})),
                "agents_executed":len(raw.get("agent_runs",[])),"tool_calls":raw.get("tool_calls_spent"),
                "wall_seconds":payload.get("wall_seconds"),"gpu_peak_gb":payload.get("gpu_peak_allocated_gb"),
                "seap_status":seap.get("status","not_run"),"seap_score":metrics.get("score_0to100"),
                "seap_normalized_mae":metrics.get("normalized_mae"),"seap_signed_pearson":metrics.get("pearson_signed")})
    # Identities are stable; exporting never overwrites completed human scoring files.
    with (out/"review_items.jsonl").open("w") as f:
        for item in items: f.write(json.dumps(item)+"\n")
    identity_fields=("item_id","item_type","model","angle","angle_name","behavior_id","element","rubric_version","job_sha256")
    fields=["item_id","reviewer_id","item_type","model","angle","angle_name","behavior_id","element","rubric_version","job_sha256",*IDS,"evidence_notes"]
    write_csv(out/"human_scores_BLANK_TEMPLATE.csv",({**{k:i[k] for k in identity_fields},
                "reviewer_id":"","evidence_notes":"",**{k:"" for k in IDS}} for i in items),fields)
    write_csv(out/"evaluation_matrix.csv",summaries)
    write_csv(out/"seap_exact_vs_approximation.csv",seap_rows,
              ["report_id","model","angle","prompt_index","head_i","head_j","f_empty","f_i","f_j","f_ij","exact","approximation"])
    # One report per model-angle cell for held-out judge validation, independent of scores/outcome.
    groups=defaultdict(list)
    for item in items:
        if item["item_type"]=="report":groups[(item["model"],item["angle"])].append(item)
    validation=[];remainder=[]
    for key,rows in sorted(groups.items()):
        rows.sort(key=lambda r:hashlib.sha256(("judge-validation-v1|"+r["item_id"]).encode()).hexdigest())
        validation.append(rows[0]);remainder.extend(rows[1:])
    calibration=[]
    for angle in range(1,26):
        available=sorted((i for i in remainder if i["angle"]==angle),key=lambda i:hashlib.sha256(i["item_id"].encode()).hexdigest())
        calibration+=available[:4]
    sample={"frozen":sum(r["status"]=="completed" for r in summaries)==manifest["expected_jobs"] and len(validation)==550 and len(calibration)==100,
            "calibration_items":[i["item_id"] for i in calibration],"validation_items":[i["item_id"] for i in validation],
            "target_calibration":100,"target_validation":550,"rule":"Freeze after full coverage; no overlap; two blinded humans plus explicit consensus reference."}
    (out/"judge_validation_sample.json").write_text(json.dumps(sample,indent=2))
    lookup={i["item_id"]:i for i in items}
    for label,ids in (("calibration",sample["calibration_items"]),("validation",sample["validation_items"])):
        write_csv(out/f"human_{label}_BLANK_TEMPLATE.csv",({**{k:lookup[i][k] for k in identity_fields},
                  "reviewer_id":"","evidence_notes":"",**{k:"" for k in IDS}} for i in ids),fields)
    plots=out/"graphs";plots.mkdir(exist_ok=True)
    counts=Counter((r["model"],r["angle"]) for r in summaries if r["status"]=="completed")
    from .catalog import COUNTS
    nseeds=len(manifest["config"]["seeds"])
    coverage=[[counts[(m,a)]/(COUNTS[a]*nseeds)*100 for a in range(1,26)] for m in models]
    _plot_matrix(coverage,models,[str(a) for a in range(1,26)],"Completed behavior coverage (%) — all 22 models × 25 angles",plots/"01_model_angle_coverage.png",0,100)
    for model in models:
        keys=[k for k in tool_sums if k[0]==model]
        if not keys:continue
        layerids=sorted({k[1] for k in keys if k[1] is not None}); layerids=[None]+layerids
        columns=sorted({k[2]+"/"+k[3] for k in keys})
        data=[]
        for l in layerids:
            row=[]
            for column in columns:
                role,tool=column.split("/",1);v=tool_sums.get((model,l,role,tool))
                row.append(100*v["ok"]/v["cells"] if v else np.nan)
            data.append(row)
        _plot_matrix(data,["global" if l is None else f"L{l}" for l in layerids],columns,
                     model+" — successful agent/tool execution (%); grey = structurally inapplicable",
                     plots/("02_agent_layer_tools_"+_slug(model)+".png"),0,100)
    behavior_counts=Counter((r["model"],r["behavior_id"]) for r in summaries if r["status"]=="completed")
    for angle in range(1,26):
        from .catalog import BUILDERS
        behaviors=[b for b in BUILDERS if metadata(b)["angle"]==angle]
        _plot_matrix([[100*behavior_counts[(m,b)]/nseeds for b in behaviors] for m in models],models,behaviors,
                     f"Angle {angle}: completed behavior coverage (%)",plots/f"08_angle_{angle:02d}_behaviors.png",0,100)
    values=defaultdict(list)
    for r in summaries:
        if r["seap_score"] is not None:values[(r["model"],r["angle"])].append(r["seap_score"])
    _plot_matrix([[np.mean(values[(m,a)]) if values[(m,a)] else np.nan for a in range(1,26)] for m in models],
                 models,[str(a) for a in range(1,26)],"S-EAP 100/(1+normalized MAE) — descriptive accuracy score",plots/"03_seap_model_angle_score.png",0,100)
    if seap_rows:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,6));x=[r["exact"] for r in seap_rows];y=[r["approximation"] for r in seap_rows]
        ax.scatter(x,y,s=8,alpha=.25);lo=min(x+y);hi=max(x+y);ax.plot([lo,hi],[lo,hi],"k--")
        ax.set(xlabel="Exact signed joint interaction",ylabel="S-EAP signed approximation",title="Paired interaction validation (pairs within prompts are dependent)")
        fig.tight_layout();fig.savefig(plots/"04_seap_exact_vs_approximation.png",dpi=150);plt.close(fig)
    result={"reports":len(summaries),"review_items":len(items),"corrupt_checkpoints":errors,
            "judge_validation":"not_yet_validated","human_scores":"awaiting_independent_review",
            "run_id":manifest["run_id"],"rubric":VERSION}
    (out/"export_summary.json").write_text(json.dumps(result,indent=2))
    if persist:
        shutil.copytree(out,Path(persist)/"review",dirs_exist_ok=True)
        if (root/"reports").exists():shutil.copytree(root/"reports",Path(persist)/"reports",dirs_exist_ok=True)
    return result
