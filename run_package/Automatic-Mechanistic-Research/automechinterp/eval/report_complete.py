"""Complete deterministic evidence reports shared by human reviewers and judges."""
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

ROLE_TOOLS = {
    "NetworkAnalyst":["profile_network","layerwise_cka_scan","redundancy_scan_flagged"],
    "WeightAgent":["attn_out_svd","mlp_out_svd","weight_norm_profile","embedding_unembedding_alignment"],
    "SafetyAgent":["refusal_direction","activation_anomaly_monitor","copy_suppression","deception_detection"],
    "LayerAgent":["patch_layer","attn_mlp_attribution","logit_lens","sae_layer_profile"],
    "ComponentAgent":["run_eap","run_acdc","run_synergy_eap","get_attention_pattern","run_sae_decompose"],
    "LensAgent":["logit_lens","jacobian_lens","logit_lens_trajectory","patchscopes"],
    "ProbeAgent":["linear_probe","sparse_probe","mdl_probe","probe_direction_causal_test"],
    "FeatureAgent":["activation_dimensionality","train_toy_sae","feature_direction_geometry","public_sae_decompose"],
    "SteeringAgent":["activation_addition","ablation_steering","leace"],
    "Skeptic":["ablate_component","exclusion_ablation","minimality_check","counterexample_search"],
    "Judge":["adjudicate"], "Orchestrator":["route_and_merge"],
}
LAYER_ROLES=["LayerAgent","ComponentAgent","LensAgent","ProbeAgent","FeatureAgent","SteeringAgent"]


def _slug(text):
    return re.sub(r"[^a-z0-9]+","_",text.lower()).strip("_")


def safe(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [safe(v) for v in value]
    if hasattr(value,"tolist"): return safe(value.tolist())
    return value


def report_id(task,model_id,run_identity=None):
    identity=run_identity or {"unfrozen_run":datetime.now(timezone.utc).isoformat()}
    h=hashlib.sha256(json.dumps(safe(identity),sort_keys=True).encode()).hexdigest()[:16]
    return f"{_slug(model_id)}__{_slug(task['behavior'])}__{h}"


def coverage_records(result):
    runs={r["agent"]:r for r in result.get("agent_runs",[])}
    n=result.get("n_layers",max([int(l) for l in result.get("layer_states",{})]+[-1])+1)
    names=["Orchestrator","NetworkAnalyst","WeightAgent","SafetyAgent"]
    names += [f"{role}{l}" for l in range(n) for role in LAYER_ROLES]
    names += ["Skeptic","Judge"]
    out=[]
    for name in names:
        role=re.sub(r"\d+$","",name); match=re.search(r"(\d+)$",name)
        layer=int(match[1]) if match else None
        run=runs.get(name)
        if name=="Skeptic": run=run or runs.get("SkepticAgent")
        if name=="Orchestrator":
            events=[{"tool":"route_and_merge","status":"ok","seconds":0,"observation":result.get("search_coverage",{})}]
            tools=["route_and_merge"]; termination="completed"
        elif name=="Judge":
            events=[{"tool":"adjudicate","status":"ok","seconds":0,"observation":result["verdict"]}] if result.get("verdict") else []
            tools=["adjudicate"];termination="completed" if events else "no_claim"
        else:
            events=run.get("telemetry",[]) if run else []
            tools=list(dict.fromkeys(ROLE_TOOLS.get(role,[])+(run.get("available_tools",[]) if run else [])+[e["tool"] for e in events]))
            termination=run.get("termination","unknown") if run else "not_selected_or_not_applicable"
        for tool in tools:
            found=[e for e in events if e["tool"]==tool]
            if not found and run and tool not in run.get("available_tools",[]):
                reason="optional_tool_not_available"
            else:
                reason=termination
            status=("error" if any(e["status"]=="error" for e in found) else
                    "skipped" if found and all(e["status"]=="skipped" for e in found) else
                    "ok" if found else "not_run")
            out.append({"agent":name,"role":role,"layer":layer,"tool":tool,"status":status,
                        "calls":len(found),"seconds":sum(e.get("seconds",0) for e in found),
                        "reason":reason if not found else "see tool evidence"})
    return out


def evidence_bundle(task,result,model_id,seap=None,capability=None):
    evidence={"E000":{"task":{k:task.get(k) for k in ("behavior","category","clean_prompt","corrupted_prompt","io_token","s_token","data_audit","dataset_hash","discovery_pairs","eval_pairs")},"model":model_id},
              "E001":{"configuration":result.get("run_configuration",{}),"execution_status":result.get("execution_status"),"search_coverage":result.get("search_coverage"),"wall_seconds":result.get("wall_seconds"),"tool_calls_spent":result.get("tool_calls_spent")},
              "E002":{"network":result.get("network_findings",{}),"weight":result.get("weight_findings",{}),"safety":result.get("safety_findings",{})},
              "E003":{"claimed_heads":result.get("claimed_heads",[]),"verification":result.get("verification_state",{}),"verdict":result.get("verdict")},
              "E004":{"seap":seap or {"status":"not_run"}},
              "E005":{"capability":capability or {"status":"not_measured"}},
              "E006":{"agent_layer_tool_coverage":coverage_records(result)}}
    for l in range(result.get("n_layers",0)):
        def layer(key): return result.get(key,{}).get(l,result.get(key,{}).get(str(l),{}))
        evidence[f"L{l:03d}"]={"layer":l,"layer_state":layer("layer_states"),"component":layer("component_states"),
                                "lens":layer("lens_findings"),"probe":layer("probe_findings"),
                                "feature":layer("feature_findings"),"steering":layer("steering_findings")}
    for i,agent in enumerate(result.get("agent_runs",[])):
        evidence[f"A{i:03d}"]=agent
    evidence["E007"]={"orchestration_trace":result.get("trace",[])}
    return safe(evidence)


def write_report(task,result,model_id,seap=None,capability=None):
    evidence=evidence_bundle(task,result,model_id,seap,capability)
    lines=[f"# {task['behavior']}","",f"**Angle:** {task['category']}  ",f"**Model:** {model_id}  ",
           f"**Execution:** {result.get('execution_status','unknown')}  ",
           "**Rubric:** shared eight metrics, integer 1–10; human and judge use identical evidence.","",
           "## Interpretation", "", "Causal verdict: "+str((result.get("verdict") or {}).get("verdict","No claim")),
           "All model layers appear below. Specialist and optional-tool omissions remain visible as not_run; they are not zero effects. Finite catalog data do not establish independent stress-test generalization.","",
           "## Agent × layer × tool matrix", "", "| Agent | Layer | Tool | Status | Calls | Seconds | Reason |",
           "|---|---:|---|---|---:|---:|---|"]
    for row in coverage_records(result):
        lines.append(f"| {row['agent']} | {row['layer'] if row['layer'] is not None else 'global'} | {row['tool']} | {row['status']} | {row['calls']} | {row['seconds']:.3f} | {row['reason']} |")
    lines += ["", "## Complete evidence", "", "Evidence IDs are shared by the human form and LLM judge. Full observations and findings follow; no top-k-only transcript replaces the underlying results."]
    for key,value in evidence.items():
        lines += ["",f"### {key}","","```json",json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),"```"]
    return "\n".join(lines)+"\n"


def save_report(report_text,out_dir,rid):
    path=Path(out_dir)/f"{rid}.md";path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(report_text);return path
