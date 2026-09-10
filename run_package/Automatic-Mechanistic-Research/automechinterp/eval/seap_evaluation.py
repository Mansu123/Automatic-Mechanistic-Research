"""Signed S-EAP comparison against exact, identically scoped joint interventions."""
from itertools import combinations
import math
import time
import numpy as np
from .causal_measurements import score_contrast, ablate_heads
from ..tools import adapter, colab_causal


def summarize_interactions(rows):
    if not rows:
        return {"status": "insufficient_data", "n_observations": 0, "score_0to100": None}
    exact = np.array([r["exact"] for r in rows], float)
    approx = np.array([r["approximation"] for r in rows], float)
    if not np.isfinite(exact).all() or not np.isfinite(approx).all():
        raise FloatingPointError("Nonfinite S-EAP interaction")
    mae = float(np.abs(exact-approx).mean())
    scale = float(np.abs(exact).mean())
    nmae = mae/scale if scale > 1e-8 else None
    active = np.abs(exact) > 1e-6
    correlation = float(np.corrcoef(exact,approx)[0,1]) if min(exact.std(),approx.std()) > 1e-10 else None
    # Resample prompts as clusters, not overlapping head pairs as independent data.
    by_prompt = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_index"],[]).append(row)
    interval = None
    if len(by_prompt)>1:
        groups=list(by_prompt.values()); rng=np.random.default_rng(1729); values=[]
        for _ in range(1000):
            sample=[r for i in rng.integers(0,len(groups),len(groups)) for r in groups[i]]
            den=np.mean([abs(r["exact"]) for r in sample])
            if den>1e-8: values.append(np.mean([abs(r["exact"]-r["approximation"]) for r in sample])/den)
        if values: interval=np.quantile(values,[.025,.975]).tolist()
    return {"status":"ok" if nmae is not None else "near_zero_reference",
            "n_observations":len(rows),"n_prompt_clusters":len(by_prompt),
            "mae":mae,"normalized_mae":nmae,"normalized_mae_ci95":interval,
            "pearson_signed":correlation,
            "sign_agreement":float((np.sign(exact[active])==np.sign(approx[active])).mean()) if active.any() else None,
            "zero_predictor_mae":scale,
            "score_0to100":100/(1+nmae) if nmae is not None else None,
            "score_definition":"100/(1+normalized_MAE); zero predictor scores 50 for nonzero references; descriptive, not a rubric score",
            "uncertainty":"Prompt-cluster bootstrap; finite template correlations and small sample remain."}


def evaluate_seap(handle, task, enabled=True, candidate_heads=6, evaluation_pairs=4):
    if not enabled: return {"status":"disabled","rows":[],"metrics":{}}
    started=time.perf_counter()
    try:
        layers=list(range(handle.n_layers)); dim=adapter.get_head_dim(handle)
        cp=task["clean_prompt"]; pos=task["io_token"]; neg=task["s_token"]
        grad, acts=colab_causal.gradients(handle,cp,pos,neg,layers)
        candidates=sorted(((l,h) for l in layers for h in range(adapter.get_num_heads(handle))),
                          key=lambda x:-abs(float((grad[x[0]][:,x[1]*dim:(x[1]+1)*dim]*acts[x[0]][:,x[1]*dim:(x[1]+1)*dim]).sum())))[:candidate_heads]
        test=task.get("eval_pairs",[])[:evaluation_pairs]
        if not test or len(candidates)<2:
            return {"status":"insufficient_data","reason":"No disjoint evaluation pairs or fewer than two heads",
                    "candidates":candidates,"rows":[],"metrics":summarize_interactions([])}
        rows=[]; pairs=list(combinations(candidates,2)); selected_layers=sorted({l for l,h in candidates})
        for index,(prompt,_,positive,negative) in enumerate(test):
            position=len(handle.tokenizer.encode(prompt.rstrip(),add_special_tokens=True))-1
            g,a=colab_causal.gradients(handle,prompt,positive,negative,selected_layers)
            f0=score_contrast(handle,prompt,positive,negative)["margin"]
            singles={}; conditional={}
            for head in candidates:
                singles[head]=score_contrast(handle,prompt,positive,negative,
                                  lambda head=head:ablate_heads(handle,[head],positions=[position]))["margin"]
                conditional[head]=colab_causal.gradients(handle,prompt,positive,negative,selected_layers,
                                                        [head],ablate_positions=[position])
            for i,j in pairs:
                fij=score_contrast(handle,prompt,positive,negative,
                                  lambda:ablate_heads(handle,[i,j],positions=[position]))["margin"]
                exact=fij-singles[i]-singles[j]+f0
                def directed(head,other):
                    l,h=head; sl=slice(h*dim,(h+1)*dim); gj,aj=conditional[other]
                    return float((g[l][:,sl]*a[l][:,sl]-gj[l][:,sl]*aj[l][:,sl]).sum())
                approx=(directed(i,j)+directed(j,i))/2
                rows.append({"prompt_index":index,"head_i":i,"head_j":j,"f_empty":f0,
                             "f_i":singles[i],"f_j":singles[j],"f_ij":fij,
                             "exact":exact,"approximation":approx})
        return {"status":"completed","metric":"full_continuation_log_probability_margin",
                "intervention":"zero each query-head projection input at the last prompt position only",
                "equation":"I(i,j)=f({i,j})-f({i})-f({j})+f(empty)",
                "approximation":"mean of directed [g_i*a_i - g_i_after_j*a_i_after_j] and its reverse",
                "selection":"top absolute first-order zero-ablation estimates on discovery prompt; frozen before evaluation",
                "candidates":candidates,"test_pairs":test,"rows":rows,
                "metrics":summarize_interactions(rows),"wall_seconds":time.perf_counter()-started,
                "scope":"Finite-catalog held-out prompt comparison; not proof of global circuit completeness."}
    except Exception as exc:
        return {"status":"error","error":f"{type(exc).__name__}: {exc}","rows":[],"metrics":{},
                "wall_seconds":time.perf_counter()-started}
