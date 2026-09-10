"""Paired ordinal judge validation; missing or constant ratings cannot pass."""
import numpy as np
from .matrix_rubric import IDS, CRITICAL, validate_scores


def quadratic_kappa(a,b):
    a=np.asarray(a,int)-1; b=np.asarray(b,int)-1
    if not len(a) or len(a)!=len(b): return None
    obs=np.zeros((10,10));np.add.at(obs,(a,b),1)
    expected=np.outer(obs.sum(1),obs.sum(0))/len(a)
    weights=((np.arange(10)[:,None]-np.arange(10)[None,:])/9)**2
    denominator=float((weights*expected).sum())
    return 1-float((weights*obs).sum())/denominator if denominator>1e-12 else None


def compare(human,judge,metadata,min_reports=550,bootstrap=1000):
    """Rows are unique report IDs. Each report is one bootstrap cluster."""
    common=sorted(set(human)&set(judge)&set(metadata))
    if not common: return {"status":"unvalidated","reason":"No paired human/judge report scores","paired_reports":0}
    for i in common:
        validate_scores(human[i]); validate_scores(judge[i])
    a=np.array([[human[i][k] for k in IDS] for i in common],int)
    b=np.array([[judge[i][k] for k in IDS] for i in common],int)
    metrics={k:{"qwk":quadratic_kappa(a[:,j],b[:,j]),
                "exact_agreement":float((a[:,j]==b[:,j]).mean()),
                "within_one_agreement":float((abs(a[:,j]-b[:,j])<=1).mean()),
                "mae":float(abs(a[:,j]-b[:,j]).mean()),
                "judge_minus_human":float((b[:,j]-a[:,j]).mean())} for j,k in enumerate(IDS)}
    kappas=[metrics[k]["qwk"] for k in IDS]
    macro=float(np.mean(kappas)) if all(v is not None for v in kappas) else None
    rng=np.random.default_rng(1729);samples=[]
    for _ in range(bootstrap):
        idx=rng.integers(0,len(common),len(common))
        ks=[quadratic_kappa(a[idx,j],b[idx,j]) for j in range(8)]
        if all(k is not None for k in ks): samples.append(np.mean(ks))
    ci=np.quantile(samples,[.025,.975]).tolist() if len(samples)>=max(20,bootstrap//2) else None
    cells={(metadata[i]["model"],int(metadata[i]["angle"])) for i in common}
    models={m for m,a in cells}
    complete_grid=len(models)==22 and cells=={(m,a) for m in models for a in range(1,26)}
    within=float((abs(a-b)<=1).mean())
    passes=(len(common)>=min_reports and complete_grid and macro is not None and macro>=.80
            and ci is not None and ci[0]>=.75 and within>=.80
            and all(metrics[k]["qwk"] is not None and metrics[k]["qwk"]>=.75 for k in CRITICAL))
    return {"status":"validated" if passes else "unvalidated","paired_reports":len(common),
            "model_angle_cells":len(cells),"per_metric":metrics,"macro_qwk":macro,"macro_qwk_ci95":ci,
            "exact_agreement":float((a==b).mean()),"within_one_agreement":within,
            "thresholds":{"min_reports":min_reports,"model_angle_cells":550,"macro_qwk":.80,
                          "macro_qwk_ci_lower":.75,"within_one":.80,"critical_metric_qwk":.75},
            "interpretation":"Validated only for the frozen rubric, judge prompt/model and sampled report distribution. Element-level use needs separate validation."}
