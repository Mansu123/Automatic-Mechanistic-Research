"""Structured falsification evidence. This module never interprets prose as proof."""
from __future__ import annotations

from ..eval.causal_measurements import score_contrast, ablate_heads, summarize


def _measure(handle, prompts, heads):
    if len(set(map(tuple, prompts))) != len(prompts):
        raise ValueError("Duplicate verification prompts are not independent evidence")
    return [score_contrast(handle, p, a, b,
                (lambda: ablate_heads(handle, heads)) if heads else None)["margin"]
            for p, a, b in prompts]


def _result(check, effects, threshold, *, greater=True, minimum_examples=20, **metadata):
    stats = summarize(effects)
    result = {"check": check, "status": "incomplete", "passed": None,
              "statistics": stats, "threshold": threshold,
              "intervention": "joint_zero_ablation_all_positions", **metadata}
    if stats["status"] == "ok" and stats["n"] >= minimum_examples and stats["ci95"]:
        low, high = stats["ci95"]
        result.update(status="ok", passed=bool(low > threshold if greater else high < threshold))
    return result


def necessity(handle, claimed, prompts, threshold=.20):
    if not claimed:
        return {"check": "necessity", "status": "incomplete", "passed": None, "reason": "empty claim"}
    clean, ablated = _measure(handle, prompts, []), _measure(handle, prompts, claimed)
    return _result("necessity", [a-b for a,b in zip(clean, ablated)], threshold,
                   raw_clean=clean, raw_ablated=ablated, claim_role="positive_joint_contribution")


def completeness(handle, claimed, universe, prompts, tolerance=.10):
    complement = sorted(set(map(tuple, universe))-set(map(tuple, claimed)))
    if not claimed or not set(map(tuple, claimed)).issubset(set(map(tuple, universe))):
        return {"check": "completeness", "status": "incomplete", "passed": None, "reason": "invalid claim universe"}
    clean, reconstructed = _measure(handle, prompts, []), _measure(handle, prompts, complement)
    # Equivalence needs an upper bound on absolute deviation, not cancellation.
    return _result("completeness", [abs(a-b) for a,b in zip(clean, reconstructed)], tolerance,
                   greater=False, raw_clean=clean, raw_reconstructed=reconstructed,
                   component_universe="attention query heads only; MLP/residual paths retained",
                   complement= complement)


def minimality(handle, claimed, universe, prompts, threshold=.05):
    if not claimed:
        return {"check": "minimality", "status": "incomplete", "passed": None, "reason": "empty claim"}
    complement = sorted(set(map(tuple, universe))-set(map(tuple, claimed)))
    circuit = _measure(handle, prompts, complement)
    records=[]
    for head in claimed:
        reduced = _measure(handle, prompts, complement+[tuple(head)])
        records.append(_result("candidate_leave_one_out", [a-b for a,b in zip(circuit,reduced)],
                               threshold, removed_head=list(head), raw_reduced=reduced))
    complete = all(r["status"]=="ok" for r in records)
    return {"check": "minimality", "status":"ok" if complete else "incomplete",
            "passed":all(r["passed"] is True for r in records) if complete else None,
            "records":records, "raw_candidate":circuit,
            "scope":"positive-contributor attention circuit; redundancy/suppressor claims need a different contract"}


def counterexamples(handle, claimed, prompts, *, data_status, threshold=.10):
    if data_status != "validated_independent_stress" or not prompts:
        return {"check":"counterexamples","status":"incomplete","passed":None,
                "reason":"No validated independent task-specific stress dataset"}
    clean, changed = _measure(handle,prompts,[]), _measure(handle,prompts,claimed)
    eligible=[(a,b) for a,b in zip(clean,changed) if a > .5]
    effects=[a-b for a,b in eligible]
    failures=sum(d < threshold for d in effects)
    record=_result("counterexamples", effects, threshold,
                   eligible_examples=len(eligible), counterexamples_found=failures,
                   raw_clean=clean,raw_ablated=changed)
    # No extrapolation beyond the finite adversarial set is implied.
    if record["status"]=="ok":
        record["passed"]=failures==0
    return record


def evidence_gate(claimed, records):
    """Missing/errored evidence cannot establish a claim, regardless of LLM vote."""
    required=("necessity","completeness","minimality","counterexamples")
    if not claimed:
        return "Speculative", "No nonempty circuit claim was supplied"
    if any(k not in records or records[k].get("status")!="ok" for k in required):
        return "Speculative", "Required independent verification is missing, incomplete, or failed"
    if records["necessity"].get("passed") is False or records["counterexamples"].get("passed") is False:
        return "Refuted", "The stated positive-contributor claim failed a preregistered falsification check"
    if records["completeness"].get("passed") is False:
        return "Speculative", "The claimed attention circuit does not reproduce the defined behavior within tolerance"
    if records["minimality"].get("passed") is False:
        return "Probable", "Joint effect has support but the candidate does not meet the stated minimality contract"
    if all(records[k].get("passed") is True for k in required):
        return "Confirmed", "All scoped structured verification checks passed on eligible independent evidence"
    return "Speculative", "Verification schema contains an unresolved result"
