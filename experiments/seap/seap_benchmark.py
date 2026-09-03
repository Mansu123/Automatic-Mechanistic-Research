"""S-EAP empirical benchmark.

For each target model:
  1. run_acdc full sweep  -> first-order circuit (+ recall vs GT for gpt2)
  2. run_synergy_eap over a candidate head set
  3. EXACT second-order ground truth via coalition ablation:
        Syn_exact(i,j) = m({i,j}) - m({i}) - m({j}) + m({})     (m = ablate-set logit diff)
     Spearman( S-EAP Syn , Syn_exact )   -> is the finite-diff approx valid?
  4. Do the GT backup name movers (gpt2) show the backup signature, and does
     adding the top synergy heads to the ACDC circuit improve faithfulness?
"""
import sys, itertools, json
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "/Users/mansuba/Desktop/MI MAIN RESEARCH/  2 Automatic-Mechanistic-Research-main")

from automechinterp.tools import adapter, tier_c
from automechinterp.stage_a import (build_ioi_task, GT_NAME_MOVERS, GT_NEGATIVE_NAME_MOVERS,
                                    GT_S_INHIBITION, GT_BACKUP_NAME_MOVERS, GT_HEADS)

torch.manual_seed(0)


def ioi_metric(handle, io_token, s_token):
    io_id = handle.tokenizer.encode(" " + io_token.strip())[-1]
    s_id = handle.tokenizer.encode(" " + s_token.strip())[-1]
    def f(logits):
        return float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())
    return f


def run_for_model(model_id, is_gpt2):
    print("=" * 80)
    print(f"MODEL: {model_id}")
    print("=" * 80)
    handle = adapter.register_model(model_id, device="cpu")
    for p in handle.model.parameters():
        p.requires_grad_(False)
    task = build_ioi_task(handle)
    n_heads = task["n_heads"]
    n_layers = handle.n_layers
    clean, corr = task["clean_prompt"], task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]
    print(f"clean='{clean}'  corrupt='{corr}'  IO='{io_t}' S='{s_t}'")
    print(f"{n_layers} layers x {n_heads} heads")

    mfn = ioi_metric(handle, io_t, s_t)
    ci = handle.tokenizer([clean], return_tensors="pt")
    xi = handle.tokenizer([corr], return_tensors="pt")
    with torch.no_grad():
        clean_m = mfn(handle.model(**ci).logits)
        corr_m = mfn(handle.model(**xi).logits)
    print(f"clean logit-diff={clean_m:.3f}  corrupt logit-diff={corr_m:.3f}")

    # ---- 1. first-order ACDC sweep ----------------------------------------
    lr = range(n_layers)
    acdc_digest, acdc_circuit = tier_c.run_acdc(handle, clean, corr, io_t, s_t, lr, n_heads,
                                               threshold=0.10)
    print("\n[ACDC]", acdc_digest)
    if is_gpt2:
        tp = set(acdc_circuit) & set(GT_HEADS)
        bk = set(acdc_circuit) & set(GT_BACKUP_NAME_MOVERS)
        print(f"ACDC recall vs GT_HEADS: {len(tp)}/{len(GT_HEADS)}  "
              f"backup-mover recall: {len(bk)}/{len(GT_BACKUP_NAME_MOVERS)}")

    # ---- candidate set --------------------------------------------------
    eap_digest = tier_c.run_eap(handle, clean, corr, io_t, s_t, lr, n_heads)
    print("\n[EAP]", eap_digest)
    # parse "L{l}H{h}=score" tokens from digest
    import re
    eap_scores = {}
    for m in re.finditer(r"L(\d+)H(\d+):\s*(-?[\d.eE+]+)", eap_digest):
        eap_scores[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    top_eap = [lh for lh, _ in sorted(eap_scores.items(), key=lambda kv: -abs(kv[1]))[:8]]
    if is_gpt2:
        candidates = sorted(set(top_eap) | set(GT_NAME_MOVERS) | set(GT_BACKUP_NAME_MOVERS)
                            | set(GT_NEGATIVE_NAME_MOVERS))
    else:
        candidates = sorted(set(top_eap) | {lh for lh, _ in
                            sorted(eap_scores.items(), key=lambda kv: -abs(kv[1]))[:12]})
    candidates = candidates[:14]
    print(f"\ncandidate heads ({len(candidates)}): {candidates}")

    # ---- 2. S-EAP -----------------------------------------------------
    seap_digest, seap_rows = tier_c.run_synergy_eap(handle, clean, corr, io_t, s_t, lr, n_heads,
                                                    ablate_candidates=candidates, k=15)
    print("\n[S-EAP]", seap_digest)
    # seap_rows: (score, (l_i,h_i), (l_j,h_j))  -- keep only pairs where both endpoints are candidates
    seap_pair = {}
    for score, i, j in seap_rows:
        if i in candidates and j in candidates:
            key = tuple(sorted([i, j]))
            seap_pair[key] = seap_pair.get(key, 0.0) + score  # symmetrize
    # phi1 (first-order) for candidates, exact via single-head ablation
    phi1 = {}
    for (l, h) in candidates:
        m_i = adapter.ablate_head_set(handle, [(l, h)], ci["input_ids"], ci["attention_mask"], mfn)
        phi1[(l, h)] = m_i - clean_m
    m_empty = clean_m

    # ---- 3. exact 2nd-order ground truth -----------------------------
    exact = {}
    for i, j in itertools.combinations(candidates, 2):
        m_ij = adapter.ablate_head_set(handle, [i, j], ci["input_ids"], ci["attention_mask"], mfn)
        exact[(i, j)] = (m_ij - m_empty) - phi1[i] - phi1[j]

    keys = [k for k in exact if k in seap_pair]
    xs = [seap_pair[k] for k in keys]
    ys = [exact[k] for k in keys]
    rho, p = spearmanr(xs, ys)
    rho_abs, p_abs = spearmanr(np.abs(xs), np.abs(ys))
    print(f"\n[VALIDATION] Spearman(S-EAP Syn, exact Syn) over {len(keys)} pairs: "
          f"rho={rho:+.3f} (p={p:.1e}) | on |Syn|: rho={rho_abs:+.3f} (p={p_abs:.1e})")

    # top exact-synergy pairs and their backup signature
    print("\nTop exact-synergy pairs:")
    for (i, j), v in sorted(exact.items(), key=lambda kv: -abs(kv[1]))[:8]:
        tag = ""
        if is_gpt2:
            roles = []
            for h in (i, j):
                if h in GT_NAME_MOVERS: roles.append("NM")
                elif h in GT_BACKUP_NAME_MOVERS: roles.append("BACKUP")
                elif h in GT_NEGATIVE_NAME_MOVERS: roles.append("negNM")
                elif h in GT_S_INHIBITION: roles.append("Sinh")
                else: roles.append("?")
            tag = f"  [{roles[0]}+{roles[1]}]"
        backup_shape = abs(phi1[i]) < 0.5 and abs(phi1[j]) < 0.5 and abs(v) > 0.5
        print(f"  {i}+{j}: exact Syn={v:+.3f}  phi1=({phi1[i]:+.2f},{phi1[j]:+.2f})"
              f"{tag}{'  <- BACKUP SIGNATURE' if backup_shape else ''}")

    # ---- 4. faithfulness: ACDC circuit vs ACDC + top synergy heads ----
    def faithfulness(circuit):
        circuit = set(circuit)
        complement = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in circuit]
        m_circ = adapter.ablate_head_set(handle, complement, ci["input_ids"], ci["attention_mask"], mfn)
        denom = clean_m - corr_m
        return (m_circ - corr_m) / denom if abs(denom) > 1e-8 else float("nan")

    synergy_heads = set()
    for (i, j), v in sorted(exact.items(), key=lambda kv: -abs(kv[1]))[:4]:
        synergy_heads.add(i); synergy_heads.add(j)
    f_acdc = faithfulness(acdc_circuit)
    f_plus = faithfulness(set(acdc_circuit) | synergy_heads)
    print(f"\n[FAITHFULNESS] (fraction of clean-corrupt logit-diff gap the circuit alone keeps)")
    print(f"  ACDC circuit ({len(acdc_circuit)} heads):            {f_acdc:+.3f}")
    print(f"  ACDC + top-4 synergy pairs ({len(set(acdc_circuit)|synergy_heads)} heads): {f_plus:+.3f}")
    print(f"  synergy heads added: {sorted(synergy_heads)}")

    return {
        "model": model_id, "clean_m": clean_m, "corr_m": corr_m,
        "acdc_circuit": [list(x) for x in acdc_circuit],
        "spearman_signed": rho, "spearman_abs": rho_abs, "n_pairs": len(keys),
        "faith_acdc": f_acdc, "faith_plus_synergy": f_plus,
        "top_exact_synergy": [[list(i), list(j), v] for (i, j), v in
                              sorted(exact.items(), key=lambda kv: -abs(kv[1]))[:8]],
    }


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    out = {}
    if which in ("both", "gpt2"):
        out["gpt2"] = run_for_model("gpt2", is_gpt2=True)
    if which in ("both", "qwen"):
        out["qwen"] = run_for_model("Qwen/Qwen2.5-0.5B-Instruct", is_gpt2=False)
    fn = f"seap_benchmark_results_{which}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print("\n\nwrote seap_benchmark_results.json")
