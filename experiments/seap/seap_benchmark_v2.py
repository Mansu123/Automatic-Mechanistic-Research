"""S-EAP benchmark v2 -- the fixes from FINDINGS.md.

  1. candidate set = heads EAP ranks non-trivially but whose EXACT single-head
     ablation effect |phi1| is near zero (the backup regime S-EAP targets),
     PLUS the confirmed first-order circuit heads (so primary x backup pairs
     are in the tested set).
  2. Syn averaged over ALL eval_prompts (6 clean/corrupt pairs), not one.
  3. paired sign-flip permutation test  H0: E[Syn]=0  (exact, 2^n perms),
     then Benjamini-Hochberg FDR across every tested pair.
  4. S-EAP pair value = mean of both directed estimates (ablate i->read j,
     ablate j->read i) -- exact Syn is symmetric, so symmetrise the approx.

Usage: python3 seap_benchmark_v2.py [gpt2|qwen|both]
"""
import sys, itertools, json, random
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "/Users/mansuba/Desktop/MI MAIN RESEARCH/  2 Automatic-Mechanistic-Research-main")
from automechinterp.tools import adapter, tier_c
from automechinterp.stage_a import (GT_NAME_MOVERS, GT_NEGATIVE_NAME_MOVERS, GT_S_INHIBITION,
                                    GT_BACKUP_NAME_MOVERS, GT_HEADS)

torch.manual_seed(0)
_NAMES = ["John", "Mary", "Alice", "Bob", "Sarah", "Tom", "Anna", "Mike"]
_TMPL = "When {A} and {B} went to the store, {S} gave a drink to"


def ioi_pairs(n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.sample(_NAMES, 2)
        clean = _TMPL.format(A=a, B=b, S=a)      # S = A, answer = B (the IO)
        corrupt = _TMPL.format(A=b, B=a, S=b)
        out.append((clean, corrupt, b, a))
    return out


def metric_fn_for(handle, io_t, s_t):
    io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id = handle.tokenizer.encode(" " + s_t.strip())[-1]
    return lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())


def bh_fdr(pvals):
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * n / (rank + 1))
        q[idx] = prev
    return q


def signflip_perm_p(vals):
    """exact one-sample sign-flip permutation test, H0: symmetric about 0."""
    v = np.asarray(vals, float)
    n = len(v)
    obs = abs(v.mean())
    cnt = 0
    for mask in range(1 << n):
        signs = np.array([1.0 if (mask >> k) & 1 else -1.0 for k in range(n)])
        if abs((v * signs).mean()) >= obs - 1e-12:
            cnt += 1
    return cnt / (1 << n)


def run(model_id, is_gpt2, n_prompts=6, M=12):
    print("=" * 80); print(f"MODEL: {model_id}   (n_prompts={n_prompts}, M={M})"); print("=" * 80)
    handle = adapter.register_model(model_id, device="cpu")
    for p in handle.model.parameters():
        p.requires_grad_(False)
    n_heads = getattr(handle.model.config, "num_attention_heads", None) or handle.model.config.n_head
    n_layers = handle.n_layers
    lr = range(n_layers)
    prompts = ioi_pairs(n_prompts, seed=1)
    ref_clean, ref_corr, ref_io, ref_s = prompts[0]
    print(f"ref clean='{ref_clean}'  IO={ref_io} S={ref_s};  {n_layers}L x {n_heads}H")

    # ---- first-order EAP ranking on the reference pair ----
    eap_digest = tier_c.run_eap(handle, ref_clean, ref_corr, ref_io, ref_s, lr, n_heads)
    import re
    eap = {(int(a), int(b)): float(c)
           for a, b, c in re.findall(r"L(\d+)H(\d+):\s*(-?[\d.eE+]+)", eap_digest)}
    # rank all heads by |EAP|; run_eap digest only prints top-8, so recompute fully:
    # (cheap: one more full EAP call already gave top-8; for the sub-threshold band
    #  we need the whole ranking -> call the linear-approx directly per layer)
    full = {}
    for l in lr:
        d = tier_c.run_eap(handle, ref_clean, ref_corr, ref_io, ref_s, range(l, l + 1), n_heads)
        for a, b, c in re.findall(r"L(\d+)H(\d+):\s*(-?[\d.eE+]+)", d):
            full[(int(a), int(b))] = float(c)
    ranked = sorted(full.items(), key=lambda kv: -abs(kv[1]))
    top_heads = [lh for lh, _ in ranked[:4]]                     # pass first-order anyway
    band = [lh for lh, _ in ranked[4:40]]                        # non-trivial, sub-threshold

    ci0 = handle.tokenizer([ref_clean], return_tensors="pt")
    m0 = metric_fn_for(handle, ref_io, ref_s)
    with torch.no_grad():
        clean_m0 = m0(handle.model(**ci0).logits)
    # exact single-head |phi1| on the reference pair to find the near-zero regime
    phi1_ref = {}
    for lh in band:
        mi = adapter.ablate_head_set(handle, [lh], ci0["input_ids"], ci0["attention_mask"], m0)
        phi1_ref[lh] = mi - clean_m0
    backup_regime = sorted(band, key=lambda lh: abs(phi1_ref[lh]))[:max(0, M - len(top_heads))]
    candidates = sorted(set(top_heads) | set(backup_regime))[:M]
    print(f"\ntop-4 first-order heads : {top_heads}")
    print(f"backup-regime picks     : {[(lh, round(phi1_ref[lh],3)) for lh in backup_regime]}")
    if is_gpt2:
        role = lambda h: ("NM" if h in GT_NAME_MOVERS else "BACKUP" if h in GT_BACKUP_NAME_MOVERS
                          else "negNM" if h in GT_NEGATIVE_NAME_MOVERS
                          else "Sinh" if h in GT_S_INHIBITION else "?")
        print(f"candidate roles         : {[(c, role(c)) for c in candidates]}")
    print(f"candidates ({len(candidates)}): {candidates}")
    pairs = list(itertools.combinations(candidates, 2))

    # ---- per-prompt S-EAP and exact Syn ----
    seap_per = {pr: [] for pr in pairs}
    exact_per = {pr: [] for pr in pairs}
    for pi, (cp, xp, io_t, s_t) in enumerate(prompts):
        mfn = metric_fn_for(handle, io_t, s_t)
        ci = handle.tokenizer([cp], return_tensors="pt")
        with torch.no_grad():
            clean_m = mfn(handle.model(**ci).logits)
        # S-EAP: ablate each candidate once, collect directed rows
        _, rows = tier_c.run_synergy_eap(handle, cp, xp, io_t, s_t, lr, n_heads,
                                         ablate_candidates=candidates, k=1)
        directed = {}
        for score, i, j in rows:
            if i in candidates and j in candidates:
                directed[(j, i)] = score          # ablate j -> read i
        # exact coalition values
        phi1 = {}
        for c in candidates:
            phi1[c] = adapter.ablate_head_set(handle, [c], ci["input_ids"], ci["attention_mask"], mfn) - clean_m
        for (i, j) in pairs:
            mij = adapter.ablate_head_set(handle, [i, j], ci["input_ids"], ci["attention_mask"], mfn)
            exact_per[(i, j)].append((mij - clean_m) - phi1[i] - phi1[j])
            s_ij = directed.get((i, j))
            s_ji = directed.get((j, i))
            vals = [x for x in (s_ij, s_ji) if x is not None]
            seap_per[(i, j)].append(float(np.mean(vals)) if vals else 0.0)
        print(f"  prompt {pi+1}/{n_prompts} done")

    # ---- aggregate + stats ----
    recs = []
    for pr in pairs:
        se = np.array(seap_per[pr]); ex = np.array(exact_per[pr])
        recs.append(dict(pair=pr,
                         seap_mean=float(se.mean()), seap_p=signflip_perm_p(se),
                         exact_mean=float(ex.mean()), exact_p=signflip_perm_p(ex)))
    recs_by_seap = sorted(recs, key=lambda r: -abs(r["seap_mean"]))
    seap_q = bh_fdr([r["seap_p"] for r in recs]); exact_q = bh_fdr([r["exact_p"] for r in recs])
    for r, sq, eq in zip(recs, seap_q, exact_q):
        r["seap_q"] = float(sq); r["exact_q"] = float(eq)

    xs = [r["seap_mean"] for r in recs]; ys = [r["exact_mean"] for r in recs]
    rho_s, p_s = spearmanr(xs, ys)
    rho_a, p_a = spearmanr(np.abs(xs), np.abs(ys))
    print(f"\n[VALIDATION over {len(pairs)} pairs, {n_prompts}-prompt mean]")
    print(f"  Spearman(S-EAP, exact) signed : rho={rho_s:+.3f}  p={p_s:.1e}")
    print(f"  Spearman(|S-EAP|, |exact|)     : rho={rho_a:+.3f}  p={p_a:.1e}")
    print(f"  pairs with exact Syn FDR q<0.10 : "
          f"{[r['pair'] for r in recs if r['exact_q'] < 0.10]}")
    print(f"  pairs with S-EAP    FDR q<0.10  : "
          f"{[r['pair'] for r in recs if r['seap_q'] < 0.10]}")

    print("\n[TOP S-EAP PAIRS]")
    for r in recs_by_seap[:12]:
        i, j = r["pair"]
        tag = f" [{role(i)}+{role(j)}]" if is_gpt2 else ""
        print(f"  {i}+{j}: S-EAP={r['seap_mean']:+.3f} (p={r['seap_p']:.2f} q={r['seap_q']:.2f}) | "
              f"exact={r['exact_mean']:+.3f} (p={r['exact_p']:.2f} q={r['exact_q']:.2f}){tag}")

    return dict(model=model_id, n_prompts=n_prompts, candidates=[list(c) for c in candidates],
                spearman_signed=rho_s, spearman_abs=rho_a, n_pairs=len(pairs),
                exact_fdr_pairs=[[list(a), list(b)] for (a, b) in
                                [r["pair"] for r in recs if r["exact_q"] < 0.10]],
                seap_fdr_pairs=[[list(a), list(b)] for (a, b) in
                               [r["pair"] for r in recs if r["seap_q"] < 0.10]],
                top_seap=[dict(pair=[list(r["pair"][0]), list(r["pair"][1])],
                               seap_mean=r["seap_mean"], seap_p=r["seap_p"], seap_q=r["seap_q"],
                               exact_mean=r["exact_mean"], exact_p=r["exact_p"], exact_q=r["exact_q"])
                          for r in recs_by_seap[:12]])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    out = {}
    if which in ("both", "gpt2"):
        out["gpt2"] = run("gpt2", True)
    if which in ("both", "qwen"):
        out["qwen"] = run("Qwen/Qwen2.5-0.5B-Instruct", False)
    with open(f"results_v2_{which}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote results_v2_{which}.json")
