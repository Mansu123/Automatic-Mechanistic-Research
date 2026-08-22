"""Creates the base/fine-tuned model pair Stage C needs (Sec. 4.5).

The proposal's real target is Mistral-7B vs BioMistral-7B (Mistral
continually pre-trained on biomedical text) -- infeasible to download/run on
this machine (confirmed earlier: a 3B model alone caused multi-hour
downloads and thermal throttling). This produces the same *kind* of pair at
a scale that fits: GPT-2 small, fine-tuned briefly on a small biomedical
corpus. Same architecture before/after (required for Stage C's layer
transplant to even make sense), same "domain adaptation without changing
the base architecture" relationship BioMistral has to Mistral.

Saves to a local directory OUTSIDE the git repo (~/.cache/automechinterp/)
-- checkpoints are ~500MB and don't belong in version control (matches the
project's existing "no big model files" convention for the HF cache).
"""
from __future__ import annotations

import os

BIOMEDICAL_CORPUS = [
    "The patient presented with acute myocardial infarction and elevated troponin levels.",
    "Administration of the antibiotic reduced the bacterial load significantly.",
    "The biopsy revealed malignant cells consistent with adenocarcinoma.",
    "Chronic hypertension increases the risk of stroke and renal failure.",
    "The clinical trial demonstrated efficacy of the monoclonal antibody in reducing inflammation.",
    "Diabetic patients require regular monitoring of glycated hemoglobin levels.",
    "The pathogen was identified as a gram-negative bacterium resistant to penicillin.",
    "MRI imaging confirmed a lesion in the left temporal lobe of the brain.",
    "The dosage of the medication was adjusted based on renal clearance.",
    "Postoperative recovery was complicated by a secondary infection at the incision site.",
    "The vaccine induced a robust immune response with elevated antibody titers.",
    "Genetic sequencing identified a mutation in the BRCA1 gene associated with breast cancer.",
    "The patient's white blood cell count indicated an active inflammatory response.",
    "Surgical intervention was required to remove the obstructing tumor mass.",
    "The randomized controlled trial compared the new therapy against standard chemotherapy.",
    "Symptoms included persistent cough, fever, and shortness of breath consistent with pneumonia.",
    "The enzyme assay showed reduced activity in patients with the metabolic disorder.",
    "Long-term use of the corticosteroid was associated with adrenal suppression.",
    "The endoscopy revealed ulceration of the gastric mucosa.",
    "Blood cultures confirmed sepsis secondary to a urinary tract infection.",
]

DEFAULT_STAND_IN_DIR = os.path.expanduser("~/.cache/automechinterp/gpt2-biomedical-stand-in")


def build_finetuned_stand_in(base_model_id: str = "gpt2", out_dir: str = DEFAULT_STAND_IN_DIR,
                              n_epochs: int = 25, lr: float = 1e-4, device: str = "cpu") -> str:
    """Fine-tunes `base_model_id` briefly on BIOMEDICAL_CORPUS and saves the
    result to `out_dir`. Idempotent: skips training if out_dir already has a
    saved model (delete it to force a re-run)."""
    if os.path.exists(os.path.join(out_dir, "config.json")):
        return out_dir

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Fine-tuning {base_model_id} on {len(BIOMEDICAL_CORPUS)} biomedical sentences "
          f"({n_epochs} epochs) -- one-time setup, cached at {out_dir} afterward...")

    tok = AutoTokenizer.from_pretrained(base_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_id, attn_implementation="eager")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    batch = tok(BIOMEDICAL_CORPUS, return_tensors="pt", padding=True, truncation=True).to(device)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100  # ignore padding in the loss

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
        out.loss.backward()
        optimizer.step()
        print(f"  epoch {epoch + 1}/{n_epochs}: loss={out.loss.item():.4f}")

    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved fine-tuned stand-in to {out_dir}")
    return out_dir
