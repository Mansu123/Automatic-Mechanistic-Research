"""Cheap per-checkpoint validation of every prompt before GPU interpretation."""
from .matrix_rubric import ANGLE_CONTROLS


def audit_task(handle, task):
    train=task["discovery_pairs"]; test=task["eval_pairs"]
    if not train or not test:
        raise ValueError("Discovery and held-out evaluation pairs must both be nonempty")
    train_texts={p for row in train for p in row[:2]}
    test_texts={p for row in test for p in row[:2]}
    if train_texts & test_texts:
        raise ValueError("Discovery/evaluation prompt leakage")
    if len({r[0] for r in test})!=len(test):
        raise ValueError("Duplicate held-out clean prompts")
    limit=getattr(handle.model.config,"max_position_embeddings",None)
    maximum=0; multiple=0
    for clean,corrupt,positive,negative in train+test:
        if clean==corrupt or positive.strip()==negative.strip():
            raise ValueError("Degenerate prompt or answer contrast")
        for prompt in (clean,corrupt):
            prefix=handle.tokenizer.encode(prompt.rstrip(),add_special_tokens=True)
            answers=[]
            for answer in (positive,negative):
                full=handle.tokenizer.encode(prompt.rstrip()+" "+answer.strip(),add_special_tokens=True)
                if not prefix or full[:len(prefix)]!=prefix or len(full)<=len(prefix):
                    raise ValueError("Answer boundary retokenizes the prompt")
                if limit and len(full)>limit:
                    raise ValueError("Prompt and full answer exceed checkpoint context length")
                answers.append(full[len(prefix):]);maximum=max(maximum,len(full))
                multiple+=len(answers[-1])>1
            if answers[0]==answers[1]:
                raise ValueError("Distinct answers collapse to identical token IDs")
    angle=int(task["category"].split(":")[0].split()[-1])
    return {"status":"passed","dataset_hash":task["dataset_hash"],
            "discovery_pairs":len(train),"evaluation_pairs":len(test),
            "max_sequence_tokens":maximum,"multi_token_answers":multiple,
            "required_angle_controls":ANGLE_CONTROLS[angle-1],
            "independent_stress_validated":task["data_audit"]["independent_stress_validated"],
            "scope":"Tokenizer, context, split and contrast integrity; does not validate answer semantics or independent stress coverage"}
