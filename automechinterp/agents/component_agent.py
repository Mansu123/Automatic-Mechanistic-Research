"""Component Agent (Sec. 3.2, 3.4): spawned on demand by a Layer Agent whose
attn_mlp_attribution showed an attention-driven effect. Runs the cheap
gradient-based EAP approximation first, then the exact (but O(heads)) ACDC
patching sweep to confirm/refine, then inspects the surviving heads'
attention patterns before forming a hypothesis -- i.e. it keeps testing
instead of accepting the first head that looks interesting, which is the
concrete behavior answering gap_work.py's "maia-confirmation-bias" and
"sasc-noninteractive" gaps.
"""
from __future__ import annotations

from ..llm_backends import make_backend
from ..tools import adapter, tier_c
from ..tools import sae as _sae
from .base import Agent, ToolCallBudget, LOG


def build_component_agent(handle: adapter.ModelHandle, layer_idx: int,
                           clean_prompt: str, corrupted_prompt: str,
                           io_token: str, s_token: str, n_heads: int,
                           backend_kind: str, budget: ToolCallBudget,
                           backend_kwargs: dict | None = None):
    state: dict = {}

    def do_eap():
        try:
            digest = tier_c.run_eap(handle, clean_prompt, corrupted_prompt, io_token, s_token,
                                     range(layer_idx, layer_idx + 1), n_heads)
        except Exception as e:  # record the failure so the policy moves on instead of retrying forever
            digest = f"ERROR: {type(e).__name__}: {e}"
        state["eap_digest"] = digest
        return digest

    def do_acdc():
        try:
            digest, circuit = tier_c.run_acdc(handle, clean_prompt, corrupted_prompt, io_token, s_token,
                                               range(layer_idx, layer_idx + 1), n_heads, threshold=0.10)
        except Exception as e:
            digest, circuit = f"ERROR: {type(e).__name__}: {e}", []
        state["acdc_digest"] = digest
        state["circuit"] = circuit
        LOG.emit("ComponentAgent", f"L{layer_idx}: exact ACDC sweep confirms/refines the cheap EAP "
                                     f"estimate instead of trusting it -> circuit={circuit}",
                 gap_id="sasc-noninteractive")
        return digest

    def do_synergy_eap():
        """After the first-order ACDC sweep, catch heads it structurally
        misses: a component whose marginal ablation effect is ~0 because a
        'backup' compensates when it's removed (Wang et al.'s IOI backup
        name-movers). Uses run_synergy_eap over the layer + its neighbours,
        seeded with direct-logit-attribution heads (which see write-direction,
        not ablation effect), and merges any head with significant synergy to
        a circuit head into state['circuit']."""
        lo = max(0, layer_idx - 2)
        hi = min(handle.model.config.num_hidden_layers if hasattr(handle.model.config, "num_hidden_layers")
                 else handle.n_layers, layer_idx + 3)
        rng = range(lo, hi)
        try:
            dla_digest, dla_top = tier_c.direct_logit_attribution(
                handle, clean_prompt, io_token, s_token, rng, n_heads, k=8)
            state["dla_digest"] = dla_digest
            circuit = list(state.get("circuit", []))
            candidates = sorted(set(dla_top) | set(circuit))
            if not candidates:
                state["synergy_eap_digest"] = "no candidates for synergy pass"
                return state["synergy_eap_digest"]
            syn_digest, rows = tier_c.run_synergy_eap(
                handle, clean_prompt, corrupted_prompt, io_token, s_token, rng, n_heads,
                ablate_candidates=candidates, k=10)
            circuit_set = set(circuit)
            recovered = []
            for score, i, j in rows:
                if abs(score) >= 0.05 and i not in circuit_set and (j in circuit_set or j in set(dla_top)):
                    recovered.append(i)
                    circuit_set.add(i)
            state["circuit"] = sorted(circuit_set)
            state["synergy_recovered_heads"] = sorted(set(recovered))
            state["synergy_eap_digest"] = syn_digest + f" | synergy-recovered (missed by ACDC)={sorted(set(recovered))}"
            LOG.emit("ComponentAgent", f"L{layer_idx}: S-EAP second-order pass recovered "
                                        f"{sorted(set(recovered))} that the first-order ACDC threshold missed "
                                        f"-> circuit now {state['circuit']}", gap_id="sasc-noninteractive")
        except Exception as e:
            state["synergy_eap_digest"] = f"ERROR: {type(e).__name__}: {e}"
        return state["synergy_eap_digest"]

    def do_attention_pattern():
        if not state.get("circuit"):
            return "no heads survived the ACDC threshold; nothing to inspect"
        top_layer, top_head = state["circuit"][0]
        # token 0 = A, token 2 = B in "A and B ... S gave ... to" -- inspect the last token's
        # attention back to the IO-name position as a proxy for name-mover behavior
        batch = handle.tokenizer([clean_prompt], return_tensors="pt")
        seq_len = batch["input_ids"].shape[1]
        digest = tier_c.get_attention_pattern(handle, top_layer, top_head, clean_prompt,
                                               target_word_idx=seq_len - 1, source_word_idx=1)
        state["attention_digest"] = digest
        return digest

    def do_sae_decompose():
        digest = tier_c.run_sae_decompose(handle, layer_idx, clean_prompt, token_idx=-1)
        state["sae_decompose_digest"] = digest
        LOG.emit("ComponentAgent", f"L{layer_idx}: gives the causally-important head semantic "
                                     "content (which SAE features it correlates with), not just an "
                                     "effect size")
        return digest

    tools = {"run_eap": do_eap, "run_acdc": do_acdc, "run_synergy_eap": do_synergy_eap,
             "get_attention_pattern": do_attention_pattern}
    sae_available = _sae.supports_sae(handle.model_id)
    if sae_available:
        tools["run_sae_decompose"] = do_sae_decompose

    def policy_fn(evidence_text: str, tool_menu: list[str]) -> dict:
        if "eap_digest" not in state:
            return {"action": "run_eap", "args": {}, "reasoning": "cheap first pass over heads in this layer"}
        if "acdc_digest" not in state:
            return {"action": "run_acdc", "args": {},
                     "reasoning": "confirm the EAP estimate with exact per-head patching before claiming anything"}
        if "synergy_eap_digest" not in state:
            return {"action": "run_synergy_eap", "args": {},
                     "reasoning": "second-order pass: catch backup heads whose marginal effect is ~0 "
                                  "so the first-order ACDC threshold dropped them"}
        if "attention_digest" not in state and state.get("circuit"):
            return {"action": "get_attention_pattern", "args": {},
                     "reasoning": "inspect what the surviving head actually attends to before hypothesizing"}
        if sae_available and "sae_decompose_digest" not in state and state.get("circuit"):
            return {"action": "run_sae_decompose", "args": {},
                     "reasoning": "give the finding semantic content via sparse feature decomposition"}
        return {"action": "stop", "args": {}, "reasoning": "component analysis complete"}

    kwargs = backend_kwargs or {}
    backend = make_backend(backend_kind, policy_fn=policy_fn, **kwargs)
    agent = Agent(backend, tools, budget, max_steps=7)
    agent.name = f"ComponentAgent{layer_idx}"
    agent.system_prompt = (
        f"You are the Component Agent for layer {layer_idx}. Find which specific attention heads "
        "causally drive this layer's effect, confirm with exact patching (not just the cheap "
        "gradient approximation), and characterize what the surviving heads attend to."
    )
    return agent, state
