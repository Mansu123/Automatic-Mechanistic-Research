"""Pluggable reasoning backends for agents (Sec. 3.2 "Tools / Model Tier").

Every agent talks to a backend through one method: decide(system, evidence,
tool_menu) -> dict describing the next action. Swapping HeuristicBackend for
HFLocalBackend("Qwen/Qwen2.5-7B-Instruct") changes *nothing* in agent code --
that is what makes the backbone-comparison ablation (Sec. 4.7) a one-line
change instead of a rewrite.

Backends:
  HeuristicBackend  - deterministic policy, no model weights, always available.
                       Encodes the same decision rules the proposal describes in
                       prose (e.g. "flag layers where CKA drop exceeds threshold")
                       so the full pipeline is runnable with zero downloads.
  HFLocalBackend    - open-source causal LM via `transformers`, run locally.
                       Defaults to Qwen2.5-7B-Instruct (config.HEAVY_MODEL_ID).
                       Parses the model's JSON action out of free text.
  OpenAIBackend     - optional, for parity with the original proposal's
                       "backbone comparison" ablation.
  AnthropicBackend  - optional, same purpose.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


class LLMBackend:
    name: str = "base"

    def decide(self, system: str, evidence: str, tool_menu: list[str]) -> dict[str, Any]:
        """Return {'action': 'tool_name'|'stop', 'args': {...}, 'reasoning': str}."""
        raise NotImplementedError


class HeuristicBackend(LLMBackend):
    """No LLM weights. A hand-written policy function drives the ReAct loop.

    Each agent (network_analyst.py, layer_agent.py, ...) supplies `policy_fn`,
    a small function of (evidence_dict, step_index) -> next action. This class
    just exists so the *interface* agents call is identical whether the brain
    behind it is a rule table or a 7B model -- see module docstring.
    """
    name = "heuristic"

    def __init__(self, policy_fn):
        self.policy_fn = policy_fn

    def decide(self, system: str, evidence: str, tool_menu: list[str]) -> dict[str, Any]:
        return self.policy_fn(evidence, tool_menu)


_ACTION_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class HFLocalBackend(LLMBackend):
    """Local open-source LLM (default: Qwen2.5-7B-Instruct) as the agent brain.

    Requires `transformers` + `accelerate` (+ `bitsandbytes` for 4-bit). Loads
    lazily on first `decide()` call so importing this module never triggers a
    multi-GB download. On CPU-only / <16GB RAM machines, pass a small model_id
    (see config.SMOKETEST_MODEL_ID) to exercise this code path without OOMing.
    """
    name = "hf_local"

    def __init__(self, model_id: str, device: str = "cpu", load_in_4bit: bool = False,
                 max_new_tokens: int = 256):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        # make_backend() caches one instance of this class per model config so
        # concurrent Layer/Component Agents share it (not one 7B load each);
        # this lock serializes their .generate() calls the same way
        # tools/adapter.py's MODEL_LOCK serializes the target model's forward
        # passes -- same reasoning, different shared model.
        import threading
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {}
        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        elif self.device != "cpu":
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.device == "cpu" and not self.load_in_4bit:
            self._model.to("cpu")
        self._model.eval()

    def _prompt(self, system: str, evidence: str, tool_menu: list[str]) -> str:
        menu = "\n".join(f"- {t}" for t in tool_menu) + "\n- stop (finish and report)"
        user = (
            f"Evidence so far:\n{evidence}\n\n"
            f"Available actions:\n{menu}\n\n"
            "Respond with ONLY a JSON object: "
            '{"action": "<tool_name_or_stop>", "args": {...}, "reasoning": "<one sentence>"}'
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def decide(self, system: str, evidence: str, tool_menu: list[str]) -> dict[str, Any]:
        with self._lock:
            self._ensure_loaded()
            import torch

            prompt = self._prompt(system, evidence, tool_menu)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            text = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._parse_action(text, tool_menu)

    @staticmethod
    def _parse_action(text: str, tool_menu: list[str]) -> dict[str, Any]:
        m = _ACTION_JSON_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
                if "action" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
        # Fallback: the model didn't produce clean JSON. Stop rather than guess.
        return {"action": "stop", "args": {}, "reasoning": f"unparseable model output: {text[:200]!r}"}


class OpenAIBackend(LLMBackend):
    """Optional heavy-tier backend via the OpenAI API. Requires OPENAI_API_KEY."""
    name = "openai"

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def decide(self, system: str, evidence: str, tool_menu: list[str]) -> dict[str, Any]:
        from openai import OpenAI
        client = OpenAI()
        menu = "\n".join(f"- {t}" for t in tool_menu) + "\n- stop"
        user = (
            f"Evidence so far:\n{evidence}\n\nAvailable actions:\n{menu}\n\n"
            'Respond with ONLY JSON: {"action": "...", "args": {...}, "reasoning": "..."}'
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)


class AnthropicBackend(LLMBackend):
    """Optional heavy-tier backend via the Anthropic API. Requires ANTHROPIC_API_KEY."""
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5"):
        self.model = model

    def decide(self, system: str, evidence: str, tool_menu: list[str]) -> dict[str, Any]:
        import anthropic
        client = anthropic.Anthropic()
        menu = "\n".join(f"- {t}" for t in tool_menu) + "\n- stop"
        user = (
            f"Evidence so far:\n{evidence}\n\nAvailable actions:\n{menu}\n\n"
            'Respond with ONLY JSON: {"action": "...", "args": {...}, "reasoning": "..."}'
        )
        resp = client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text
        m = _ACTION_JSON_RE.search(text)
        return json.loads(m.group(0)) if m else {"action": "stop", "args": {}, "reasoning": text[:200]}


_hf_local_cache: dict[tuple, HFLocalBackend] = {}


def make_backend(kind: str, policy_fn=None, **kwargs) -> LLMBackend:
    """Factory used by hierarchy.py so `config.LLM_BACKEND` selects the brain
    for every agent in one place. Every Layer/Component Agent calls this
    separately (dozens of times in one run) -- for hf_local specifically we
    cache one HFLocalBackend per (model_id, device, quantization) combo so
    the model's weights are loaded once and shared, not reloaded per agent."""
    if kind == "heuristic":
        assert policy_fn is not None, "HeuristicBackend requires a policy_fn"
        return HeuristicBackend(policy_fn)
    if kind == "hf_local":
        cache_key = tuple(sorted(kwargs.items()))
        if cache_key not in _hf_local_cache:
            _hf_local_cache[cache_key] = HFLocalBackend(**kwargs)
        return _hf_local_cache[cache_key]
    if kind == "openai":
        return OpenAIBackend(**kwargs)
    if kind == "anthropic":
        return AnthropicBackend(**kwargs)
    raise ValueError(f"unknown backend kind: {kind}")
