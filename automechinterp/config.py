"""Global configuration for AutoMechInterp.

Model roles (Sec. 4.2 of the proposal):
  TARGET_MODEL_ID  -> the network being interpreted. GPT-2 small for Stage A
                       (ground-truth circuits exist: IOI, induction, etc).
  HEAVY_MODEL_ID    -> the open-source LLM backing the heavy-reasoning tier
                       (Orchestrator / Network Analyst / Skeptic / Judge).
                       Proposal used Claude Sonnet; here we use an open 7B
                       model so the whole pipeline runs with no API key.
  SMOKETEST_MODEL_ID -> a tiny stand-in for HEAVY_MODEL_ID used only to prove
                       the local-LLM backend code path works on machines that
                       cannot fit a 7B model in RAM (see llm_backends.py).

Everything below is overridable via environment variables so the same code
runs on a laptop (heuristic backend, GPT-2 target) or on a workstation/Colab
A100 (Qwen2.5-7B-Instruct backend, Gemma/Mistral target) without edits.
"""
import os
import torch

TARGET_MODEL_ID = os.environ.get("AMI_TARGET_MODEL", "gpt2")
HEAVY_MODEL_ID = os.environ.get("AMI_HEAVY_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SMOKETEST_MODEL_ID = os.environ.get("AMI_SMOKETEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# LLM_BACKEND: "heuristic" (no LLM, deterministic ReAct policy -- always available),
#              "hf_local"  (loads HEAVY_MODEL_ID or SMOKETEST_MODEL_ID via transformers),
#              "openai" / "anthropic" (optional API backends, see llm_backends.py)
LLM_BACKEND = os.environ.get("AMI_LLM_BACKEND", "heuristic")

DEVICE = os.environ.get("AMI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Section 3.6: budget-aware termination
GLOBAL_TOOL_CALL_BUDGET = int(os.environ.get("AMI_TOOL_BUDGET", "120"))
PER_LAYER_AGENT_BUDGET = int(os.environ.get("AMI_LAYER_BUDGET", "10"))

# Section 4.6 thresholds used by the heuristic Network Analyst / Skeptic / Judge
CKA_DROP_THRESHOLD = 0.15          # flags a layer boundary as "interesting"
REDUNDANCY_DROP_THRESHOLD = 0.05   # dropping the layer must hurt the task metric
ABLATION_EFFECT_THRESHOLD = 0.20   # component ablation must move the metric this much
EXCLUSION_LEAK_THRESHOLD = 0.10    # ablating everything else must move it less than this
