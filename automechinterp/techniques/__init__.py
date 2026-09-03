"""automechinterp.techniques -- a model-agnostic mechanistic-interpretability
toolkit, one module per curriculum category (learnmechinterp.com).

Every function here takes an `adapter.ModelHandle` (see automechinterp/tools/
adapter.py) and runs on any HF decoder LM the adapter can register -- GPT-2,
Qwen2, Llama, Pythia -- via forward/backward hooks only.

Design rules:
  * no training loops that need a corpus + hours (toy SAE / tuned-lens fits
    are the exception -- deliberately tiny, seconds not hours, clearly labelled)
  * no dependency on a pretrained artifact that only exists for one model
    (public SAE releases, edited-fact datasets); where a technique genuinely
    needs one, it raises NotImplementedError with the requirement spelled out
  * reuse automechinterp/tools/* rather than reimplementing ACDC/EAP/patching

`runner.run_all(model_id, task)` executes every applicable technique and
writes a Markdown report.
"""
