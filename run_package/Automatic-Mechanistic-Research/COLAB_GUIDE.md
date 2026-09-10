# Colab evaluation release — 6 September 2026

The release runs **22 pinned checkpoints × 203 behaviors × one seed = 4,466 jobs**. Those behaviors span **25 angles**; the angle count is not another multiplier. `colab_config.json` and `automechinterp/eval/catalog.py` define the complete suite. `ALL_BEHAVIORS` is strictly angle-ordered; the legacy 16-item subset is never the default for this runner.

## Run in Colab

1. Open `AutoMechInterp_Colab_A100_Ready.ipynb`, select A100 GPU and high RAM, and run cells in order. Upload **MIR_colab_bundle.zip** when prompted. The ZIP contains the notebooks and the `Automatic-Mechanistic-Research` directory at its root.
2. Setup preserves Colab's CUDA PyTorch in a separate environment, installs pinned dependencies and runs the regression tests. Planning prints all 22 revisions and 203 behavior IDs. The GPT-2 smoke cell must pass before the full sweep.
3. The full cell runs every checkpoint sequentially. All layers receive Layer Agent diagnostics. Component agents use up to four selected layers and specialists up to two by default. Selected and unselected agents/tools remain explicit in reports, review forms and coverage graphs. This is adaptive execution, not every optional technique on every layer. Increasing those limits changes the run identity and workload.
4. Completed jobs are checksummed and atomically saved under `MyDrive/MIR_runs_v3/<run_id>`. Rerun the same notebook/configuration to resume. The runtime/dependency versions are part of the identity; an environment change starts a different run. Failed jobs are retried. Keep the same ZIP while a sweep is running.
5. Use `AutoMechInterp_CPU_Judge_Review.ipynb` in a CPU runtime after extraction. Set **OPENROUTER_API_KEY** in Colab Secrets and grant notebook access. Copy the GPU run ID. No key is embedded in this release.
6. Calibrate the rubric on the 100-report calibration set, freeze it, then double-score the disjoint 550-report validation set. Two humans score independently before seeing judge results; retain their ratings and an explicit `consensus` reference. Copy the templates into working CSVs and preserve `item_id`, `rubric_version`, and `job_sha256`.
7. To score every element, use `human_scores_BLANK_TEMPLATE.csv` and judge `SAMPLE='all'`, `ITEM_TYPES='report,layer,agent,tool,seap'`. This is substantially more than 4,466 API calls: each layer, agent and tool is a separate item. Use resumable batches. Report-level judge validation does not validate element-level substitution for humans.

GPU extraction uses the deterministic **heuristic** agent policy. GLM is the independent evidence judge. This run does not test LLM-driven discovery agents. Some toolkit techniques require external datasets or model pairs and remain unavailable/not-run; the pipeline does not fabricate their measurements.

## Shared human and LLM rubric

Both reviewers receive the same evidence and eight metrics, each scored with an integer from **1 through 10**. Matching metrics make direct comparisons interpretable. `matrix_rubric.py` contains metric-specific two-point bands plus criteria for every individual score. Missing ratings remain blank, never zero.

| Metric | Evaluation target |
|---|---|
| Localization and scope | Evidence for the investigated layers, components and positions |
| Causal validity | Correct intervention definitions, signed effects, baselines and uncertainty |
| Independent verification | Necessity, sufficiency/completeness, minimality, joint controls and counterexamples |
| Behavior and angle controls | Held-out behavior variants and the angle-specific controls |
| Evidence faithfulness | Traceability of claims to actual measurements, including errors and nulls |
| Mechanistic explanatory depth | Supported computation steps or a precise diagnosis of an unresolved mechanism |
| Uncertainty and limitations | Confidence proportional to evidence and clear scope limits |
| Clarity and reproducibility | Pinned model, task, configuration, prompts, tool scope and evidence |

Use the lower score of a band when its criteria are only partially satisfied. A rigorously established null can score well; merely disclosing absent evidence does not earn a high score. For an element, score its own work in context rather than assigning it another agent's achievements. GLM must cite existing evidence IDs and give a reason for every metric; invalid output stays unscored.

**Report pass:** mean ≥7.5, every metric ≥5, causal validity / verification / faithfulness each ≥7, and completed execution. **Whole-run health:** full execution and report-scoring coverage, with ≥80% of reports passing. LLM-only health also requires a validated judge. These are proposed preregistered operating thresholds, not universally established scientific cutoffs.

**Official judge-validation gate:** 550 held-out paired reports, one per model–angle cell; macro quadratic weighted kappa ≥0.80; 95% report-cluster bootstrap lower bound ≥0.75; within-one-point agreement ≥80%; each critical metric's kappa ≥0.75. Exact agreement, MAE and signed judge bias are also reported. Exact agreement and within-one agreement are different statistics. Constant or insufficient ratings cannot validate the judge. The 100 calibration reports are disjoint and excluded from the gate. This sample supports an overall study, not precise conclusions for each individual model–angle cell.

Keep the rubric/prompt fixed during validation. Changing either requires a fresh calibration/validation plan. A hosted model alias can change upstream; response model/provider metadata is retained, but the release cannot freeze OpenRouter's model weights.

## Outputs

| Output under the persisted run | Contents |
|---|---|
| `coverage.json`, `coverage.csv` | Exact expected/completed jobs; must reach 4,466/4,466 |
| `acceptance/`, `task_audits/` | Real-checkpoint hook/gradient checks and tokenizer/split/context audits |
| `jobs/`, `reports/` | Full evidence and report for each job, including every layer and agent/tool status |
| `review/evaluation_matrix.csv` | Per-job execution, layer/agent counts, runtime, memory and S-EAP measurements |
| `review/agent_layer_tool_matrix.csv` | Full report × agent × layer × tool execution inventory |
| `review/review_items.jsonl`, `human_*_BLANK_TEMPLATE.csv` | Same-evidence review items and all eight human scores per item |
| `review/all_element_scores_and_health.csv` | Human/LLM scores for all scored item types, with provisional judge status |
| `review/human_vs_judge_matrix.csv`, `judge_validation.json` | Paired metrics and agreement/validation results |
| `review/human_health.json`, `judge_health.json` | Per-arm pass/health verdict (`human_review/` and `llm_review/` aggregators) |
| `review/graphs/01*`, `02*`, `08*` | Model–angle, agent–layer–tool, and all 25 behavior coverage graphs |
| `review/graphs/03*`, `04*` | S-EAP model–angle accuracy and signed exact-versus-approximate interactions |
| `review/graphs/05*`–`07*`, `09*` | Rubric/health comparisons, agreement, and layer/agent/tool/S-EAP score graphs |

The two scoring arms mirror each other: `human_review/` (reviewers) and `llm_review/` (GLM 5.2 via OpenRouter), both scoring the same evidence on the same eight-metric 1–10 rubric. `python generate_figures.py --run-dir <run>` renders three publication figures from this run's CSVs.

Score graphs appear only when corresponding scores exist. Grey cells mean unavailable/unscored, never a measured zero. `BLANK_TEMPLATE` files regenerate on export; completed working CSVs must use different filenames.

## S-EAP interpretation

The separate evaluator freezes candidate heads using a discovery prompt, then evaluates exact and approximate interactions on held-out prompt pairs under identical zero-ablation scope: query-head projection inputs at the last prompt position.

`I(i,j) = f({i,j}) - f({i}) - f({j}) + f(empty)`, where the set denotes **ablated heads** and `f` is the full-continuation log-probability margin. This convention differs from an active-coalition game; interpret the sign under the stated convention. The approximation averages both directed gradient-times-activation differences. Output includes signed correlation/agreement, MAE, normalized MAE, prompt-cluster bootstrap intervals and a zero-predictor baseline.

The descriptive score is `100 / (1 + normalized_MAE)`: perfect agreement scores 100; a zero predictor scores 50 when the reference is nonzero. A near-zero reference yields an unavailable score. This is separate from the 1–10 rubric and does not prove synergy discovery or a circuit's completeness. The default six candidates and four evaluation prompts are a bounded screen, not exhaustive pair enumeration or an adequately powered confirmatory study. The full statistically controlled S-ACDC procedure remains outside this implementation.

## What is verified, and what remains

- All **4,466** real pinned tokenizer/configuration combinations passed prompt-boundary, context-length, token-distinctness and discovery/evaluation-split checks. No weights were needed for this audit.
- All **1,827** BF16 random-weight architecture/behavior cases passed pipeline and S-EAP execution checks: 203 behaviors on nine architecture fixtures, covering the eight families in the roster plus a Gemma regression fixture.
- **31** local regression tests passed, covering causal hooks, grouped-query head geometry, BF16 paths, complete continuations, resume integrity, rubric scoring, agreement and export contracts.
- A cached real GPT-2 CPU smoke job passed extraction and persistence. It uses a reduced smoke configuration and is not an A100 BF16 benchmark or a full-model sweep.
- The earlier persisted audit found 230 tool-error reports and only six represented angles among 239 reports. These historical outputs never count as completed jobs in this release.
- Full A100 execution on the 22 pretrained checkpoints, independent stress-data validation, actual human ratings and authenticated GLM scoring remain to be performed. The finite catalog covers every named behavior, but its related templates do not test every possible perturbation or establish independent generalization. No claim of judge validation or complete scientific coverage is made before those measurements.

Reproduce locally/in Colab:

```bash
python -m unittest discover -s tests -v
python run_colab.py --plan
python check_tokenizer_matrix.py --output tokenizer_audit.json
python validate_architecture_matrix.py --output-dir architecture_validation
```

## Research basis

Faithfulness and actual causal interventions are central because circuit overlap alone can misrepresent circuit quality; gradient approximations require direct validation. See [Have Faith in Faithfulness](https://arxiv.org/abs/2403.17806) and the [MIB benchmark](https://mib-bench.github.io/). They motivate separating approximation accuracy, causal evidence and report quality.

LLM judges can exhibit biases and require human comparison on the same task distribution; general chat agreement does not validate GLM for mechanistic reports. See [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685). The ordinal agreement implementation is tested against [scikit-learn quadratic Cohen's kappa](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html).

The [OpenRouter model catalog](https://openrouter.ai/api/v1/models), checked 6 September 2026, lists `z-ai/glm-5.2:free`, zero prompt/completion prices, a 256,000-token context and structured responses. Availability/pricing is checked again before judging. [Free-model limits](https://openrouter.ai/docs/faq) can make the full element study take many days; CPU scoring preserves A100 time. No automatic paid/model fallback is configured.
