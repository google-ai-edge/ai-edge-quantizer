---
name: auto_quant
description: >-
  Systematically explore quantization configurations for a TFLite model using the AI Edge Quantizer API, finding the optimal recipe that balances file size and accuracy. Use this skill whenever the user wants to quantize a model, optimize a recipe, explore quantization tradeoffs, minimize size bounds, or perform selective quantization using AI Edge Quantizer (or AEQ) framework.
---

# Agent Contribution Guide: AI Edge Quantizer Recipe Exploration

Welcome, AI Agent! This skill governs how you autonomously explore and generate
the optimal selective quantization recipe for any given LiteRT FlatBuffer
(`.tflite`) model using the AI Edge Quantizer (AEQ) framework. Strict adherence
to these protocols ensures code is high-quality, memory safe, and meets
verification protocols.

Your goal is to build an iterative search loop in Python that actively
interrogates the baseline model for numerical degradation, updates the recipe on
the fly to protect highly sensitive tensors, and produces a bounding-box of
Pareto-optimal models for the user.

## Skill Index

Leverage these project-specific skill guides to navigate and execute tasks
effectively. You must read all of these files before proceeding.

| File                                                       | Contents        |
| ---------------------------------------------------------- | --------------- |
| [`README.md`](../../README.md)                             | Overview of the |
:                                                            : AEQ framework   :
:                                                            : and the repo's  :
:                                                            : structure       :
| [`README.md`](../../../../README.md#operator-coverage)     | What            |
:                                                            : configurations  :
:                                                            : are supported   :
:                                                            : for each        :
:                                                            : operator        :
| [`algorithms.md`](algorithms.md)                           | Registry of     |
:                                                            : available       :
:                                                            : quantization    :
:                                                            : algorithms      :
| [`error_metric_selection.md`](error_metric_selection.md)   | Registry of     |
:                                                            : validation      :
:                                                            : error metrics & :
:                                                            : guidelines on   :
:                                                            : task-specific   :
:                                                            : selection and   :
:                                                            : interpretation  :
| [`file_naming.md`](file_naming.md)                         | Standardized    |
:                                                            : conventions for :
:                                                            : naming exported :
:                                                            : models,         :
:                                                            : recipes, masks, :
:                                                            : and validation  :
:                                                            : reports         :
| [`quantization_pattern.md`](quantization_pattern.md)       | Empirical       |
:                                                            : patterns\: what :
:                                                            : works, what     :
:                                                            : doesn't, and    :
:                                                            : why             :
| [`snr_best_practices.md`](snr_best_practices.md)           | How to compute, |
:                                                            : parse, and      :
:                                                            : threshold SNR   :
:                                                            : to locate       :
:                                                            : fragile layers  :
| [`regex_targeting.md`](regex_targeting.md)                 | How to write    |
:                                                            : readable,       :
:                                                            : unified         :
:                                                            : block-level     :
:                                                            : regex patterns  :
:                                                            : to avoid        :
:                                                            : scattered       :
:                                                            : precision       :
:                                                            : overlaps        :
| [`size_estimation.md`](size_estimation.md)                 | How to compute  |
:                                                            : theoretical     :
:                                                            : compressed      :
:                                                            : model size      :
| [`experiment_runner.md`](experiment_runner.md)             | Memory-safe     |
:                                                            : experiment      :
:                                                            : loop, helpers,  :
:                                                            : average         :
:                                                            : bitwidth        :
| [`image_segmentation_eval.md`](image_segmentation_eval.md) | Mandatory       |
:                                                            : End-to-End      :
:                                                            : metrics &       :
:                                                            : spatial heatmap :
:                                                            : generation code :
:                                                            : for             :
:                                                            : segmentation    :
:                                                            : models          :
| [`pareto_curve_plotting.md`](pareto_curve_plotting.md)     | Guidelines on   |
:                                                            : formatting the  :
:                                                            : Pareto          :
:                                                            : visualization,  :
:                                                            : drawing         :
:                                                            : frontiers, and  :
:                                                            : handling        :
:                                                            : outliers        :
| [`output_report.md`](output_report.md)                     | How to format   |
:                                                            : and organize    :
:                                                            : the output      :
:                                                            : produced        :

## Phase 0: Understand the Framework

Before beginning, explicitly use specific file reading tools to read all files
in Skill Index to learn about standard quantization properties, AEQ heuristics,
disk size implications, and file naming conventions.

## 1. Implementation Plan Guidelines

Because complex selective quantization tasks can cause context loss during
execution, your first action MUST be to draft a highly explicit, self-contained
Implementation Plan. Since you will follow this plan closely, you must bake all
necessary context directly into it.

**You MUST output this plan as an artifact file** (e.g.,
`implementation_plan.md` via `write_to_file`) **and HALT your execution**,
explicitly asking the user for permission to proceed before you make *any* code
edits or run *any* loops. You must strictly follow this plan once approved.

Your printed plan must contain:

### Step 1: Context & Role Declaration

*   **Role Judgment**: Declare your role (Contributor, Maintainer, or Reviewer).
*   **Context Summary**: Explicitly write out a 2-3 sentence summary of the
    specific constraints you read from the skill docs. You MUST explicitly state
    that your artifact names will obey the strict rules from `file_naming.md`
    and explicitly confirm you will produce a Pareto visualization per
    `pareto_curve_plotting.md`.

### Step 2: Technical Design & File List

*   **Target Files**: Explicitly list all files you plan to create or modify.
    Please adhere strictly to the `output_report.md` requirements.
*   **Optimization Constraints**: Briefly outline how you will handle target
    file size bounds, acceptable error thresholds across MULTIPLE metrics, and
    generic regex targeting to prevent data leakage.

### Step 3: Experiment Harness Architecture

*   **Metric Selection**: Infer from the float model what metrics you should use
    for validation following the guidelines of `error_metric_selection.md`.
*   **Experiment Blueprint**: Explicitly outline your script structure focusing
    on proper API sequential updates, memory-safe `Quantizer` instantiation per
    permutation, strictly sequential `qt.validate()` calls (one validation at a
    time), explicit garbage collection (`import gc; gc.collect()`), and
    selecting layer candidates. **Most importantly, explicitly confirm you are
    implementing the stringent Two-Phase loop mandated by `experiment_runner.md`
    (Phase 1: Freezing float skips, Phase 2: Iterative Greedy INT4 squashing
    sorted by robustness). Do not hallucinate trial-and-error hacks.**

### Step 4: Verification Plan

*   **Subagent Hand-off**: Document your explicit commitment to invoke a
    `Gemini-3.6-Flash` Reviewer `subagent` (e.g. via `invoke_subagent` tool) to
    audit your execution script *before* you present the final report, following
    the Two-Pass Review Loop (Section 6). Do NOT skip the subagent call before
    providing the final report. ONLY invoke this subagent after you think you're
    done with the quantization exploration process.

### Step 5: Post-Execution Walkthrough

*   Acknowledge that the final response will adhere to Section 8 formatting
    guidelines.

## Phase 1: Setup & Bounding Box

1.  **Ask the User for Bounds**: If the user did not specify, explicitly ask for
    either their **Minimum Model Compression Ratio Model Size** OR their **Max
    Tolerable Error Bound**. Our objective is to find either the best quality
    model given the minimum compression ratio or the smallest model given the
    max tolerable error bound.
2.  **Acquire Verification Preprocessing**: In order to use `qt.validate()`, you
    need a numpy tensor input. You may copy the image processing logic from
    existing manual human scripts in the directory, for example `processing.py`.
    *   **CRITICAL CONSTRAINT (NO LEAKAGE)**: You MUST NOT copy the human's
        manual selective recipe adjustments from existing sources or tutorials
        to blindly inject layer names. You must discover which layers to skip
        dynamically based on your own validation metric parsing!
3.  **Effort & Sizing Logic**: The user's size bounds represent a *limit*, NOT a
    *target to match exactly*. You are authorized to return models that are
    smaller than requested! The primary goal is to **maximize accuracy** (or the
    target metric) while guaranteeing the model is small enough. If you can make
    it even smaller while retaining acceptable accuracy, do so. Never
    artificially inflate the model just to hit a bound. Try multiple iterations
    until you have found the best possible quantization recipe.

## Phase 2: Building the Iteration Harness

Do not try to guess the recipe blindly or write concurrent sweeps. You must
construct a dedicated, repeatable Python evaluation script (e.g.
`explore_aeq_model.py`) that operates on ONE recipe at a time. Do not run
parallel configurations or concurrent validations.

1.  Load the float baseline `.tflite` model and run data through preprocessing.
2.  Instantiate `quantizer.Quantizer()` and load **ONE default baseline**
    starting recipe via the API.
3.  Inject targeted `qt.update_quantization_recipe(...)` rules sequentially on
    the single `qt` object.
4.  Call `qt.quantize()` and `qt.export_model()`. You MUST output every single
    model variation with a detailed, unique file name following the guidelines
    in `file_naming.md`. Save the `.json` recipe too.
5.  Call `qt.validate(...)` strictly sequentially (one recipe validation at a
    time) to prevent memory overload. Never execute multiple `qt.validate(...)`
    calls concurrently or in parallel. Evaluating multiple error metrics within
    a single validation call is supported. **CRITICAL: Look at metrics in your
    script to decide which tensors to target.**

## Phase 3: The Empirical Search Loop (Execution)

1.  **Two-Phase Iterative Greedy Loop**: Read `experiment_runner.md` for the
    exact step-by-step logic.
    -   **Phase 1 (Finalize Float Skips)**: Identify fragile layers using
        sensitivity analysis (primary metric, e.g., SNR) and incrementally add
        them to `no_quantize` until overall model quality is guaranteed via
        **MSE**. Freeze the float skip-layers set once MSE is good.
    -   **Phase 2 (Iterative Greedy INT4 Squashing)**: Rank remaining
        non-skipped layers by robustness and iteratively squash them
        block-by-block to 4-bit, validating model MSE after each step to prevent
        compounding noise. Revert and stop if MSE exceeds tolerance.
        *   **CRITICAL**: When designing the Phase 2 condition, determine
            threshold bounds dynamically relative to the optimal configuration
            found in Phase 1. Do not use hardcoded scalar assumptions because
            End-to-End metrics can output naturally large arbitrary scalars!
2.  **Loop Execution**: Build out the script to execute both Phase 1 (skip set
    finalization via MSE) and Phase 2 (iterative greedy INT4 squashing on
    non-skipped layers) automatically.

## Phase 4: Exploring the Pareto Frontier

Provide options. You must output, record, and **export to disk EVERY SINGLE**
recipe permutation and quantized model `.tflite` generated during your
exploration (including failed baseline attempts, intermediate steps, and the
final optimized models). Do not just keep them in memory. This output directory
is necessary so the user can plot a rich Pareto curve.

Among all explored recipes, you must explicitly highlight and **recommend the
Top 3** recipes in your final report, mapped to these key profiles:

*   **Target Profile (INT8 + Skips)**: Evaluates remaining non-skipped layers
    strictly at 8-bit, inheriting the finalized skip list.
*   **Size Profile (INT4 + Skips)**: Pushes remaining non-skipped layers to
    4-bit, inheriting the finalized skip list.
*   **Balanced Profile (Mixed-Precision)**: Result of differential sensitivity
    analysis on remaining non-skipped layers (pushing robust layers to 4-bit and
    moderately sensitive layers to 8-bit), inheriting the finalized
    `no_quantize` skip list.

## Agent Verification Protocol (Two-Pass Loop)

To ensure code quality, verifications must follow a strict Two-Pass Recursive
Review:

1.  **Pre-flight Self-Check**: Ensure the python code parses and runs
    successfully before reviewing.
2.  **Pass 1 (Comprehensive Audit)**: You MUST literally invoke a reviewer
    subagent. Do not just hallucinate a review yourself. Pass your python script
    implementation to the subagent via the prompt, and instruct it to
    mathematically evaluate bit depths, proper API usage (not raw JSON hacking),
    isolation of quantizers, and multi-metric evaluations. Wait for its message
    back.
3.  **Pass 2 (Delta Verification)**: Provide updated code to the subagent (using
    `send_message`) and verify all fixes are applied without regressions.
4.  If an impasse is reached, yield to human.

## Role-Based Execution Guidelines

*   **Contributor**: Adhere to `experiment_runner.md` strictly. Prevent Data
    Leakage by never hardcoding explicitly named tutorial layers in your
    implementations; find and parse the actual broken scopes generically.
*   **Maintainer**: Ensure backwards compatibility when updating scripts.
*   **Reviewer**: Organize feedback into clear categories: `[Quantization
    Quality]`, `[AEQ API Health]`, `[Experiment Rigor]`, `[Reporting]`. Reject
    code that lacks multi-metric tracking or multiple Pareto options.

    *   **Rule**: You MUST reject any script that uses manual trial-and-error
    regex arrays to inflate size instead of the proper Two-Phase greedy loop.
    *   **Rule**: You MUST reject any script that doesn't explicitly program the
    Greedy Phase 2 squashing loop sorting layers by robustness.
    *   **Rule**: You MUST reject the code if the artifact paths don't strictly
    generate `_validation_metrics.json` and follow the
    `[execution]_[precision]...[ver]` schema exactly as defined in
    `file_naming.md` and `output_report.md`.

## Post-Execution Walkthrough Guidelines

After execution, format your final response strictly:

1.  **Phase 1: Executive Summary & Bound Achievements**: Technical TL;DR.
    Provide detailed mathematical reasoning (using the selected validation
    metrics) justifying your layer assignments (skip vs int8 vs int4) for the
    models on the frontier.**
2.  **Phase 2: Architectural Footprint**: Paths of generated scripts, quantized
    models `.tflite` files, quantization recipes `.json` files, Pareto graph,
    and `quantization_report.md`.
3.  **Phase 3: Precision & Sanity Guarantee**: Confirm validations evaluated
    distinct error paths jointly to decide configurations and verify data
    leakage checks (generic regex abstraction used over hallucinated names).
4.  **Phase 4: Verification Audit Trail**: Explicit subagent pass/fail status
    scorecard and delta fixes.
