# Experiment Runner Guidelines

This document describes principles for writing experiment sweeps to benchmark
different quantization recipes using AI Edge Quantizer.

Unlike PyTorch graph mode quantization that operates on `nn.Module` objects in
memory, AI Edge Quantizer operates by taking a `.tflite` model, applying a
recipe, and exporting a new quantized `.tflite` model. Because models can be
large, memory management and state isolation are important.

## Guidelines for Experiment Sweeps

1.  **State Isolation & Memory Management:** Always re-instantiate the
    `Quantizer` object inside your sweep loop for each new configuration. This
    ensures that memory is cleaned up correctly and quantization state does not
    bleed across iterations.
    *   **Prevent Memory Overload & Workstation Crashes**: Never run multiple
        `qt.validate()` calls concurrently or launch parallel model validation
        sweeps (e.g. via multithreading, multiprocessing, or concurrent tasks).
        Running multiple validations concurrently creates massive simultaneous
        memory allocations that can exhaust system RAM and crash the
        workstation.
    *   **Sequential Execution**: Always execute validations strictly **one at a
        time** (sequentially). Wait for one model's validation to finish before
        starting the next configuration sweep. Evaluating multiple metrics
        within a single validation call is supported.
    *   **Explicit Garbage Collection**: Call `import gc; gc.collect()` and
        clear unneeded variable references (`del qt`, `del validation_results`)
        after every validation run and before instantiating the next `Quantizer`
        object.
    *   **Sample Dataset Size**: Limit the validation dataset sample count /
        batch size during iterative exploration sweeps to maintain a low RSS
        memory footprint.
2.  **The Conservative Two-Phase Search Loop:** Selective quantization follows a
    conservative, two-phase process:

    *   **Phase 1 (Finalize Skips via MSE Quality Guarantee)**: Conduct
        sensitivity analysis using the model's primary metric (e.g., SNR for
        vision models, KL divergence for LLMs) on the baseline to identify
        fragile candidate layers/blocks. Incrementally add candidate sensitive
        blocks to the `no_quantize` skip list and validate after each addition
        using **MSE (Mean Squared Error)**. Continue iterating and adding
        candidate fragile layers to the skip list until the model's MSE metric
        reaches an acceptable quality threshold. Once the MSE quality criteria
        are satisfied, **freeze the skip-layers set**. Do NOT go back to add
        more layers to the skip list in subsequent steps. You are ONLY allowed
        to proceed to Phase 2 once you're locked in on the skip-layers set for
        this phase.
    *   **Phase 2 (Iterative Greedy INT4 Squashing on Remaining Layers)**: With
        the `no_quantize` skip-layers set strictly finalized and frozen,
        optimize the remaining non-skipped layers using an **iterative greedy
        search** to prevent inter-layer noise accumulation. It's important that
        you MUST adhere to the skip-layers set from Phase 1 and only do this
        greedy INT4 on the remaining non-skip layers. The greedy search can be
        conducted as followed:
        1.  Rank all remaining non-skipped layers/blocks by relative robustness
            using sensitive analysis with primary metric (e.g., SNR for image
            problems). Strictly exclude purely weightless operations (such as
            activation functions, normalization routines, pooling operations,
            reshaping steps, and tensor concatenations) from candidate lists.
            Exclusively target parameterized neural layers that contain
            trainable weights (such as convolutions, attention projections, and
            dense linear transformations) to ensure every mixed-precision
            iteration actively compresses physical parameter arrays and reduces
            disk footprint.
        2.  Iteratively process candidate robust blocks from most robust to
            least robust:
            -   In each step, squash the next candidate robust block to 4-bit
                (`num_bits: 4` via `add_dynamic_config`).
            -   Validate the model checking total model **MSE** and primary
                metric.
            -   If overall model MSE remains within acceptable bounds, commit
                the 4-bit assignment and proceed to the next candidate block.
            -   If squashing a block causes total model MSE to exceed acceptable
                bounds, revert that block to 8-bit and **halt Phase 2**.
        3.  Export Pareto frontier profiles:
            -   **Target Profile (INT8 + Skips)**: Non-skipped layers at 8-bit +
                finalized skip list.
            -   **Size Profile (INT4 + Skips)**: Non-skipped layers to 4-bit +
                finalized skip list.
            -   **Balanced Profile (Iterative Greedy Mixed-Precision)**:
                Finalized 4-bit squashed blocks + 8-bit remaining non-skipped +
                finalized skip list.

3.  **Recipe Overrides & Resolution Precedence:** In AEQ (`RecipeManager`),
    recipe rules are evaluated sequentially where **later rules override earlier
    matching rules**. Therefore:

    -   Default / broad rules (such as `regex=".*"` INT4/INT8 configs) MUST be
        added FIRST.
    -   Specific `algorithm_key="no_quantize"` skip rules MUST be added AFTER
        broad quantization rules so that FP32 fallbacks are preserved and not
        accidentally overwritten by `.*` or 4-bit overrides.
    -   Avoid applying 4-bit overrides to regexes matching layers in the
        `no_quantize` skip list.

4.  **Tracking Artifacts:** Explicitly save the `.tflite` models and track the
    configuration (by exporting the recipe via `qt.get_quantization_recipe()`)
    alongside the validation results for **every single recipe permutation
    tried** (not just the optimal ones) for easy comparison and Pareto curve
    plotting later.

5.  **Visual Evaluation (Mask Generation):** For imaging problems, mathematical
    metrics alone aren't enough to verify quality. Your experiment sweep must
    run inference (using the exported `.tflite` models) on a sample image and
    save the resulting segmentation output masks in `results_fig/` so the user
    can visually verify the qualitative degradation.

## IMPORTANT!!!! Notes on regex

Once you identify the target tensors to update quantization recipe, **NEVER**
build a regex by simply joining their full absolute tensor paths using
`re.escape()`. Doing so creates unreadable megabytes of strings, and alternating
between different precisions within the same block is highly inefficient at
runtime.

Instead, you **must read and follow the block-level extraction rules** in
[`regex_targeting.md`](regex_targeting.md) to target the entire common parent
module (like a stage or residual block parent) using clean, hardware-friendly
regex filters.
