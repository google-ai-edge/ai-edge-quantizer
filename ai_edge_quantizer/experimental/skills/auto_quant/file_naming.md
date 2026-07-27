--------------------------------------------------------------------------------

name: aeq-file-naming description: >-

## Provides strict, standardized conventions for naming exported LiteRT models, recipes, masks, and validation reports during Edge Quantization search sweeps, ensuring generated artifacts clearly communicate the underlying precision layout and avoid naming collisions.

# AEQ File Naming Conventions

When exploring the Pareto frontier, an agent routinely generates dozens of
intermediate quantization profiles, outputting `.tflite` models, `.json`
recipes, and visual masks.

A strictly patterned file naming convention prevents output directories from
becoming incomprehensible, prevents agents from accidentally overwriting earlier
sweep iterations with generic names like `model_quantized.tflite`, and
intuitively identifies the precision properties of the model at a glance.

--------------------------------------------------------------------------------

## 1. The Naming Schema

Every generated file belonging to a specific configuration sweep **must inherit
the identical base name** and append suffix extensions identifying the artifact
type.

The base name MUST be constructed using the following structure:
`[execution_method]_[precision]_[scoping]_[algorithm]_[modifier]_[ver]`

### Component Specifications

#### A. Execution Method

Indicates the primary runtime quantization execution method.

*   `dynamic`: Activations evaluated in float on the fly.
*   `static`: Activations explicitly quantized offline using a calibration
    dataset.
*   `weightonly`: Pure storage space optimizations (e.g. decompression back to
    float before execution, common on GPUs).

#### B. Primary Precision

Indicates the dominant bit-depth of the remaining non-skipped channels.

*   `int8`: Weights squashed to 8 bits.
*   `int4`: Weights squashed to 4 bits.
*   `mixed`: Explicit combination of diverse variable precisions (e.g.,
    `mixed_4_8`).

#### C. Layer Scoping

Identifies whether any layers were skipped (protected at float) during the
operation.

*   `baseline`: The default. The entire network received the primary
    algorithm/precision without skips.
*   `selective`: Important boundary layers or heavily degraded dense layers were
    skipped (`no_quantize` injected) or manually pinned to higher precisions to
    bypass degradation.

#### D. Algorithm (Optional)

Identifies advanced algorithm explicitly overridden in the `op_config`.

*   `hadamard`: Forced Hadamard rotation algorithm.
*   `octav`: Forced OCTAV (Optimal Clipping for Tensors And Vectors) algorithm.

#### E. Special Modifiers (Optional)

Identifies advanced experimental topologies explicitly overridden in the
`op_config`.

*   `sym`: Forced symmetric mapping.
*   `asym`: Forced asymmetric mapping using non-zero Zero Points.
*   `tensor`: Downgraded to tensor-wise granularity from the default
    channel-wise granularity.

#### F. Version / Baseline Marker

Indicates the chronological point in the Pareto search loop for each specific
precision phase.

*   `baseline`: Identifies the unconstrained baseline quantization (e.g.,
    `dynamic_int8_baseline.tflite`).
*   `ver{N}`: Identifies the sequential version within each search phase:
    *   **Phase 1 (INT8 Skip Search)**: Each incremental skip iteration saves as
        `dynamic_int8_selective_ver1`, `dynamic_int8_selective_ver2`, etc.
    *   **Phase 2 (Mixed/INT4 Squashing)**: Iterative greedy INT4 squashing
        saves as `dynamic_mixed_4_8_selective_ver1`,
        `dynamic_int4_selective_ver1`, etc.

--------------------------------------------------------------------------------

## 2. Examples of Correct Naming

*   `dynamic_int8_baseline.tflite`
    *   Dynamic, 8-bit, pure baseline configuration on the whole network.
*   `dynamic_int8_selective_ver1.tflite` / `.json` / `_mask.png`
    *   Phase 1, Iteration 1: First candidate `no_quantize` skip layer injected
        to rescue SNR/MSE.
*   `dynamic_int8_selective_ver2.tflite` / `.json` / `_mask.png`
    *   Phase 1, Iteration 2: Second candidate `no_quantize` skip layer added to
        finalize FP32 skip list.
*   `dynamic_mixed_4_8_selective_ver1.tflite`
    *   Phase 2, Iteration 1: Inherits finalized skip list, squashes robust
        blocks to 4-bit while leaving remaining layers at 8-bit.
*   `dynamic_int4_selective_ver1.tflite`
    *   Phase 2, Iteration 2: Inherits finalized skip list, squashes all
        remaining non-skipped layers to 4-bit.

--------------------------------------------------------------------------------

## 3. Required File Extensions

Whenever a target baseline runs through an evaluation loop, you must explicitly
output these four artifacts utilizing the uniform base name:

1.  `{base_name}.tflite` $\rightarrow$ Exported to `model/quantized/`
2.  `{base_name}.json` $\rightarrow$ Exported to `model/quantized/recipes/`
3.  `{base_name}_mask.png` $\rightarrow$ Exported to `results_fig/` (If visual
    segmentation/detection testing).
4.  `{base_name}_validation_metrics.json` $\rightarrow$ Exported to `reports/`
    (Optional: If tracking raw metric output data separated from the primary
    Markdown report).

