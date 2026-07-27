# Quantization Exploration Patterns

Empirical patterns observed when compressing `.tflite` models via AI Edge
Quantizer. These heuristics guide the sweep order and help interpret compression
vs. quality results.

## Pattern 1: Dynamic Quantization is often the most practical starting point

Dynamic Quantization (e.g. `recipe.dynamic_wi8_afp32()`) quantizes weights only
(e.g. at 8 bits) but handles the real-time inference activations in float format
automatically at the kernel level.

*   **Pros**: Significant size reduction (usually ~4x for INT8) and memory
    usage, without requiring any calibration dataset. It is strongly recommended
    as the baseline starting point for CPU/GPU targets.
*   **Cons**: Can be slightly slower on some highly optimized NPUs compared to
    static quantization, and the on-the-fly dequantization introduces unique
    computational noise.

**Implication**: Start tuning with dynamic quantization. Only transition to
fully **Static Quantization** if your target hardware explicitly lacks integer
processing capabilities for partial graphs or if NPU latency bounds are
exceedingly strict (and you have representative calibration data available).

## Pattern 2: Validation metrics are a strong signal for evaluating quantization recipes

Tracking the difference between original signals and compressed outputs using
the appropriate mathematical metric (quantization noise prediction) predicts
fidelity drops cleanly.

*   **Layer Sensitivity and Mixed-Precision**: Not all layers respond to
    quantization equally. If aggressive quantization applied to a layer produces
    stable, favorable metrics based on modality (e.g. high SNR, low KL
    divergence), that layer is robust. If the metrics plummet, the layer is
    heavily sensitive. Instead of a single recipe, you can sweep layer-by-layer
    metrics to apply **mixed-precision**: aggressively quantizing robust layers
    and protecting degraded layers in higher precision.
*   **Comparing Algorithms**: If choosing between algorithms (like standard
    Min-Max, GPTQ, etc.), tracking layer-wise metrics allows algorithm
    evaluation offline directly on weights/activations without needing thousands
    of prompt-based evaluations.

**Implication**: Rely heavily on `quantizer.ValidationErrorMetric` outputs
defined in `error_metric_selection.md` to debug why accuracy dropped, and to
identify specifically which layers are fragile.

## Pattern 3: Navigating the Pareto Curve (Mixed-Precision)

Instead of relying on a completely different algorithm like `weight_only` when
accuracy drops, explore the **Pareto Curve** (Quality vs. Size) by mixing bit
depths dynamically.

Start with a baseline `dynamic_wi8_afp32()` and find which sensitive layers to
skip entirely (`no_quantize`). Then, for the non-skipped layers, do not treat it
as simply "8-bit or 4-bit for everything."

*   **Target Profile**: Uses `num_bits: 8` for all non-skipped layers. (High
    Quality)
*   **Size Profile**: Aggressively pushes all non-skipped layers to `num_bits:
    4`. (High Compression)
*   **Balanced Profile**: Identifies which layers are robust to 4-bit by
    comparing the Target and Size profiles' layer-wise SNR. Layers that don't
    degrade are pushed to `num_bits: 4`, leaving moderately sensitive ones at
    `num_bits: 8`.

**Implication**: Explore intermediate points between your baseline (Target) and
most aggressive (Size) bounds by iterating layer-wise precision combinations
(`num_bits: 8` vs `num_bits: 4`) via `op_config` on the non-skipped layers.
Build a true Pareto curve instead of just flipping to weight-only algorithms.

## Pattern 4: Granularity is the biggest scale lever

The more independently each weight group can be represented (specifically the
scaling values mapping integer domains), the less numerical degradation occurs.

*   **TENSORWISE**: One uniform scale/zero-point for an entire tensor. (Maximum
    compression, maximum loss chance).
*   **CHANNELWISE**: A distinct scale per output channel/feature map row.

**Implication**: Start with `CHANNELWISE` granularity (which is the default in
8-bit configs). Only ever drop to `TENSORWISE` if you are fighting for the last
raw megabytes on memory-constrained microcontrollers.

## Pattern 5: Small bit-widths (INT4) amplify scale/zero-point choices

When pushing beyond 4x compression into 8x (e.g., `recipe.dynamic_wi4_afp32()`),
the difference between **Symmetric** (scale only, 0-point anchored) and
**Asymmetric** (scale + 0-point shift) becomes stark. With INT4, we only have 16
literal bins to map weights into. If the original weights are skewed positively,
symmetric mapping wastes half the integer domain representing theoretically
negative weights that don't exist.

**Implication**: At INT8, symmetric is often perfectly acceptable. At INT4,
ensure `symmetric=False` defaults are preserved to capture the weight
distributions faithfully during Weight-Only quantization
(`weight_only_wi4_afp32`).
However, for **Dynamic Range Quantization** (`dynamic_wi4_afp32` with runtime
integer computation), LiteRT integer compute kernels strictly require symmetric
weights (`symmetric: True`). When applying INT4 overrides to dynamic recipes,
invoke `qt.add_dynamic_config(..., num_bits=4)` directly via the python API, as
it automatically enforces supported kernel symmetry and prevents validation
failures. Do NOT try to change bit-depths by passing magic strings into
`algorithm_key`.

## Pattern 6: Metric Mismatches (XNNPACK vs Reference)

You can call `.validate()` using `use_xnnpack=False` (default, mathematically
pure Reference kernels) or `use_xnnpack=True` (optimized Edge CPU kernel
simulation). Often, XNNPACK has highly optimized macro-blocks determining
mathematically fast but approximate outputs (e.g., using specialized
accumulators). If the model validates well on Reference but badly on XNNPACK,
your weights are likely inducing overflow/underflows within those accumulators.


