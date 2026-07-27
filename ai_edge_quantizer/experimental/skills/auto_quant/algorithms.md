# AEQ Algorithms Registry

This document serves as a registry for all the quantization algorithms offered
in AI Edge Quantizer (AEQ), corresponding to the `AlgorithmName` enum in
`algorithm_manager.py`.

## Supported Algorithms

*   **NO_QUANTIZE**: Skips quantization for the specified operations.
*   **MIN_MAX_UNIFORM_QUANT**: Standard uniform quantization using min/max
    bounds for calibration.
*   **FLOAT_CASTING**: Casts operations to float.
*   **DEQUANTIZED_WEIGHT_RECOVERY**: Reconstructs/recovers weights during
    dequantization to optimize model quality.
*   **OCTAV**: Optimizes activation ranges (Optimal Clipping and Tuning of
    Activation Variables).
*   **HADAMARD_ROTATION**: Implements Hadamard rotation using runtime custom
    ops.
*   **DECOMPOSED_HADAMARD_ROTATION**: Implements Hadamard rotation entirely
    using mathematically equivalent decomposed standard ops.
*   **MSE**: Computes quantization parameters by minimizing Mean Squared Error
    (MSE).
*   **GPTQ**: Specifically tailored post-training quantization method (Accurate
    Post-Training Quantization for Generative Pre-trained Transformers).

When configuring a quantization recipe via `quantizer.py`, specify the algorithm
using `AlgorithmName.<ALGORITHM>`.

