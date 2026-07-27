<!-- disableFinding(LINE_OVER_80) -->

--------------------------------------------------------------------------------

name: error-metric-selection description: >-

## Guides the agent to dynamically select, apply, and interpret the correct validation error metrics (MSE, SNR, Cosine Similarity, KL Divergence, Median Difference Ratio) for sensitivity sweeps and quantization validation based on the task description and model modality.

# Error Metrics Registry and Selection Guide

This skill serves as the registry of available error metrics to validate model
numerical degradation in the quantizer API (accessible in the enum
`quantizer.ValidationErrorMetric`) and guides you on how to dynamically select
and interpret them during sensitivity sweeps and quantization evaluations.

By default, blindly using Mean Squared Error (MSE) or Signal-to-Noise Ratio
(SNR) for every model modality can hide critical degradation (e.g., probability
drift in LLMs or structural shift in embeddings) or over-report harmless errors.

--------------------------------------------------------------------------------

## 1. Under-the-Hood Calculation Behavior

When you call `qt.validate(error_metrics=[...])`, AEQ runs side-by-side
inference on the float reference model (`ref`) and the quantized target model
(`tgt`) using the same input samples. For each tensor, it executes the
validation functions passing them as: `fn(target_data, reference_data)`

> [!WARNING] **Memory Safety & Workstation Crash Prevention**: Never execute
> multiple `qt.validate()` calls concurrently or in parallel threads/processes.
> Concurrent validations hold multiple active model graphs and large activation
> tensor buffers in RAM, which can exhaust system memory and crash the
> workstation.
> **Always execute validations strictly sequentially** — one recipe validation
> at a time. Evaluating multiple metrics within a single validation call is
> supported.

Understanding the registry formulas guarantees correct thresholding:

1.  **`ValidationErrorMetric.MSE` (Mean Squared Error)**

    *   **Formula**: `mean((tgt - ref)^2)`
    *   **Behavior**: Sensitive to absolute scale and heavily penalized by
        outlier values. Perfect when the absolute numerical deviation directly
        translates to quality loss.

2.  **`ValidationErrorMetric.SNR` (Signal-to-Noise Ratio)**

    *   **Formula**: `mean(ref^2) / (mean((tgt - ref)^2) + epsilon)`
    *   **Scale Conversion**: Evaluated as a linear ratio under the hood. You
        must convert it to Decibels (dB) in your script using: `10.0 *
        np.log10(value)`.
    *   **Behavior**: Scale-invariant and power-normalized. Focuses on the
        relative strength of the signal versus the quantization noise.

3.  **`ValidationErrorMetric.COSINE_SIMILARITY` (Cosine Similarity)**

    *   **Formula**: `dot(tgt, ref) / (norm(tgt) * norm(ref))`
    *   **Behavior**: Ignores magnitude completely and measures only the
        directional/geometric alignment of vectors in high-dimensional space.
        Returns values between `-1.0` and `1.0`.

4.  **`ValidationErrorMetric.KL_DIVERGENCE` (Kullback-Leibler Divergence)**

    *   **Formula**: `sum(ref_clipped * log((ref_clipped + epsilon) /
        (tgt_clipped + epsilon)))` (values clipped to non-negative domains).
    *   **Behavior**: Measures information loss/entropy shift between the target
        distribution `tgt` and reference distribution `ref`. Extremely sensitive
        to relative probability drifts.

5.  **`ValidationErrorMetric.MEDIAN_DIFF_RATIO` (Median Difference Ratio)**

    *   **Formula**: `median(abs(tgt - ref) / (abs(ref) + epsilon))`
    *   **Behavior**: Scale-normalized and highly outlier-resistant. Represents
        typical percentage error per tensor element.

--------------------------------------------------------------------------------

## 2. Modality & Task Selection Matrix

Before running a sensitivity sweep, identify the model's primary task and select
the optimal validation metrics combination:

Modality / Task                               | Primary Metric                     | Secondary / Diagnostic Metric                      | Selection Rationale
:-------------------------------------------- | :--------------------------------- | :------------------------------------------------- | :------------------
**Generative LLMs / Text Generation**         | `KL_DIVERGENCE` (on output logits) | `COSINE_SIMILARITY` (on layers / attention blocks) | Autoregressive generation cascades small logit shifts into incorrect tokens. KL Divergence monitors probability distribution drift. Cosine Similarity tracks semantic embedding/feature rotation.
**Classification (Image / Text / Audio)**     | `KL_DIVERGENCE` (on output logits) | `COSINE_SIMILARITY` or `MSE`                       | Classification evaluates logits transformed by Softmax. KL Divergence directly checks class probability distributions for shifts.
**Embedding Generation (CLIP, Face Rec)**     | `COSINE_SIMILARITY`                | `MSE`                                              | Search, retrieval, and matching pipelines rely exclusively on vector angles (cosine distance). Magnitude changes are irrelevant as embeddings are normalized during inference.
**Computer Vision (Segmentation, Detection)** | `SNR` (converted to dB)            | `MSE`                                              | Dense regression outputs require power-normalized comparison across resolutions. MSE tracks absolute coordinate/pixel drifts.
**Audio / Speech (TTS, Vocoders, STT)**       | `SNR` (converted to dB)            | `MSE`                                              | Waveform and spectrogram generation are highly vulnerable to noise floor artifacts. SNR directly relates to acoustic clarity.
**Regression (Depth, Keypoints, Heatmaps)**   | `MEDIAN_DIFF_RATIO`                | `MSE`                                              | Continuously valued predictions. Median Diff Ratio prevents outlier points from masking typical layer errors.

--------------------------------------------------------------------------------

## 3. Sensitivity Decisions & Threshold Rules

When analyzing validation score lists, flag tensors for protection based on
these guideline values:

*   **Cosine Similarity**: Any tensor falling below `0.99` (or `0.995` for
    embedding models) points to vector rotation. Skip or pin to INT8.
*   **KL Divergence**: Monitor standard outliers. If mean KL stays below `0.01`
    but certain tensors spike to `0.5+`, target those spikes immediately using
    `regex` overrides.
*   **SNR (dB)**: SNR does *not* have a stable fixed threshold; acceptable SNR
    scales based on model architecture and bit-width. **Do not use hardcoded
    decibel thresholds.** Instead, evaluate the *relative distribution*. If the
    model's median SNR is high, but a few outlier layers plunge significantly
    below the median, those relative outliers are your highly degraded tensors
    requiring fallback configurations.
*   **Median Difference Ratio**: Tensors exceeding a ratio of `0.05` (5% typical
    relative median change) represent substantial local drift.

