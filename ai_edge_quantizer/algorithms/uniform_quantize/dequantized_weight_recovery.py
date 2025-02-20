# Copyright 2024 The AI Edge Quantizer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Recovers quantized weights from dequantized weights (often from QAT)."""

from typing import Optional
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor


ALGORITHM_KEY = "dequantized_weight_recovery"
_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation
_IntType = uniform_quantize_tensor.IntType


def _validate_recovered_scale(
    dequant_vals: np.ndarray, scale: np.ndarray, tol: float = 1e-4
):
  """Validates if the recovered quantized values match the dequantized values.

  Args:
      dequant_vals: The dequantized weight values.
      scale: The scale values.
      tol: The tolerance for the difference between the recovered and original
        values.

  Raises:
      RuntimeError: If the maximum difference between the recovered and
        original values exceeds the tolerance.
  """
  quant_vals = np.round(dequant_vals / scale)  # no need to clamp.
  recovered_vals = quant_vals * scale
  diff = np.abs(recovered_vals - dequant_vals).flatten()
  max_diff = diff.max()
  if max_diff > tol:
    raise RuntimeError(
        "Failed to recover the original quantized values from dequantized"
        f" values. Max diff between recovered and original values: {max_diff}"
    )


def _get_scale(arr: np.ndarray, min_scale: float) -> float:
  """Helper function to calculate scale from a 1D array."""
  # Make sure the array includes zero (symmetric quantization).
  arr = np.append(arr, 0)
  unique_vals = np.unique(arr)
  if unique_vals.size > 1:
    diffs = np.diff(unique_vals)
    return float(
        np.maximum(np.min(diffs), min_scale)
    )  # Cast to float to ensure return type consistency
  return min_scale


def get_zp_scale_from_2d_dequantized_symmetric_weights(
    dequant_vals: np.ndarray,
    quantized_dimension: Optional[int] = None,
    min_scale: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
  """Calculates scale and zero point from 2D dequantized, symmetric weights.

  Handles both per-tensor and per-channel (axis) quantization.

  Args:
      dequant_vals: The 2D dequantized weight values (numpy array).
      quantized_dimension:  The dimension along which quantization was performed
        (0 or 1), or None for per-tensor quantization.
      min_scale: The minimum allowed scale value.

  Returns:
      A tuple containing:
          - zero_points: Zero points (all zeros for symmetric quantization).
          - scales: Scales (scalar for per-tensor, array for per-channel).

  Raises:
      ValueError: If `dequant_vals` is not 2D, or if
          `quantized_dimension` is not 0, 1, or None.
  """

  if dequant_vals.ndim != 2:
    raise ValueError(
        f"Only 2D weights are supported. Got {dequant_vals.ndim} dimensions."
    )

  if quantized_dimension not in (0, 1, None):
    raise ValueError(
        f"quantized_dimension must be 0, 1, or None. Got {quantized_dimension}"
    )

  # Use absolute values for symmetric quantization.
  dequant_vals = np.abs(dequant_vals)

  if quantized_dimension is None:
    # Per-tensor quantization: One scale for the entire tensor.
    scales = _get_scale(dequant_vals.flatten(), min_scale)
    scales = np.array([[scales]])

  else:
    # Per-channel quantization: A scale for each slice along the dimension.
    scales = []
    for i in range(dequant_vals.shape[quantized_dimension]):
      if quantized_dimension == 0:
        vec = dequant_vals[i, :]
      else:  # quantized_dimension == 1
        vec = dequant_vals[:, i]
      scales.append(_get_scale(vec, min_scale))

    # Reshape for correct broadcasting.
    scales = (
        np.array(scales).reshape(-1, 1)
        if quantized_dimension == 0
        else np.array(scales).reshape(1, -1)
    )

  zero_points = np.zeros_like(scales, dtype=np.int32)
  _validate_recovered_scale(dequant_vals, scales)
  return zero_points, scales
