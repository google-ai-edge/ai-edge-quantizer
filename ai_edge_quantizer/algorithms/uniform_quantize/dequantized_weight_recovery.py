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

import dataclasses
from typing import Any, Optional
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from ai_edge_quantizer.algorithms.utils import common_utils

ALGORITHM_KEY = "dequantized_weight_recovery"
_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation
_IntType = uniform_quantize_tensor.IntType


def _validate_recovered_weights(
    original_vals: np.ndarray,
    quant_vals: np.ndarray,
    scale: np.ndarray,
    tol: float = 1e-4,
):
  """Validates if recovered weights (from the quantized values) are close enough to the original ones.

  Args:
    original_vals: Original values before quantization.
    quant_vals: Quantized values.
    scale: Scale used for quantization.
    tol: Tolerance for the difference between original and recovered values.

  Raises:
    RuntimeError: If the maximum difference between original and recovered
    values exceeds the tolerance.
  """
  recovered_vals = quant_vals * scale
  diff = np.abs(recovered_vals - original_vals).flatten()
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


def get_zp_scale_from_dequantized_symmetric_weights(
    dequant_vals: np.ndarray,
    quantized_dimension: Optional[int] = None,
    min_scale: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
  """Calculates scale and zero point from dequantized and symmetric weights.

  Handles both per-tensor and per-channel (axis) quantization.

  Args:
      dequant_vals: The dequantized weight values (numpy array).
      quantized_dimension:  The dimension along which quantization was performed
        (0 or 1), or None for per-tensor quantization.
      min_scale: The minimum allowed scale value.

  Returns:
      A tuple containing:
          - zero_points: Zero points (all zeros for symmetric quantization).
          - scales: Scales (scalar for per-tensor, array for per-channel).

  Raises:
      ValueError: If `quantized_dimension` is not 0, 1, or None.
  """

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
    # Create a broadcasted array for per-channel scales. It should have the same
    # number of dimensions as the input, with 1 in all dimensions except for the
    # quantized dimension, which retains its original size.
    scales = np.empty(
        tuple(
            [
                1
                if i != quantized_dimension
                else dequant_vals.shape[quantized_dimension]
                for i in range(dequant_vals.ndim)
            ]
        )
    )
    for i in range(dequant_vals.shape[quantized_dimension]):
      slices = [slice(None)] * dequant_vals.ndim
      slices[quantized_dimension] = i
      vec = dequant_vals[tuple(slices)]
      scales[tuple(slices)] = _get_scale(vec, min_scale)

  zero_points = np.zeros_like(scales, dtype=np.int32)
  return zero_points, scales


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: Optional[np.ndarray] = None,
    tensor_qsv: Optional[dict[str, Any]] = None,
) -> qtyping.UniformQuantParams:
  """Get the quantization parameters for a tensor.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor.
    tensor_qsv: A dictionary containing the min/max of the tensor.

  Returns:
    The quantization parameters for the tensor.

  Raises:
    ValueError: If the quantization granularity is blockwise, or if the tensor
    is not a symmetric weight tensor.
  """
  # Fallback to naive_min_max_quantize.py for non-weight tensors.
  if tensor_content is None:
    return naive_min_max_quantize.get_tensor_quant_params(
        op_info, tensor_quant_config, tensor_content, tensor_qsv
    )

  if tensor_quant_config.granularity == qtyping.QuantGranularity.BLOCKWISE:
    raise ValueError(
        "Blockwise quantization is not supported for dequantized weight"
        " recovery."
    )
  if not tensor_quant_config.symmetric:
    raise ValueError(
        "Only symmetric weights are supported for dequantized weight recovery."
    )

  quantized_dim = None
  if tensor_quant_config.granularity == qtyping.QuantGranularity.CHANNELWISE:
    quantized_dim = common_utils.get_weight_quantized_dim(
        op_info, tensor_content
    )

  zp, scale = get_zp_scale_from_dequantized_symmetric_weights(
      dequant_vals=tensor_content,
      quantized_dimension=quantized_dim,
  )
  quant_params = qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=quantized_dim,
  )
  quantized_vars = uniform_quantize_tensor.uniform_quantize(
      tensor_content, quant_params
  )
  _validate_recovered_weights(tensor_content, quantized_vars, scale)
  return dataclasses.replace(quant_params, quantized_data=quantized_vars)


def calibrate(
    tfl_op: Any,
    graph_info: qtyping.GraphInfo,
    tensor_content_map: dict[str, np.ndarray],
    inputs_to_ignore: Optional[list[int]] = None,
    outputs_to_ignore: Optional[list[int]] = None,
) -> dict[str, qtyping.QSV]:
  """Collect quantization statistics variable (QSV, e.g., min/max) for the op.

  Args:
    tfl_op: The tfl operation.
    graph_info: Graph information needed to perform quantization for the op.
    tensor_content_map: A map of tensor name to tensor content.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.

  Returns:
    A dictionary with key as tensor name and value as the collected QSV.
  """
  # Reuse the min/max calibration algorithm from naive_min_max_quantize.py since
  # only weights need to be handled differently.
  return naive_min_max_quantize.min_max_calibrate(
      tfl_op,
      graph_info,
      tensor_content_map,
      inputs_to_ignore,
      outputs_to_ignore,
  )


def init_qsvs(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    inputs_to_ignore: Optional[list[int]] = None,
    outputs_to_ignore: Optional[list[int]] = None,
) -> qtyping.QSV:
  """Initialize the QSVs.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.

  Returns:
    QSVs.
  """
  # Reuse the min/max calibration algorithm from naive_min_max_quantize.py since
  # only weights need to be handeled differently.
  return naive_min_max_quantize.init_qsvs(
      op_info, graph_info, inputs_to_ignore, outputs_to_ignore
  )
