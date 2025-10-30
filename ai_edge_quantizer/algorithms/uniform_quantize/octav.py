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

"""Implements the OCTAV quantization."""

import dataclasses
from typing import Any, Optional, Sequence, Union
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import common_quantize
from ai_edge_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from ai_edge_quantizer.algorithms.utils import common_utils

ALGORITHM_KEY = "OCTAV"


def _guess_clipping_with_octav(
    x: np.ndarray,
    bits: int,
    axis: Union[int, Sequence[int]],
    max_iterations: int,
    exponent_divisor: float,
    early_stop: bool = True,
) -> np.ndarray:
  """Returns a tensor of absolute clipping constants for a tensor using OCTAV.

  This method implements equation (6) from the OCTAV paper:
  https://arxiv.org/abs/2206.06501

  Args:
    x: Tensor data to return guesses for.
    bits: Number of bits used during quantization.
    axis: Axis to reduce the tensor along to get the guesses.
    max_iterations: Number of Newton-Raphson iterations to use.
    exponent_divisor: What factor to divide the 4^-bits term by. In the paper,
      3.0 is optimal for signed ints and 12.0 for unsigned ints.
    early_stop: If True, stop the iteration if the guess doesn't change.

  Returns:
    A tensor of shape [num_channels] with clipping constant guesses.
  """
  magnitude = np.abs(x)
  x_reduced = np.mean(x, axis=axis, keepdims=True)
  old_guess = np.zeros(x_reduced.shape)
  guess = np.ones(x_reduced.shape)
  for _ in range(max_iterations):
    if early_stop and np.allclose(guess, old_guess):
      break
    guess_broadcasted = np.broadcast_to(guess, magnitude.shape)
    guess_mask = np.asarray(magnitude < guess_broadcasted, dtype=x.dtype)
    numerator = np.sum(
        magnitude * np.asarray(1.0 - guess_mask), axis=axis, keepdims=True
    )
    denominator1 = (4.0 ** (-bits) / exponent_divisor) * np.sum(
        guess_mask, axis=axis, keepdims=True
    )
    denominator2 = np.sum(1.0 - guess_mask, axis=axis, keepdims=True)
    old_guess = guess
    guess = numerator / (denominator1 + denominator2)

  return guess


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: Optional[np.ndarray] = None,
    tensor_qsv: Optional[dict[str, Any]] = None,
) -> qtyping.UniformQuantParams:
  """Returns the quantization parameters for a tensor.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor. When None, it means the tensor is
      not a weight tensor (e.g. static quantization) so we fallback to using
      naive_min_max_quantize.
    tensor_qsv: A dictionary containing the min/max of the tensor.

  Raises:
    ValueError: If the blockwise quantization is requested.
    ValueError: If the asymmetric quantization is requested.
    ValueError: `tensor_qsv` must contain min/max values, or `tensor_content`
      must be provided so that they can be inferred.
  """
  # Fallback to naive_min_max_quantize.py for non-weight tensors.
  if tensor_content is None:
    return naive_min_max_quantize.get_tensor_quant_params(
        op_info, tensor_quant_config, tensor_content, tensor_qsv
    )

  if not tensor_quant_config.symmetric:
    raise ValueError(
        f"Unsupported symmetry: {tensor_quant_config.symmetric}. OCTAV"
        " supports symmetric quantization only for now."
    )

  if not tensor_qsv:
    # We need min/max to calculate quantization parameters, which
    # should be collected during the calibration process. However,
    # weight-only and DRQ do not require calibration, thus it is
    # possible that this information is missing here. In that case we
    # collect min/max on the spot.
    tensor_min_max = common_quantize.init_tensor_min_max(
        tensor_content,
        op_info,
    )
  else:
    tensor_min_max = tensor_qsv

  if "min" not in tensor_min_max or "max" not in tensor_min_max:
    raise ValueError(
        "min and max must be provided to produce tensor quantization"
        " parameters. Check if the correct calibration results are passed into"
        " the ParamsGenerator."
    )

  quantized_dim = common_utils.get_weight_quantized_dim(
      op_info, tensor_content, tensor_quant_config.granularity
  )
  if uniform_quantize_tensor.is_blockwise(tensor_quant_config.granularity):
    reshaped_data, reduce_dims = (
        uniform_quantize_tensor.reshape_data_for_blockwise(
            tensor_content,
            op_info.op_name,
            tensor_quant_config.granularity,
        )
    )
  else:
    reshaped_data = tensor_content
    reduce_dims = common_utils.get_reduce_dims(
        quantized_dim, tensor_content.shape
    )
  clipping_constants = _guess_clipping_with_octav(
      reshaped_data,
      tensor_quant_config.num_bits,
      reduce_dims,
      max_iterations=10,
      exponent_divisor=3.0 if tensor_quant_config.symmetric else 12.0,
  )
  # We created a new dimension in order to reduce properly for blockwise
  # quantization, so we need to reshape the clipping constants back to the
  # min/max shape for the next step.
  if uniform_quantize_tensor.is_blockwise(tensor_quant_config.granularity):
    clipping_constants = clipping_constants.reshape(tensor_min_max["min"].shape)

  zp, scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
      tensor_min_max["min"],
      tensor_min_max["max"],
      tensor_quant_config.num_bits,
      tensor_quant_config.symmetric,
      tensor_quant_config.granularity,
      clipping_constants,
  )

  quant_params = qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=quantized_dim,
      block_size=uniform_quantize_tensor.extract_block_size_from_granularity(
          tensor_quant_config.granularity
      ),
  )

  quantized_vars = uniform_quantize_tensor.uniform_quantize(
      tensor_content,
      quant_params,
      is_blockwise_quant=uniform_quantize_tensor.is_blockwise(
          tensor_quant_config.granularity
      ),
  )

  return dataclasses.replace(quant_params, quantized_data=quantized_vars)
