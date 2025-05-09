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

"""Uniform quantize in tensor level."""

import dataclasses
from typing import Optional, Sequence
import ml_dtypes
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.utils import tfl_flatbuffer_utils


@dataclasses.dataclass(frozen=True)
class IntType:
  num_bits: int
  signed: bool


def get_quantized_range(qtype: IntType) -> tuple[float, float]:
  """Calculates range of the quantized type."""
  if qtype.signed:
    qmax = 2 ** (qtype.num_bits - 1) - 1
    qmin = -(2 ** (qtype.num_bits - 1))
  else:
    qmax = (2**qtype.num_bits) - 1
    qmin = 0
  return float(qmin), float(qmax)


def _round_and_clip(
    tensor: np.ndarray, qtype: IntType, narrow: bool
) -> np.ndarray:
  """Round and clip the tensor to the given type, but don't cast it."""
  qmin, qmax = get_quantized_range(qtype)
  if narrow:
    if qtype.signed:
      return np.clip(
          np.rint(tensor),
          qmin + 1,
          qmax,
      )
    else:
      raise ValueError("Unsigned data type should not have narrow range.")
  else:
    return np.clip(np.rint(tensor), qmin, qmax)


def assign_quantized_type(tensor: np.ndarray, qtype: IntType) -> np.ndarray:
  """Cast the tensor to the quantized type."""
  if qtype.num_bits <= 8:
    qtype = np.int8 if qtype.signed else np.uint8
  elif qtype.num_bits <= 16:
    qtype = np.int16 if qtype.signed else np.uint16
  elif qtype.num_bits <= 32:
    qtype = np.int32 if qtype.signed else np.uint32
  else:
    qtype = np.int64 if qtype.signed else np.uint64
  return tensor.astype(qtype)


def fix_quantization_params_rank(
    tensor_data: np.ndarray,
    quantization_params: qtyping.UniformQuantParams,
) -> qtyping.UniformQuantParams:
  """Fix the rank of quantization parameters (scale/zero points).

  Scale and zero points need to be the same rank as tensor_data to avoid
  ambiguous broadcasting.

  Args:
    tensor_data: The tensor to be quantized.
    quantization_params: The quantization parameters.

  Returns:
    quantization_params with broadcasted scales and zero_points.
  """
  scales, zero_points = (
      quantization_params.scale,
      quantization_params.zero_point,
  )
  if tensor_data.ndim == scales.ndim:
    return quantization_params

  if tensor_data.ndim == 0:
    # Scalar tensor requires scalar scale and zero_point.
    if scales.size != 1 or zero_points.size != 1:
      raise ValueError(
          "Scale and zero_point must contain single element for scalar tensor."
          f" Got scale: {scales}, zero_point: {zero_points}"
      )
    scales = np.array(scales.item())
    zero_points = np.array(zero_points.item())
  else:
    dims = [
        dim
        for dim in range(tensor_data.ndim)
        if dim != quantization_params.quantized_dimension
    ]
    scales = np.expand_dims(scales, axis=dims)
    zero_points = np.expand_dims(zero_points, axis=dims)

  return qtyping.UniformQuantParams(
      scale=scales,
      zero_point=zero_points,
      num_bits=quantization_params.num_bits,
      symmetric=quantization_params.symmetric,
      quantized_dimension=quantization_params.quantized_dimension,
      quantized_data=quantization_params.quantized_data,
  )


def _get_tensor_shape_for_blockwise(
    tensor_shape: Sequence[int], quantized_dim: int, block_size: int
) -> list[int]:
  """Get the tensor shape for blockwise quantization.

  This function splits the quantize dimension of the tensor into blocks and the
  dim/blocks. Hence, min/max of the tensor can be calculated for each block
  using existing functions.

  Args:
    tensor_shape: The original shape of the tensor.
    quantized_dim: The dimension to be quantized blockwise.
    block_size: The size of the block.

  Returns:
    The new tensor shape for calculating scale and zp for blockwise
    quantization.
  """
  new_shape = []
  for index, val in enumerate(tensor_shape):
    if index == quantized_dim:
      new_shape.append(int(val / block_size))
      new_shape.append(block_size)
    else:
      new_shape.append(val)
  return new_shape


def reshape_data_for_blockwise(
    tensor_data: np.ndarray, op_name: qtyping.TFLOperationName, block_size: int
) -> tuple[np.ndarray, int]:
  """Reshapes data for blockwise quantization.

  Args:
    tensor_data: The original tensor data.
    op_name: The name of the TFL op.
    block_size: The size of the block.

  Returns:
    A tuple containing the reshaped tensor data and the new reduce dimension.
  """
  quantized_dim = tfl_flatbuffer_utils.TFL_OP_TO_BLOCKWISE_WEIGHT_QUANTIZED_DIM[
      op_name
  ]
  new_shape = _get_tensor_shape_for_blockwise(
      tensor_data.shape, quantized_dim, block_size
  )
  reshaped_data = tensor_data.reshape(new_shape)
  return reshaped_data, quantized_dim + 1


def _broadcast_scale_zp_for_blockwise(
    tensor_content: np.ndarray,
    quant_params: qtyping.UniformQuantParams,
) -> qtyping.UniformQuantParams:
  """Broadcasts scale and zp for blockwise quantization.

  Args:
    tensor_content: The original tensor data.
    quant_params: The quantization parameters.
      `quant_params.quantized_dimension` must be specified.
      `quant_params.block_size` must be specified and positive.

  Returns:
    The updated quantization parameters with broadcasted scale and zp for
    correct constant quantization.
  """
  if quant_params.quantized_dimension is None:
    raise ValueError("Quantized dimension must be specified.")
  if quant_params.block_size is None or quant_params.block_size <= 0:
    raise ValueError("Block size must be specified and positive.")
  quantized_dim = quant_params.quantized_dimension
  expanded_tensor_shape = _get_tensor_shape_for_blockwise(
      tensor_content.shape, quantized_dim, quant_params.block_size
  )
  expanded_scale = np.reshape(
      np.broadcast_to(
          np.expand_dims(quant_params.scale, quantized_dim + 1),
          expanded_tensor_shape,
      ),
      tensor_content.shape,
  )
  expanded_zp = np.reshape(
      np.broadcast_to(
          np.expand_dims(quant_params.zero_point, quantized_dim + 1),
          expanded_tensor_shape,
      ),
      tensor_content.shape,
  )
  return qtyping.UniformQuantParams(
      scale=expanded_scale,
      zero_point=expanded_zp,
      num_bits=quant_params.num_bits,
      symmetric=quant_params.symmetric,
      quantized_dimension=quantized_dim,
      block_size=quant_params.block_size,
  )


def uniform_quantize(
    tensor_data: np.ndarray,
    quantization_params: qtyping.UniformQuantParams,
    is_blockwise: bool = False,
):
  """Uniform quantize a tensor.

  Args:
    tensor_data: The tensor to be quantized.
    quantization_params: The quantization parameters.
    is_blockwise: Whether the tensor is blockwise quantized.

  Returns:
    The quantized tensor.
  """
  # The reshaping for blockwise quantization is unique hence we do this here
  # to avoid unexpected broadcast behavior downstream.
  if is_blockwise:
    quantization_params = _broadcast_scale_zp_for_blockwise(
        tensor_data, quantization_params
    )

  # quant params in flatbuffer is flattened, expand the rank to be the same
  # as the tensor rank to avoid ambiguous broadcasting.
  quantization_params = fix_quantization_params_rank(
      tensor_data, quantization_params
  )
  _is_valid_quantization_params(tensor_data, quantization_params)
  scales, zero_points = (
      quantization_params.scale,
      quantization_params.zero_point,
  )
  inverse_scales = 1.0 / scales
  # TODO: b/332574603 - support unsigned data type.
  qtype = IntType(quantization_params.num_bits, signed=True)
  # Symmetric means narrow range (e.g., -127 to 127)
  narrow_range = quantization_params.symmetric
  required_dtype = np.signedinteger if qtype.signed else np.unsignedinteger
  if not np.issubdtype(zero_points.dtype, required_dtype):
    raise ValueError(
        f"zero_points need to be {required_dtype}."
        f" But the actual type is {zero_points.dtype}."
    )
  ret = np.multiply(tensor_data, inverse_scales) + zero_points
  ret = _round_and_clip(ret, qtype, narrow_range)
  ret = assign_quantized_type(ret, qtype)
  return ret


def uniform_dequantize(
    tensor_data: np.ndarray,
    quantization_params: qtyping.UniformQuantParams,
):
  """Uniform dequantize a tensor.

  Args:
    tensor_data: The tensor to be dequantized.
    quantization_params: The quantization parameters.

  Returns:
    The dequantized tensor.
  """
  # quant params in flatbuffer is flattened, expand the rank to be the same
  # as the tensor rank to avoid ambiguous broadcasting.
  quantization_params = fix_quantization_params_rank(
      tensor_data, quantization_params
  )
  _is_valid_quantization_params(tensor_data, quantization_params)
  return np.multiply(
      tensor_data - quantization_params.zero_point, quantization_params.scale
  )


def symmetric_quantize_bias_tensor(
    bias_content: np.ndarray,
    input_tensor_quant_params: qtyping.UniformQuantParams,
    weight_tensor_quant_params: qtyping.UniformQuantParams,
) -> qtyping.UniformQuantParams:
  """Quantize bias tensor (symmetrically, i.e., zero_point = 0).

  We quantize bias to a much higher bit width, e.g., int32 for int8 weights. We
  can afford the cost of being symmetric all the time. This configuration fits
  TFL kernel designs.

  Args:
    bias_content: The bias content.
    input_tensor_quant_params: The quantization parameters of input tensor.
    weight_tensor_quant_params: The quantization parameters of weight tensor.

  Returns:
    The quantized bias tensor.
  """
  input_tensor_scale = input_tensor_quant_params.scale
  weight_tensor_scale = weight_tensor_quant_params.scale
  # Bias is always 1D, make sure the scale has 1D shape as well.
  effective_output_scale = np.squeeze(input_tensor_scale * weight_tensor_scale)
  # Squeeze can produce scalar, but we want 1D tensor.
  if not effective_output_scale.shape:
    effective_output_scale = np.expand_dims(effective_output_scale, axis=0)

  # symmetric
  bias_zp = np.zeros_like(effective_output_scale, dtype=np.int32)
  bias_number_bits = 64 if input_tensor_quant_params.num_bits == 16 else 32
  symmetric = True
  quantized_dimension = None if len(effective_output_scale) == 1 else 0
  bias_quant_params = qtyping.UniformQuantParams(
      scale=effective_output_scale,
      zero_point=bias_zp,
      num_bits=bias_number_bits,
      symmetric=symmetric,
      quantized_dimension=quantized_dimension,
  )

  quantized_vars = uniform_quantize(bias_content, bias_quant_params)

  # UniformQuantParams is frozen dataclass, need to recreate.
  return qtyping.UniformQuantParams(
      scale=effective_output_scale,
      zero_point=bias_zp,
      num_bits=bias_number_bits,
      quantized_dimension=quantized_dimension,
      symmetric=symmetric,
      quantized_data=quantized_vars,
  )


def tensor_zp_scale_from_min_max(
    min_value,
    max_value,
    num_bits: int,
    symmetric: bool,
    granularity: qtyping.QuantGranularity,
    clipping_values: Optional[np.ndarray] = None,
):
  """Get zero point and scale from min and max value.

  Args:
    min_value: The minimum value of the tensor (channelwise and blockwise
      supported).
    max_value: The maximum value of the tensor (channelwise and blockwise
      supported).
    num_bits: The number of bits of the tensor.
    symmetric: Whether the tensor is symmetric.
    granularity: The granularity of the tensor.
    clipping_values: Absolute clipping values to apply to the tensor. This will
      clip the tensors to the range [-clipping_values, clipping_values]. This
      should be the same shape as min_value and max_value. If None, no clipping
      will be applied.

  Returns:
    The zero point and scale of the tensor.
  """
  # TODO: b/332574603 - support unsigned data type.
  qtype = IntType(
      num_bits,
      signed=True,
  )
  qmin, qmax = get_quantized_range(qtype)
  min_bound = 1e-4  # 1e-6 precision for int8 and 1e-8 for int16.

  if granularity == qtyping.QuantGranularity.BLOCKWISE:
    # Blockwise quantization uses float16 scale, with 7 bit mantissa,
    # so the maximum representable value is 65280.
    float16_max = np.broadcast_to(np.array(65280), min_value.shape)
    clipping_values = (
        float16_max
        if clipping_values is None
        else np.minimum(clipping_values, float16_max)
    )

  if symmetric:
    bound = np.maximum(np.abs(min_value), np.abs(max_value))
    bound = np.maximum(bound, min_bound)
    if clipping_values is not None:
      bound = np.clip(bound, -clipping_values, clipping_values)
    if not qtype.signed:
      half_q = (qmax - 1) / 2
      scale = bound / half_q
      zp = np.ones_like(scale) * (half_q + 1)
    else:
      scale = bound / qmax
      zp = np.zeros_like(scale, dtype=np.int32)
  else:
    # Include 0 to the range to support zero-padding.
    # See: https://arxiv.org/pdf/1712.05877.pdf
    # This ensures bound_min <= 0 <= bound_max.
    bound_max = np.maximum(max_value, np.zeros_like(max_value))
    bound_min = np.minimum(min_value, np.zeros_like(min_value))
    bound = np.maximum(bound_max - bound_min, min_bound)
    if clipping_values is not None:
      bound = np.clip(bound, -clipping_values, clipping_values)
    scale = bound / (qmax - qmin)
    zp = qmin - bound_min / scale
    zp = np.rint(zp)

  if granularity == qtyping.QuantGranularity.BLOCKWISE:
    # Round the scale values to 7 bit mantissa.
    scale = (
        scale.astype(ml_dtypes.bfloat16).astype(np.float16).astype(np.float32)
    )

  # It's safe to cast zp to qtype without clipping because we can infer
  # qmin <= zp <= qmax from bound_min <= 0 <= bound_max.
  zp = assign_quantized_type(zp, qtype)
  return zp, scale


def _is_valid_quantization_params(
    tensor_data: np.ndarray,
    quantization_params: qtyping.UniformQuantParams,
) -> None:
  """Checks if the quantization parameters are valid.

  A valid quantization params requires:
    1. scale and zero point have the same shape (TFL Runtime requirement).
    2. scale and zero point have the same rank as the tensor content (avoid
    ambiguous broadcasting).

  Args:
    tensor_data: The tensor to be quantized.
    quantization_params: The quantization parameters.

  Returns:
    True if the quantization parameters are valid.
  """
  if quantization_params.scale.shape != quantization_params.zero_point.shape:
    raise ValueError(
        "scale and zero_point must have the same shape. Got"
        f" {quantization_params.scale.shape} and"
        f" {quantization_params.zero_point.shape}"
    )

  tensor_rank = tensor_data.ndim
  scale_rank = quantization_params.scale.ndim
  zero_point_rank = quantization_params.zero_point.ndim
  if (tensor_rank != scale_rank) or (tensor_rank != zero_point_rank):
    raise ValueError(
        f"Ranks of scales ({scale_rank}) and zps"
        f" ({zero_point_rank}) must be the same as the tensor rank"
        f" ({tensor_rank})."
    )
  if (
      quantization_params.block_size != 0
      and tensor_data.shape[quantization_params.quantized_dimension]
      % quantization_params.block_size
      != 0
  ):
    raise ValueError(
        "Tensor dimension must be divisible by block size. Got dimension:"
        f" {tensor_data.shape[quantization_params.quantized_dimension]} and"
        f" block size: {quantization_params.block_size}"
    )
