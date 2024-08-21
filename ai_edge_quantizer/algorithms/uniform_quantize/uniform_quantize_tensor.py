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
import numpy as np
from ai_edge_quantizer import qtyping


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


def uniform_quantize_for_emulated_subchannel(
    tensor_data: np.ndarray,
    quantization_params: qtyping.UniformQuantParams,
    block_size: int,
) -> np.ndarray:
  """Uniform quantize a tensor for emulated subchannel.

  emulation involves reshaping the tensor and quantizing value on a different
  axes. Hence, we use a different quantization function.

  Args:
    tensor_data: The tensor to be quantized.
    quantization_params: The quantization parameters.
    block_size: The block size of the emulated subchannel.

  Returns:
    The quantized tensor.
  """
  scales, zero_points = (
      quantization_params.scale,
      quantization_params.zero_point,
  )
  transposed_and_reshaped_tensor = np.reshape(
      np.transpose(tensor_data, (1, 0)),
      (
          1,
          int(tensor_data.shape[1] / block_size),
          block_size,
          tensor_data.shape[0],
      ),
  )
  inverse_scales = 1.0 / scales
  qtype = IntType(quantization_params.num_bits, signed=True)
  # Symmetric means narrow range (e.g., -127 to 127)
  narrow_range = quantization_params.symmetric
  required_dtype = np.signedinteger if qtype.signed else np.unsignedinteger
  if not np.issubdtype(zero_points.dtype, required_dtype):
    raise ValueError(
        f"zero_points need to be {required_dtype}."
        f" But the actual type is {zero_points.dtype}."
    )
  ret = (
      np.multiply(transposed_and_reshaped_tensor, inverse_scales) + zero_points
  )
  ret = _round_and_clip(ret, qtype, narrow_range)
  ret = assign_quantized_type(ret, qtype)
  return ret


def uniform_quantize(
    tensor_data: np.ndarray,
    quantization_params: qtyping.UniformQuantParams,
):
  """Uniform quantize a tensor.

  Args:
    tensor_data: The tensor to be quantized.
    quantization_params: The quantization parameters.

  Returns:
    The quantized tensor.
  """
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
    min_value, max_value, num_bits: int, symmetric: bool
):
  """Get zero point and scale from min and max value.

  Args:
    min_value: The minimum value of the tensor (channel-wise supported).
    max_value: The maximum value of the tensor (channel-wise supported).
    num_bits: The number of bits of the tensor.
    symmetric: Whether the tensor is symmetric.

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

  if symmetric:
    bound = np.maximum(np.abs(min_value), np.abs(max_value))
    bound = np.maximum(bound, min_bound)
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
    scale = bound / (qmax - qmin)
    zp = qmin - bound_min / scale
    zp = np.rint(zp)

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
