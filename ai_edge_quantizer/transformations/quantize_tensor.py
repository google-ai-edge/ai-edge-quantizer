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

"""quantize a given tensor."""

from typing import Optional

import ml_dtypes
import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import transformation_utils


# TODO: b/335014051 - Support distinguishing INT, FLOAT & UINT, BFLOAT.
def quant_params_to_tflite_type(
    bitwidth: int,
) -> Optional[qtyping.TensorType]:
  """Returns the TFLite dtype for the given bit width.

  Args:
    bitwidth: Bit width from UniformQuantParams.

  Returns:
    The corresponding TFLite tensor type.
  """
  if bitwidth == 4:
    return qtyping.TensorType.INT4
  elif bitwidth <= 8:
    return qtyping.TensorType.INT8
  elif bitwidth <= 16:
    return qtyping.TensorType.INT16
  elif bitwidth <= 32:
    return qtyping.TensorType.INT32
  elif bitwidth <= 64:
    return qtyping.TensorType.INT64
  else:
    raise ValueError(f"Unsupported quant params: {bitwidth}")


def nonlinear_quant_params_to_tflite_type(
    bitwidth: int,
) -> Optional[qtyping.TensorType]:
  """Returns the TFLite dtype for the given bit width.

  Args:
    bitwidth: bitwidth from NonLinearQuantParams

  Returns:
    the corresponding tflite tensortype
  """
  if bitwidth == 16:
    return qtyping.TensorType.FLOAT16
  elif bitwidth == 32:
    return qtyping.TensorType.FLOAT32
  else:
    raise ValueError(f"Unsupported nonlinear params: {bitwidth}")


def _perform_channelwise_quantization(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.QuantizationParametersT:
  """Perform channelwise quantization and fill the quantization parameters.

  Args:
    transformation_input: Input structure that contains all information needed
      for the transformation.

  Returns:
    The quantization parameters.
  """
  assert isinstance(
      transformation_input.quant_params, qtyping.UniformQuantParams
  )
  flatbuffer_quantization = qtyping.QuantizationParametersT()
  flatbuffer_quantization.scale = np.ravel(
      transformation_input.quant_params.scale
  ).astype(np.float32, copy=False)
  if transformation_input.quant_params.zero_point is not None:
    flatbuffer_quantization.zeroPoint = np.ravel(
        transformation_input.quant_params.zero_point
    ).astype(np.int64, copy=False)
  if transformation_input.quant_params.quantized_dimension is not None:
    flatbuffer_quantization.quantizedDimension = (
        transformation_input.quant_params.quantized_dimension
    )

  return flatbuffer_quantization


def _perform_blockwise_quantization(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.QuantizationParametersT:
  """Perform blockwise quantization and fill the quantization parameters.

  Args:
    transformation_input: Input structure that contains all information needed
      for the transformation.

  Returns:
    The quantization parameters.
  """
  assert isinstance(
      transformation_input.quant_params, qtyping.UniformQuantParams
  )
  flatbuffer_quantization = qtyping.QuantizationParametersT()
  flatbuffer_quantization.detailsType = (
      qtyping.QuantizationDetails.BlockwiseQuantization
  )
  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  blockwise_details = qtyping.BlockwiseQuantizationT()
  # Downcast and round the scale to fp16 with 7 bit mantissa.
  scale_tensor_id = transformation_utils.add_new_constant_tensor(
      tensor.name + b"_scales",
      transformation_input.quant_params.scale.astype(ml_dtypes.bfloat16).astype(
          np.float16
      ),
      qtyping.TensorType.FLOAT16,
      transformation_input.subgraph,
      transformation_input.buffers,
  )
  blockwise_details.scales = scale_tensor_id
  # Blockwise quantization does not support zero point yet, so this points to
  # a -1 buffer index.
  # TODO: b/404909258 - Add optional zero point to blockwise quantization.
  blockwise_details.zeroPoints = -1
  blockwise_details.blockSize = transformation_input.quant_params.block_size
  flatbuffer_quantization.details = blockwise_details
  # TODO: b/443830202 - Hardcoding to 0 for now.
  flatbuffer_quantization.quantizedDimension = 0
  return flatbuffer_quantization


def quantize_tensor(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Quantize the tensor at the tensor_id in the given subgraph.

  Args:
    transformation_input: Input structure that contains all information needed
      for the transformation.

  Returns:
    TransformationInfo:
      op_id: The producer index for tensor.
      num_ops_added: The total number of ops inserted by this operation, which
        is 0.
  """
  tensor: qtyping.TensorT = transformation_input.subgraph.tensors[
      transformation_input.tensor_id
  ]
  # TODO: b/336385820 - Suppport quantize buffer directly when quantized_data
  # is not provided.
  if tensor.buffer:
    if transformation_input.quant_params.quantized_data is not None:
      transformation_input.buffers[tensor.buffer].data = (
          transformation_utils.pack_data(
              transformation_input.quant_params.num_bits,
              np.ravel(
                  np.asarray(transformation_input.quant_params.quantized_data)
              ).view(np.uint8),
          )
      )

  if isinstance(transformation_input.quant_params, qtyping.UniformQuantParams):
    if transformation_input.quant_params.block_size == 0:
      flatbuffer_quantization = _perform_channelwise_quantization(
          transformation_input
      )
    else:
      flatbuffer_quantization = _perform_blockwise_quantization(
          transformation_input
      )
    tensor.quantization = flatbuffer_quantization
    tensor.type = quant_params_to_tflite_type(
        transformation_input.quant_params.num_bits
    )
  if isinstance(
      transformation_input.quant_params, qtyping.NonLinearQuantParams
  ):
    tensor.type = nonlinear_quant_params_to_tflite_type(
        transformation_input.quant_params.num_bits
    )

  return qtyping.TransformationInfo(
      0, num_ops_added=0, output_tensor_id=transformation_input.tensor_id
  )
