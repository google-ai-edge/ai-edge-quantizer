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

from typing import Optional, cast
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import transformation_utils
from ai_edge_litert import schema_py_generated  # pylint: disable=g-direct-tensorflow-import


# TODO: b/335014051 - Support distinguishing INT, FLOAT & UINT, BFLOAT.
def quant_params_to_tflite_type(
    bitwidth: int,
) -> Optional[schema_py_generated.TensorType]:
  """Given specifications from quant param return the corresponding TFLite dtype.

  Args:
    bitwidth: Bit width from UniformQuantParams.

  Returns:
    The corresponding TFLite tensor type.
  """
  if bitwidth == 4:
    return schema_py_generated.TensorType.INT4
  elif bitwidth <= 8:
    return schema_py_generated.TensorType.INT8
  elif bitwidth <= 16:
    return schema_py_generated.TensorType.INT16
  elif bitwidth <= 32:
    return schema_py_generated.TensorType.INT32
  elif bitwidth <= 64:
    return schema_py_generated.TensorType.INT64
  else:
    raise ValueError(f"Unsupported quant params: {bitwidth}")


def nonlinear_quant_params_to_tflite_type(
    bitwidth: int,
) -> Optional[schema_py_generated.TensorType]:
  """Given specifications from quant param return the corresponding tflite dtype.

  Args:
    bitwidth: bitwidth from NonLinearQuantParams

  Returns:
    the corresponding tflite tensortype
  """
  if bitwidth == 16:
    return schema_py_generated.TensorType.FLOAT16
  elif bitwidth == 32:
    return schema_py_generated.TensorType.FLOAT32
  else:
    raise ValueError(f"Unsupported nonlinear params: {bitwidth}")


def _pack_data(bitwidth: int, flattened_data: np.ndarray) -> np.ndarray:
  """Pack the data to the corresponding bit width.

  Currently only support 4 bits. If no packing is needed, the original data is
  returned.

  Args:
    bitwidth: Bit width from NonLinearQuantParams.
    flattened_data: The data to be packed.

  Returns:
    Packed data.
  """
  if bitwidth == 4:
    even_data = flattened_data[::2] & 0x0F
    odd_data = np.left_shift(flattened_data[1::2], 4).astype(np.uint8)
    if odd_data.shape[0] == even_data.shape[0] - 1:
      odd_data = np.pad(odd_data, (0, 1), constant_values=0)
    return np.bitwise_or(even_data, odd_data)
  else:
    return flattened_data


def _perform_channelwise_quantization(
    transformation_input: transformation_utils.TransformationInput,
) -> schema_py_generated.QuantizationParametersT():
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
  flatbuffer_quantization = schema_py_generated.QuantizationParametersT()
  flatbuffer_quantization.scale = list(
      transformation_input.quant_params.scale.flatten().astype(np.float32)
  )  # Flatbuffer requires scale as list[float].
  if transformation_input.quant_params.zero_point is not None:
    flatbuffer_quantization.zeroPoint = list(
        transformation_input.quant_params.zero_point.flatten().astype(np.int64)
    )  # Flatbuffer requires zeroPoint as list[int64]
  if transformation_input.quant_params.quantized_dimension is not None:
    flatbuffer_quantization.quantizedDimension = (
        transformation_input.quant_params.quantized_dimension
    )

  return flatbuffer_quantization


def _perform_blockwise_quantization(
    transformation_input: transformation_utils.TransformationInput,
) -> schema_py_generated.QuantizationParametersT():
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
  flatbuffer_quantization = schema_py_generated.QuantizationParametersT()
  flatbuffer_quantization.detailsType = (
      schema_py_generated.QuantizationDetails.BlockwiseQuantization
  )
  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  blockwise_details = schema_py_generated.BlockwiseQuantizationT()
  scale_tensor_id = transformation_utils.add_new_constant_tensor(
      tensor.name + b"_scale",
      transformation_input.quant_params.scale,
      schema_py_generated.TensorType.FLOAT16,
      transformation_input.subgraph,
      transformation_input.buffers,
  )
  blockwise_details.scales = scale_tensor_id
  blockwise_details.blockSize = transformation_input.quant_params.block_size
  # blockwise quantization allows optional zero point.
  if transformation_input.quant_params.zero_point is not None:
    zero_point_tensor_id = transformation_utils.add_new_constant_tensor(
        tensor.name + b"_zero_point",
        transformation_input.quant_params.zero_point,
        schema_py_generated.TensorType.INT32,
        transformation_input.subgraph,
        transformation_input.buffers,
    )
    blockwise_details.zeroPoints = zero_point_tensor_id
  flatbuffer_quantization.details = blockwise_details
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
  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  # TODO: b/336385820 - Suppport quantize buffer directly when quantized_data
  # is not provided.
  if tensor.buffer:
    if transformation_input.quant_params.quantized_data is not None:
      transformation_input.buffers[tensor.buffer].data = _pack_data(
          transformation_input.quant_params.num_bits,
          np.frombuffer(
              cast(
                  np.ndarray, transformation_input.quant_params.quantized_data
              ).tobytes(),
              dtype=np.uint8,
          ).flatten(),
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
