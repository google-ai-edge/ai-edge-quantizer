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

"""flatbuffer utils for the Quantizer."""

from typing import Any, Optional, Union

import immutabledict
import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_litert import schema_py_generated  # pylint:disable=g-direct-tensorflow-import
from tensorflow.lite.tools import flatbuffer_utils  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.platform import gfile  # pylint: disable=g-direct-tensorflow-import

_TFLOpName = qtyping.TFLOperationName

TFL_OP_NAME_TO_CODE = immutabledict.immutabledict({
    _TFLOpName.FULLY_CONNECTED: (
        schema_py_generated.BuiltinOperator.FULLY_CONNECTED
    ),
    _TFLOpName.BATCH_MATMUL: schema_py_generated.BuiltinOperator.BATCH_MATMUL,
    _TFLOpName.CONV_2D: schema_py_generated.BuiltinOperator.CONV_2D,
    _TFLOpName.DEPTHWISE_CONV_2D: (
        schema_py_generated.BuiltinOperator.DEPTHWISE_CONV_2D
    ),
    _TFLOpName.CONV_2D_TRANSPOSE: (
        schema_py_generated.BuiltinOperator.TRANSPOSE_CONV
    ),
    _TFLOpName.EMBEDDING_LOOKUP: (
        schema_py_generated.BuiltinOperator.EMBEDDING_LOOKUP
    ),
    _TFLOpName.SOFTMAX: schema_py_generated.BuiltinOperator.SOFTMAX,
    _TFLOpName.AVERAGE_POOL_2D: (
        schema_py_generated.BuiltinOperator.AVERAGE_POOL_2D
    ),
    _TFLOpName.RESHAPE: schema_py_generated.BuiltinOperator.RESHAPE,
    _TFLOpName.TANH: schema_py_generated.BuiltinOperator.TANH,
    _TFLOpName.TRANSPOSE: schema_py_generated.BuiltinOperator.TRANSPOSE,
    _TFLOpName.GELU: schema_py_generated.BuiltinOperator.GELU,
    _TFLOpName.ADD: schema_py_generated.BuiltinOperator.ADD,
    _TFLOpName.SUB: schema_py_generated.BuiltinOperator.SUB,
    _TFLOpName.MUL: schema_py_generated.BuiltinOperator.MUL,
    _TFLOpName.MEAN: schema_py_generated.BuiltinOperator.MEAN,
    _TFLOpName.RSQRT: schema_py_generated.BuiltinOperator.RSQRT,
    _TFLOpName.CONCATENATION: schema_py_generated.BuiltinOperator.CONCATENATION,
    _TFLOpName.STRIDED_SLICE: schema_py_generated.BuiltinOperator.STRIDED_SLICE,
    _TFLOpName.SPLIT: schema_py_generated.BuiltinOperator.SPLIT,
    _TFLOpName.LOGISTIC: schema_py_generated.BuiltinOperator.LOGISTIC,
    _TFLOpName.SLICE: schema_py_generated.BuiltinOperator.SLICE,
    _TFLOpName.SUM: schema_py_generated.BuiltinOperator.SUM,
    _TFLOpName.SELECT_V2: schema_py_generated.BuiltinOperator.SELECT_V2,
})

TFL_OP_CODE_TO_NAME = immutabledict.immutabledict(
    dict((reversed(item) for item in TFL_OP_NAME_TO_CODE.items()))
)

# Quantized dimension for per-channel quantization.
# See https://www.tensorflow.org/lite/performance/quantization_spec.
TFL_OP_TO_WEIGHT_QUANTIZED_DIM = immutabledict.immutabledict({
    _TFLOpName.FULLY_CONNECTED: 0,
    _TFLOpName.DEPTHWISE_CONV_2D: 3,
    _TFLOpName.CONV_2D: 0,
    _TFLOpName.EMBEDDING_LOOKUP: 0,
    _TFLOpName.CONV_2D_TRANSPOSE: 0,
})

NUM_TFL_DATATYPES = 18
TENSOR_CODE_TO_TYPE = {}
for dtype_code in range(NUM_TFL_DATATYPES):
  TENSOR_CODE_TO_TYPE[dtype_code] = flatbuffer_utils.type_to_name(dtype_code)
TENSOR_CODE_TO_TYPE = immutabledict.immutabledict(TENSOR_CODE_TO_TYPE)
TENSOR_TYPE_TO_CODE = immutabledict.immutabledict(
    (reversed(item) for item in TENSOR_CODE_TO_TYPE.items())
)

# Expose functions in tensorflow.lite.tools.flatbuffer_utils
write_model = flatbuffer_utils.write_model


def read_model(tflite_model: Union[str, bytearray]) -> Any:
  """Read and convert the TFLite model into a flatbuffer object.

  Args:
    tflite_model: TFLite model path or bytearray.

  Raises:
    ValueError: Unsupported tflite_model type.

  Returns:
    flatbuffer_model: the flatbuffer_model.
  """
  if isinstance(tflite_model, str):
    return flatbuffer_utils.read_model(tflite_model)
  elif isinstance(tflite_model, bytes) or isinstance(tflite_model, bytearray):
    return flatbuffer_utils.read_model_from_bytearray(tflite_model)
  else:
    raise ValueError(
        "Unsupported tflite_model type: %s" % type(tflite_model).__name__
    )


def get_model_content(tflite_path: str) -> bytes:
  """Get the model content (bytes) from the path.

  Args:
    tflite_path: Path to the .tflite.

  Returns:
    The model bytes.
  """
  with gfile.Open(tflite_path, "rb") as tflite_file:
    return tflite_file.read()


def get_model_buffer(tflite_path: str) -> bytearray:
  """Get the model buffer from the path.

  Args:
    tflite_path: path to the .tflite.

  Returns:
    model_buffer: the model buffer.
  """
  with gfile.Open(tflite_path, "rb") as tflite_file:
    return bytearray(tflite_file.read())


def parse_op_tensors(op: Any, subgraph_tensors: list[Any]) -> list[Any]:
  """Parse the op tensors.

  Args:
    op: the op that need to be parsed.
    subgraph_tensors: list of tensors in the subgraph.

  Returns:
    tensors: list of tensors that are associated with the op.
  """

  tensors = []
  for tensor_idx in list(op.outputs) + list(op.inputs):
    if tensor_idx != -1:
      tensors.append(subgraph_tensors[tensor_idx])
  return tensors


def parse_fc_bmm_conv_tensors(
    op: Any,
    subgraph_tensors: list[Any],
    input_index: int = 0,
    weight_index: int = 1,
    bias_index: int = 2,
    output_index: int = 0,
) -> tuple[Any, Any, Any, Any]:
  """Parse tensors in FullyConnected, BatchMatmul, and Convolutions.

  Args:
    op: the TFLite op, must be fully_connected, batch_matmul, or convolution.
    subgraph_tensors: tensors in the subgraph.
    input_index: index for the input tensor.
    weight_index: index for the weight tensor.
    bias_index: index for the bias tensor.
    output_index: index for the output tensor.

  Returns:
    input_tensor, weight_tensor, bias_tensor, output_tensor
  """

  input_tensor = subgraph_tensors[op.inputs[input_index]]
  weight_tensor = subgraph_tensors[op.inputs[weight_index]]
  bias_tensor = None
  if bias_index < len(op.inputs) and op.inputs[bias_index] != -1:
    bias_tensor = subgraph_tensors[op.inputs[bias_index]]
  output_tensor = subgraph_tensors[op.outputs[output_index]]
  return input_tensor, weight_tensor, bias_tensor, output_tensor


# flatbuffer_model has Any type since tensorflow.lite.tools.flatbuffer_utils
# is not type annotated.
def buffer_to_tensors(flatbuffer_model: Any) -> dict[int, list[Any]]:
  """Get the buffer to tensor map for a tflite model.

  Args:
    flatbuffer_model: the flatbuffer_model.

  Returns:
    buffer_to_tensor_map: key as buffer index, value as list of tensors share
    the buffer
  """
  buffer_to_tensor_map = {}
  for subgraph in flatbuffer_model.subgraphs:
    for op in subgraph.operators:
      for tensor in parse_op_tensors(op, subgraph.tensors):
        if tensor.buffer not in buffer_to_tensor_map:
          buffer_to_tensor_map[tensor.buffer] = []
        buffer_to_tensor_map[tensor.buffer].append(tensor)
  return buffer_to_tensor_map


def get_tensor_name(tensor: Any) -> str:
  """Get the tensor name for a fb tensor.

  Args:
    tensor: tensor in flatbuffer.

  Returns:
    tensor_name: name of the buffer
  """
  return tensor.name.decode("utf-8")


def get_tensor_data(tensor: Any, buffers: list[Any]) -> Optional[np.ndarray]:
  """Get the tensor data.

  Args:
    tensor: tensor in flatbuffer.
    buffers: list of buffers

  Returns:
    tensor_data: data inside the tensor
  """
  tensor_buffer = buffers[tensor.buffer]
  buffer_data = tensor_buffer.data
  if buffer_data is None:
    return None
  data = np.frombuffer(
      buffer_data, dtype=TENSOR_CODE_TO_TYPE[tensor.type].lower()
  )
  data = np.reshape(data, tensor.shape)
  return data


def has_same_quantization(tensor1: Any, tensor2: Any) -> bool:
  """Check if two tensors have the same quantization.

  Args:
    tensor1: tensor in flatbuffer.
    tensor2: tensor in flatbuffer.

  Returns:
    True if two tensors have the same quantization.
  """

  def to_tuple(val):
    if val is None:
      val = []
    return tuple(val)

  same_type = tensor1.type == tensor2.type

  # Return True if both tensors are not quantized.
  if tensor1.quantization is None and tensor2.quantization is None:
    return True
  if tensor1.quantization.scale is None and tensor2.quantization.scale is None:
    return True

  same_scale = to_tuple(tensor1.quantization.scale) == to_tuple(
      tensor2.quantization.scale
  )
  same_zero_point = to_tuple(tensor1.quantization.zeroPoint) == to_tuple(
      tensor2.quantization.zeroPoint
  )
  same_quantized_dimension = (
      tensor1.quantization.quantizedDimension
      == tensor2.quantization.quantizedDimension
  )
  return (
      same_type and same_scale and same_zero_point and same_quantized_dimension
  )


def is_float_model(flatbuffer_model: Any) -> bool:
  """Checks that the model is float and not already quantized."""
  for subgraph in flatbuffer_model.subgraphs:
    for tensor in subgraph.tensors:
      if tensor.quantization is None:
        continue
      if tensor.quantization.scale is not None:
        return False
  return True


def get_subgraph_input_output_operators(
    subgraph: Any,
) -> list[qtyping.IOOperator]:
  """Get the input/output operators for the subgraph.

  Args:
    subgraph: The subgraph object.

  Returns:
    Input and output operators for the subgraph.
  """
  input_op = qtyping.IOOperator(
      inputs=[],
      outputs=subgraph.inputs,
      op_key=qtyping.TFLOperationName.INPUT,
  )
  output_op = qtyping.IOOperator(
      inputs=subgraph.outputs,
      outputs=[],
      op_key=qtyping.TFLOperationName.OUTPUT,
  )
  return [input_op, output_op]
