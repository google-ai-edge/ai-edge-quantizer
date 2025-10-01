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
from ai_edge_litert import schema_py_generated as schema  # pylint:disable=g-direct-tensorflow-import
from tensorflow.lite.tools import flatbuffer_utils  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.platform import gfile  # pylint: disable=g-direct-tensorflow-import

_TFLOpName = qtyping.TFLOperationName

TFL_OP_NAME_TO_CODE = immutabledict.immutabledict({
    _TFLOpName.FULLY_CONNECTED: schema.BuiltinOperator.FULLY_CONNECTED,
    _TFLOpName.BATCH_MATMUL: schema.BuiltinOperator.BATCH_MATMUL,
    _TFLOpName.CONV_2D: schema.BuiltinOperator.CONV_2D,
    _TFLOpName.DEPTHWISE_CONV_2D: schema.BuiltinOperator.DEPTHWISE_CONV_2D,
    _TFLOpName.CONV_2D_TRANSPOSE: schema.BuiltinOperator.TRANSPOSE_CONV,
    _TFLOpName.EMBEDDING_LOOKUP: schema.BuiltinOperator.EMBEDDING_LOOKUP,
    _TFLOpName.SOFTMAX: schema.BuiltinOperator.SOFTMAX,
    _TFLOpName.AVERAGE_POOL_2D: schema.BuiltinOperator.AVERAGE_POOL_2D,
    _TFLOpName.RESHAPE: schema.BuiltinOperator.RESHAPE,
    _TFLOpName.TANH: schema.BuiltinOperator.TANH,
    _TFLOpName.TRANSPOSE: schema.BuiltinOperator.TRANSPOSE,
    _TFLOpName.GELU: schema.BuiltinOperator.GELU,
    _TFLOpName.ADD: schema.BuiltinOperator.ADD,
    _TFLOpName.SUB: schema.BuiltinOperator.SUB,
    _TFLOpName.MUL: schema.BuiltinOperator.MUL,
    _TFLOpName.MEAN: schema.BuiltinOperator.MEAN,
    _TFLOpName.RSQRT: schema.BuiltinOperator.RSQRT,
    _TFLOpName.CONCATENATION: schema.BuiltinOperator.CONCATENATION,
    _TFLOpName.STRIDED_SLICE: schema.BuiltinOperator.STRIDED_SLICE,
    _TFLOpName.SPLIT: schema.BuiltinOperator.SPLIT,
    _TFLOpName.LOGISTIC: schema.BuiltinOperator.LOGISTIC,
    _TFLOpName.SLICE: schema.BuiltinOperator.SLICE,
    _TFLOpName.SUM: schema.BuiltinOperator.SUM,
    _TFLOpName.SELECT: schema.BuiltinOperator.SELECT,
    _TFLOpName.SELECT_V2: schema.BuiltinOperator.SELECT_V2,
    _TFLOpName.STABLEHLO_COMPOSITE: schema.BuiltinOperator.STABLEHLO_COMPOSITE,
    _TFLOpName.DYNAMIC_UPDATE_SLICE: (
        schema.BuiltinOperator.DYNAMIC_UPDATE_SLICE
    ),
    _TFLOpName.PAD: schema.BuiltinOperator.PAD,
    _TFLOpName.SQUARED_DIFFERENCE: schema.BuiltinOperator.SQUARED_DIFFERENCE,
    _TFLOpName.MAX_POOL_2D: schema.BuiltinOperator.MAX_POOL_2D,
    _TFLOpName.RESIZE_BILINEAR: schema.BuiltinOperator.RESIZE_BILINEAR,
    _TFLOpName.GATHER_ND: schema.BuiltinOperator.GATHER_ND,
    _TFLOpName.PACK: schema.BuiltinOperator.PACK,
    _TFLOpName.UNPACK: schema.BuiltinOperator.UNPACK,
    _TFLOpName.DIV: schema.BuiltinOperator.DIV,
    _TFLOpName.BROADCAST_TO: schema.BuiltinOperator.BROADCAST_TO,
    _TFLOpName.SQRT: schema.BuiltinOperator.SQRT,
    _TFLOpName.GATHER: schema.BuiltinOperator.GATHER,
    _TFLOpName.HARD_SWISH: schema.BuiltinOperator.HARD_SWISH,
    _TFLOpName.MAXIMUM: schema.BuiltinOperator.MAXIMUM,
    _TFLOpName.PADV2: schema.BuiltinOperator.PADV2,
    _TFLOpName.REDUCE_MIN: schema.BuiltinOperator.REDUCE_MIN,
    _TFLOpName.EQUAL: schema.BuiltinOperator.EQUAL,
    _TFLOpName.NOT_EQUAL: schema.BuiltinOperator.NOT_EQUAL,
    _TFLOpName.MIRROR_PAD: schema.BuiltinOperator.MIRROR_PAD,
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

TFL_OP_TO_BLOCKWISE_WEIGHT_QUANTIZED_DIM = immutabledict.immutabledict({
    _TFLOpName.FULLY_CONNECTED: 1,
    _TFLOpName.EMBEDDING_LOOKUP: 1,
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
  """Returns a map from buffer id to tensors that use it."""
  buffer_to_tensor_map = {}
  for subgraph in flatbuffer_model.subgraphs:
    for op in subgraph.operators:
      for tensor in parse_op_tensors(op, subgraph.tensors):
        if tensor.buffer not in buffer_to_tensor_map:
          buffer_to_tensor_map[tensor.buffer] = []
        if tensor not in buffer_to_tensor_map[tensor.buffer]:
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
  if tensor.shape is not None:
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


def get_op_side_effect_subgraphs(
    op: Union[schema.Operator, schema.OperatorT],
) -> list[int]:
  """Get indices of any subgraphs invoked as a side effect of the operator.

  Args:
    op: The operator object.

  Returns:
    A list of subgraph indices invoked by the operator. Empty if the operator
    does not invoke any subgraphs.
  """
  if opts := flatbuffer_utils.get_options_as(
      op, schema.StableHLOCompositeOptionsT
  ):
    return [opts.decompositionSubgraphIndex]
  # Can add other nested ops here (control flow ops, etc).
  return []


def get_op_name_by_index(
    flatbuffer_model: Any, subgraph_id: int, op_index: int
) -> str:
  """Get the op name from the flatbuffer model."""
  op = flatbuffer_model.subgraphs[subgraph_id].operators[op_index]
  builtin_code = flatbuffer_model.operatorCodes[op.opcodeIndex].builtinCode
  return TFL_OP_CODE_TO_NAME[builtin_code]
