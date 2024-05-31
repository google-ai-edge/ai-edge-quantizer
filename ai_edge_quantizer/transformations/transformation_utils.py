"""Utility functions for graph transformations."""

import dataclasses
from typing import Union

import numpy as np

from ai_edge_quantizer import qtyping
from tensorflow.lite.python import schema_py_generated  # pylint: disable=g-direct-tensorflow-import


@dataclasses.dataclass
class TransformationInput:
  """Standard input for a graph transformation.

  Attributes:
    tensor_id: the tensor index to insert dequant op after
    op_codes: list of operatorCode in the model, if dequantize op doesn't exist,
      we need to insert the op code into the list
    buffers: list of buffer in the original TFlite model for buffer quantization
    subgraph: flatbuffer subgraph object which the tensor resides.
    producer: op id for the producer of the tensor.
    consumers: op ids for consumers of the new dequant op.
    quant_params: quantization parameters to be applied on the orignal tensor
  """

  tensor_id: int
  op_codes: list[schema_py_generated.OperatorCodeT]
  buffers: list[schema_py_generated.BufferT]
  subgraph: schema_py_generated.SubGraphT
  producer: int
  consumers: list[int]
  quant_params: Union[qtyping.UniformQuantParams, qtyping.NonLinearQuantParams]


def add_op_code(
    op_code: schema_py_generated.OperatorCodeT,
    model_op_codes: list[schema_py_generated.OperatorCodeT],
) -> int:
  """Add an op code into a model if it's not present.

  Args:
    op_code: The op code to be added.
    model_op_codes: The op codes of the model.

  Returns:
    The index of the op code in the model.
  """
  for i, model_op_code in enumerate(model_op_codes):
    if model_op_code.builtinCode == op_code:
      return i
  model_op_codes.append(schema_py_generated.OperatorCodeT())
  model_op_codes[-1].builtinCode = op_code
  return len(model_op_codes) - 1


def add_new_constant_tensor(
    tensor_name: str,
    data: np.ndarray,
    tensor_type: schema_py_generated.TensorType,
    subgraph: schema_py_generated.SubGraphT,
    buffers: list[schema_py_generated.BufferT],
) -> int:
  """Add a new constant tensor to the model.

  Args:
    tensor_name: The name of the new tensor.
    data: The data of the new tensor.
    tensor_type: The type of the new tensor.
    subgraph: The subgraph where the new tensor is added.
    buffers: The buffers of the model.

  Returns:
    The index of the new tensor in the subgraph.
  """
  tensor_buffer = schema_py_generated.BufferT()
  tensor_buffer.data = np.frombuffer(data.tobytes(), dtype=np.uint8).flatten()
  tensor_buffer.offset = 0
  tensor_buffer.size = 0
  tensor_buffer_id = len(buffers)
  buffers.append(tensor_buffer)

  new_tensor = schema_py_generated.TensorT()
  new_tensor.shape = data.shape
  new_tensor.buffer = tensor_buffer_id
  new_tensor.type = tensor_type
  new_tensor.name = tensor_name
  new_tensor_id = len(subgraph.tensors)
  subgraph.tensors.append(new_tensor)
  return new_tensor_id


def add_new_activation_tensor(
    tensor_name: str,
    shape: list[int],
    tensor_type: schema_py_generated.TensorType,
    subgraph: schema_py_generated.SubGraphT,
) -> int:
  """Add a new activation tensor to the model.

  Args:
    tensor_name: The name of the new tensor.
    shape: The shape of the new tensor.
    tensor_type: The type of the new tensor.
    subgraph: The subgraph where the new tensor is added.

  Returns:
    The index of the new tensor in the subgraph.
  """
  new_tensor = schema_py_generated.TensorT()
  new_tensor.shape = shape
  new_tensor.type = tensor_type
  new_tensor.name = tensor_name
  new_tensor.buffer = 0
  new_tensor_id = len(subgraph.tensors)
  subgraph.tensors.append(new_tensor)
  return new_tensor_id
