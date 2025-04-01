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

"""Utility functions for graph transformations."""

import dataclasses
from typing import Optional, Union

import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_litert import schema_py_generated  # pylint: disable=g-direct-tensorflow-import


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


def add_new_constant_buffer(
    data: np.ndarray,
    buffers: list[schema_py_generated.BufferT],
) -> int:
  """Add a new constant buffer to the model.

  Args:
    data: The data of the new tensor.
    buffers: The buffers of the model.

  Returns:
    The index of the new buffer in the model.
  """
  new_buffer = schema_py_generated.BufferT()
  new_buffer.data = np.frombuffer(data.tobytes(), dtype=np.uint8).flatten()
  new_buffer.offset = 0
  new_buffer.size = 0
  new_buffer_id = len(buffers)
  buffers.append(new_buffer)

  return new_buffer_id


def add_new_constant_tensor(
    tensor_name: str,
    data: np.ndarray,
    tensor_type: schema_py_generated.TensorType,
    subgraph: schema_py_generated.SubGraphT,
    buffers: list[schema_py_generated.BufferT],
    tensor_shape: Optional[list[int]] = None,
) -> int:
  """Add a new constant tensor to the model.

  Args:
    tensor_name: The name of the new tensor.
    data: The data of the new tensor.
    tensor_type: The type of the new tensor.
    subgraph: The subgraph where the new tensor is added.
    buffers: The buffers of the model.
    tensor_shape: The shape of the new tensor. If not provided, the shape of the
      data will be used.

  Returns:
    The index of the new tensor in the subgraph.
  """
  new_buffer_id = add_new_constant_buffer(data, buffers)

  new_tensor = schema_py_generated.TensorT()
  if tensor_shape is None:
    tensor_shape = data.shape
  new_tensor.shape = tensor_shape
  new_tensor.buffer = new_buffer_id
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
