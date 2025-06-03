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

import copy
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
    custom_op_name: Optional[str] = None,
) -> int:
  """Add an op code into a model if it's not present.

  Args:
    op_code: The op code to be added.
    model_op_codes: The op codes of the model.
    custom_op_name: The custom string of the op code. If None, the op code will
      be added as a builtin op code.

  Returns:
    The index of the op code in the model.
  """
  if (
      op_code == schema_py_generated.BuiltinOperator.CUSTOM
      and custom_op_name is None
  ):
    raise ValueError('Custom string is required for custom op code.')

  for i, model_op_code in enumerate(model_op_codes):
    # If the model already has the op code, just return the index.
    if model_op_code.builtinCode == op_code:
      if custom_op_name is not None:
        if model_op_code.customCode == custom_op_name:
          return i
      else:
        # Built-in op
        return i

  model_op_codes.append(schema_py_generated.OperatorCodeT())
  model_op_codes[-1].builtinCode = op_code
  if custom_op_name is not None:
    model_op_codes[-1].customCode = custom_op_name
  return len(model_op_codes) - 1


def get_constant_buffer(
    data: np.ndarray,
    buffers: list[schema_py_generated.BufferT],
    force_duplicate_buffer: bool = False,
) -> int:
  """Get the index of the constant buffer that contains the given data.

  creating new buffer if provided data is not found in buffers list.

  Args:
    data: The data of the new tensor.
    buffers: The buffers of the model.
    force_duplicate_buffer: Whether to add a new buffer even if the same buffer
      already exists.

  Returns:
    The index of the new buffer in the model.
  """

  if isinstance(data, np.ndarray):
    # in the case where the data is passed from quantization_params.
    new_data = np.frombuffer(data.tobytes(), dtype=np.uint8).flatten()
  elif isinstance(data, bytes):
    # in the case where the data is coming from duplicating buffers, we need to
    # make a copy of the data to avoid having two buffers pointing to the same
    # data.
    new_data = copy.deepcopy(data)
  else:
    raise ValueError('data passed in must be either np.ndarray or bytes.')
  # TODO: b/417811116 - we should make this more efficient.
  if not force_duplicate_buffer:
    for index, buffer in enumerate(buffers):
      if np.array_equal(buffer.data, new_data):
        return index
  new_buffer = schema_py_generated.BufferT()
  new_buffer.data = new_data
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
    force_duplicate_buffer: bool = False,
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
    force_duplicate_buffer: Whether to add a new buffer even if the same buffer
      already exists.

  Returns:
    The index of the new tensor in the subgraph.
  """
  new_buffer_id = get_constant_buffer(data, buffers, force_duplicate_buffer)

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
  # If there's a dynamic shape, we need to read from the shapeSignature field
  # instead of shape. Shape should contain just 1 for the dynamic dimension but
  # shapeSignature should contain the true shape.
  if -1 in shape:
    new_tensor.shapeSignature = shape
    new_tensor.shape = [1 if i == -1 else i for i in shape]
  else:
    new_tensor.shape = shape
  new_tensor.type = tensor_type
  new_tensor.name = tensor_name
  new_tensor.buffer = 0
  new_tensor_id = len(subgraph.tensors)
  subgraph.tensors.append(new_tensor)
  return new_tensor_id
