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

"""Duplicate tensor transformation."""

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import transformation_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils


def duplicate_tensor(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Duplicates the tensor."""
  tensor_id = transformation_input.tensor_id
  tensor = transformation_input.subgraph.tensors[tensor_id]
  tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
  buffer_data = transformation_input.buffers[tensor.buffer].data
  if buffer_data is None:
    raise ValueError(
        'Duplicate Tensor transformation supports only constant tensors.'
        f' Tensor {tensor_name} is not constant.'
    )

  # Using number of tensors as a unique identifier for the duplicated tensor
  # to avoid name collision when the tensor is duplicated multiple times.
  duplicated_tensor_name = (
      f'{tensor_name}_duplicated_{len(transformation_input.subgraph.tensors)}'
  )
  duplicated_tensor_id = transformation_utils.add_new_constant_tensor(
      tensor_name=duplicated_tensor_name,
      data=buffer_data,
      tensor_type=tensor.type,
      tensor_shape=tensor.shape,
      subgraph=transformation_input.subgraph,
      buffers=transformation_input.buffers,
  )

  # Update the consumers' input tensor id to the duplicated tensor id.
  for consumer in transformation_input.consumers:
    consumer_inputs = transformation_input.subgraph.operators[consumer].inputs
    for i in range(len(consumer_inputs)):
      if consumer_inputs[i] == tensor_id:
        consumer_inputs[i] = duplicated_tensor_id
        break

  return qtyping.TransformationInfo(
      op_id=0, num_ops_added=0, output_tensor_id=duplicated_tensor_id
  )
