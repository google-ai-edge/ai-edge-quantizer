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

"""Duplicate buffer transformation."""

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import transformation_utils


def duplicate_buffer(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Duplicates the buffer of the tensor."""
  tensor_id = transformation_input.tensor_id
  tensor = transformation_input.subgraph.tensors[tensor_id]
  duplicated_buffer_id = transformation_utils.add_new_constant_buffer(
      data=transformation_input.buffers[tensor.buffer].data,
      buffers=transformation_input.buffers,
  )
  tensor.buffer = duplicated_buffer_id

  return qtyping.TransformationInfo(
      op_id=0, num_ops_added=0, output_tensor_id=tensor_id
  )
