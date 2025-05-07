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

"""Hadamard rotation pattern transformation."""

from flatbuffers import flexbuffers
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import transformation_utils
from ai_edge_litert import schema_py_generated  # pylint: disable=g-direct-tensorflow-import


def _to_flexbuffer(
    hadamard_size: int,
    random_binary_vector: list[np.int8],
) -> bytes:
  """Converts hadamard_size to flexbuffer."""
  fbb = flexbuffers.Builder()
  with fbb.Map():
    fbb.Int('hadamard_size', hadamard_size)
    fbb.VectorFromElements('random_binary_vector', random_binary_vector)
  return fbb.Finish()


def _is_producer_embedding_lookup(
    transformation: transformation_utils.TransformationInput,
) -> bool:
  if transformation.producer == -1:
    return False
  else:
    return (
        transformation.op_codes[
            transformation.subgraph.operators[
                transformation.producer
            ].opcodeIndex
        ].builtinCode
        == schema_py_generated.BuiltinOperator.EMBEDDING_LOOKUP
    )


def _is_fully_connected(
    transformation: transformation_utils.TransformationInput, op_id: int
) -> bool:
  return (
      transformation.op_codes[
          transformation.subgraph.operators[op_id].opcodeIndex
      ].builtinCode
      == schema_py_generated.BuiltinOperator.FULLY_CONNECTED
  )


def _update_embedding_lookup_consumers(
    transformation: transformation_utils.TransformationInput,
    new_tensor_id: int,
) -> bool:
  """Updates the consumers of the embedding lookup op to use the new tensor.

  Args:
    transformation: The transformation input to update the consumers of.
    new_tensor_id: The new tensor id to use as the input to the embedding lookup
      consumers.
  """
  for consumer in transformation.consumers:
    # If the consumer is a graph output and not an op, we can ignore it here
    # since the graph output will be updated later.
    if consumer == -1:
      continue
    consumer_op = transformation.subgraph.operators[consumer]
    # Find the input that was attached to the insertion point, and replace it
    # with the new tensor.
    for i, _ in enumerate(consumer_op.inputs):
      if consumer_op.inputs[i] == transformation.tensor_id:
        consumer_op.inputs[i] = new_tensor_id


def _update_fully_connected_consumers(
    transformation: transformation_utils.TransformationInput,
    new_tensor_id: int,
) -> bool:
  """Updates the fully connected op(s) to use the new tensor.

  Since the new tensor is inserted to the fully_connected's input, we need to
  scan each consumer (in case of multiple fully_connected ops), and update
  the input tensor to the new tensor.

  Args:
    transformation: The transformation input to update the consumers of.
    new_tensor_id: The new tensor id to use as the input to the fully connected
      consumers.

  Returns:
    True if the fully connected op(s) were updated to use the new tensor.
  """
  updated = False
  for consumer in transformation.consumers:
    if _is_fully_connected(transformation, consumer):
      transformation.subgraph.operators[consumer].inputs[0] = new_tensor_id
      updated = True
      continue
  return updated


def insert_hadamard_rotation(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Inserts a custom aeq.hadamard_rotation op on this tensor.

  This function works for float32 tensors only.

  Args:
    transformation_input: The transformation input to insert the custom op on.

  Returns:
    The transformation info of the inserted custom op.

  Raises:
    ValueError: If the transformation input is not a uniform quantization
    transformation.
    ValueError: If the Hadamard quantization params are not set.
    ValueError: If the tensor is not a float32 tensor.
    ValueError: If no supported ops were found as the tensor's producer or
    consumers.
  """
  if not isinstance(
      transformation_input.quant_params, qtyping.UniformQuantParams
  ):
    raise ValueError('Hadamard rotation supports uniform quantization only')

  if transformation_input.quant_params.hadamard is None:
    raise ValueError(
        'Hadamard rotation quantization params are not set but op insertion is'
        ' requested.'
    )

  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  if tensor.type != schema_py_generated.TensorType.FLOAT32:
    raise ValueError(
        'The Hadamard rotation op supports float32 tensors only. Got'
        f' {tensor.type} tensor.'
    )

  # Create new custom op with the current tensor as input and a new activation
  # tensor as output.
  custom_op_code_idx = transformation_utils.add_op_code(
      schema_py_generated.BuiltinOperator.CUSTOM,
      transformation_input.op_codes,
      'aeq.hadamard_rotation',
  )
  custom_op = schema_py_generated.OperatorT()
  custom_op.opcodeIndex = custom_op_code_idx
  custom_op.inputs = [transformation_input.tensor_id]
  custom_op.customOptions = _to_flexbuffer(
      transformation_input.quant_params.hadamard.hadamard_size,
      transformation_input.quant_params.hadamard.random_binary_vector.tolist(),
  )
  new_tensor_id = transformation_utils.add_new_activation_tensor(
      tensor.name + b'_rotated',
      tensor.shapeSignature
      if tensor.shapeSignature is not None
      else tensor.shape,
      schema_py_generated.TensorType.FLOAT32,
      transformation_input.subgraph,
  )
  custom_op.outputs = [new_tensor_id]

  # Update the users of this tensor to use the new tensor.
  if _is_producer_embedding_lookup(transformation_input):
    _update_embedding_lookup_consumers(transformation_input, new_tensor_id)
  elif not _update_fully_connected_consumers(
      transformation_input, new_tensor_id
  ):
    raise ValueError(
        'The Hadamard rotation op supports embedding lookup and fully connected'
        ' ops only, but no such ops were found.'
    )

  # If the tensor is a graph output, we need to replace the tensor with the
  # new tensor.
  for i, output in enumerate(transformation_input.subgraph.outputs):
    if output == transformation_input.tensor_id:
      transformation_input.subgraph.outputs[i] = new_tensor_id

  # Find the actual insertion point. The insertion point should be after the
  # producer op and before the first consumer op. The max() operation ensures
  # that we're not using -1 as the insertion point.
  first_consumer_id = min(transformation_input.consumers)
  op_id = max(transformation_input.producer + 1, first_consumer_id)

  # Insert the custom op.
  transformation_input.subgraph.operators.insert(op_id, custom_op)

  return qtyping.TransformationInfo(
      op_id=op_id,
      num_ops_added=1,
      output_tensor_id=new_tensor_id,
  )
