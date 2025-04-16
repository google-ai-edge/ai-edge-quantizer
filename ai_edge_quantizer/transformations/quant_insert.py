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

"""Apply dequantization transformations to the given op/tensor.

Inserts dequantize node after the given tensor to enable float execution of
the tensor consumer
"""

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import quantize_tensor
from ai_edge_quantizer.transformations import transformation_utils
from ai_edge_litert import schema_py_generated  # pylint: disable=g-direct-tensorflow-import


def insert_quant(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Insert quant op after the given tensor in the subgraph.

  Args:
    transformation_input: input structure that contains all information needed
      for the transformation.

  Returns:
    TransformationInfo:
      op_id: the index where the dequant op is added
      num_ops_added: the total number of ops inserted by this operation, which
        is 1
  """
  quant_op_code_idx = transformation_utils.add_op_code(
      schema_py_generated.BuiltinOperator.QUANTIZE,
      transformation_input.op_codes,
  )

  # create output tensor for the quantize op
  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  new_tensor_id = transformation_utils.add_new_activation_tensor(
      tensor.name + b'_quantized',
      tensor.shape,
      schema_py_generated.TensorType.FLOAT32,
      transformation_input.subgraph,
  )

  # quantize the output tensor
  ## we need a new transformation input because we don't want to modify the
  ## original input
  quantize_tensor.quantize_tensor(
      transformation_utils.TransformationInput(
          new_tensor_id,
          transformation_input.op_codes,
          transformation_input.buffers,
          transformation_input.subgraph,
          transformation_input.producer,
          transformation_input.consumers,
          transformation_input.quant_params,
      )
  )

  # create quantize_op
  quant_op = schema_py_generated.OperatorT()
  quant_op.opcodeIndex = quant_op_code_idx
  quant_op.outputs = [new_tensor_id]
  quant_op.inputs = [transformation_input.tensor_id]

  # update the original consumers of the op to take the dequant op,
  # and find the first consumer of the new tensor
  original_consumers, current_consumers = transformation_input.consumers
  current_first_consumer_id = min(current_consumers)
  original_first_consumer_id = min(original_consumers)
  for consumer_id in current_consumers:
    op = transformation_input.subgraph.operators[consumer_id]
    for input_idx in range(len(op.inputs)):
      if op.inputs[input_idx] == transformation_input.tensor_id:
        op.inputs[input_idx] = new_tensor_id

  # if the output is also an output to the graph, we need to update that as well
  for output_idx, output in enumerate(transformation_input.subgraph.outputs):
    if output == transformation_input.tensor_id:
      transformation_input.subgraph.outputs[output_idx] = new_tensor_id

  # add dequant into the subgraph op list,
  # must insert the op right before it's first consumer
  # in the case of output goes to graph output, we need to ensure the dequant
  # op is inserted after the producer
  original_producer, current_producer = transformation_input.producer
  insertion_op_id = max(current_producer + 1, current_first_consumer_id)
  insertion_op_id_in_original_graph = max(
      original_producer + 1, original_first_consumer_id
  )
  transformation_input.subgraph.operators.insert(insertion_op_id, quant_op)
  return qtyping.TransformationInfo(
      op_id=insertion_op_id_in_original_graph,
      num_ops_added=1,
      output_tensor_id=new_tensor_id,
  )
