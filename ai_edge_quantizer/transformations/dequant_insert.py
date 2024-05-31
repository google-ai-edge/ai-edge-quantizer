"""Apply dequantization transformations to the given op/tensor.

Inserts dequantize node after the given tensor to enable float execution of
the tensor consumer
"""

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import quantize_tensor
from ai_edge_quantizer.transformations import transformation_utils
from tensorflow.lite.python import schema_py_generated  # pylint: disable=g-direct-tensorflow-import


def insert_dequant(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Insert dequant op after the given tensor in the subgraph.

  So far, this will enforce all the consumer of the provided tensor to take
  float input. This behaviour will change once we support op_id.

  Args:
    transformation_input: input structure that contains all information needed
      for the transformation.

  Returns:
    TransformationInfo:
      op_id: the index where the dequant op is added
      num_ops_added: the total number of ops inserted by this operation, which
        is 1
  """
  dequant_op_code_idx = transformation_utils.add_op_code(
      schema_py_generated.BuiltinOperator.DEQUANTIZE,
      transformation_input.op_codes,
  )
  # create output tensor for the dequant op
  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  new_tensor_id = transformation_utils.add_new_activation_tensor(
      tensor.name + b'_dequant',
      tensor.shape,
      schema_py_generated.TensorType.FLOAT32,
      transformation_input.subgraph,
  )

  # create dequantize_op
  dequant_op = schema_py_generated.OperatorT()
  dequant_op.opcodeIndex = dequant_op_code_idx
  dequant_op.outputs = [new_tensor_id]
  dequant_op.inputs = [transformation_input.tensor_id]

  # quantize the source tensor
  quantize_tensor.quantize_tensor(transformation_input)

  # update the original consumers of the op to take the dequant op,
  # and find the first consumer of the new tensor
  first_consumer_id = min(transformation_input.consumers)
  for consumer_id in transformation_input.consumers:
    op = transformation_input.subgraph.operators[consumer_id]
    for input_idx in range(len(op.inputs)):
      if op.inputs[input_idx] == transformation_input.tensor_id:
        op.inputs[input_idx] = new_tensor_id

  # add dequant into the subgraph op list,
  # must insert the op right before it's first consumer
  transformation_input.subgraph.operators.insert(first_consumer_id, dequant_op)
  return qtyping.TransformationInfo(
      op_id=first_consumer_id, num_ops_added=1, output_tensor_id=new_tensor_id
  )
