"""Apply dequantization transformations to the given op/tensor.

Inserts dequantize node after the given tensor to enable float execution of
the tensor consumer
"""

from quantization_toolkit import typing as qtyping
from quantization_toolkit.transformations import quantize_tensor
from tensorflow.lite.python import schema_py_generated  # pylint: disable=g-direct-tensorflow-import


def insert_dequant(
    tensor_id: int,
    op_codes: list[schema_py_generated.OperatorCodeT],
    buffers: list[schema_py_generated.BufferT],
    subgraph: schema_py_generated.SubGraphT,
    producer: int,
    consumers: list[int],
    quant_params: qtyping.UniformQuantParams,
) -> qtyping.TransformationInfo:
  """Insert dequant op after the given tensor in the subgraph.

  So far, this will enforce all the consumer of the provided tensor to take
  float input. This behaviour will change once we support op_id.

  Args:
    tensor_id: the tensor index to insert dequant op after
    op_codes: list of operatorCode in the model, if dequantize op doesn't exist,
      we need to insert the op code into the list
    buffers: list of buffer in the original TFlite model for buffer quantization
    subgraph: flatbuffer subgraph object which the tensor resides.
    producer: op id for the producer of the tensor.
    consumers: op ids for consumers of the new dequant op.
    quant_params: quantization parameters to be applied on the orignal tensor

  Returns:
    TransformationInfo:
      op_id: the index where the dequant op is added
      num_ops_added: the total number of ops inserted by this operation, which
        is 1
  """
  dequant_op_code = schema_py_generated.BuiltinOperator.DEQUANTIZE
  dequant_op_code_idx = len(op_codes)
  for i, op_code in enumerate(op_codes):
    if op_code.builtinCode == dequant_op_code:
      dequant_op_code_idx = i
      break
  if dequant_op_code_idx == len(op_codes):
    dequant_op_code_struct = schema_py_generated.OperatorCodeT()
    dequant_op_code_struct.builtinCode = dequant_op_code
    op_codes.append(dequant_op_code_struct)

  # create output tensor for the dequant op
  tensor = subgraph.tensors[tensor_id]
  new_tensor = schema_py_generated.TensorT()
  new_tensor.shape = tensor.shape
  new_tensor.buffer = 0
  new_tensor.type = schema_py_generated.TensorType.FLOAT32
  new_tensor.name = tensor.name + b'_dequant'

  # new tensor is always appended at the end
  new_tensor_id = len(subgraph.tensors)
  subgraph.tensors.append(new_tensor)

  # create dequantize_op
  dequant_op = schema_py_generated.OperatorT()
  dequant_op.opcodeIndex = dequant_op_code_idx
  dequant_op.outputs = [new_tensor_id]
  dequant_op.inputs = [tensor_id]

  # quantize the source tensor
  quantize_tensor.quantize_tensor(
      tensor_id, op_codes, buffers, subgraph, producer, consumers, quant_params
  )

  # update the original consumers of the op to take the dequant op,
  # and find the first consumer of the new tensor
  first_consumer_id = min(consumers)
  for consumer_id in consumers:
    op = subgraph.operators[consumer_id]
    for input_idx in range(len(op.inputs)):
      if op.inputs[input_idx] == tensor_id:
        op.inputs[input_idx] = new_tensor_id

  # add dequant into the subgraph op list,
  # must insert the op right before it's first consumer
  subgraph.operators.insert(first_consumer_id, dequant_op)
  return qtyping.TransformationInfo(
      op_id=first_consumer_id, num_ops_added=1, output_tensor_id=new_tensor_id
  )
