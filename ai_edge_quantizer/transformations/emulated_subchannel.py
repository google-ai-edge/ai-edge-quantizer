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

"""Transformation pattern for emulated subchannel quantization."""

from typing import cast
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import transformation_utils
from tensorflow.lite.python import schema_py_generated  # pylint: disable=g-direct-tensorflow-import


def emulated_subchannel(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Emulated subchannel quantization for fully_connected op.

  The input tensor must also be the weight tensor of the fully_connected op.

  after the transformation, the fully connected op will be replaced by:
  reshape -> batch_matmul -> mul -> sum -> add (if bias is present)

  Args:
    transformation_input: input structure that contains all information needed
      for the transformation.

  Returns:
    The transformation info.
  """
  # only apply to a single fully_connected op
  if len(transformation_input.consumers) > 1:
    raise ValueError('Emulated Subchannel transformation only support one op')
  if isinstance(
      transformation_input.quant_params, qtyping.NonLinearQuantParams
  ):
    raise ValueError(
        'Emulated Subchannel transformation only support uniform quantization'
    )
  if (
      transformation_input.op_codes[
          transformation_input.subgraph.operators[
              transformation_input.consumers[0]
          ].opcodeIndex
      ].builtinCode
      != schema_py_generated.BuiltinOperator.FULLY_CONNECTED
  ):
    raise ValueError(
        'Emulated Subchannel transformation only support fully_connected op'
    )
  if transformation_input.producer != -1:
    raise ValueError(
        'Emulated Subchannel transformation only support constant tensor'
    )

  # insert all tne necessary op codes into the model
  reshape_op_code_idx = transformation_utils.add_op_code(
      schema_py_generated.BuiltinOperator.RESHAPE, transformation_input.op_codes
  )
  bmm_op_code_idx = transformation_utils.add_op_code(
      schema_py_generated.BuiltinOperator.BATCH_MATMUL,
      transformation_input.op_codes,
  )
  mul_op_code_idx = transformation_utils.add_op_code(
      schema_py_generated.BuiltinOperator.MUL, transformation_input.op_codes
  )
  sum_op_code_idx = transformation_utils.add_op_code(
      schema_py_generated.BuiltinOperator.SUM, transformation_input.op_codes
  )

  original_fc_op_idx = transformation_input.consumers[0]
  if cast(
      schema_py_generated.FullyConnectedOptionsT,
      transformation_input.subgraph.operators[
          original_fc_op_idx
      ].builtinOptions,
  ).fusedActivationFunction != (
      schema_py_generated.ActivationFunctionType.NONE
  ):
    raise ValueError(
        'Emulated Subchannel transformation only support'
        ' fusedActivationFunction NONE for now'
    )

  weight_tensor = transformation_input.subgraph.tensors[
      transformation_input.tensor_id
  ]

  # modify the weight tensor with the correct quantization parameters
  transformation_input.buffers[weight_tensor.buffer].data = np.frombuffer(
      cast(
          np.ndarray, transformation_input.quant_params.quantized_data
      ).tobytes(),
      dtype=np.uint8,
  )
  weight_tensor.shape = cast(
      np.ndarray, transformation_input.quant_params.quantized_data
  ).shape
  weight_tensor.quantization.scale = np.ones(shape=[1], dtype=np.float32)
  weight_tensor.quantization.zeroPoint = np.zeros(
      shape=[1], dtype=np.int64
  ).flatten()

  # assuming zero point is 0, so no need to add a zero point tensor
  for val in transformation_input.quant_params.zero_point.flatten():
    if val != 0:
      raise ValueError(
          'Emulated Subchannel transformation only support zero point 0 for now'
      )

  scale_tensor_id = transformation_utils.add_new_constant_tensor(
      weight_tensor.name + b'_scale',
      transformation_input.quant_params.scale,
      schema_py_generated.TensorType.FLOAT32,
      transformation_input.subgraph,
      transformation_input.buffers,
  )

  # for fully connected op, the reduce axis is always 1
  reduce_axes_data = np.array([1], dtype=np.int32)
  reduce_axes_tensor_id = transformation_utils.add_new_constant_tensor(
      weight_tensor.name + b'_reduce_axes',
      reduce_axes_data,
      schema_py_generated.TensorType.INT32,
      transformation_input.subgraph,
      transformation_input.buffers,
  )

  # find the input and output tensor of the fully connected op
  activation_input_id = transformation_input.subgraph.operators[
      transformation_input.consumers[0]
  ].inputs[0]
  activation_output_id = transformation_input.subgraph.operators[
      transformation_input.consumers[0]
  ].outputs[0]
  activation_input = transformation_input.subgraph.tensors[activation_input_id]
  activation_output = transformation_input.subgraph.tensors[
      activation_output_id
  ]

  if len(activation_input.shape) != 3:
    raise ValueError(
        'Emulated Subchannel transformation only support 3D input tensor'
    )
  bmm_input_shape = [
      activation_input.shape[0] * activation_input.shape[1],
      weight_tensor.shape[1],
      1,
      weight_tensor.shape[2],
  ]
  intermediate_tensor_shape = [
      activation_input.shape[0] * activation_input.shape[1],
      weight_tensor.shape[1],
      1,
      weight_tensor.shape[3],
  ]
  sum_output_shape = [
      activation_input.shape[0] * activation_input.shape[1],
      1,
      1,
      weight_tensor.shape[3],
  ]

  # create constant tensors for reshape
  reshape1_shape_id = transformation_utils.add_new_constant_tensor(
      activation_output.name + b'_reshape_op1_shape',
      np.array(bmm_input_shape, dtype=np.int32),
      schema_py_generated.TensorType.INT32,
      transformation_input.subgraph,
      transformation_input.buffers,
  )
  reshape2_shape_id = transformation_utils.add_new_constant_tensor(
      activation_output.name + b'_reshape_op2_shape',
      np.array(activation_output.shape, dtype=np.int32),
      schema_py_generated.TensorType.INT32,
      transformation_input.subgraph,
      transformation_input.buffers,
  )

  # create all intermediate tensors
  bmm_input_id = transformation_utils.add_new_activation_tensor(
      activation_output.name + b'_bmm_input',
      bmm_input_shape,
      schema_py_generated.TensorType.FLOAT32,
      transformation_input.subgraph,
  )
  mul_input_id = transformation_utils.add_new_activation_tensor(
      activation_output.name + b'_mul_input',
      intermediate_tensor_shape,
      schema_py_generated.TensorType.FLOAT32,
      transformation_input.subgraph,
  )
  sum_input_id = transformation_utils.add_new_activation_tensor(
      activation_output.name + b'_reduce_sum_input',
      intermediate_tensor_shape,
      schema_py_generated.TensorType.FLOAT32,
      transformation_input.subgraph,
  )
  reshape_op2_input_id = transformation_utils.add_new_activation_tensor(
      activation_output.name + b'_reshape_op2_input',
      sum_output_shape,
      schema_py_generated.TensorType.FLOAT32,
      transformation_input.subgraph,
  )

  # reshape
  reshape_op1 = schema_py_generated.OperatorT()
  reshape_op1.opcodeIndex = reshape_op_code_idx
  reshape_op1_option = schema_py_generated.ReshapeOptionsT()
  reshape_op1_option.newShape = bmm_input_shape
  reshape_op1.inputs = [activation_input_id, reshape1_shape_id]
  reshape_op1.outputs = [bmm_input_id]
  reshape_op1.builtinOptionsType = (
      schema_py_generated.BuiltinOptions.ReshapeOptions
  )  # reshape option index
  reshape_op1.builtinOptions = reshape_op1_option

  # batch_matmul
  bmm_op = schema_py_generated.OperatorT()
  bmm_op.opcodeIndex = bmm_op_code_idx
  bmm_op.inputs = [bmm_input_id, transformation_input.tensor_id]
  bmm_op.outputs = [mul_input_id]
  bmm_op.builtinOptionsType = (
      schema_py_generated.BuiltinOptions.BatchMatMulOptions
  )
  bmm_op.builtinOptions = schema_py_generated.BatchMatMulOptionsT()

  # mul
  mul_op = schema_py_generated.OperatorT()
  mul_op.opcodeIndex = mul_op_code_idx
  mul_option = schema_py_generated.MulOptionsT()
  mul_option.fusedActivationFunction = (
      schema_py_generated.ActivationFunctionType.NONE
  )
  mul_op.inputs = [mul_input_id, scale_tensor_id]
  mul_op.outputs = [sum_input_id]
  mul_op.builtinOptionsType = schema_py_generated.BuiltinOptions.MulOptions
  mul_op.builtinOptions = mul_option

  # sum
  sum_op = schema_py_generated.OperatorT()
  sum_op.opcodeIndex = sum_op_code_idx
  sum_op.inputs = [sum_input_id, reduce_axes_tensor_id]
  sum_op.outputs = [reshape_op2_input_id]
  sum_op.builtinOptionsType = schema_py_generated.BuiltinOptions.ReducerOptions
  sum_op.builtinOptions = schema_py_generated.ReducerOptionsT()
  sum_op.builtinOptions.keepDims = True

  # reshape
  reshape_op2 = schema_py_generated.OperatorT()
  reshape_op2.opcodeIndex = reshape_op_code_idx
  reshape_op2_option = schema_py_generated.ReshapeOptionsT()
  reshape_op2_option.newShape = activation_output.shape
  reshape_op2.inputs = [reshape_op2_input_id, reshape2_shape_id]
  reshape_op2.outputs = [activation_output_id]
  reshape_op2.builtinOptionsType = (
      schema_py_generated.BuiltinOptions.ReshapeOptions
  )
  reshape_op2.builtinOptions = reshape_op2_option

  transformation_input.subgraph.operators.insert(
      original_fc_op_idx, reshape_op1
  )
  transformation_input.subgraph.operators.insert(original_fc_op_idx + 1, bmm_op)
  transformation_input.subgraph.operators.insert(original_fc_op_idx + 2, mul_op)
  transformation_input.subgraph.operators.insert(original_fc_op_idx + 3, sum_op)
  transformation_input.subgraph.operators.insert(
      original_fc_op_idx + 4, reshape_op2
  )

  # if there is a bias tensor, we need an add to process it

  if (
      len(
          transformation_input.subgraph.operators[original_fc_op_idx + 5].inputs
      )
      > 2
      and transformation_input.subgraph.operators[
          original_fc_op_idx + 5
      ].inputs[2]
      != -1
  ):
    add_op_code_idx = transformation_utils.add_op_code(
        schema_py_generated.BuiltinOperator.ADD, transformation_input.op_codes
    )
    reshape_op2_output_id = transformation_utils.add_new_activation_tensor(
        activation_output.name + b'_reshape_op2_output',
        activation_output.shape,
        schema_py_generated.TensorType.FLOAT32,
        transformation_input.subgraph,
    )
    reshape_op2.outputs = [reshape_op2_output_id]
    add_op = schema_py_generated.OperatorT()
    add_op.opcodeIndex = add_op_code_idx
    add_option = schema_py_generated.AddOptionsT()
    add_op.builtinOptionsType = schema_py_generated.BuiltinOptions.AddOptions
    add_op.builtinOptions = add_option
    add_op.inputs = [
        reshape_op2_output_id,
        transformation_input.subgraph.operators[original_fc_op_idx + 5].inputs[
            2
        ],
    ]
    add_op.outputs = [activation_output_id]
    transformation_input.subgraph.operators.insert(
        original_fc_op_idx + 5, add_op
    )
    del transformation_input.subgraph.operators[original_fc_op_idx + 6]
    return qtyping.TransformationInfo(
        original_fc_op_idx, 6, activation_output_id
    )
  else:
    del transformation_input.subgraph.operators[original_fc_op_idx + 5]
    return qtyping.TransformationInfo(
        original_fc_op_idx, 5, activation_output_id
    )
