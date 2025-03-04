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

"""Quantization helpers common to all uniform quantization algorithms.

This file contains quantization helpers common to all uniform quantization
algorithms. The materialize_op functions require algorithm-specific logic to
produce the quantization parameters (e.g. scale, zero point) for each tensor,
which is encapsulated in get_tensor_quant_params_fn. Each algorithm is required
to implement the get_tensor_quant_params_fn with the
qtyping.GetTensorQuantParamsFuncSignature signature.
"""

from typing import Any
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from ai_edge_quantizer.algorithms.utils import common_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation
_OpQuantConstraint = common_utils.OpQuantConstraint
_ComputePrecision = qtyping.ComputePrecision


def check_op_quantization_config(
    op_name: _TFLOpName,
    op_quant_config: qtyping.OpQuantizationConfig,
    config_check_policy: qtyping.ConfigCheckPolicyDict,
) -> None:
  """Checks the op quantization config.

  Args:
    op_name: The name of the op.
    op_quant_config: The quantization config for the op.
    config_check_policy: The policy to check the op quantization config.

  Raises:
    ValueError: If the op quantization config is invalid.
  """
  if op_quant_config.weight_tensor_config is None:
    raise ValueError(
        "Weight tensor quantization is required for min/max uniform"
        " quantization."
    )
  if op_quant_config.weight_tensor_config.dtype != qtyping.TensorDataType.INT:
    raise ValueError(
        "Weights need to have integer type for min/max uniform quantization. If"
        " you wish to perform float casting quantization (e.g., fp16 weight"
        " only), please set algorithm key as 'float_casting'."
    )

  if op_quant_config.min_weight_elements < 0:
    raise ValueError(
        f"min_weight_elements must be non-negative for op: {op_name} with"
        f" config: {op_quant_config}."
    )

  if op_quant_config.compute_precision in [
      _ComputePrecision.INTEGER,
      _ComputePrecision.FLOAT,
  ]:
    # Use policy-based mechanism to validate op.
    common_utils.check_if_valid_op_config(
        op_name, op_quant_config, config_check_policy
    )
  common_utils.check_subchannel_config(op_name, op_quant_config)


def materialize_input(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in the virtual input op."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
  )


def materialize_output(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in the virtual output op."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
  )


def materialize_add(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.add."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
  )


def materialize_sub(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.sub."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
  )


def materialize_mul(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.mul."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
  )


def materialize_softmax_and_logistic(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.softmax and tfl.logistic."""
  # Hard code scales and zp values as they are hard coded in TFL kernels.
  # Softmax:
  #   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/activations.cc#L548
  # Logistic:
  #   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/activations.cc#L421
  output_activation_constraints = {
      8: qtyping.UniformQuantParams(
          num_bits=8,
          quantized_dimension=None,
          scale=np.array(1.0 / 256),
          zero_point=np.array(-128),
          symmetric=False,
      ),
      16: qtyping.UniformQuantParams(
          num_bits=16,
          quantized_dimension=None,
          scale=np.array(1.0 / 32768),
          zero_point=np.array(0),
      ),
  }

  return common_utils.materialize_op_with_output_activation_constraint(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      output_activation_constraints,
      get_tensor_quant_params_fn,
  )


def materialize_batch_matmul(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.batch_matmul."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
  )


def materialize_embedding_lookup(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.embedding_lookup."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      inputs_to_ignore=[0],  # Lookup index does not need to be quantized.
  )


def materialize_reshape(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.reshape."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Shape tensor does not need to be quantized.
  )


def materialize_average_pool_2d(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.average_pool_2d."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
  )


def _materialize_bias_for_conv_ops(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    op_tensor_params: list[qtyping.TensorTransformationParams],
    op_input_index: int = 0,
    op_weight_index: int = 1,
    op_bias_index: int = 2,
):
  """Materializes bias tensors in conv ops by updating `op_tensor_params`.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    op_tensor_params: Partially populated quantization configuration for the
      tensors associated with the op in the order of input, weight, output.
    op_input_index: Index for the input tensor in the op.
    op_weight_index: Index for the weight tensor in the op.
    op_bias_index: Index for the bias tensor in the op.
  """
  _, _, bias_tensor, _ = tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
      op_info.op,
      graph_info.subgraph_tensors,
      op_input_index,
      op_weight_index,
      op_bias_index,
  )
  if bias_tensor is not None:
    bias_quant_params = None
    # Fused bias needs to be quantized for SRQ.
    # Check if SRQ.
    if (
        op_info.op_quant_config.compute_precision == _ComputePrecision.INTEGER
        and op_info.op_quant_config.activation_tensor_config is not None
    ):
      bias_content = tfl_flatbuffer_utils.get_tensor_data(
          bias_tensor,
          graph_info.buffers,
      )
      bias_quant_params = (
          uniform_quantize_tensor.symmetric_quantize_bias_tensor(
              bias_content,
              op_tensor_params[op_input_index].consumers[0].parameters,
              op_tensor_params[op_weight_index].consumers[0].parameters,
          )
      )
    # We only quantize bias under SRQ. Setting is_constant=True for SRQ only
    # to avoid quantize bias for DRQ and weight-only cases.
    is_constant = (
        # Check if SRQ.
        op_info.op_quant_config.compute_precision == _ComputePrecision.INTEGER
        and op_info.op_quant_config.activation_tensor_config is not None
    )
    op_tensor_params[op_bias_index] = (
        common_utils.get_tensor_transformation_params(
            tfl_flatbuffer_utils.get_tensor_name(bias_tensor),
            op_info,
            is_inbounding_tensor=True,
            quant_params=bias_quant_params,
            is_constant=is_constant,
        )
    )


def _are_weights_too_small(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    weight_index: int,
) -> bool:
  """Checks if weights are too small to be quantized."""
  tensor = graph_info.subgraph_tensors[op_info.op.inputs[weight_index]]
  tensor_data = tfl_flatbuffer_utils.get_tensor_data(
      tensor,
      graph_info.buffers,
  )
  return (
      tensor_data is not None
      and np.size(tensor_data) < op_info.op_quant_config.min_weight_elements
  )


def materialize_slice(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.slice."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[
          1,
          2,
      ],  # Begin and size indices do not need to be quantized.
  )


def materialize_select_v2(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.select_v2."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      constraint=_OpQuantConstraint.SAME_AS_OUTPUT_SCALE,
      inputs_to_ignore=[
          0,
      ],  # Condition tensor does not need to be quantized.
  )


def materialize_dynamic_update_slice(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.dynamic_update_slice."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      constraint=_OpQuantConstraint.SAME_AS_OUTPUT_SCALE,
      inputs_to_ignore=[
          2,
      ],  # start_indices do not need to be quantized.
  )


def materialize_sum(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.sum."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Axis index does not need to be quantized.
  )


def materialize_fc_conv(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
    input_index: int = 0,
    weight_index: int = 1,
    bias_index: int = 2,
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in fully_connected, conv_2d and depthwise_conv_2d.

  Args:
    get_tensor_quant_params_fn: A function to get the quantization parameters
      for a tensor.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.
    input_index: Index for the input tensor in the op.
    weight_index: Index for the weight tensor in the op.
    bias_index: Index for the bias tensor in the op.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  ignored_inputs = [bias_index]  # Bias tensor is quantized separately.
  if _are_weights_too_small(op_info, graph_info, weight_index):
    ignored_inputs.append(weight_index)

  op_tensor_params = common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      inputs_to_ignore=ignored_inputs,
  )

  _materialize_bias_for_conv_ops(
      op_info,
      graph_info,
      op_tensor_params,
      op_input_index=input_index,
      op_weight_index=weight_index,
      op_bias_index=bias_index,
  )

  return op_tensor_params


def materialize_conv2d_transpose(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.conv2d_transpose.

  Args:
    get_tensor_quant_params_fn: A function to get the quantization parameters
      for a tensor.
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  ignored_shape_index = 0
  weight_index = 1
  input_index = 2
  bias_index = 3

  ignored_inputs = [
      ignored_shape_index,
      bias_index,  # Bias tensor is quantized separately.
  ]
  if _are_weights_too_small(op_info, graph_info, weight_index):
    ignored_inputs.append(weight_index)

  op_tensor_params = common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      inputs_to_ignore=ignored_inputs,
  )
  if len(op_tensor_params) < 2:
    raise ValueError(
        "Materialize standard op should return at least two tensors for"
        " conv2d_transpose."
    )
  _materialize_bias_for_conv_ops(
      op_info,
      graph_info,
      op_tensor_params,
      op_input_index=input_index,
      op_weight_index=weight_index,
      op_bias_index=bias_index,
  )

  return op_tensor_params


def materialize_tanh(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.tanh."""
  # Hard code scales and zero point values as they are hard coded in:
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/ir/tfl_ops.td#L3430
  output_activation_constraints = {}
  for num_bits in [8, 16]:
    output_activation_constraints[num_bits] = qtyping.UniformQuantParams(
        num_bits=num_bits,
        quantized_dimension=None,
        scale=np.array(1.0 / (1 << (num_bits - 1))),
        zero_point=np.array(0),
        # Activation is always asymmetric for 8 bit and symmetric for 16 bits.
        symmetric=num_bits == 16,
    )
  return common_utils.materialize_op_with_output_activation_constraint(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      output_activation_constraints,
      get_tensor_quant_params_fn,
  )


def materialize_transpose(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.transpose."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1],  # Permutation tensor does not need to be quantized.
  )


def materialize_gelu(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.gelu."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
  )


def materialize_strided_slice(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.strided_slice."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[1, 2, 3],  # Ignore the begin, end, and strides tensors.
  )


def materialize_mean(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.mean."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      inputs_to_ignore=[1],  # Axis tensor does not need to be quantized.
  )


def materialize_rsqrt(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.rsqrt."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
  )


def materialize_concatenation(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.concatenation."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      constraint=_OpQuantConstraint.SAME_AS_OUTPUT_SCALE,
  )


def materialize_split(
    get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature,
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: dict[str, Any],
) -> list[qtyping.TensorTransformationParams]:
  """Materialize tensors in tfl.split."""
  return common_utils.materialize_standard_op(
      op_info,
      graph_info,
      tensor_name_to_qsv,
      get_tensor_quant_params_fn,
      constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
      inputs_to_ignore=[0],  # Split dimension does not need to be quantized.
  )
