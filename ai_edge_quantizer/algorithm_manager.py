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

"""Quantizer Algorithm Manager Interface."""

import enum
from ai_edge_quantizer import algorithm_manager_api
from ai_edge_quantizer import default_policy
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.nonlinear_quantize import float_casting
from ai_edge_quantizer.algorithms.uniform_quantize import naive_min_max_quantize

_TFLOpName = qtyping.TFLOperationName

_alg_manager_instance = algorithm_manager_api.AlgorithmManagerApi()

# Expose instance functions.
get_quantization_func = _alg_manager_instance.get_quantization_func
get_supported_ops = _alg_manager_instance.get_supported_ops
get_init_qsv_func = _alg_manager_instance.get_init_qsv_func
register_op_quant_config_validation_func = (
    _alg_manager_instance.register_op_quant_config_validation_func
)
register_config_check_policy_func = (
    _alg_manager_instance.register_config_check_policy
)
register_quantized_op = _alg_manager_instance.register_quantized_op
is_op_registered = _alg_manager_instance.is_op_registered
is_algorithm_registered = _alg_manager_instance.is_algorithm_registered
check_op_quantization_config = (
    _alg_manager_instance.check_op_quantization_config
)


# Quantization algorithms.
class AlgorithmName(str, enum.Enum):
  NO_QUANTIZE = "no_quantize"
  MIN_MAX_UNIFORM_QUANT = naive_min_max_quantize.ALGORITHM_KEY
  FLOAT_CASTING = float_casting.ALGORITHM_KEY


# Register MIN_MAX_UNIFORM_QUANT algorithm.
register_op_quant_config_validation_func(
    AlgorithmName.MIN_MAX_UNIFORM_QUANT,
    naive_min_max_quantize.check_op_quantization_config,
)

# Register a config check policy for MIN_MAX_UNIFORM_QUANT algorithm.
register_config_check_policy_func(
    AlgorithmName.MIN_MAX_UNIFORM_QUANT,
    default_policy.DEFAULT_CONFIG_CHECK_POLICY,
)


for op_name, materialize_func in zip(
    (
        _TFLOpName.INPUT,
        _TFLOpName.OUTPUT,
        _TFLOpName.FULLY_CONNECTED,
        _TFLOpName.BATCH_MATMUL,
        _TFLOpName.CONV_2D,
        _TFLOpName.DEPTHWISE_CONV_2D,
        _TFLOpName.CONV_2D_TRANSPOSE,
        _TFLOpName.RESHAPE,
        _TFLOpName.AVERAGE_POOL_2D,
        _TFLOpName.EMBEDDING_LOOKUP,
        _TFLOpName.SOFTMAX,
        _TFLOpName.TANH,
        _TFLOpName.TRANSPOSE,
        _TFLOpName.GELU,
        _TFLOpName.ADD,
        _TFLOpName.SUB,
        _TFLOpName.MUL,
        _TFLOpName.MEAN,
        _TFLOpName.RSQRT,
        _TFLOpName.CONCATENATION,
        _TFLOpName.STRIDED_SLICE,
        _TFLOpName.SPLIT,
        _TFLOpName.LOGISTIC,  # Sigmoid
    ),
    (
        naive_min_max_quantize.materialize_input,
        naive_min_max_quantize.materialize_output,
        naive_min_max_quantize.materialize_fc_conv,
        naive_min_max_quantize.materialize_batch_matmul,
        naive_min_max_quantize.materialize_fc_conv,
        naive_min_max_quantize.materialize_fc_conv,
        naive_min_max_quantize.materialize_conv2d_transpose,
        naive_min_max_quantize.materialize_reshape,
        naive_min_max_quantize.materialize_average_pool_2d,
        naive_min_max_quantize.materialize_embedding_lookup,
        naive_min_max_quantize.materialize_softmax_and_logistic,
        naive_min_max_quantize.materialize_tanh,
        naive_min_max_quantize.materialize_transpose,
        naive_min_max_quantize.materialize_gelu,
        naive_min_max_quantize.materialize_add,
        naive_min_max_quantize.materialize_sub,
        naive_min_max_quantize.materialize_mul,
        naive_min_max_quantize.materialize_mean,
        naive_min_max_quantize.materialize_rsqrt,
        naive_min_max_quantize.materialize_concatenation,
        naive_min_max_quantize.materialize_strided_slice,
        naive_min_max_quantize.materialize_split,
        naive_min_max_quantize.materialize_softmax_and_logistic,
    ),
):
  register_quantized_op(
      AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      op_name,
      naive_min_max_quantize.init_qsvs,
      calibration_func=naive_min_max_quantize.min_max_calibrate,
      materialize_func=materialize_func,
  )

# Register FLOAT_CASTING algorithm.
register_op_quant_config_validation_func(
    AlgorithmName.FLOAT_CASTING,
    float_casting.check_op_quantization_config,
)

# Register a config check policy for FLOAT_CASTING algorithm.
# TODO: b/353780772 - Replace an empty policy for FLOAT_CASTING algorithm.
register_config_check_policy_func(
    AlgorithmName.FLOAT_CASTING, qtyping.ConfigCheckPolicyDict()
)

for op_name, materialize_func in zip(
    (
        _TFLOpName.FULLY_CONNECTED,
        _TFLOpName.CONV_2D,
        _TFLOpName.DEPTHWISE_CONV_2D,
        _TFLOpName.CONV_2D_TRANSPOSE,
        _TFLOpName.EMBEDDING_LOOKUP,
    ),
    (
        float_casting.materialize_fc_conv,
        float_casting.materialize_fc_conv,
        float_casting.materialize_fc_conv,
        float_casting.materialize_conv2d_transpose,
        float_casting.materialize_embedding_lookup,
    ),
):
  register_quantized_op(
      AlgorithmName.FLOAT_CASTING,
      op_name,
      float_casting.init_qsvs,
      calibration_func=float_casting.calibrate,
      materialize_func=materialize_func,
  )
