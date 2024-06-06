"""Quantizer Algorithm Manager Interface."""

import enum
from ai_edge_quantizer import algorithm_manager_api
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.nonlinear_quantize import float_casting
from ai_edge_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor


_TFLOpName = qtyping.TFLOperationName

_alg_manager_instance = algorithm_manager_api.AlgorithmManagerApi()

# Expose instance functions.
get_quantization_func = _alg_manager_instance.get_quantization_func
get_supported_ops = _alg_manager_instance.get_supported_ops
get_init_qsv_func = _alg_manager_instance.get_init_qsv_func
register_op_quant_config_validation_func = (
    _alg_manager_instance.register_op_quant_config_validation_func
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
moving_average_update_qsv = (
    uniform_quantize_tensor.update_tensor_qsv_moving_average
)

for op_name, materialize_func in zip(
    (
        _TFLOpName.FULLY_CONNECTED,
        _TFLOpName.BATCH_MATMUL,
        _TFLOpName.CONV_2D,
        _TFLOpName.RESHAPE,
        _TFLOpName.AVERAGE_POOL_2D,
        _TFLOpName.EMBEDDING_LOOKUP,
        _TFLOpName.SOFTMAX,
    ),
    (
        naive_min_max_quantize.materialize_fc_conv,
        naive_min_max_quantize.materialize_batch_matmul,
        naive_min_max_quantize.materialize_fc_conv,
        naive_min_max_quantize.materialize_reshape,
        naive_min_max_quantize.materialize_average_pool_2d,
        naive_min_max_quantize.materialize_embedding_lookup,
        naive_min_max_quantize.materialize_softmax,
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
for op_name, materialize_func in zip(
    (
        _TFLOpName.FULLY_CONNECTED,
        _TFLOpName.CONV_2D,
    ),
    (
        float_casting.materialize_fc_conv,
        float_casting.materialize_fc_conv,
    ),
):
  register_quantized_op(
      AlgorithmName.FLOAT_CASTING,
      op_name,
      float_casting.init_qsvs,
      calibration_func=float_casting.calibrate,
      materialize_func=materialize_func,
  )
