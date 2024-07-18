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

import collections
from absl.testing import parameterized
from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.utils import min_max_quantize_utils

_OpExecutionMode = qtyping.OpExecutionMode
_QuantTransformation = qtyping.QuantTransformation
_TFLOpName = qtyping.TFLOperationName
_OpQuantConfig = qtyping.OpQuantizationConfig
_TensorQuantConfig = qtyping.TensorQuantizationConfig


# TODO: b/335008966 - increase test coverage.
class MinMaxQuantizeUtilsTest(parameterized.TestCase):

  @parameterized.product(
      execution_mode=[
          _OpExecutionMode.WEIGHT_ONLY,
          _OpExecutionMode.DRQ,
          _OpExecutionMode.SRQ,
      ],
      is_inbounding_tensor=[True, False],
      is_constant=[True, False],
  )
  def test_get_tensor_transformations(
      self, execution_mode, is_inbounding_tensor, is_constant
  ):
    op_quant_config = qtyping.OpQuantizationConfig(
        execution_mode=execution_mode,
    )
    transformations = min_max_quantize_utils.get_tensor_transformations(
        op_quant_config, is_inbounding_tensor, is_constant
    )
    if execution_mode == _OpExecutionMode.WEIGHT_ONLY:
      if is_inbounding_tensor and is_constant:
        self.assertSequenceEqual(
            transformations,
            [
                _QuantTransformation.ADD_DEQUANTIZE,
            ],
        )
      else:
        self.assertSequenceEqual(
            transformations,
            [_QuantTransformation.NO_QUANTIZE],
        )

    if execution_mode == _OpExecutionMode.DRQ:
      if is_inbounding_tensor and is_constant:
        self.assertSequenceEqual(
            transformations, [_QuantTransformation.QUANTIZE_TENSOR]
        )
      else:
        self.assertSequenceEqual(
            transformations,
            [_QuantTransformation.NO_QUANTIZE],
        )

    if execution_mode == _OpExecutionMode.SRQ:
      if is_inbounding_tensor:
        if is_constant:
          self.assertSequenceEqual(
              transformations, [_QuantTransformation.QUANTIZE_TENSOR]
          )
        else:
          self.assertSequenceEqual(
              transformations, [_QuantTransformation.ADD_QUANTIZE]
          )
      else:
        self.assertSequenceEqual(
            transformations, [_QuantTransformation.ADD_DEQUANTIZE]
        )

  @parameterized.parameters((_TFLOpName.FULLY_CONNECTED), (_TFLOpName.CONV_2D))
  def test_check_weight_only_config_succeeds(self, op_name):
    min_max_quantize_utils.check_weight_only_config(op_name)

  @parameterized.parameters((_TFLOpName.RESHAPE), (_TFLOpName.AVERAGE_POOL_2D))
  def test_check_weight_only_config_fails(self, op_name):
    with self.assertRaises(ValueError):
      min_max_quantize_utils.check_weight_only_config(op_name)

  @parameterized.product(
      op_name=(_TFLOpName.FULLY_CONNECTED, _TFLOpName.CONV_2D),
      weight_num_bits=(4, 8),
      weight_channel_wise=(True, False),
  )
  def test_check_drq_config_succeeds(
      self, op_name, weight_num_bits, weight_channel_wise
  ):
    # TODO: b/353365054 - Remove this check after int4 DRQ is supported for
    # conv2d.
    if op_name == _TFLOpName.CONV_2D and weight_num_bits == 4:
      return
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_num_bits, channel_wise=weight_channel_wise
        ),
        execution_mode=_OpExecutionMode.DRQ,
    )
    min_max_quantize_utils.check_drq_config(op_name, op_quant_config)

  @parameterized.parameters((_TFLOpName.RESHAPE), (_TFLOpName.AVERAGE_POOL_2D))
  def test_check_drq_config_unsupported_op_raise_error(self, op_name):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(num_bits=8, channel_wise=True),
        execution_mode=_OpExecutionMode.DRQ,
    )
    error_message = "Unsupported op for dynamic range quantization"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_drq_config(op_name, op_quant_config)

  @parameterized.parameters((_TFLOpName.FULLY_CONNECTED), (_TFLOpName.CONV_2D))
  def test_check_drq_config_wrong_bits_raise_error(self, op_name):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(num_bits=2, channel_wise=False),
        execution_mode=_OpExecutionMode.DRQ,
    )
    error_message = "Only int4/int8 symmetric DRQ is supported for op"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_drq_config(op_name, op_quant_config)

  @parameterized.parameters((_TFLOpName.FULLY_CONNECTED), (_TFLOpName.CONV_2D))
  def test_check_drq_config_asymmetric_weights_raise_error(self, op_name):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8, symmetric=False, channel_wise=False
        ),
        execution_mode=_OpExecutionMode.DRQ,
    )
    error_message = "Only int4/int8 symmetric DRQ is supported for op"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_drq_config(op_name, op_quant_config)

  @parameterized.product(
      op_name=(_TFLOpName.FULLY_CONNECTED, _TFLOpName.CONV_2D),
      act_num_bits=(8, 16),
      weight_num_bits=(4, 8),
      weight_channel_wise=(True, False),
      symmetric_act=(True, False),
  )
  def test_check_srq_config_succeeds(
      self,
      op_name,
      act_num_bits,
      weight_num_bits,
      weight_channel_wise,
      symmetric_act,
  ):
    # Asym int16 activation is not supported.
    if not symmetric_act and act_num_bits == 16:
      return
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=act_num_bits, symmetric=symmetric_act
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_num_bits, channel_wise=weight_channel_wise
        ),
        execution_mode=_OpExecutionMode.SRQ,
    )
    min_max_quantize_utils.check_srq_config(op_name, op_quant_config)

  def test_check_srq_config_unsupported_op_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
        weight_tensor_config=_TensorQuantConfig(num_bits=8, channel_wise=True),
        execution_mode=_OpExecutionMode.SRQ,
    )
    error_message = "Unsupported op for static range quantization"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_srq_config(
          _TFLOpName.CUSTOM_OP, op_quant_config
      )

  def test_check_srq_config_no_act_config_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=None,
        weight_tensor_config=_TensorQuantConfig(num_bits=8, channel_wise=True),
        execution_mode=_OpExecutionMode.SRQ,
    )
    error_message = "activation_tensor_config is required for SRQ"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_srq_config(
          _TFLOpName.FULLY_CONNECTED, op_quant_config
      )

  def test_check_srq_config_wrong_act_bits_config_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=14, symmetric=True
        ),
        weight_tensor_config=_TensorQuantConfig(num_bits=8, channel_wise=True),
        execution_mode=_OpExecutionMode.SRQ,
    )
    error_message = "Only int8/int16 activation SRQ is supported"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_srq_config(
          _TFLOpName.FULLY_CONNECTED, op_quant_config
      )

  def test_check_srq_config_asym_int16_act_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=16, symmetric=False
        ),
        weight_tensor_config=_TensorQuantConfig(num_bits=8, channel_wise=True),
        execution_mode=_OpExecutionMode.SRQ,
    )
    error_message = (
        "Int16 activation SRQ requires symmetric activation quantization"
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_srq_config(
          _TFLOpName.FULLY_CONNECTED, op_quant_config
      )

  def test_check_srq_config_wrong_weight_bits_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=16, symmetric=True
        ),
        weight_tensor_config=_TensorQuantConfig(num_bits=2, channel_wise=True),
        execution_mode=_OpExecutionMode.SRQ,
    )
    error_message = "Currently only int4/int8 symmetric weight are supported"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_srq_config(
          _TFLOpName.FULLY_CONNECTED, op_quant_config
      )

  def test_check_srq_config_asym_weight_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8, symmetric=False, channel_wise=True
        ),
        execution_mode=_OpExecutionMode.SRQ,
    )
    error_message = "Currently only int4/int8 symmetric weight are supported"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_srq_config(
          _TFLOpName.FULLY_CONNECTED, op_quant_config
      )

  @parameterized.product(
      op_name=[
          _TFLOpName.FULLY_CONNECTED,
          _TFLOpName.CONV_2D,
      ],
      activation_tensor_config=[
          None,
          _TensorQuantConfig(num_bits=8, symmetric=False),
          _TensorQuantConfig(num_bits=16, symmetric=True),
      ],
      execution_mode=[
          _OpExecutionMode.WEIGHT_ONLY,
          _OpExecutionMode.DRQ,
          _OpExecutionMode.SRQ,
      ],
  )
  def test_check_supported_int4_config_succeeds(
      self, op_name, activation_tensor_config, execution_mode
  ):
    # Exclude invalid SRQ config.
    if (
        activation_tensor_config is not None
        and execution_mode != _OpExecutionMode.SRQ
    ) or (
        activation_tensor_config is None
        and execution_mode == _OpExecutionMode.SRQ
    ):
      return
    # TODO: b/353365054 - Remove this check after int4 DRQ is supported for
    # conv2d.
    if execution_mode == _OpExecutionMode.DRQ and op_name == _TFLOpName.CONV_2D:
      return
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=activation_tensor_config,
        weight_tensor_config=_TensorQuantConfig(
            num_bits=4, symmetric=True, channel_wise=True
        ),
        execution_mode=execution_mode,
    )
    # Raise error if the config is not supported.
    if execution_mode == _OpExecutionMode.DRQ:
      min_max_quantize_utils.check_drq_config(op_name, op_quant_config)
    elif execution_mode == _OpExecutionMode.WEIGHT_ONLY:
      min_max_quantize_utils.check_weight_only_config(op_name)
    elif execution_mode == _OpExecutionMode.SRQ:
      min_max_quantize_utils.check_srq_config(op_name, op_quant_config)

  @parameterized.product(
      op_name=[_TFLOpName.BATCH_MATMUL],
      activation_tensor_config=[
          None,
          _TensorQuantConfig(num_bits=8, symmetric=False),
          _TensorQuantConfig(num_bits=16, symmetric=True),
      ],
      execution_mode=[
          _OpExecutionMode.DRQ,
          _OpExecutionMode.SRQ,
      ],
  )
  def test_check_unsupported_int4_config_raise_error(
      self, op_name, activation_tensor_config, execution_mode
  ):
    # Exclude invalid SRQ config.
    if (
        activation_tensor_config is not None
        and execution_mode != _OpExecutionMode.SRQ
    ) or (
        activation_tensor_config is None
        and execution_mode == _OpExecutionMode.SRQ
    ):
      return
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=activation_tensor_config,
        weight_tensor_config=_TensorQuantConfig(
            num_bits=4, symmetric=True, channel_wise=True
        ),
        execution_mode=execution_mode,
    )
    with self.assertRaises(ValueError):
      if execution_mode == _OpExecutionMode.DRQ:
        min_max_quantize_utils.check_drq_config(op_name, op_quant_config)
      elif execution_mode == _OpExecutionMode.SRQ:
        min_max_quantize_utils.check_srq_config(op_name, op_quant_config)

  def test_materialize_op_with_output_activation_constraint_fails_for_multiple_output_op(
      self,
  ):
    # Create a mock op with multiple outputs.
    MockOp = collections.namedtuple("MockOp", ["outputs"])
    mock_op_info = qtyping.OpInfo(
        op=MockOp(outputs=[1, 2]),
        op_name=_TFLOpName.SOFTMAX,
        subgraph_op_index=0,
        op_quant_config=_OpQuantConfig(),
    )

    with self.assertRaisesRegex(
        ValueError, "only supports ops with a single output tensor"
    ):
      min_max_quantize_utils.materialize_op_with_output_activation_constraint(
          op_info=mock_op_info,
          graph_info=qtyping.GraphInfo([], []),
          tensor_name_to_qsv={},
          output_activation_constraints={},
      )


if __name__ == "__main__":
  googletest.main()
