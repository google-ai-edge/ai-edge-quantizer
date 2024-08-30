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
from ai_edge_quantizer import default_policy
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.utils import min_max_quantize_utils

_ComputePrecision = qtyping.ComputePrecision
_QuantTransformation = qtyping.QuantTransformation
_TFLOpName = qtyping.TFLOperationName
_OpQuantConfig = qtyping.OpQuantizationConfig
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_DEFAULT_CONFIG_CHECK_POLICY = default_policy.DEFAULT_CONFIG_CHECK_POLICY


# TODO: b/335008966 - increase test coverage.
class MinMaxQuantizeUtilsTest(parameterized.TestCase):

  @parameterized.product(
      test_case=[
          # Tuple holds computation precision, whether to use SRQ and whether
          # to use explicit dequantize.
          (_ComputePrecision.FLOAT, False, True),  # WEIGHT_ONLY.
          (_ComputePrecision.INTEGER, False, False),  # DRQ.
          (_ComputePrecision.INTEGER, True, False),  # SRQ.
      ],
      is_inbounding_tensor=[True, False],
      is_constant=[True, False],
  )
  def test_get_tensor_transformations(
      self, test_case, is_inbounding_tensor, is_constant
  ):
    compute_precision, is_srq, explicit_dequantize = test_case
    weight_tensor_config = _TensorQuantConfig(num_bits=8)
    op_quant_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=weight_tensor_config if is_srq else None,
        compute_precision=compute_precision,
        explicit_dequantize=explicit_dequantize,
    )
    transformations = min_max_quantize_utils.get_tensor_transformations(
        op_quant_config, is_inbounding_tensor, is_constant
    )
    # Check if WEIGHT_ONLY.
    if (
        compute_precision == _ComputePrecision.FLOAT
        and op_quant_config.explicit_dequantize
    ):
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

    # Check if DRQ.
    if compute_precision == _ComputePrecision.INTEGER and not is_srq:
      if is_inbounding_tensor and is_constant:
        self.assertSequenceEqual(
            transformations, [_QuantTransformation.QUANTIZE_TENSOR]
        )
      else:
        self.assertSequenceEqual(
            transformations,
            [_QuantTransformation.NO_QUANTIZE],
        )

    # Check if SRQ.
    if compute_precision == _ComputePrecision.INTEGER and is_srq:
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
    self.assertIn(op_name, _DEFAULT_CONFIG_CHECK_POLICY.keys())

  @parameterized.parameters((_TFLOpName.RESHAPE), (_TFLOpName.AVERAGE_POOL_2D))
  def test_check_weight_only_config_raises_when_invalid_config(self, op_name):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
        ),
        compute_precision=_ComputePrecision.FLOAT,
    )
    error_message = (
        f"Quantization config for op: {op_name} with config:"
        f" {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  @parameterized.product(
      op_name=(_TFLOpName.FULLY_CONNECTED, _TFLOpName.CONV_2D),
      weight_num_bits=(4, 8),
      granularity=(
          qtyping.QuantGranularity.TENSORWISE,
          qtyping.QuantGranularity.CHANNELWISE,
      ),
  )
  def test_check_drq_config_succeeds(
      self, op_name, weight_num_bits, granularity
  ):
    # TODO: b/353365054 - Remove this check after int4 DRQ is supported for
    # conv2d.
    if op_name == _TFLOpName.CONV_2D and weight_num_bits == 4:
      return
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_num_bits,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # DRQ.
    )
    min_max_quantize_utils.check_if_valid_op_config(
        op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
    )

  @parameterized.parameters((_TFLOpName.RESHAPE), (_TFLOpName.AVERAGE_POOL_2D))
  def test_check_drq_config_unsupported_op_raise_error(self, op_name):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # DRQ.
    )
    error_message = (
        f"Quantization config for op: {op_name} with config:"
        f" {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  @parameterized.parameters((_TFLOpName.FULLY_CONNECTED), (_TFLOpName.CONV_2D))
  def test_check_drq_config_wrong_bits_raise_error(self, op_name):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=2,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # DRQ.
    )
    error_message = (
        f"Quantization config for op: {op_name} with config:"
        f" {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  @parameterized.parameters((_TFLOpName.FULLY_CONNECTED), (_TFLOpName.CONV_2D))
  def test_check_drq_config_asymmetric_weights_raise_error(self, op_name):
    op_quant_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            symmetric=False,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # DRQ.
    )
    error_message = (
        f"Quantization config for op: {op_name} with config:"
        f" {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  @parameterized.product(
      op_name=(_TFLOpName.FULLY_CONNECTED, _TFLOpName.CONV_2D),
      act_num_bits=(8, 16),
      weight_num_bits=(4, 8),
      granularity=(
          qtyping.QuantGranularity.TENSORWISE,
          qtyping.QuantGranularity.CHANNELWISE,
      ),
      symmetric_act=(True, False),
  )
  def test_check_srq_config_succeeds(
      self,
      op_name,
      act_num_bits,
      weight_num_bits,
      granularity,
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
            num_bits=weight_num_bits,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    min_max_quantize_utils.check_if_valid_op_config(
        op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
    )

  def test_check_srq_config_unsupported_op_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    error_message = (
        f"Unsupported op for {op_quant_config.compute_precision}:"
        f" {_TFLOpName.CUSTOM_OP}"
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_if_valid_op_config(
          _TFLOpName.CUSTOM_OP, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  def test_check_srq_config_wrong_act_bits_config_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=14, symmetric=True
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    error_message = (
        f"Quantization config for op: {_TFLOpName.FULLY_CONNECTED} with config:"
        f" {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_if_valid_op_config(
          _TFLOpName.FULLY_CONNECTED,
          op_quant_config,
          _DEFAULT_CONFIG_CHECK_POLICY,
      )

  def test_check_srq_config_asym_int16_act_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=16, symmetric=False
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    error_message = (
        f"Quantization config for op: {_TFLOpName.FULLY_CONNECTED} with config:"
        f" {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_if_valid_op_config(
          _TFLOpName.FULLY_CONNECTED,
          op_quant_config,
          _DEFAULT_CONFIG_CHECK_POLICY,
      )

  def test_check_srq_config_wrong_weight_bits_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=16, symmetric=True
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=2,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    error_message = (
        f"Quantization config for op: {_TFLOpName.FULLY_CONNECTED} with config:"
        f" {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_if_valid_op_config(
          _TFLOpName.FULLY_CONNECTED,
          op_quant_config,
          _DEFAULT_CONFIG_CHECK_POLICY,
      )

  def test_check_srq_config_asym_weight_raise_error(self):
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            symmetric=False,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    error_message = (
        f"Quantization config for op: {_TFLOpName.FULLY_CONNECTED} with config:"
        f" {op_quant_config} was not found in the policy."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      min_max_quantize_utils.check_if_valid_op_config(
          _TFLOpName.FULLY_CONNECTED,
          op_quant_config,
          _DEFAULT_CONFIG_CHECK_POLICY,
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
      compute_precision=[
          _ComputePrecision.FLOAT,
          _ComputePrecision.INTEGER,
      ],
  )
  def test_check_supported_int4_config_succeeds(
      self, op_name, activation_tensor_config, compute_precision
  ):
    # Exclude invalid SRQ config.
    if (
        activation_tensor_config is not None
        and compute_precision != _ComputePrecision.INTEGER
    ) or (
        activation_tensor_config is None
        and compute_precision == _ComputePrecision.FLOAT
    ):
      return
    # TODO: b/353365054 - Remove this check after int4 DRQ is supported for
    # conv2d.
    if (
        # Check if DRQ and CONV_2D.
        compute_precision == _ComputePrecision.INTEGER
        and activation_tensor_config is None
        and op_name == _TFLOpName.CONV_2D
    ):
      return
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=activation_tensor_config,
        weight_tensor_config=_TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=compute_precision,
    )
    # Raise error if the config is not supported.
    # Check if DRQ.
    if (
        compute_precision == _ComputePrecision.INTEGER
        and op_quant_config.activation_tensor_config is None
    ):
      min_max_quantize_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )
    # Check if WEIGHT_ONLY.
    elif (
        compute_precision == _ComputePrecision.FLOAT
        and op_quant_config.explicit_dequantize
    ):
      self.assertIn(op_name, _DEFAULT_CONFIG_CHECK_POLICY.keys())
    # Check if SRQ.
    if (
        compute_precision == _ComputePrecision.INTEGER
        and op_quant_config.activation_tensor_config is not None
    ):
      min_max_quantize_utils.check_if_valid_op_config(
          op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
      )

  @parameterized.product(
      op_name=[_TFLOpName.BATCH_MATMUL],
      activation_tensor_config=[
          None,
          _TensorQuantConfig(num_bits=8, symmetric=False),
          _TensorQuantConfig(num_bits=16, symmetric=True),
      ],
      test_case=[
          # Tuple holds compute precision and whether to use drq.
          (_ComputePrecision.INTEGER, True),
          (_ComputePrecision.INTEGER, False),
      ],
  )
  def test_check_unsupported_int4_config_raise_error(
      self, op_name, activation_tensor_config, test_case
  ):
    compute_precision, is_drq = test_case
    # Exclude invalid SRQ config.
    if (activation_tensor_config is not None and is_drq) or (
        activation_tensor_config is None and not is_drq
    ):
      return
    op_quant_config = _OpQuantConfig(
        activation_tensor_config=activation_tensor_config,
        weight_tensor_config=_TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=compute_precision,
    )

    with self.assertRaises(ValueError):
      if is_drq:
        min_max_quantize_utils.check_if_valid_op_config(
            op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
        )
      elif not is_drq:
        min_max_quantize_utils.check_if_valid_op_config(
            op_name, op_quant_config, _DEFAULT_CONFIG_CHECK_POLICY
        )

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
