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

"""Tests for recipe_manager.py."""

from absl.testing import parameterized
from tensorflow.python.platform import googletest
from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe_manager

_OpExecutionMode = qtyping.OpExecutionMode
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_TensorDataType = qtyping.TensorDataType
_AlgorithmName = recipe_manager.AlgorithmName


# Sample functions for test cases.
def _sample_init_qsvs(*_, **__):
  return 1.0, dict()


def _sample_calibration_func(*_, **__):
  return 2.0, dict()


def _sample_materialize_func(*_, **__):
  return 3.0, dict()


def _sample_check_op_config_func(_, op_config):
  if op_config.weight_tensor_config.num_bits == 17:
    raise ValueError('Unsupported number of bits.')


def _add_default_int8xint8_integer_recipe(recipe_manager_object):
  recipe_manager_object.add_quantization_config(
      regex='.*',
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      op_config=qtyping.OpQuantizationConfig(
          activation_tensor_config=_TensorQuantConfig(
              num_bits=8, symmetric=False
          ),
          weight_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
          execution_mode=_OpExecutionMode.SRQ,
      ),
  )


# register some currently unsupported ops for testing purposes
def _register_testing_op(algorithm_key, tfl_op):
  algorithm_manager.register_op_quant_config_validation_func(
      algorithm_key, _sample_check_op_config_func
  )
  algorithm_manager.register_quantized_op(
      algorithm_key,
      tfl_op,
      _sample_init_qsvs,
      _sample_calibration_func,
      _sample_materialize_func,
  )


class ConfiguratorTest(parameterized.TestCase, googletest.TestCase):
  """Test cases for the flax quantizer Configurator."""

  def setUp(self):
    super().setUp()
    self._recipe_manager = recipe_manager.RecipeManager()
    self._testing_ops = [
        _TFLOpName.BATCH_MATMUL,
        _TFLOpName.FULLY_CONNECTED,
        _TFLOpName.DEPTHWISE_CONV_2D,
    ]
    for op in self._testing_ops:
      _register_testing_op(_AlgorithmName.MIN_MAX_UNIFORM_QUANT, op)
      _register_testing_op('GPTQ', op)

  def test_add_get_quantization_config(self):
    # Int8 DRQ all ops under "Dense".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            execution_mode=_OpExecutionMode.DRQ
        ),
    )

    # Int8 weight-only FullyConnected configuration under "Dense_3".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_3/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            execution_mode=_OpExecutionMode.WEIGHT_ONLY
        ),
    )
    # Int4 DRQ BatchMatmul configuration under "Dense_3".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_3/.*',
        operation_name=_TFLOpName.BATCH_MATMUL,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=4),
            execution_mode=_OpExecutionMode.DRQ,
        ),
    )

    # Return NO_QUANT if not match
    alg_key, _ = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_1/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.NO_QUANTIZE)
    alg_key, _ = self._recipe_manager.get_quantization_configs(
        _TFLOpName.DEPTHWISE_CONV_2D, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.NO_QUANTIZE)

    # Check _TFLOperationKey.ALL
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.DEPTHWISE_CONV_2D, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.DRQ)

    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.DRQ)

    # Check conflicts handling
    # Int8 Weight-only for FC under "Dense", this should only overwrite FC but
    # leave others unchanged.
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            execution_mode=_OpExecutionMode.WEIGHT_ONLY
        ),
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.WEIGHT_ONLY)
    alg_key, _ = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)

    # Reset all ops, this time with 4 bits DRQ.
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=4),
            execution_mode=_OpExecutionMode.DRQ,
        ),
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.DRQ)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 4)
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.DRQ)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 4)

    # Overwrite all FC
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=3),
        ),
    )
    # FC config is overridden
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 3)
    # but not bmm
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 4)

  def test_add_unsupported_quantization_config(self):
    error_message = 'Unsupported operation'
    # Add unregistered operation
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._recipe_manager.add_quantization_config(
          regex='.*/Dense/.*',
          operation_name=_TFLOpName.CUSTOM_OP,
          algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
          op_config=qtyping.OpQuantizationConfig(
              execution_mode=_OpExecutionMode.DRQ
          ),
      )
    # Add unregistered algorithm
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._recipe_manager.add_quantization_config(
          regex='.*/Dense/.*',
          operation_name=_TFLOpName.FULLY_CONNECTED,
          algorithm_key='AWQ',
          op_config=qtyping.OpQuantizationConfig(
              execution_mode=_OpExecutionMode.DRQ
          ),
      )

  def test_add_unsupported_num_bits_raise_error(self):
    error_message = 'Unsupported number of bits.'
    # Add unregistered operation
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._recipe_manager.add_quantization_config(
          regex='.*/Dense/.*',
          operation_name=_TFLOpName.FULLY_CONNECTED,
          algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
          op_config=qtyping.OpQuantizationConfig(
              weight_tensor_config=_TensorQuantConfig(num_bits=17),
          ),
      )

  def test_add_unsupported_skip_successful(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_3/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=17),
            execution_mode=_OpExecutionMode.DRQ,
            skip_checks=True,
        ),
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertIsNone(op_config.activation_tensor_config)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 17)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.DRQ)

  def test_set_full_integer_quantization_config(self):
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    # Full integer setting is global
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    op_act_config = op_config.activation_tensor_config
    self.assertIsNotNone(op_act_config)
    self.assertEqual(op_act_config.num_bits, 8)
    self.assertEqual(op_act_config.symmetric, False)
    self.assertEqual(op_act_config.channel_wise, False)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 8)
    self.assertEqual(op_config.weight_tensor_config.symmetric, True)
    self.assertEqual(op_config.weight_tensor_config.channel_wise, False)

    # Change weight settings for Dense_3 FC
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_3/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=3),
        ),
    )
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertIsNone(op_config.activation_tensor_config)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 3)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.WEIGHT_ONLY)

    # Change the global setting to  int16
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=_TensorQuantConfig(
                num_bits=16, symmetric=True
            ),
            weight_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
            execution_mode=_OpExecutionMode.SRQ,
        ),
    )
    # This does not impact the special dense_3 case
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense_3/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertIsNone(op_config.activation_tensor_config)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 3)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.WEIGHT_ONLY)

    # All the others will be int16
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.CONV_2D, 'model/Dense_31/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    op_act_config = op_config.activation_tensor_config
    self.assertIsNotNone(op_act_config)
    self.assertEqual(op_act_config.num_bits, 16)
    self.assertEqual(op_act_config.symmetric, True)
    self.assertEqual(op_act_config.channel_wise, False)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 8)
    self.assertEqual(op_config.weight_tensor_config.symmetric, True)
    self.assertEqual(op_config.weight_tensor_config.channel_wise, False)

  def test_get_full_quantization_config(self):
    # Int8 asymetric full integer model.
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    # Default all BMM.
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=_TFLOpName.BATCH_MATMUL,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
    )

    # Int8 DRQ FULLY_CONNECTED ops under "Dense".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            execution_mode=_OpExecutionMode.DRQ
        ),
    )

    # Overwrite DRQ ALL ops under "Dense".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=4),
            execution_mode=_OpExecutionMode.WEIGHT_ONLY,
        ),
    )

    # Overwrite "Dense_1" to only quantize FullyConnected.
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_1/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=6),
            execution_mode=_OpExecutionMode.WEIGHT_ONLY,
        ),
    )

    # Add BMM to "Dense_1".
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_1/.*',
        operation_name=_TFLOpName.BATCH_MATMUL,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=3),
            execution_mode=_OpExecutionMode.WEIGHT_ONLY,
        ),
    )

    expected_full_quantization_config = [
        {
            'regex': '.*',
            'operation': '*',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'activation_tensor_config': {
                    'num_bits': 8,
                    'symmetric': False,
                    'channel_wise': False,
                    'dtype': 'INT',
                },
                'weight_tensor_config': {
                    'num_bits': 8,
                    'symmetric': True,
                    'channel_wise': False,
                    'dtype': 'INT',
                },
                'execution_mode': 'SRQ',
                'skip_checks': False,
            },
        },
        {
            'regex': '.*',
            'operation': 'BATCH_MATMUL',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 8,
                    'symmetric': True,
                    'channel_wise': False,
                },
                'execution_mode': 'WEIGHT_ONLY',
                'skip_checks': False,
            },
        },
        {
            'regex': '.*/Dense/.*',
            'operation': '*',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 4,
                    'symmetric': True,
                    'channel_wise': False,
                },
                'execution_mode': 'WEIGHT_ONLY',
                'skip_checks': False,
            },
        },
        {
            'regex': '.*/Dense_1/.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 6,
                    'symmetric': True,
                    'channel_wise': False,
                },
                'execution_mode': 'WEIGHT_ONLY',
                'skip_checks': False,
            },
        },
        {
            'regex': '.*/Dense_1/.*',
            'operation': 'BATCH_MATMUL',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 3,
                    'symmetric': True,
                    'channel_wise': False,
                },
                'execution_mode': 'WEIGHT_ONLY',
                'skip_checks': False,
            },
        },
    ]
    self.assertEqual(
        expected_full_quantization_config,
        self._recipe_manager.get_quantization_recipe(),
    )

  def test_get_quantization_configs_with_no_quantize_overwrite(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
        ),
    )
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.NO_QUANTIZE,
    )

    # Fully connected will be overwritten to no quantization.
    alg_key, _ = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.NO_QUANTIZE)
    # Other ops will be quantized.
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.CONV_2D, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.WEIGHT_ONLY)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 8)

  def test_load_from_full_quantization_config(self):
    full_quantization_config = [
        {
            'regex': '.*',
            'operation': 'BATCH_MATMUL',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 8,
                    'symmetric': True,
                    'channel_wise': True,
                },
                'execution_mode': 'WEIGHT_ONLY',
            },
        },
        {
            'regex': '.*/Dense/.*',
            'operation': '*',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 4,
                    'symmetric': False,
                    'channel_wise': True,
                },
                'execution_mode': 'DRQ',
            },
        },
    ]
    self._recipe_manager.load_quantization_recipe(full_quantization_config)

    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense10/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.WEIGHT_ONLY)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 8)

    # Dense will be overwritten by the last setting
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.DRQ)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 4)

  def test_get_unsupported_op_fall_back_to_default(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense/.*',
        operation_name=_TFLOpName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=17),
        ),
    )
    alg_key, _ = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense10/op'
    )
    # int17 is not supported, fall back to float.
    self.assertEqual(alg_key, _AlgorithmName.NO_QUANTIZE)

  def test_load_from_full_quantization_config_full_integer(self):
    full_quantization_config = [
        {
            'regex': '.*',
            'operation': '*',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'activation_tensor_config': {
                    'num_bits': 8,
                    'symmetric': False,
                    'channel_wise': False,
                    'dtype': 'INT',
                },
                'weight_tensor_config': {
                    'num_bits': 8,
                    'symmetric': True,
                    'channel_wise': False,
                    'dtype': 'INT',
                },
                'execution_mode': 'SRQ',
            },
        },
        {
            'regex': '.*',
            'operation': 'BATCH_MATMUL',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 8,
                    'symmetric': True,
                    'channel_wise': True,
                },
                'execution_mode': 'WEIGHT_ONLY',
            },
        },
        {
            'regex': '.*/Dense/.*',
            'operation': '*',
            'algorithm_key': _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': 'INT',
                    'num_bits': 4,
                    'symmetric': False,
                    'channel_wise': True,
                },
                'execution_mode': 'DRQ',
            },
        },
    ]
    self._recipe_manager.load_quantization_recipe(full_quantization_config)

    # BMMs will be overridden to weight-only
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.BATCH_MATMUL, 'model/Dense10/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertIsNone(op_config.activation_tensor_config)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.WEIGHT_ONLY)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 8)

    # Dense will be overwritten by the last setting
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.FULLY_CONNECTED, 'model/Dense/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertIsNone(op_config.activation_tensor_config)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.DRQ)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 4)

    # Other ops will have default quantization settings
    alg_key, op_config = self._recipe_manager.get_quantization_configs(
        _TFLOpName.CONV_2D, 'model/Dense11/op'
    )
    self.assertEqual(alg_key, _AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    op_act_config = op_config.activation_tensor_config
    self.assertIsNotNone(op_act_config)
    self.assertEqual(op_act_config.num_bits, 8)
    self.assertEqual(op_config.execution_mode, _OpExecutionMode.SRQ)
    self.assertEqual(op_config.weight_tensor_config.num_bits, 8)

  def test_need_calibration_false(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_1/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            execution_mode=_OpExecutionMode.DRQ
        ),
    )
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_2/.*',
        operation_name=_TFLOpName.CONV_2D,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            execution_mode=_OpExecutionMode.WEIGHT_ONLY
        ),
    )
    self.assertFalse(self._recipe_manager.need_calibration())

  def test_need_calibration_true(self):
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_1/.*',
        operation_name=_TFLOpName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            execution_mode=_OpExecutionMode.DRQ
        ),
    )
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_2/.*',
        operation_name=_TFLOpName.CONV_2D,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            execution_mode=_OpExecutionMode.WEIGHT_ONLY
        ),
    )
    self._recipe_manager.add_quantization_config(
        regex='.*/Dense_3/.*',
        operation_name=_TFLOpName.BATCH_MATMUL,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            execution_mode=_OpExecutionMode.SRQ
        ),
    )
    self.assertTrue(self._recipe_manager.need_calibration())


if __name__ == '__main__':
  googletest.main()
