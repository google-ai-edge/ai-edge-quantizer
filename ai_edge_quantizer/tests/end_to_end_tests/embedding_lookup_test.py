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

"""E2E tests for the quantizer for model with embedding_lookup."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils

_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig

_RNG = np.random.default_rng(66)


def _get_dummy_data(num_samples):
  data = []
  for _ in range(num_samples):
    data.append({'inputs': _RNG.uniform(size=(1,)).astype(np.int32)})
  return data


# def _get_test_data(num_samples: int = 8):
#   return _get_dummy_data(num_samples)


class EmbeddingLookupTest(test_utils.BaseOpTestCase):

  def setUp(self):
    super().setUp()
    self.float_model_path = test_utils.get_path_to_datafile(
        '../models/embedding_lookup.tflite'
    )
    self._quantizer = quantizer.Quantizer(self.float_model_path)

  @parameterized.parameters(
      ('../../recipes/default_af32w8float_recipe.json', 1700),
      ('../../recipes/default_af32w4float_recipe.json', 1600),
      ('../../recipes/dynamic_legacy_wi8_afp32_recipe.json', 1400),
      ('../../recipes/dynamic_wi8_afp32_recipe.json', 1400),
  )
  def test_embedding_lookup_model_int_weight_only(
      self, recipe_path, expected_model_size
  ):
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertFalse(self._quantizer.need_calibration)
    quant_result = self._quantizer.quantize()
    # Check model size.
    self.assertLess(len(quant_result.quantized_model), expected_model_size)

    # TODO: b/364405203 - Enable after 0 signature works.
    # comparison_result = self._quantizer.validate(
    #     error_metrics='mse',
    #     signature_test_data=_get_test_data(),
    #     signature_key='main',
    # )
    # self._check_comparion_result(
    #     comparion_result,
    #     weight_tolerance=1e-2,
    #     output_tolerance=1e-4,
    # )

  def test_embedding_lookup_model_fp16_weight_only(self):
    self._quantizer.update_quantization_recipe(
        regex='.*',
        algorithm_key=quantizer.AlgorithmName.FLOAT_CASTING,
        operation_name=_OpName.EMBEDDING_LOOKUP,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT
            ),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY
            explicit_dequantize=True,
        ),
    )
    quant_result = self._quantizer.quantize()
    self.assertLess(len(quant_result.quantized_model), 2000)

    # TODO: b/364405203 - Enable after 0 signature works.
    # comparion_result = self._quantizer.validate(
    #     error_metrics='mse',
    #     signature_test_data=_get_test_data(),
    # )
    # self._check_comparion_result(
    #     comparion_result,
    #     weight_tolerance=1e-5,
    #     output_tolerance=1e-5,
    # )

  @parameterized.parameters(
      ('../../recipes/default_a8w8_recipe.json', 1400),
      ('../../recipes/default_a16w8_recipe.json', 1400),
  )
  def test_embedding_lookup_model_full_integer(
      self, recipe_path, expected_model_size
  ):
    calibration_result = {
        'Identity_1': {'min': -2.0, 'max': 2.0},
    }
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertTrue(self._quantizer.need_calibration)
    quant_result = self._quantizer.quantize(calibration_result)
    self.assertLess(len(quant_result.quantized_model), expected_model_size)
    # TODO: b/364405203 - Enable after 0 signature works.
    # comparion_result = self._quantizer.validate(
    #     error_metrics='mse',
    #     signature_test_data=_get_test_data(),
    # )
    # self._check_comparion_result(
    #     comparion_result,
    #     weight_tolerance=1e-5,
    #     output_tolerance=1e-5,
    # )

  # def _check_comparion_result(
  #     self,
  #     comparion_result,
  #     weight_tolerance,
  #     output_tolerance,
  # ):
  #   # TODO: b/357959309 - Use comparison result directly for testing.
  #   comparion_result = comparion_result.get_all_tensor_results()
  #   weight_mse = comparion_result[
  #       'jax2tf_export_func_/...y_yz-_...z/pjit__einsum_/MatMul;jax2tf_export_func_/pjit__one_hot_/Equal;jax2tf_export_func_/pjit__one_hot_/Cast_1'
  #   ]
  #   self.assertLess(weight_mse, weight_tolerance)
  #   output_mse = comparion_result['Identity_1']
  #   self.assertLess(output_mse, output_tolerance)

  @parameterized.product(weight_bit_width=[4, 8])
  def test_hadamard_rotation_accuracy_and_size_within_tolerance(
      self, weight_bit_width
  ):
    self._quantizer.load_quantization_recipe([
        {
            'regex': '.*',
            'operation': qtyping.TFLOperationName.EMBEDDING_LOOKUP,
            'algorithm_key': quantizer.AlgorithmName.HADAMARD_ROTATION,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': qtyping.TensorDataType.INT,
                    'num_bits': weight_bit_width,
                    'granularity': qtyping.QuantGranularity.CHANNELWISE,
                },
                'compute_precision': qtyping.ComputePrecision.FLOAT,
                'explicit_dequantize': True,
            },
        },
    ])
    quant_result = self._quantizer.quantize()
    with self.subTest(name='ModelSizeReduction'):
      self.assertLess(len(quant_result.quantized_model), 1600)

    validation_result = self._quantizer.validate(
        test_data={
            'lookup': [{
                'lookup': np.random.randint(0, 15, size=(1,), dtype=np.int32)
            }],
        },
        error_metrics='mse',
    )
    with self.subTest(name='OutputErrors'):
      self.assert_output_errors_below_tolerance(validation_result, 1e-5)


if __name__ == '__main__':
  googletest.main()
