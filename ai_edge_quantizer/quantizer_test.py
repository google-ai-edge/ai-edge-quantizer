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

"""Tests for quantizer."""

import json
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils

_OpExecutionMode = qtyping.OpExecutionMode
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_TensorDataType = qtyping.TensorDataType
_AlgorithmName = quantizer.AlgorithmName

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('')
_RNG = np.random.default_rng(66)


def _get_calibration_data(num_samples: int = 256):
  calibration_data = []
  for _ in range(num_samples):
    calibration_data.append(
        {'conv2d_input': _RNG.uniform(size=(1, 28, 28, 1)).astype(np.float32)}
    )
  return calibration_data


class QuantizerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/conv_fc_mnist.tflite'
    )
    self._test_recipe_path = os.path.join(
        TEST_DATA_PREFIX_PATH,
        'recipes/default_af32w8float_recipe.json',
    )
    with open(self._test_recipe_path) as json_file:
      self._test_recipe = json.load(json_file)
    self._quantizer = quantizer.Quantizer(
        self._test_model_path, self._test_recipe_path
    )

  def test_update_quantization_recipe_succeeds(self):
    self._quantizer.load_quantization_recipe(self._test_recipe_path)
    scope_regex = '.*/Dense/.*'
    new_op_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=_TensorQuantConfig(num_bits=4, symmetric=True),
        execution_mode=_OpExecutionMode.DRQ,
    )
    self._quantizer.update_quantization_recipe(
        regex=scope_regex,
        operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=new_op_config,
    )
    updated_recipe = self._quantizer.get_quantization_recipe()
    self.assertLen(updated_recipe, 2)

    added_config = updated_recipe[-1]
    self.assertEqual(added_config['regex'], scope_regex)
    self.assertEqual(
        added_config['op_config']['execution_mode'],
        new_op_config.execution_mode,
    )

  def test_load_quantization_recipe_succeeds(self):
    qt = quantizer.Quantizer(self._test_model_path, None)
    qt.load_quantization_recipe(self._test_recipe_path)
    self.assertEqual(qt.get_quantization_recipe(), self._test_recipe)

    # Load a different recipe.
    new_recipe_path = os.path.join(
        TEST_DATA_PREFIX_PATH,
        'recipes/default_af32w8int_recipe.json',
    )
    with open(new_recipe_path) as json_file:
      new_recipe = json.load(json_file)
    qt.load_quantization_recipe(new_recipe_path)
    self.assertEqual(qt.get_quantization_recipe(), new_recipe)

  @parameterized.parameters(
      'recipes/default_a8w8_recipe.json',
      'recipes/default_a16w8_recipe.json',
  )
  def test_calibrate_required_recipe_succeeds(self, recipe_path):
    recipe_path = os.path.join(TEST_DATA_PREFIX_PATH, recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertTrue(self._quantizer.need_calibration)
    # Calibrate with empty state.
    calib_data = _get_calibration_data()
    calibration_result = self._quantizer.calibrate(calib_data)
    self.assertLen(calibration_result, 13)

  @parameterized.parameters(
      'recipes/default_a8w8_recipe.json',
      'recipes/default_a16w8_recipe.json',
  )
  def test_reloaded_calibration_succeeds(self, recipe_path):
    recipe_path = os.path.join(TEST_DATA_PREFIX_PATH, recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    calib_data = _get_calibration_data()
    calibration_result = self._quantizer.calibrate(calib_data)
    # Load and calibrate again.
    updated_calibration_result = self._quantizer.calibrate(
        calib_data, previous_calibration_result=calibration_result
    )
    self.assertLen(updated_calibration_result, 13)
    self.assertNotEqual(
        calibration_result['StatefulPartitionedCall:0'],
        updated_calibration_result['StatefulPartitionedCall:0'],
    )

  @parameterized.parameters(
      'recipes/default_af32w8int_recipe.json',
      'recipes/default_af32w8float_recipe.json',
  )
  def test_calibrate_nonrequired_recipe_succeeds(self, recipe_path):
    recipe_path = os.path.join(TEST_DATA_PREFIX_PATH, recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertFalse(self._quantizer.need_calibration)
    # Empty calibration result if no calibration is required.
    calibration_result = self._quantizer.calibrate(_get_calibration_data())
    self.assertEmpty(calibration_result)

  def test_quantize_no_calibration_succeeds(self):
    self._quantizer.load_quantization_recipe(self._test_recipe_path)
    self.assertIsNone(self._quantizer._result.quantized_model)
    quant_result = self._quantizer.quantize()
    self.assertEqual(quant_result.recipe, self._test_recipe)
    self.assertIsNotNone(quant_result.quantized_model)

  @parameterized.parameters(
      'recipes/default_a8w8_recipe.json',
      'recipes/default_a16w8_recipe.json',
  )
  def test_quantize_calibration_needed_succeeds(self, recipe_path):
    recipe_path = os.path.join(TEST_DATA_PREFIX_PATH, recipe_path)
    with open(recipe_path) as json_file:
      recipe = json.load(json_file)

    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertTrue(self._quantizer.need_calibration)
    calibration_result = self._quantizer.calibrate(_get_calibration_data())

    self.assertIsNone(self._quantizer._result.quantized_model)
    quant_result = self._quantizer.quantize(calibration_result)
    self.assertEqual(quant_result.recipe, recipe)
    self.assertIsNotNone(quant_result.quantized_model)

  @parameterized.parameters(
      'recipes/default_a8w8_recipe.json',
      'recipes/default_a16w8_recipe.json',
  )
  def test_quantize_calibration_needed_raise_error(self, recipe_path):
    recipe_path = os.path.join(TEST_DATA_PREFIX_PATH, recipe_path)

    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertTrue(self._quantizer.need_calibration)
    error_message = (
        'Model quantization statistics values (QSVs) are required for the input'
        ' recipe.'
    )
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      self._quantizer.quantize()

  def test_quantize_no_recipe_raise_error(self):
    qt = quantizer.Quantizer(self._test_model_path, None)
    error_message = 'Can not quantize without a quantization recipe.'
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      qt.quantize()

  def test_save_succeeds(self):
    model_name = 'test_model'
    save_path = '/tmp/'
    self._quantizer.load_quantization_recipe(self._test_recipe_path)
    result = self._quantizer.quantize()
    result.save(save_path, model_name)
    saved_recipe_path = os.path.join(
        save_path, model_name, model_name + '_recipe.json'
    )
    with open(saved_recipe_path) as json_file:
      saved_recipe = json.load(json_file)
    self.assertEqual(saved_recipe, self._test_recipe)

  def test_save_no_quantize_raise_error(self):
    error_message = 'No quantized model to save.'
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      self._quantizer._result.save('/tmp/', 'test_model')

  def test_compare_succeeds(self):
    self._quantizer.quantize()
    comparison_result = self._quantizer.compare()
    self.assertIsNotNone(comparison_result)
    self.assertIn('sequential/dense_1/MatMul', comparison_result)

  def test_save_compare_result_succeeds(self):
    self._quantizer.quantize()
    test_json_path = '/tmp/test_compare_result.json'
    comparison_result = self._quantizer.compare()
    self._quantizer.save_comparison_result(
        comparison_result, test_json_path, [0, 1]
    )
    with open(test_json_path) as json_file:
      json_dict = json.load(json_file)
    self.assertIn('results', json_dict)
    results = json_dict['results']
    self.assertIn('sequential/dense_1/MatMul', results)


class QuantizerBytearrayInputs(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/conv_fc_mnist.tflite'
    )
    self._test_recipe_path = os.path.join(
        TEST_DATA_PREFIX_PATH,
        'recipes/default_af32w8float_recipe.json',
    )
    with open(self._test_model_path, 'rb') as f:
      model_content = bytearray(f.read())
    with open(self._test_recipe_path, 'r') as f:
      self._test_recipe = json.load(f)
    self._quantizer = quantizer.Quantizer(model_content, self._test_recipe)

  def test_quantize_compare_succeeds(self):
    quant_result = self._quantizer.quantize()
    self.assertEqual(quant_result.recipe, self._test_recipe)
    self.assertIsNotNone(quant_result.quantized_model)
    comparison_result = self._quantizer.compare()
    self.assertIsNotNone(comparison_result)
    self.assertIn('sequential/dense_1/MatMul', comparison_result)

  def test_compare_succeeds(self):
    self._quantizer.quantize()
    comparison_result = self._quantizer.compare()
    self.assertIsNotNone(comparison_result)
    self.assertIn('sequential/dense_1/MatMul', comparison_result)


if __name__ == '__main__':
  googletest.main()
