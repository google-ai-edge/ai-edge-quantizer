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

import json
import os
from absl import flags
import numpy as np
from tensorflow.python.platform import googletest
from ai_edge_quantizer import model_validator
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import validation_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('.')


class ComparisonResultTest(googletest.TestCase):

  def setUp(self):
    # TODO: b/358437395 - Remove this line once the bug is fixed.
    flags.FLAGS.mark_as_parsed()
    super().setUp()
    self.test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/two_signatures.tflite'
    )
    self.test_model = tfl_flatbuffer_utils.get_model_buffer(
        self.test_model_path
    )
    self.test_quantized_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH,
        'tests/models/two_signatures_a8w8.tflite',
    )
    self.test_quantized_model = tfl_flatbuffer_utils.get_model_buffer(
        self.test_quantized_model_path
    )
    self.test_data = {
        'add': {'add_x:0': 1e-3, 'Add/y': 0.25, 'PartitionedCall:0': 1e-3},
        'multiply': {
            'multiply_x:0': 1e-3,
            'Mul/y': 0.32,
            'PartitionedCall_1:0': 1e-2,
        },
    }
    self.test_dir = self.create_tempdir()
    self.comparison_result = model_validator.ComparisonResult(
        self.test_model, self.test_quantized_model
    )

  def test_add_new_signature_results_succeeds(self):
    for signature_key, test_result in self.test_data.items():
      self.comparison_result.add_new_signature_results(
          'mean_squared_difference',
          test_result,
          signature_key,
      )
    self.assertLen(
        self.comparison_result.available_signature_keys(), len(self.test_data)
    )

    for signature_key in self.test_data:
      signature_result = self.comparison_result.get_signature_comparison_result(
          signature_key
      )
      input_tensors = signature_result.input_tensors
      output_tensors = signature_result.output_tensors
      constant_tensors = signature_result.constant_tensors
      intermediate_tensors = signature_result.intermediate_tensors

      self.assertLen(input_tensors, 1)
      self.assertLen(output_tensors, 1)
      self.assertLen(constant_tensors, 1)
      self.assertEmpty(intermediate_tensors)

  def test_add_new_signature_results_fails_same_signature_key(self):
    self.comparison_result.add_new_signature_results(
        'mean_squared_difference',
        self.test_data['add'],
        'add',
    )
    error_message = 'add is already in the comparison_results.'
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self.comparison_result.add_new_signature_results(
          'mean_squared_difference',
          self.test_data['add'],
          'add',
      )

  def test_get_signature_comparison_result_fails_with_invalid_signature_key(
      self,
  ):
    self.comparison_result.add_new_signature_results(
        'mean_squared_difference',
        self.test_data['add'],
        'add',
    )
    error_message = 'multiply is not in the comparison_results.'
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self.comparison_result.get_signature_comparison_result('multiply')

  def test_get_all_tensor_results_succeeds(self):
    for signature_key, test_result in self.test_data.items():
      self.comparison_result.add_new_signature_results(
          'mean_squared_difference',
          test_result,
          signature_key,
      )
    all_tensor_results = self.comparison_result.get_all_tensor_results()
    self.assertLen(all_tensor_results, 6)
    self.assertIn('add_x:0', all_tensor_results)
    self.assertIn('Add/y', all_tensor_results)
    self.assertIn('PartitionedCall:0', all_tensor_results)
    self.assertIn('multiply_x:0', all_tensor_results)
    self.assertIn('Mul/y', all_tensor_results)
    self.assertIn('PartitionedCall_1:0', all_tensor_results)

  def test_save_comparison_result_succeeds(self):
    for signature_key, test_result in self.test_data.items():
      self.comparison_result.add_new_signature_results(
          'mean_squared_difference',
          test_result,
          signature_key,
      )
    model_name = 'test_model'
    self.comparison_result.save(self.test_dir.full_path, model_name)
    test_json_path = os.path.join(
        self.test_dir.full_path, model_name + '_comparison_result.json'
    )
    with open(test_json_path) as json_file:
      json_dict = json.load(json_file)

    # Check model size stats.
    self.assertIn('reduced_size_bytes', json_dict)
    self.assertEqual(
        json_dict['reduced_size_bytes'],
        len(self.test_model) - len(self.test_quantized_model),
    )
    self.assertIn('reduced_size_percentage', json_dict)
    self.assertEqual(
        json_dict['reduced_size_percentage'],
        (len(self.test_model) - len(self.test_quantized_model))
        / len(self.test_model)
        * 100,
    )

    for signature_key in self.test_data:
      self.assertIn(signature_key, json_dict)
      signature_result = json_dict[signature_key]
      self.assertIn('error_metric', signature_result)
      self.assertEqual(
          signature_result['error_metric'], 'mean_squared_difference'
      )
      self.assertIn('constant_tensors', signature_result)
      if signature_key == 'add':
        self.assertIn('Add/y', signature_result['constant_tensors'])
        self.assertNotIn('Mul/y', signature_result['constant_tensors'])
      elif signature_key == 'multiply':
        self.assertIn('Mul/y', signature_result['constant_tensors'])
        self.assertNotIn('Add/y', signature_result['constant_tensors'])


class ModelValidatorCompareTest(googletest.TestCase):

  def setUp(self):
    # TODO: b/358437395 - Remove this line once the bug is fixed.
    flags.FLAGS.mark_as_parsed()
    super().setUp()
    self.reference_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/single_fc_bias.tflite'
    )
    self.target_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH,
        'tests/models/single_fc_bias_sub_channel_weight_only_sym_weight.tflite',
    )
    self.reference_model = tfl_flatbuffer_utils.get_model_buffer(
        self.reference_model_path
    )
    self.target_model = tfl_flatbuffer_utils.get_model_buffer(
        self.target_model_path
    )
    self.signature_key = 'serving_default'  # single signature.
    self.test_data = test_utils.create_random_normal_input_data(
        self.reference_model_path
    )
    self.test_dir = self.create_tempdir()

  def test_model_validator_compare(self):
    error_metric = 'mean_squared_difference'
    comparison_result = model_validator.compare_model(
        self.reference_model,
        self.target_model,
        self.test_data,
        error_metric,
        validation_utils.mean_squared_difference,
    )
    result = comparison_result.get_signature_comparison_result(
        self.signature_key
    )
    self.assertEqual(result.error_metric, 'mean_squared_difference')
    input_tensors = result.input_tensors
    output_tensors = result.output_tensors
    constant_tensors = result.constant_tensors
    intermediate_tensors = result.intermediate_tensors

    self.assertLen(input_tensors, 1)
    self.assertLen(output_tensors, 1)
    self.assertLen(constant_tensors, 2)
    self.assertEmpty(intermediate_tensors)

    self.assertAlmostEqual(input_tensors['serving_default_input_2:0'], 0)
    self.assertAlmostEqual(constant_tensors['arith.constant1'], 0)
    self.assertLess(output_tensors['StatefulPartitionedCall:0'], 1e-5)

  def test_create_json_for_model_explorer(self):
    error_metric = 'mean_squared_difference'
    comparison_result = model_validator.compare_model(
        self.reference_model,
        self.target_model,
        self.test_data,
        error_metric,
        validation_utils.mean_squared_difference,
    )
    mv_json = model_validator.create_json_for_model_explorer(
        comparison_result, [0, 1, 2, 3]
    )
    self.assertContainsSubset(
        '"thresholds": [{"value": 0, "bgColor": "rgb(200, 0, 0)"}, {"value":'
        ' 1, "bgColor": "rgb(200, 63, 0)"}, {"value": 2, "bgColor": "rgb(200,'
        ' 126, 0)"}, {"value": 3, "bgColor": "rgb(200, 189, 0)"}]',
        mv_json,
    )

  def test_create_json_for_model_explorer_no_thresholds(self):
    error_metric = 'mean_squared_difference'
    comparison_result = model_validator.compare_model(
        self.reference_model,
        self.target_model,
        self.test_data,
        error_metric,
        validation_utils.mean_squared_difference,
    )
    mv_json = model_validator.create_json_for_model_explorer(
        comparison_result, []
    )
    self.assertContainsSubset('"thresholds": []', mv_json)


class ModelValidatorMultiSignatureModelTest(googletest.TestCase):

  def setUp(self):
    # TODO: b/358437395 - Remove this line once the bug is fixed.
    flags.FLAGS.mark_as_parsed()
    super().setUp()
    self.reference_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/two_signatures.tflite'
    )
    self.target_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH,
        'tests/models/two_signatures_a8w8.tflite',
    )
    self.reference_model = tfl_flatbuffer_utils.get_model_buffer(
        self.reference_model_path
    )
    self.target_model = tfl_flatbuffer_utils.get_model_buffer(
        self.target_model_path
    )
    self.test_data = {
        'add': [{'x': np.array([2.0]).astype(np.float32)}],
        'multiply': [{'x': np.array([1.0]).astype(np.float32)}],
    }
    self.test_dir = self.create_tempdir()

  def test_model_validator_compare_succeeds(self):
    error_metric = 'mean_squared_difference'
    result = model_validator.compare_model(
        self.reference_model,
        self.target_model,
        self.test_data,
        error_metric,
        validation_utils.mean_squared_difference,
    )
    for signature_key in self.test_data:
      signature_result = result.get_signature_comparison_result(signature_key)
      input_tensors = signature_result.input_tensors
      output_tensors = signature_result.output_tensors
      constant_tensors = signature_result.constant_tensors
      intermediate_tensors = signature_result.intermediate_tensors

      self.assertLen(input_tensors, 1)
      self.assertLen(output_tensors, 1)
      self.assertLen(constant_tensors, 1)
      self.assertEmpty(intermediate_tensors)

      if signature_key == 'add':
        self.assertLess(input_tensors['add_x:0'], 1e-3)
        self.assertAlmostEqual(constant_tensors['Add/y'], 0)
        self.assertLess(output_tensors['PartitionedCall:0'], 1e-3)
      elif signature_key == 'multiply':
        self.assertLess(input_tensors['multiply_x:0'], 1e-3)
        self.assertAlmostEqual(constant_tensors['Mul/y'], 0)
        self.assertLess(output_tensors['PartitionedCall_1:0'], 1e-2)

  def test_create_json_for_model_explorer(self):
    error_metric = 'mean_squared_difference'
    comparison_result = model_validator.compare_model(
        self.reference_model,
        self.target_model,
        self.test_data,
        error_metric,
        validation_utils.mean_squared_difference,
    )
    thresholds = [0, 1, 2, 3]
    mv_json = model_validator.create_json_for_model_explorer(
        comparison_result, thresholds
    )
    self.assertContainsSubset(
        '"thresholds": [{"value": 0, "bgColor": "rgb(200, 0, 0)"}, {"value":'
        ' 1, "bgColor": "rgb(200, 63, 0)"}, {"value": 2, "bgColor": "rgb(200,'
        ' 126, 0)"}, {"value": 3, "bgColor": "rgb(200, 189, 0)"}]',
        mv_json,
    )

  def test_create_json_for_model_explorer_no_thresholds(self):
    error_metric = 'mean_squared_difference'
    comparison_result = model_validator.compare_model(
        self.reference_model,
        self.target_model,
        self.test_data,
        error_metric,
        validation_utils.mean_squared_difference,
    )
    mv_json = model_validator.create_json_for_model_explorer(
        comparison_result, []
    )
    self.assertContainsSubset('"thresholds": []', mv_json)


if __name__ == '__main__':
  googletest.main()
