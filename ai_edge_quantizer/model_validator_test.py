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

"""Test for model_validator."""

import json
import os
from absl import flags
from tensorflow.python.platform import googletest
from ai_edge_quantizer import model_validator
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import validation_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('.')


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
    self.dataset = test_utils.create_random_normal_input_data(
        self.reference_model_path
    )
    self.test_dir = self.create_tempdir()

  def test_model_validator_compare(self):
    error_metric = 'mean_squared_difference'
    for signature_key, input_dataset in self.dataset.items():
      comparison_result = model_validator.compare_model(
          self.reference_model_path,
          self.target_model_path,
          input_dataset,
          error_metric,
          validation_utils.mean_squared_difference,
          signature_key=signature_key,
      )
      self.assertEqual(
          comparison_result.error_metric, 'mean_squared_difference'
      )
      self.assertLen(comparison_result.input_tensors, 1)
      self.assertLen(comparison_result.output_tensors, 1)
      self.assertLen(comparison_result.constant_tensors, 2)
      self.assertEmpty(comparison_result.intermediate_tensors)

      self.assertAlmostEqual(
          comparison_result.input_tensors['serving_default_input_2:0'], 0
      )
      self.assertAlmostEqual(
          comparison_result.constant_tensors['arith.constant1'], 0
      )
      self.assertLess(
          comparison_result.output_tensors['StatefulPartitionedCall:0'], 1e-5
      )

  def test_create_json_for_model_explorer(self):
    error_metric = 'mean_squared_difference'
    for signature_key, input_dataset in self.dataset.items():
      comparison_result = model_validator.compare_model(
          self.reference_model_path,
          self.target_model_path,
          input_dataset,
          error_metric,
          validation_utils.mean_squared_difference,
          signature_key=signature_key,
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
    for signature_key, input_dataset in self.dataset.items():
      comparison_result = model_validator.compare_model(
          self.reference_model_path,
          self.target_model_path,
          input_dataset,
          error_metric,
          validation_utils.mean_squared_difference,
          signature_key=signature_key,
      )
      mv_json = model_validator.create_json_for_model_explorer(
          comparison_result, []
      )
      self.assertContainsSubset('"thresholds": []', mv_json)

  def test_save_comparison_result_succeeds(self):
    error_metric = 'mean_squared_difference'
    model_name = 'test_model'
    for signature_key, input_dataset in self.dataset.items():
      comparison_result = model_validator.compare_model(
          self.reference_model_path,
          self.target_model_path,
          input_dataset,
          error_metric,
          validation_utils.mean_squared_difference,
          signature_key=signature_key,
      )
      comparison_result.save(self.test_dir.full_path, model_name)

      # Test json for comparison result.
      test_json_path = os.path.join(
          self.test_dir.full_path, model_name + '_comparison_result.json'
      )
      with open(test_json_path) as json_file:
        json_dict = json.load(json_file)
      self.assertIn('error_metric', json_dict)
      self.assertEqual(json_dict['error_metric'], 'mean_squared_difference')
      self.assertIn('constant_tensors', json_dict)
      self.assertIn('arith.constant', json_dict['constant_tensors'])

      # Test json for model explorer.
      test_json_path = os.path.join(
          self.test_dir.full_path,
          model_name + '_comparison_result_me_input.json',
      )
      with open(test_json_path) as json_file:
        json_dict = json.load(json_file)
      self.assertIn('results', json_dict)
      results = json_dict['results']
      self.assertIn('arith.constant', results)


if __name__ == '__main__':
  googletest.main()
