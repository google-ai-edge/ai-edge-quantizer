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

import os
from tensorflow.python.platform import googletest
from ai_edge_quantizer import model_validator
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import validation_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('.')


class ModelValidatorCompareTest(googletest.TestCase):

  def setUp(self):
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

  def test_model_validator_compare(self):
    for signature_key, input_dataset in self.dataset.items():
      comparison_result = model_validator.compare_model(
          self.reference_model_path,
          self.target_model_path,
          input_dataset,
          validation_utils.mean_squared_difference,
          signature_key=signature_key,
      )
      self.assertLen(comparison_result, 4)
      self.assertAlmostEqual(comparison_result['serving_default_input_2:0'], 0)
      self.assertAlmostEqual(comparison_result['arith.constant1'], 0)
      self.assertLess(comparison_result['StatefulPartitionedCall:0'], 1e-5)

  def test_create_json_for_model_explorer(self):
    for signature_key, input_dataset in self.dataset.items():
      comparison_result = model_validator.compare_model(
          self.reference_model_path,
          self.target_model_path,
          input_dataset,
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
    for signature_key, input_dataset in self.dataset.items():
      comparison_result = model_validator.compare_model(
          self.reference_model_path,
          self.target_model_path,
          input_dataset,
          validation_utils.mean_squared_difference,
          signature_key=signature_key,
      )
      mv_json = model_validator.create_json_for_model_explorer(
          comparison_result, []
      )
      self.assertContainsSubset('"thresholds": []', mv_json)


if __name__ == '__main__':
  googletest.main()
