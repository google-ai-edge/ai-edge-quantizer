"""Test for model_validator."""

import os
from tensorflow.python.platform import googletest
from ai_edge_quantizer import model_validator
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils
from ai_edge_quantizer.utils import validation_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('.')


class ModelValidatorCompareTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.reference_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'test_models/single_fc_bias.tflite'
    )
    self.target_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH,
        'test_models/single_fc_bias_sub_channel_weight_only_sym_weight.tflite',
    )
    reference_model_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self.reference_model_path
    )
    self.dataset = test_utils.create_random_normal_dataset(
        reference_model_interpreter.get_input_details(), 3, 666
    )

  def test_model_validator_compare(self):
    comparison_result = model_validator.compare_model(
        self.reference_model_path,
        self.target_model_path,
        self.dataset,
        False,
        validation_utils.mean_squared_difference,
    )
    self.assertLen(comparison_result, 4)
    self.assertAlmostEqual(comparison_result['serving_default_input_2:0'], 0)
    self.assertAlmostEqual(comparison_result['arith.constant1'], 0)
    self.assertAlmostEqual(
        comparison_result['StatefulPartitionedCall:0'], 8.421804e-06
    )

  def test_create_json_for_model_explorer(self):
    comparison_result = model_validator.compare_model(
        self.reference_model_path,
        self.target_model_path,
        self.dataset,
        False,
        validation_utils.mean_squared_difference,
    )
    thresholds = [0, 1, 2, 3]
    mv_json = model_validator.create_json_for_model_explorer(
        comparison_result, thresholds
    )
    self.assertContainsSubset(
        '"thresholds": [{"value": 0, "bgColor":'
        ' "rgb(200, 0, 0)"}, {"value": 1, "bgColor": "rgb(200, 63, 0)"},'
        ' {"value": 2, "bgColor": "rgb(200, 126, 0)"}, {"value": 3, "bgColor":'
        ' "rgb(200, 189, 0)"}]',
        mv_json,
    )

  def test_create_json_for_model_explorer_no_thresholds(self):
    comparison_result = model_validator.compare_model(
        self.reference_model_path,
        self.target_model_path,
        self.dataset,
        False,
        validation_utils.mean_squared_difference,
    )
    mv_json = model_validator.create_json_for_model_explorer(
        comparison_result, []
    )
    self.assertContainsSubset('"thresholds": []', mv_json)


if __name__ == '__main__':
  googletest.main()
