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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import googletest
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import calibration_utils
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_RNG = np.random.default_rng(66)

_CALIBRATION_DATASET = {
    "signature_1": [{
        "cache_0": np.zeros(shape=(1, 100, 4, 4), dtype=np.float32),
        "cache_1": np.zeros(shape=(1, 100, 4, 4), dtype=np.float32),
        "positions": np.zeros(shape=(1, 100), dtype=np.int32),
        "tokens": np.zeros(shape=(1, 100), dtype=np.int32),
    }],
    "signature_2": [{
        "cache_0": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
        "cache_1": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
        "positions": (
            _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32)
        ),
        "tokens": _RNG.integers(low=0, high=10, size=(1, 100)).astype(np.int32),
    }],
}


def _get_quant_parameters(
    quantized_model: bytes, signature_data: dict[str, list[str]]
) -> list[np.ndarray]:
  """Returns the quantization parameters from the quantized model."""
  quant_params = []
  tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
      quantized_model
  )
  for signature_key, signature_names in signature_data.items():
    signature_runner = tfl_interpreter.get_signature_runner(signature_key)

    for signature_name in signature_names:
      input_details = signature_runner.get_input_details()
      output_details = signature_runner.get_output_details()
      if signature_name in input_details.keys():
        quant_param = input_details[signature_name]["quantization_parameters"][
            "scales"
        ].squeeze()
        quant_params.append(quant_param)
      elif signature_name in output_details.keys():
        output_details = signature_runner.get_output_details()
        quant_param = output_details[signature_name]["quantization_parameters"][
            "scales"
        ].squeeze()
        quant_params.append(quant_param)
      else:
        raise ValueError(
            f"Signature name {signature_name} not found in the model."
        )
  return quant_params


class CalibrationQsvAlignmentUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_smoothing_factor",
          smoothing_factor=0,
          expected_vals={"min": -1000, "max": 800},
      ),
      dict(
          testcase_name="one_smoothing_factor",
          smoothing_factor=1,
          expected_vals={"min": -10, "max": 8},
      ),
      dict(
          testcase_name="normal_smoothing_factor",
          smoothing_factor=0.99,
          expected_vals={"min": -19.9, "max": 15.92},
      ),
  )
  def test_update_tensor_qsv_moving_average(
      self, smoothing_factor, expected_vals
  ):
    old_qsv = {"min": -10, "max": 8}
    # Large values to mimic outlier.
    new_qsv = {"min": -1000, "max": 800}
    updated_qsv = calibration_utils.moving_average_update(
        old_qsv, new_qsv, smoothing_factor=smoothing_factor
    )
    self.assertAlmostEqual(updated_qsv["min"], expected_vals["min"])
    self.assertAlmostEqual(updated_qsv["max"], expected_vals["max"])

  @parameterized.named_parameters(
      dict(
          testcase_name="scalar",
          old_qsv={"min": -10, "max": 8},
          new_qsv={"min": -1000, "max": 1},
          expected_qsv={"min": -1000, "max": 8},
      ),
      dict(
          testcase_name="2darray",
          old_qsv={"min": [[-19], [20]], "max": [[21], [250]]},
          new_qsv={"min": [[-1000], [25]], "max": [[33], [100]]},
          expected_qsv={"min": [[-1000], [20]], "max": [[33], [250]]},
      ),
  )
  def test_update_tensor_qsv_min_max(self, old_qsv, new_qsv, expected_qsv):
    updated_qsv = calibration_utils.min_max_update(old_qsv, new_qsv)
    if isinstance(expected_qsv["min"], list):
      self.assertEqual(list(updated_qsv["min"]), expected_qsv["min"])
      self.assertEqual(list(updated_qsv["max"]), expected_qsv["max"])
    else:
      self.assertEqual(updated_qsv["min"], expected_qsv["min"])
      self.assertEqual(updated_qsv["max"], expected_qsv["max"])

  def test_calibration_utils_init_fails(self):
    model_path = "non_existent_model.tflite"
    with self.assertRaisesWithPredicateMatch(
        tf.errors.NotFoundError, lambda err: f"{model_path}" in str(err)
    ):
      calibration_utils.CalibrationQsvAlignmentUtils(model_path)

  def test_calibration_utils_init_succeeds(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/single_add.tflite"
    )
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    self.assertNotEmpty(calib_utils._signature_runners)
    self.assertNotEmpty(calib_utils._same_as_input_scale_ops)

  def test_search_tensor_by_signature_name_succeeds_on_unconstrained_op(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/single_add.tflite"
    )
    expected_tensor_name = "PartitionedCall:0"
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    tensor_name = calib_utils._search_tensor_by_signature_name(
        "serving_default", "add"
    )
    self.assertEqual(tensor_name, [expected_tensor_name])

  def test_search_tensor_by_signature_name_succeeds_on_constrained_op(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/single_slice.tflite"
    )
    expected_tensor_name = "slice_input_tensor:0"
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    tensor_name = calib_utils._search_tensor_by_signature_name(
        "slice", "output_0"
    )
    self.assertEqual(tensor_name, [expected_tensor_name])

  def test_align_quant_stats_succeeds(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/toy_model_with_kv_cache_multi_signature.tflite"
    )
    recipe_path = test_utils.get_path_to_datafile(
        "../recipes/default_a8w8_recipe.json"
    )
    signature_data = {
        "signature_1": ["output_1_1"],
        "signature_2": ["output_1_1"],
    }

    # Obtain the calibration results.
    qt = quantizer.Quantizer(model_path, recipe_path)
    qsv = qt.calibrate(_CALIBRATION_DATASET)

    # First quantize the model without aligning the quantization parameters.
    quantized_model = qt.quantize(qsv).quantized_model
    quant_params = _get_quant_parameters(quantized_model, signature_data)
    self.assertFalse(
        all(x == quant_params[0] for x in quant_params)
    )  # not equal quantization params.

    # Align the quantization parameters and quantize again.
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    calib_utils.align_quant_stats(qsv, signature_data)
    quantized_model = qt.quantize(qsv).quantized_model
    quant_params = _get_quant_parameters(quantized_model, signature_data)
    self.assertTrue(
        all(x == quant_params[0] for x in quant_params)
    )  # equal quantization params.

  def test_update_quant_stats_succeeds(self):
    model_path = test_utils.get_path_to_datafile(
        "../tests/models/toy_model_with_kv_cache_multi_signature.tflite"
    )
    recipe_path = test_utils.get_path_to_datafile(
        "../recipes/default_a8w8_recipe.json"
    )
    signature_data = {
        "signature_1": ["output_1_1"],
        "signature_2": ["output_1_1"],
    }

    # Obtain the calibration results.
    qt = quantizer.Quantizer(model_path, recipe_path)
    qsv = qt.calibrate(_CALIBRATION_DATASET)

    # First quantize the model without updating the `signature_1`.
    quantized_model = qt.quantize(qsv).quantized_model
    quant_params = _get_quant_parameters(quantized_model, signature_data)
    self.assertFalse(
        all(x == quant_params[0] for x in quant_params)
    )  # not equal quantization params.

    # Update the `signature_1` with stats from `signature_2`.
    calib_utils = calibration_utils.CalibrationQsvAlignmentUtils(model_path)
    min_val, max_val = calib_utils.align_quant_stats(  # for min and max only.
        qsv,
        {
            "signature_2": ["output_1_1"],
        },
    )
    calib_utils.update_quant_stats(
        qsv,
        {
            "signature_1": ["output_1_1"],
        },
        min_val,
        max_val,
    )
    quantized_model = qt.quantize(qsv).quantized_model
    quant_params = _get_quant_parameters(quantized_model, signature_data)
    self.assertTrue(
        all(x == quant_params[0] for x in quant_params)
    )  # equal quantization params.


if __name__ == "__main__":
  googletest.main()
