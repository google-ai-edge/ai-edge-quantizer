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

"""Tests for calibrator."""

from collections.abc import Generator
import os
from typing import Any

import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import calibrator
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe_manager
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_ComputePrecision = qtyping.ComputePrecision
_AlgorithmName = recipe_manager.AlgorithmName

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("")
_TENSOR_QUANT_CONFIG = qtyping.TensorQuantizationConfig

TEST_MIN_VAL, TEST_MAX_VAL = -1, 1

_RNG = np.random.default_rng(66)


def _representative_dataset_gen(size=(1, 8), num_samples=10):
  for _ in range(num_samples):
    vals = np.random.rand(*size).astype(np.float32)
    vals[0][0], vals[0][1] = (
        TEST_MIN_VAL,
        TEST_MAX_VAL,
    )  # fix min/max for testing
    yield {"input_1": vals}


def _get_calibration_data(
    dataset_gen: Generator[dict[str, Any], Any, None],
) -> dict[str, Any]:
  calibration_samples = [sample for sample in dataset_gen]
  calibration_data = {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
  }
  return calibration_data


def _add_default_int8xint8_integer_recipe(recipe_manager_object):
  recipe_manager_object.add_quantization_config(
      regex=".*",
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      op_config=qtyping.OpQuantizationConfig(
          activation_tensor_config=_TENSOR_QUANT_CONFIG(
              num_bits=8, symmetric=False
          ),
          weight_tensor_config=_TENSOR_QUANT_CONFIG(num_bits=8, symmetric=True),
          compute_precision=_ComputePrecision.INTEGER,
      ),
  )


class CalibratorTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    self._test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/single_fc.tflite"
    )
    self._calibrator = calibrator.Calibrator(self._test_model_path)
    self._recipe_manager = recipe_manager.RecipeManager()
    dataset_gen = _representative_dataset_gen()
    self._representative_dataset = _get_calibration_data(dataset_gen)

  def test_calibrator_state_manipulation(self):
    # load/get qsvs
    sample_qsv = {"serving_default_input_1:0": {"min": -10, "max": 8}}
    self._calibrator.load_model_qsvs(sample_qsv)
    model_tensor_qsvs = self._calibrator.get_model_qsvs()
    self.assertLen(model_tensor_qsvs, 1)
    self.assertIn("serving_default_input_1:0", model_tensor_qsvs)  # input
    input_qsv = model_tensor_qsvs["serving_default_input_1:0"]
    self.assertEqual(input_qsv["min"], -10)
    self.assertEqual(input_qsv["max"], 8)

    # reset qsvs
    self._calibrator.reset_model_qsvs()
    model_tensor_qsvs = self._calibrator.get_model_qsvs()
    self.assertEmpty(model_tensor_qsvs)

  def test_calibrate_single_fc_success(self):
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager
    )
    model_tensor_qsvs = self._calibrator.get_model_qsvs()

    self.assertLen(model_tensor_qsvs, 2)
    self.assertIn("serving_default_input_1:0", model_tensor_qsvs)  # input
    input_qsv = model_tensor_qsvs["serving_default_input_1:0"]
    self.assertSequenceAlmostEqual(
        input_qsv["min"].flatten(), [TEST_MIN_VAL], delta=1e-5
    )
    self.assertSequenceAlmostEqual(
        input_qsv["max"].flatten(), [TEST_MAX_VAL], delta=1e-5
    )
    self.assertIn("StatefulPartitionedCall:0", model_tensor_qsvs)  # output
    output_qsv = model_tensor_qsvs["StatefulPartitionedCall:0"]
    # Relu, only check the min
    self.assertSequenceAlmostEqual(output_qsv["min"].flatten(), [0])

  def test_calibration_cache_is_empty_when_off(self):
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    self.assertEmpty(self._calibrator.get_cached_output())
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager, cache_output=False
    )
    self.assertEmpty(self._calibrator.get_cached_output())

  def test_calibration_cache_when_on(self):
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    self.assertEmpty(self._calibrator.get_cached_output())
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager, cache_output=True
    )
    self.assertLen(self._calibrator.get_cached_output(), 10)

  def test_calibration_cache_is_empty_after_reset(self):
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager, cache_output=True
    )
    self._calibrator.clear_cached_output()
    self.assertEmpty(self._calibrator.get_cached_output())

  def test_calibrate_unsupported_ops_success(self):
    # Many ops in the following model are not supported currently.
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/branching_conv_fc.tflite"
    )
    test_calibrator = calibrator.Calibrator(test_model_path)
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    dataset_gen = _representative_dataset_gen(size=(3, 4, 4, 1))
    test_calibrator.calibrate(
        _get_calibration_data(dataset_gen),
        self._recipe_manager,
        cache_output=True,
    )
    self.assertLen(test_calibrator.get_cached_output(), 10)

  def test_calibrate_reshape_with_empty_shape_success(self):
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/reshape_with_empty_shape.tflite"
    )
    test_calibrator = calibrator.Calibrator(test_model_path)
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    calib_data = tfl_interpreter_utils.create_random_normal_input_data(
        test_model_path, num_samples=4
    )
    test_calibrator.calibrate(calib_data, self._recipe_manager)
    self.assertNotEmpty(test_calibrator.get_model_qsvs())


class CalibratorAlreadyQuantizedModelTest(googletest.TestCase):

  def test_check_is_float_model_succeeds_when_model_is_float(self):
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/conv_fc_mnist.tflite"
    )
    _ = calibrator.Calibrator(test_model_path)

  def test_check_is_float_model_raises_error_when_model_is_quantized(self):
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/mnist_quantized.tflite"
    )
    with self.assertRaisesRegex(
        ValueError,
        "The input model for calibration is not a float model.",
    ):
      _ = calibrator.Calibrator(test_model_path)


class CalibratorToyGemma2Test(googletest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)

    self._test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH,
        "tests/models/toy_model_with_kv_cache_multi_signature.tflite",
    )

    self._toy_gemma2_calibration_dataset = {
        "signature_1": [{
            "cache_0": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            "cache_1": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            "positions": _RNG.integers(low=0, high=10, size=(1, 100)).astype(
                np.int32
            ),
            "tokens": _RNG.integers(low=0, high=10, size=(1, 100)).astype(
                np.int32
            ),
        }],
        "signature_2": [{
            "cache_0": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            "cache_1": _RNG.random(size=(1, 100, 4, 4), dtype=np.float32),
            "positions": _RNG.integers(low=0, high=10, size=(1, 100)).astype(
                np.int32
            ),
            "tokens": _RNG.integers(low=0, high=10, size=(1, 100)).astype(
                np.int32
            ),
        }],
    }

  def test_toy_gemma2_calibration_success(self):
    calib = calibrator.Calibrator(self._test_model_path)
    recipe_mngr = recipe_manager.RecipeManager()
    _add_default_int8xint8_integer_recipe(recipe_mngr)
    calib.calibrate(
        self._toy_gemma2_calibration_dataset,
        model_recipe_manager=recipe_mngr,
    )
    self.assertLen(calib.get_model_qsvs(), 202)


if __name__ == "__main__":
  googletest.main()
