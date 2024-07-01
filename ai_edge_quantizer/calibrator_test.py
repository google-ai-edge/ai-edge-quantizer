"""Tests for calibrator."""

import os

import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import calibrator
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe_manager
from ai_edge_quantizer.utils import test_utils

_OpExecutionMode = qtyping.OpExecutionMode
_AlgorithmName = recipe_manager.AlgorithmName

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("")
_TENSOR_QUANT_CONFIG = qtyping.TensorQuantizationConfig

TEST_MIN_VAL, TEST_MAX_VAL = -1, 1


def _representative_dataset_gen(size=(1, 8), num_samples=10):
  for _ in range(num_samples):
    vals = np.random.rand(*size).astype(np.float32)
    vals[0][0], vals[0][1] = (
        TEST_MIN_VAL,
        TEST_MAX_VAL,
    )  # fix min/max for testing
    yield {"input_1": vals}


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
          execution_mode=_OpExecutionMode.SRQ,
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
    self._representative_dataset = _representative_dataset_gen()

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

  def test_calibrator_initialize_qsv(self):
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    # Overwrite the single op to fc
    self._recipe_manager.add_quantization_config(
        regex=".*Stateful.*",
        operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TENSOR_QUANT_CONFIG(
                num_bits=4, channel_wise=True
            ),
        ),
    )
    self._calibrator._initialize_model_qsvs(self._recipe_manager)
    model_tensor_qsvs = self._calibrator.get_model_qsvs()

    self.assertLen(model_tensor_qsvs, 4)
    self.assertIn("serving_default_input_1:0", model_tensor_qsvs)  # input
    input_qsv = model_tensor_qsvs["serving_default_input_1:0"]
    self.assertTupleEqual(input_qsv["min"].shape, (1, 1))
    self.assertEqual(input_qsv["min"], np.array(0.0))
    self.assertTupleEqual(input_qsv["max"].shape, (1, 1))
    self.assertEqual(input_qsv["max"], np.array(6.0))

    self.assertIn("sequential/dense/MatMul", model_tensor_qsvs)  # weight
    weight_tensor_qsv = model_tensor_qsvs["sequential/dense/MatMul"]
    mins_maxs_shape = (16, 1)
    self.assertTupleEqual(weight_tensor_qsv["min"].shape, mins_maxs_shape)
    self.assertAlmostEqual(weight_tensor_qsv["min"][0][0], -0.40436327)
    self.assertTupleEqual(weight_tensor_qsv["max"].shape, mins_maxs_shape)
    self.assertAlmostEqual(weight_tensor_qsv["max"][0][0], 0.46138108)

    self.assertIn(
        "sequential/dense/BiasAdd/ReadVariableOp", model_tensor_qsvs
    )  # bias
    bias_tensor_qsv = model_tensor_qsvs[
        "sequential/dense/BiasAdd/ReadVariableOp"
    ]
    mins_maxs_shape = (16,)
    self.assertTupleEqual(bias_tensor_qsv["min"].shape, mins_maxs_shape)
    self.assertAlmostEqual(bias_tensor_qsv["min"][0], -0.26978338)
    self.assertTupleEqual(bias_tensor_qsv["max"].shape, mins_maxs_shape)
    # Here bias min/max will be the same as each element is a scalar
    # Bias will be quantized with input_scale * weight_scale.
    self.assertSequenceEqual(
        list(bias_tensor_qsv["max"].flatten()),
        list(bias_tensor_qsv["min"].flatten()),
    )

    self.assertIn("StatefulPartitionedCall:0", model_tensor_qsvs)  # output
    output_qsv = model_tensor_qsvs["StatefulPartitionedCall:0"]
    self.assertEqual(output_qsv["min"], [0])
    self.assertEqual(output_qsv["max"], np.array(6.0))

  def test_calibrate_single_fc_success(self):
    _add_default_int8xint8_integer_recipe(self._recipe_manager)
    self._calibrator.calibrate(
        self._representative_dataset, self._recipe_manager
    )
    model_tensor_qsvs = self._calibrator.get_model_qsvs()

    self.assertLen(model_tensor_qsvs, 4)
    self.assertIn("serving_default_input_1:0", model_tensor_qsvs)  # input
    input_qsv = model_tensor_qsvs["serving_default_input_1:0"]
    self.assertSequenceAlmostEqual(
        input_qsv["min"].flatten(), [-0.09561], delta=1e-5
    )
    self.assertSequenceAlmostEqual(
        input_qsv["max"].flatten(), [5.52191], delta=1e-5
    )

    self.assertIn("sequential/dense/MatMul", model_tensor_qsvs)  # weight
    weight_qsv = model_tensor_qsvs["sequential/dense/MatMul"]
    self.assertSequenceAlmostEqual(weight_qsv["min"].flatten(), [-0.49114203])
    self.assertSequenceAlmostEqual(weight_qsv["max"].flatten(), [0.4903704])

    self.assertIn(
        "sequential/dense/BiasAdd/ReadVariableOp", model_tensor_qsvs
    )  # bias
    bias_qsv = model_tensor_qsvs["sequential/dense/BiasAdd/ReadVariableOp"]
    self.assertSequenceAlmostEqual(bias_qsv["min"].flatten(), [-0.38401994])
    self.assertSequenceAlmostEqual(bias_qsv["max"].flatten(), [0.31727126])

    self.assertIn("StatefulPartitionedCall:0", model_tensor_qsvs)  # output
    output_qsv = model_tensor_qsvs["StatefulPartitionedCall:0"]
    # Relu, only check the min
    self.assertSequenceAlmostEqual(output_qsv["min"].flatten(), [0])

  def test_calibrate_unsupported_ops_fails(self):
    # Many ops in the following model are not supported currently.
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/branching_conv_fc.tflite"
    )
    test_calibrator = calibrator.Calibrator(test_model_path)

    _add_default_int8xint8_integer_recipe(self._recipe_manager)

    error_message = (
        "Full integer calibration requires all ops in the model to be"
        " supported."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      test_calibrator.calibrate(
          _representative_dataset_gen(size=(1, 28, 28, 1)), self._recipe_manager
      )

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


if __name__ == "__main__":
  googletest.main()
