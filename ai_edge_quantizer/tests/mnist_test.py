"""E2E tests for the quantizer using a toy MNIST model."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils

_OpExecutionMode = qtyping.OpExecutionMode
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig

_RNG = np.random.default_rng(66)


def _get_dummy_data(num_samples):
  data = []
  for _ in range(num_samples):
    data.append(
        {'conv2d_input': _RNG.uniform(size=(1, 28, 28, 1)).astype(np.float32)}
    )
  return data


def _get_calibration_data(num_samples: int = 256):
  return _get_dummy_data(num_samples)


def _get_test_data(num_samples: int = 8):
  return _get_dummy_data(num_samples)


class MNISTTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.float_model_path = test_utils.get_path_to_datafile(
        'models/conv_fc_mnist.tflite'
    )
    self._quantizer = quantizer.Quantizer(self.float_model_path)

  @parameterized.product(
      execution_mode=[_OpExecutionMode.WEIGHT_ONLY, _OpExecutionMode.DRQ],
      symmetric_weight=[True, False],
      channel_wise_weight=[True, False],
  )
  def test_mnist_toy_model_int8_weight_only(
      self, execution_mode, symmetric_weight, channel_wise_weight
  ):

    # asym DRQ is not supported
    # TODO: b/335254997 - fail when trying to use unsupported recipe.
    if execution_mode == _OpExecutionMode.DRQ and not symmetric_weight:
      return
    self._quantizer.update_quantization_recipe(
        regex='.*',
        operation_name=_OpName.FULLY_CONNECTED,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=symmetric_weight,
                channel_wise=channel_wise_weight,
            ),
            execution_mode=execution_mode,
        ),
    )
    _ = self._quantizer.quantize()
    # Check model size.
    self.assertLess(len(self._quantizer._result.quantized_model), 55000)

    comparion_result = self._quantizer.compare(error_metrics='mse')
    self._check_comparion_result(
        comparion_result,
        weight_tolerance=1e-2 if channel_wise_weight else 1e-1,
        logits_tolerance=1e-2 if channel_wise_weight else 1e-1,
        output_tolerance=1e-4 if channel_wise_weight else 1e-2,
    )

  @parameterized.product(
      execution_mode=[_OpExecutionMode.WEIGHT_ONLY, _OpExecutionMode.DRQ],
      symmetric_weight=[True, False],
  )
  def test_mnist_toy_model_int4_weight_only(
      self, execution_mode, symmetric_weight
  ):

    # Asym DRQ is not supported.
    # TODO: b/335254997 - Fail when trying to use unsupported recipe.
    if execution_mode == _OpExecutionMode.DRQ and not symmetric_weight:
      return
    self._quantizer.update_quantization_recipe(
        regex='.*',
        operation_name=_OpName.FULLY_CONNECTED,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=4,
                symmetric=symmetric_weight,
                channel_wise=True,
            ),
            execution_mode=execution_mode,
        ),
    )
    _ = self._quantizer.quantize()
    # Check model size.
    self.assertLess(len(self._quantizer._result.quantized_model), 30000)

    comparion_result = self._quantizer.compare(error_metrics='mse')
    # TODO: b/346787369 - Update the weight tolerance for int4.
    self._check_comparion_result(
        comparion_result,
        weight_tolerance=1000,
        logits_tolerance=2,
        output_tolerance=1e-2,
    )

  def test_mnist_toy_model_fp16_weight_only(self):
    self._quantizer.update_quantization_recipe(
        regex='.*',
        algorithm_key=quantizer.AlgorithmName.FLOAT_CASTING,
        operation_name=_OpName.FULLY_CONNECTED,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT
            ),
            execution_mode=_OpExecutionMode.WEIGHT_ONLY,
        ),
    )
    _ = self._quantizer.quantize()
    # Check model size.
    self.assertLess(len(self._quantizer._result.quantized_model), 105000)

    comparion_result = self._quantizer.compare(error_metrics='mse')
    self._check_comparion_result(
        comparion_result,
        weight_tolerance=1e-5,
        logits_tolerance=1e-5,
        output_tolerance=1e-5,
    )

  @parameterized.parameters(
      'recipes/conv_fc_mnist_a8w8_recipe.json',
      'recipes/conv_fc_mnist_a16w8_recipe.json',
  )
  def test_mnist_toy_model_full_intege(self, recipe_path):
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertTrue(self._quantizer.need_calibration)
    calibration_result = self._quantizer.calibrate(_get_calibration_data())
    quant_result = self._quantizer.quantize(calibration_result)
    # Check model size.
    self.assertLess(len(quant_result.quantized_model), 55000)

    comparion_result = self._quantizer.compare(
        error_metrics='mse', signature_test_data=_get_test_data()
    )
    self._check_comparion_result(
        comparion_result,
        weight_tolerance=1e-2,
        logits_tolerance=1e-1,
        output_tolerance=1e-4,
    )

  # TODO: b/345503484 - Check weight tensor type of the quantized model.
  def _check_comparion_result(
      self,
      comparion_result,
      weight_tolerance,
      logits_tolerance,
      output_tolerance,
  ):
    # Check weight tensors.
    conv_weight_mse = comparion_result['sequential/conv2d/Conv2D']
    self.assertLess(conv_weight_mse, weight_tolerance)
    fc1_weight_mse = comparion_result['arith.constant1']
    self.assertLess(fc1_weight_mse, weight_tolerance)
    fc2_weight_mse = comparion_result['arith.constant']
    self.assertLess(fc2_weight_mse, weight_tolerance)
    # check logits.
    logits_mse = comparion_result['sequential/dense_1/MatMul']
    self.assertLess(logits_mse, logits_tolerance)
    # check final output.
    output_mse = comparion_result['StatefulPartitionedCall:0']
    self.assertLess(output_mse, output_tolerance)


if __name__ == '__main__':
  googletest.main()
