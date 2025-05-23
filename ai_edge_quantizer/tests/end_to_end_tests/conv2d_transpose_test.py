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

"""E2E tests for the quantizer for model with conv2d tranpose."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig

_RNG = np.random.default_rng(66)


def _get_dummy_data(num_samples):
  samples = []
  for _ in range(num_samples):
    samples.append(
        {'input_6': _RNG.uniform(size=(1, 16, 16, 1)).astype(np.float32)}
    )
  data = {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: samples,
  }
  return data


def _get_calibration_data(num_samples: int = 128):
  return _get_dummy_data(num_samples)


def _get_test_data(num_samples: int = 8):
  return _get_dummy_data(num_samples)


class Conv2DTransposeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.float_model_path = test_utils.get_path_to_datafile(
        '../models/single_conv2d_transpose_bias.tflite'
    )
    self._quantizer = quantizer.Quantizer(self.float_model_path)

  @parameterized.product(
      symmetric_weight=[True, False],
      granularity=[
          qtyping.QuantGranularity.CHANNELWISE,
          qtyping.QuantGranularity.TENSORWISE,
      ],
  )
  def test_conv2d_transpose_model_int8_weight_only(
      self, symmetric_weight, granularity
  ):
    self._quantizer.update_quantization_recipe(
        regex='.*',
        operation_name=_OpName.CONV_2D_TRANSPOSE,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=symmetric_weight,
                granularity=granularity,
            ),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY.
            explicit_dequantize=True,
        ),
    )
    quant_result = self._quantizer.quantize()
    # Check model size.
    self.assertLess(len(quant_result.quantized_model), 10000)

    comparion_result = self._quantizer.validate(error_metrics='mse')
    self._check_comparion_result(
        comparion_result,
        weight_tolerance=1e-2 if granularity else 1e-1,
        output_tolerance=1e-4 if granularity else 1e-2,
    )

  @parameterized.parameters(
      '../../recipes/dynamic_legacy_wi8_afp32_recipe.json',
      '../../recipes/dynamic_wi8_afp32_recipe.json',
  )
  def test_conv2d_transpose_model_int8_drq(self, recipe_path):
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)

    # Check model size.
    quant_result = self._quantizer.quantize()
    self.assertLess(len(quant_result.quantized_model), 10000)

    comparion_result = self._quantizer.validate(error_metrics='mse')
    self._check_comparion_result(
        comparion_result,
        weight_tolerance=1e-2,
        output_tolerance=1e-4,
    )

  @parameterized.parameters(
      '../../recipes/default_a8w8_recipe.json',
      '../../recipes/default_a16w8_recipe.json',
  )
  def test_conv2d_transpose_model_full_integer(self, recipe_path):
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertTrue(self._quantizer.need_calibration)
    calibration_result = self._quantizer.calibrate(_get_calibration_data())
    quant_result = self._quantizer.quantize(calibration_result)
    # Check model size.
    self.assertLess(len(quant_result.quantized_model), 10000)

    comparion_result = self._quantizer.validate(
        error_metrics='mse', test_data=_get_test_data()
    )
    self._check_comparion_result(
        comparion_result,
        weight_tolerance=1e-2,
        output_tolerance=1e-4,
    )

  def test_conv2d_transpose_model_fp16_weight_only(self):
    self._quantizer.update_quantization_recipe(
        regex='.*',
        algorithm_key=quantizer.AlgorithmName.FLOAT_CASTING,
        operation_name=_OpName.CONV_2D_TRANSPOSE,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT
            ),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY.
            explicit_dequantize=True,
        ),
    )
    quant_result = self._quantizer.quantize()
    # Check model size.
    self.assertLess(len(quant_result.quantized_model), 10000)

    comparion_result = self._quantizer.validate(error_metrics='mse')
    self._check_comparion_result(
        comparion_result,
        weight_tolerance=1e-5,
        output_tolerance=1e-5,
    )

  # TODO: b/345503484 - Check weight tensor type of the quantized model.
  def _check_comparion_result(
      self,
      comparion_result,
      weight_tolerance,
      output_tolerance,
  ):
    # TODO: b/357959309 - Use comparison result directly for testing.
    comparion_result = comparion_result.get_all_tensor_results()
    # Check weight tensors.
    weight_mse = comparion_result[
        'sequential_5/conv2d_transpose_3/conv2d_transpose'
    ]
    self.assertLess(weight_mse, weight_tolerance)
    bias_mse = comparion_result[
        'sequential_5/conv2d_transpose_3/BiasAdd;sequential_5/conv2d_transpose_3/conv2d_transpose;sequential_5/conv2d_transpose_3/BiasAdd/ReadVariableOp'
    ]
    self.assertLess(bias_mse, weight_tolerance)
    # Check final output.
    output_mse = comparion_result['StatefulPartitionedCall:0']
    self.assertLess(output_mse, output_tolerance)


if __name__ == '__main__':
  googletest.main()
