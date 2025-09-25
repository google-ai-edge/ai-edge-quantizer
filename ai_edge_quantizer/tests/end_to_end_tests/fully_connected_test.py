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

"""E2E tests for the quantizer for model with fully connected op."""

import os

from absl.testing import parameterized

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils


_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig
_AlgorithmName = quantizer.AlgorithmName

_TEST_MODEL_FOLDER = test_utils.get_path_to_datafile(
    '../models/dequantized_weights/'
)


# TODO: b/398335637 - Better ways to construct op configs for different cases.
class FullyConnectedTest(test_utils.BaseOpTestCase):

  def setUp(self):
    super().setUp()
    self._op_name = _OpName.FULLY_CONNECTED

  @parameterized.product(
      # (algorithm_name, weight_tolerance, output_tolerance)
      algorithm_and_tolerances=[
          (_AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY, 1e-5, 1e-5),
          (_AlgorithmName.MIN_MAX_UNIFORM_QUANT, 1e-2, 1e-1),
          (_AlgorithmName.OCTAV, 1e-3, 1e-1),
          # Only check size reduction for MSE.
          (_AlgorithmName.MSE, 10, 10),
      ],
      # (model_name, granularity)
      model_and_granularity=[
          (
              'tensor_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.TENSORWISE,
          ),
          (
              'channel_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.CHANNELWISE,
          ),
      ],
      weight_bit_width=[4, 8],
  )
  def test_weight_only_quantization_accuracy_and_size_within_tolerance(
      self, algorithm_and_tolerances, model_and_granularity, weight_bit_width
  ):
    algorithm_key, weight_tolerance, output_tolerance = algorithm_and_tolerances
    model, granularity = model_and_granularity
    model_path = os.path.join(_TEST_MODEL_FOLDER, model)
    op_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_bit_width,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.FLOAT,
        explicit_dequantize=True,
    )
    expected_model_size_reduction = (
        80 if op_config.weight_tensor_config.num_bits == 4 else 65
    )
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )

  @parameterized.product(
      # (algorithm_name, weight_tolerance, output_tolerance)
      algorithm_and_tolerances=[
          (_AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY, 1e-5, 1e-2),
          (_AlgorithmName.MIN_MAX_UNIFORM_QUANT, 1e-2, 1e-1),
          (_AlgorithmName.OCTAV, 1e-3, 1e-1),
          (_AlgorithmName.MSE, 10, 10),
      ],
      # (model_name, granularity)
      model_and_granularity=[
          (
              'tensor_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.TENSORWISE,
          ),
          (
              'channel_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.CHANNELWISE,
          ),
      ],
      weight_bit_width=[4, 8],
  )
  def test_dynamic_quantization_accuracy_and_size_within_tolerance(
      self, algorithm_and_tolerances, model_and_granularity, weight_bit_width
  ):

    algorithm_key, weight_tolerance, output_tolerance = algorithm_and_tolerances
    model, granularity = model_and_granularity
    model_path = os.path.join(_TEST_MODEL_FOLDER, model)
    op_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_bit_width,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    expected_model_size_reduction = (
        80 if op_config.weight_tensor_config.num_bits == 4 else 65
    )
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )

  @parameterized.product(
      # (algorithm_name, weight_tolerance, output_tolerance)
      algorithm_and_tolerances=[
          (_AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY, 1e-5, 1e-1),
          (_AlgorithmName.MIN_MAX_UNIFORM_QUANT, 1e-2, 1e-1),
          (_AlgorithmName.OCTAV, 1e-3, 1e-1),
          # Only check size reduction for MSE.
          (_AlgorithmName.MSE, 10, 10),
      ],
      # (model_name, granularity)
      model_and_granularity=[
          (
              'tensor_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.TENSORWISE,
          ),
          (
              'channel_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.CHANNELWISE,
          ),
      ],
      weight_bit_width=[4, 8],
  )
  def test_a8static_quantization_accuracy_and_size_within_tolerance(
      self, algorithm_and_tolerances, model_and_granularity, weight_bit_width
  ):
    algorithm_key, weight_tolerance, output_tolerance = algorithm_and_tolerances
    model, granularity = model_and_granularity
    model_path = os.path.join(_TEST_MODEL_FOLDER, model)
    op_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=8,
            symmetric=False,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_bit_width,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    expected_model_size_reduction = (
        80 if op_config.weight_tensor_config.num_bits == 4 else 65
    )
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )

  @parameterized.product(
      # (algorithm_name, weight_tolerance, output_tolerance)
      algorithm_and_tolerances=[
          (_AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY, 1e-5, 1e-1),
          (_AlgorithmName.MIN_MAX_UNIFORM_QUANT, 1e-2, 1e-1),
          (_AlgorithmName.OCTAV, 1e-3, 1e-1),
          # Only check size reduction for MSE.
          (_AlgorithmName.MSE, 10, 10),
      ],
      # (model_name, granularity)
      model_and_granularity=[
          (
              'tensor_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.TENSORWISE,
          ),
          (
              'channel_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.CHANNELWISE,
          ),
      ],
      weight_bit_width=[4, 8],
  )
  def test_a16static_quantization_accuracy_and_size_within_tolerance(
      self, algorithm_and_tolerances, model_and_granularity, weight_bit_width
  ):
    algorithm_key, weight_tolerance, output_tolerance = algorithm_and_tolerances
    model, granularity = model_and_granularity
    model_path = os.path.join(_TEST_MODEL_FOLDER, model)
    op_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=16,
            symmetric=True,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_bit_width,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    expected_model_size_reduction = (
        80 if op_config.weight_tensor_config.num_bits == 4 else 65
    )
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )

  @parameterized.product(weight_bit_width=[4, 8])
  def test_hadamard_rotation_accuracy_and_size_within_tolerance(
      self, weight_bit_width
  ):
    algorithm_key = _AlgorithmName.HADAMARD_ROTATION
    # Soft skip weight errors because they're rotated hence not expected to
    # match.
    weight_tolerance = 1
    output_tolerance = 2e-2
    model_path = test_utils.get_path_to_datafile(
        '../models/conv_fc_mnist.tflite'
    )
    granularity = qtyping.QuantGranularity.CHANNELWISE
    op_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_bit_width,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    expected_model_size_reduction = (
        80 if op_config.weight_tensor_config.num_bits == 4 else 65
    )
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )


if __name__ == '__main__':
  googletest.main()
