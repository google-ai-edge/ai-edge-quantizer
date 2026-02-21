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

"""E2E tests for the quantizer for model input and output."""

from absl.testing import parameterized
import absl.testing.absltest as absltest
import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils


_OpExecutionMode = qtyping.OpExecutionMode
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_Granularity = qtyping.QuantGranularity
_OpQuantConfig = qtyping.OpQuantizationConfig
_ComputePrecision = qtyping.ComputePrecision

_RNG = np.random.default_rng(66)


def _get_dummy_data(num_samples):
  samples = []
  for _ in range(num_samples):
    samples.append(
        {'input_1': _RNG.uniform(size=(1, 32, 32)).astype(np.float32)}
    )

  data = {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: samples,
  }
  return data


def _get_calibration_data(num_samples: int = 128):
  return _get_dummy_data(num_samples)


def _get_test_data(num_samples: int = 8):
  return _get_dummy_data(num_samples)


class InputOutputTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.float_model_path = test_utils.get_path_to_datafile(
        '../models/single_mean.tflite'
    )
    self._quantizer = quantizer.Quantizer(self.float_model_path)

  @parameterized.named_parameters(
      dict(
          testcase_name='INT8_input',
          activation_tensor_config=_TensorQuantConfig(
              num_bits=8,
              symmetric=False,
              granularity=_Granularity.TENSORWISE,
          ),
          op=_OpName.INPUT,
      ),
      dict(
          testcase_name='INT8_output',
          activation_tensor_config=_TensorQuantConfig(
              num_bits=8,
              symmetric=False,
              granularity=_Granularity.TENSORWISE,
          ),
          op=_OpName.OUTPUT,
      ),
      dict(
          testcase_name='INT16_input',
          activation_tensor_config=_TensorQuantConfig(
              num_bits=16,
              symmetric=True,
              granularity=_Granularity.TENSORWISE,
          ),
          op=_OpName.INPUT,
      ),
      dict(
          testcase_name='INT16_output',
          activation_tensor_config=_TensorQuantConfig(
              num_bits=16,
              symmetric=True,
              granularity=_Granularity.TENSORWISE,
          ),
          op=_OpName.OUTPUT,
      ),
  )
  def test_input_output_explicit_set_quantize(
      self, activation_tensor_config, op
  ):
    self._quantizer.update_quantization_recipe(
        regex='.*',
        operation_name=op,
        op_config=_OpQuantConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=True,
                granularity=_Granularity.TENSORWISE,
            ),
            compute_precision=_ComputePrecision.INTEGER,
        ),
    )
    calibration_result = {}
    if self._quantizer.need_calibration:
      calibration_result = self._quantizer.calibrate(_get_calibration_data())
    quantization_result = self._quantizer.quantize(calibration_result)
    quantized_model = tfl_flatbuffer_utils.read_model(
        bytearray(quantization_result.quantized_model)
    )
    self.assertLen(quantized_model.subgraphs, 1)
    subgraph = quantized_model.subgraphs[0]
    subgraph_tensors = subgraph.tensors
    self.assertLen(subgraph.inputs, 1)
    input_tensor = subgraph_tensors[subgraph.inputs[0]]
    output_tensor = subgraph_tensors[subgraph.outputs[0]]
    # Check input/output tensor type.
    if op == _OpName.INPUT:
      # See schema_py_generated.py for type code.
      if activation_tensor_config is None:
        self.assertEqual(input_tensor.type, 0)  # float32.
      elif activation_tensor_config.num_bits == 8:
        self.assertEqual(input_tensor.type, 9)  # int8.
      elif activation_tensor_config.num_bits == 16:
        self.assertEqual(input_tensor.type, 7)  # int16.
      self.assertEqual(output_tensor.type, 0)
    else:
      self.assertEqual(input_tensor.type, 0)  # float32.
      if activation_tensor_config is None:
        self.assertEqual(output_tensor.type, 0)  # float32.
      elif activation_tensor_config.num_bits == 8:
        self.assertEqual(output_tensor.type, 9)  # int8.
      elif activation_tensor_config.num_bits == 16:
        self.assertEqual(output_tensor.type, 7)  # int16.

    # Check accuracy.
    comparison_result = self._quantizer.validate(
        error_metrics='mse', test_data=_get_test_data()
    )
    self._check_comparison_result(comparison_result, output_tolerance=1e-4)

  @parameterized.parameters(
      ('../../recipes/default_a8w8_recipe.json', 9),
      ('../../recipes/default_a16w8_recipe.json', 7),
      ('../../recipes/default_af32w8float_recipe.json', 0),
      ('../../recipes/dynamic_wi8_afp32_recipe.json', 0),
      ('../../recipes/dynamic_legacy_wi8_afp32_recipe.json', 0),
  )
  def test_input_output_with_default_recipe(
      self, recipe_path, activation_type_code
  ):
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    calibration_result = {}
    if self._quantizer.need_calibration:
      calibration_result = self._quantizer.calibrate(_get_calibration_data())
    quantization_result = self._quantizer.quantize(calibration_result)
    quantized_model = tfl_flatbuffer_utils.read_model(
        bytearray(quantization_result.quantized_model)
    )

    # Check input tensor type.
    self.assertLen(quantized_model.subgraphs, 1)
    subgraph = quantized_model.subgraphs[0]
    subgraph_tensors = subgraph.tensors
    self.assertLen(subgraph.inputs, 1)
    input_tensor = subgraph_tensors[subgraph.inputs[0]]
    output_tensor = subgraph_tensors[subgraph.outputs[0]]
    # See schema_py_generated.py for type code.
    self.assertEqual(input_tensor.type, activation_type_code)
    self.assertEqual(output_tensor.type, activation_type_code)
    # check accuracy.
    comparison_result = self._quantizer.validate(
        error_metrics='mse', test_data=_get_test_data()
    )
    self._check_comparison_result(comparison_result, output_tolerance=1e-4)

  def _check_comparison_result(self, comparison_result, output_tolerance):
    comparison_result = comparison_result.get_all_tensor_results()
    output_mse = comparison_result['PartitionedCall:0']
    self.assertLess(output_mse, output_tolerance)


if __name__ == '__main__':
  absltest.main()
