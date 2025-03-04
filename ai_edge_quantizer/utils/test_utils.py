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

"""Utils for tests."""

import inspect as _inspect
import os.path as _os_path
import sys as _sys

from absl.testing import parameterized

from ai_edge_quantizer import model_validator
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import tfl_interpreter_utils

_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig
_AlgorithmName = quantizer.AlgorithmName


def get_path_to_datafile(path):
  """Get the path to the specified file in the data dependencies.

  The path is relative to the file calling the function.

  Args:
    path: a string resource path relative to the calling file.

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  data_files_path = _os_path.dirname(_inspect.getfile(_sys._getframe(1)))  # pylint: disable=protected-access
  path = _os_path.join(data_files_path, path)
  path = _os_path.normpath(path)
  return path


class BaseOpTestCase(parameterized.TestCase):
  """Base class for op-level tests."""

  def quantize_and_validate(
      self,
      model_path: str,
      algorithm_key: _AlgorithmName,
      op_name: _OpName,
      op_config: _OpQuantConfig,
      num_validation_samples: int = 4,
      error_metric: str = 'mse',
  ) -> model_validator.ComparisonResult:
    """Quantizes and validates the given model with the given configurations.

    Args:
      model_path: The path to the model to be quantized.
      algorithm_key: The algorithm to be used for quantization.
      op_name: The name of the operation to be quantized.
      op_config: The configuration for the operation to be quantized.
      num_validation_samples: The number of samples to use for validation.
      error_metric: The error error_metric to use for validation.

    Returns:
      The comparison result of the validation.
    """
    quantizer_instance = quantizer.Quantizer(model_path)
    quantizer_instance.update_quantization_recipe(
        algorithm_key=algorithm_key,
        regex='.*',
        operation_name=op_name,
        op_config=op_config,
    )
    if quantizer_instance.need_calibration:
      calibration_data = tfl_interpreter_utils.create_random_normal_input_data(
          quantizer_instance.float_model, num_samples=num_validation_samples * 8
      )
      calibration_result = quantizer_instance.calibrate(calibration_data)
      quantization_result = quantizer_instance.quantize(calibration_result)
    else:
      quantization_result = quantizer_instance.quantize()
    test_data = tfl_interpreter_utils.create_random_normal_input_data(
        quantization_result.quantized_model, num_samples=num_validation_samples
    )
    return quantizer_instance.validate(test_data, error_metric)

  def assert_model_size_reduction_above_min_pct(
      self,
      validation_result: model_validator.ComparisonResult,
      min_pct: float,
  ):
    """Checks the model size reduction (percentage) against the given expectation."""
    _, reduction_pct = validation_result.get_model_size_reduction()
    self.assertGreater(reduction_pct, min_pct)

  def assert_weights_errors_below_tolerance(
      self,
      validation_result: model_validator.ComparisonResult,
      weight_tolerance: float,
  ):
    """Checks the weight tensors' numerical behavior against the given tolerance."""
    self.assertNotEmpty(validation_result.available_signature_keys())
    for signature_key in validation_result.available_signature_keys():
      signature_result = validation_result.get_signature_comparison_result(
          signature_key
      )
      for result in signature_result.constant_tensors.values():
        self.assertLess(result, weight_tolerance)

  def assert_output_errors_below_tolerance(
      self,
      validation_result: model_validator.ComparisonResult,
      output_tolerance: float,
  ):
    """Checks the output tensor numerical behavior against the given tolerance."""
    self.assertNotEmpty(validation_result.available_signature_keys())
    for signature_key in validation_result.available_signature_keys():
      signature_result = validation_result.get_signature_comparison_result(
          signature_key
      )
      for result in signature_result.output_tensors.values():
        self.assertLess(result, output_tolerance)
