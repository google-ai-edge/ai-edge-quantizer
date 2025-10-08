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

"""AI Edge Quantizer API."""

from collections.abc import Iterable
import dataclasses
import json
import logging
import os
from typing import Any, Optional, Union

from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import calibrator
from ai_edge_quantizer import default_policy
from ai_edge_quantizer import model_modifier
from ai_edge_quantizer import model_validator
from ai_edge_quantizer import params_generator
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe_manager
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils
from ai_edge_quantizer.utils import validation_utils
from tensorflow.python.platform import gfile  # pylint: disable=g-direct-tensorflow-import


# Expose algorithm names to users.
AlgorithmName = algorithm_manager.AlgorithmName

_QuantRecipe = recipe_manager.ModelQuantizationRecipe
_TFLOpName = qtyping.TFLOperationName
_OpQuantizationConfig = qtyping.OpQuantizationConfig
_TensorQuantizationConfig = qtyping.TensorQuantizationConfig
_TensorTransformationParams = dict[str, qtyping.TensorTransformationParams]
_SignatureInput = dict[str, Any]  # input_argument_name -> tensor_value.
_CalibrationResult = dict[str, qtyping.QSV]


@dataclasses.dataclass(frozen=True)
class QuantizationResult:
  """Quantization result.

  Attributes:
    recipe: Quantization recipe.
    quantized_model: Quantized model.
  """

  recipe: _QuantRecipe
  quantized_model: Optional[bytearray]

  def save(
      self, save_folder: str, model_name: str, overwrite: bool = False
  ) -> None:
    """Saves the quantized model and the quantization recipe.

    Args:
      save_folder: Path to the folder to save the quantized model and the
        quantization recipe.
      model_name: Name of the model.
      overwrite: Whether to overwrite the model if it already exists.

    Raises:
      RuntimeError: If no quantized model is available.
    """
    if not gfile.Exists(save_folder):
      gfile.MakeDirs(save_folder)

    model_save_path = os.path.join(save_folder, f'{model_name}.tflite')
    self.export_model(model_save_path, overwrite)

    recipe_save_path = os.path.join(save_folder, model_name + '_recipe.json')
    recipe = json.dumps(self.recipe)
    with gfile.GFile(recipe_save_path, 'w') as output_file_handle:
      output_file_handle.write(recipe)

  def export_model(self, filepath: str, overwrite: bool = False) -> None:
    """Exports the quantized model to a .tflite flatbuffer.

    Args:
      filepath: Path (including file name) that the exported model should be
        serialized to.
      overwrite: Whether to overwrite the model if it already exists.

    Raises:
      RuntimeError: If no quantized model is available.
      ValueError: If the model already exists in the folder and overwrite is
        False.
    """
    if self.quantized_model is None:
      raise RuntimeError(
          'No quantized model to save. Make sure .quantize() is called.'
      )
    if gfile.Exists(filepath):
      if overwrite:
        logging.warning(
            'The model %s already exists in the folder. Overwriting the model'
            ' since overwrite=True.',
            filepath,
        )
      else:
        raise ValueError(
            f'The model {filepath} already exists in the folder. Please'
            ' consider change the model name or specify overwrite=True to'
            ' overwrite the model if needed.'
        )
    with gfile.GFile(filepath, 'wb') as output_file_handle:
      output_file_handle.write(self.quantized_model)


class Quantizer:
  """AI Edge Quantizer API.

  Attributes:
    float_model: TFLite model file path or bytearray.
    quantization_recipe: Quantization recipe .json filepath or in loaded json
      format.
    previous_quantized_model: Optional previously quantized TFLite model file
      path or bytearray. This is useful for validating a quantized model
      without quantizing it again.
  """

  def __init__(
      self,
      float_model: Union[str, bytearray],
      quantization_recipe: Optional[Union[str, _QuantRecipe]] = None,
      previous_quantized_model: Optional[Union[str, bytearray]] = None,
  ):
    """Initializes the quantizer.

    Args:
      float_model: Path to the float tflite model.
      quantization_recipe: Quantization recipe in .json filepath or loaded json
        format.
      previous_quantized_model: Path to an optional previously quantized tflite
        model. This is useful for validating a quantized model without
        quantizing it again.
    """
    # Use `float model` as bytes for memory efficiency.
    self.float_model: bytes = (
        tfl_flatbuffer_utils.get_model_content(float_model)
        if isinstance(float_model, str)
        else float_model
    )
    if previous_quantized_model is not None:
      self.previous_quantized_model: bytes = (
          tfl_flatbuffer_utils.get_model_content(previous_quantized_model)
          if isinstance(previous_quantized_model, str)
          else previous_quantized_model
      )
    else:
      self.previous_quantized_model = None

    self._recipe_manager: recipe_manager.RecipeManager = (
        recipe_manager.RecipeManager()
    )
    if quantization_recipe is not None:
      self.load_quantization_recipe(quantization_recipe)
    self._result: QuantizationResult = QuantizationResult([{}], None)
    self._quantize_called = False

  def load_quantization_recipe(self, recipe: Union[str, _QuantRecipe]) -> None:
    """Loads a quantization recipe.

    The existing recipe will be overwritten.

    Args:
      recipe: Quantization recipe in json format.
    """
    if isinstance(recipe, str):
      with gfile.Open(recipe) as json_file:
        recipe = json.load(json_file)
    self._recipe_manager.load_quantization_recipe(recipe)

  def load_config_policy(self, filename: str) -> None:
    """Loads a JSON policy.

    The existing policy will be overwritten.

    Args:
      filename: Config policy filename.
    """
    with gfile.Open(filename, 'r') as f:
      policy = default_policy.update_default_config_policy(f.read())

    # Register the policy for MIN_MAX_UNIFORM_QUANT algorithm.
    algorithm_manager.register_config_check_policy_func(
        AlgorithmName.MIN_MAX_UNIFORM_QUANT, policy
    )

  def get_quantization_recipe(self) -> _QuantRecipe:
    """Gets the quantization recipe.

    Returns:
      A quantization recipe.
    """
    return self._recipe_manager.get_quantization_recipe()

  def update_quantization_recipe(
      self,
      regex: str,
      operation_name: _TFLOpName,
      op_config: Optional[_OpQuantizationConfig] = None,
      algorithm_key: str = algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
  ):
    """Adds a quantization configuration to the recipe.

    Conflict arises when we are trying to set an operation under a certain regex
    which is already existed in the config dictionary. Under such circumstance,
    the new config is used to replace the previous one.

    We also have special treatment for _TFLOperationKey.ALL. If the new config
    is on _TFLOperationKey.ALL and there are existing op configs inside the same
    scope, we clear the previous configs and use _TFLOperationKey.ALL.

    Args:
      regex: Regular expression for layer name matching.
      operation_name: Target TFLite operation. * for all supported TFLite
        operation.
      op_config: Quantization configuration which will be used to update the
        default configuration. None or empty dict means the default
        configuration will be used.
      algorithm_key: Algorithm key to be applied.
    """
    self._recipe_manager.add_quantization_config(
        regex, operation_name, op_config, algorithm_key
    )

  def add_dynamic_config(
      self,
      regex: str,
      operation_name: _TFLOpName,
      num_bits: int,
      granularity: qtyping.QuantGranularity = qtyping.QuantGranularity.CHANNELWISE,
      algorithm_key: str = algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
  ):
    """Adds a dynamic quantization configuration to the recipe.

    During dynamic quantization, activations are not processed by AEQ and
    remain in float format. The runtime kernel is expected to quantize these
    activations on-the-fly, as indicated by compute_precision=Integer and
    explicit_dequantize=False.

    The model quality may suffer due to the on-the-fly quantization. If quality
    is a concern, consider using weight-only
    quantization.

    Args:
      regex: Regular expression for layer name (op's output tensor name)
        matching.
      operation_name: Target TFLite operation.
      num_bits: Number of bits for quantization.
      granularity: Granularity of quantization.
      algorithm_key: Algorithm key to be applied.
    """
    self._recipe_manager.add_dynamic_config(
        regex, operation_name, num_bits, granularity, algorithm_key
    )

  def add_weight_only_config(
      self,
      regex: str,
      operation_name: _TFLOpName,
      num_bits: int,
      granularity: qtyping.QuantGranularity = qtyping.QuantGranularity.CHANNELWISE,
      algorithm_key: str = algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
  ):
    """Adds a weight only quantization configuration to the recipe.

    In weight-only quantization, weights are quantized, but the actual operation
    (op) computation remains in float. The quantized weight is explicitly
    dequantized before being fed into the op. This is achieved by inserting a
    dequantize op between the quantized weight and the consuming op. To enable
    this, both compute_precision will be set to Float and explicit_dequantize to
    True.

    Weight-only quantization is useful for reducing model size but may
    not decrease latency due to float computation. However, quantized model
    generally has better quality than other quantization options (e.g., dynamic
    range quantization) due to no loss of precision on activations. If latency
    is a concern, consider using dynamic quantization.

    Args:
      regex: Regular expression for layer name matching.
      operation_name: Target TFLite operation.
      num_bits: Number of bits for quantization.
      granularity: Granularity of quantization.
      algorithm_key: Algorithm key to be applied.
    """
    self._recipe_manager.add_weight_only_config(
        regex, operation_name, num_bits, granularity, algorithm_key
    )

  def add_static_config(
      self,
      regex: str,
      operation_name: _TFLOpName,
      activation_num_bits: int,
      weight_num_bits: int,
      weight_granularity: qtyping.QuantGranularity = qtyping.QuantGranularity.CHANNELWISE,
      algorithm_key: str = algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
  ):
    """Adds a static quantization configuration to the recipe.

    In static quantization, both weights and activations are quantized. This
    requires a calibration step to determine the quantization parameters (e.g.,
    min/max ranges) for activations. The quantized model uses integer arithmetic
    for computations, which can lead to significant latency reductions.

    However, calibration is needed to determine the quantization parameters for
    activations, which requires sample data and may lead to quality loss. If
    there is no hardware requirement for full integer quantization, consider
    using dynamic quantization for simplicity.

    Args:
      regex: Regular expression for layer name matching.
      operation_name: Target TFLite operation.
      activation_num_bits: Number of bits for activation quantization.
      weight_num_bits: Number of bits for weight quantization.
      weight_granularity: Granularity of weight quantization.
      algorithm_key: Algorithm key to be applied.
    """
    self._recipe_manager.add_static_config(
        regex,
        operation_name,
        activation_num_bits,
        weight_num_bits,
        weight_granularity,
        algorithm_key,
    )

  @property
  def need_calibration(self) -> bool:
    """Checks if the current recipe needs calibration."""
    return self._recipe_manager.need_calibration()

  def calibrate(
      self,
      calibration_data: dict[str, Iterable[_SignatureInput]],
      previous_calibration_result: Optional[_CalibrationResult] = None,
      num_threads: int = 16,
  ) -> _CalibrationResult:
    """Calibrates the float model (required by static range quantization).

    Args:
      calibration_data: Calibration data for a model signature.
      previous_calibration_result: Previous calibration result to be loaded. The
        calibration process will be resumed from the previous result.
      num_threads: Number of threads to use for calibration.

    Returns:
      Calibration result ({tensor_name: tensor QSVs (e.g.,min/max)}).

    Raises:
      ValueError: If the calibration result is insufficient.
    """
    if not self.need_calibration:
      return {}

    calib = calibrator.Calibrator(self.float_model, num_threads=num_threads)
    if previous_calibration_result is not None:
      calib.load_model_qsvs(previous_calibration_result)
    calib.calibrate(calibration_data, self._recipe_manager)
    return calib.get_model_qsvs()

  def _ensure_model_qsv_sufficient(
      self, calibration_result: _CalibrationResult
  ):
    """Checks if the calibration result has sufficient QSV."""

    # Find all tensor names with empty entries.
    empty_qsvs = [key for key, value in calibration_result.items() if not value]

    # Go over every signature and check if empty entry tensor belongs to it.
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self.float_model
    )
    for signature_key in tfl_interpreter.get_signature_list():
      subgraph_idx = tfl_interpreter_utils.get_signature_main_subgraph_index(
          tfl_interpreter, signature_key
      )

      for tensor_detail in tfl_interpreter.get_tensor_details(subgraph_idx):
        tensor_name = tensor_detail['name']
        if tensor_name in empty_qsvs:
          raise ValueError(
              f'Missing QSVs (min/max) for tensor {tensor_name} in Signature'
              f" '{signature_key}'. Please check if Signature"
              f' {signature_key} has been calibrated.'
          )

  def quantize(
      self, calibration_result: Optional[_CalibrationResult] = None
  ) -> QuantizationResult:
    """Quantizes the float model.

    Args:
      calibration_result: Calibration result to be used for quantization (if
        needed, check with self.need_calibration).

    Returns:
      Quantization result.

    Raises:
      RuntimeError: If quantization recipe is empty.
    """
    self._quantize_called = True
    if calibration_result is not None:
      self._ensure_model_qsv_sufficient(calibration_result)

    if not self.get_quantization_recipe():
      raise RuntimeError('Can not quantize without a quantization recipe.')
    quant_params = self._get_quantization_params(calibration_result)
    quantized_model = self._get_quantized_model(quant_params)
    self._result = QuantizationResult(
        self.get_quantization_recipe(), quantized_model
    )
    return self._result

  def validate(
      self,
      test_data: Optional[dict[str, Iterable[_SignatureInput]]] = None,
      error_metrics: str = 'mse',
      use_xnnpack: bool = True,
      num_threads: int = 16,
  ) -> model_validator.ComparisonResult:
    """Numerical validation of the quantized model for a model signature.

    Side by side numerical comparison will be performed on all tensors in the
    quantized model against ones from the float model. If no test data is
    provided, random normal distributed data will be used. This test is intended
    to be SANITY check for the quality of the quantized model. End to end task
    specific test should be performed as the golden standard of the quantized
    model quality. The comparison result will be saved in json format if
    json_save_path is provided.

    Args:
      test_data: A dictionary of signature key and its correspending test input
        data that will be used for validation. If set to None, random normal
        distributed data will be used for all signatures in the model.
      error_metrics: Error metrics to be used for comparison.
      use_xnnpack: Whether to use the xnnpack library for validation.
      num_threads: Number of threads to use for validation.

    Returns:
      The comparison result.
    """
    if test_data is None:
      # Create test data for all signatures in the model.
      test_data = tfl_interpreter_utils.create_random_normal_input_data(
          self.float_model, num_samples=1
      )
    if self._quantize_called:
      quantized_model = self._result.quantized_model
    else:
      quantized_model = self.previous_quantized_model

    if quantized_model is None:
      raise ValueError('No quantized model available to validate.')
    return model_validator.compare_model(
        self.float_model,
        quantized_model,
        test_data,
        error_metrics,
        validation_utils.get_validation_func(error_metrics),
        use_xnnpack=use_xnnpack,
        num_threads=num_threads,
    )

  def _get_quantization_params(
      self, calibration_result: Optional[_CalibrationResult] = None
  ) -> _TensorTransformationParams:
    """Gets the quantization parameters.

    Args:
      calibration_result: Calibration result to be used for quantization (if
        needed, check with self.need_calibration).

    Returns:
      A dictionary containing the quantization parameters.
    """
    params_generator_instance = params_generator.ParamsGenerator(
        self.float_model
    )
    return params_generator_instance.generate_quantization_parameters(
        self._recipe_manager, calibration_result
    )

  def _get_quantized_model(
      self, quant_params: _TensorTransformationParams
  ) -> bytearray:
    """Gets the quantized model.

    Args:
      quant_params: A dictionary containing the quantization parameters.

    Returns:
      The quantized model.
    """
    model_modifier_instance = model_modifier.ModelModifier(self.float_model)
    return model_modifier_instance.modify_model(quant_params)
