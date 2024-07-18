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
import os
from typing import Any, Optional, Union
from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import calibrator
from ai_edge_quantizer import model_modifier
from ai_edge_quantizer import model_validator
from ai_edge_quantizer import params_generator
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe_manager
from ai_edge_quantizer.utils import test_utils
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

  def save(self, save_folder: str, model_name: str) -> None:
    """Saves the quantized model and the quantization recipe.

    Args:
      save_folder: Path to the folder to save the quantized model and the
        quantization recipe.
      model_name: Name of the model.

    Raises:
      RuntimeError: If no quantized model is available.
    """
    if self.quantized_model is None:
      raise RuntimeError(
          'No quantized model to save. Make sure .quantize() is called.'
      )

    # Nested to group recipe and model under the same folder.
    save_folder = os.path.join(save_folder, model_name)
    gfile.MakeDirs(save_folder)

    model_save_path = os.path.join(
        save_folder, model_name + '_quantized.tflite'
    )
    with gfile.GFile(model_save_path, 'wb') as output_file_handle:
      output_file_handle.write(self.quantized_model)

    recipe = json.dumps(self.recipe)
    recipe_save_path = os.path.join(save_folder, model_name + '_recipe.json')
    with gfile.GFile(recipe_save_path, 'w') as output_file_handle:
      output_file_handle.write(recipe)


class Quantizer:
  """AI Edge Quantizer API.

  Attributes:
    float_model: TFLite model file path or bytearray.
    quantization_recipe: Quantization recipe .json filepath or in loaded json
      format.
  """

  def __init__(
      self,
      float_model: Union[str, bytearray],
      quantization_recipe: Optional[Union[str, _QuantRecipe]] = None,
  ):
    """Initializes the quantizer.

    Args:
      float_model: Path to the float tflite model.
      quantization_recipe: Quantization recipe in .json filepath or loaded json
        format.
    """
    self.float_model: Union[str, bytearray] = float_model
    self._recipe_manager: recipe_manager.RecipeManager = (
        recipe_manager.RecipeManager()
    )
    if quantization_recipe is not None:
      self.load_quantization_recipe(quantization_recipe)
    self._result: QuantizationResult = QuantizationResult([{}], None)

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

  @property
  def need_calibration(self) -> bool:
    """Checks if the current recipe needs calibration."""
    return self._recipe_manager.need_calibration()

  def calibrate(
      self,
      calibration_data: Iterable[_SignatureInput],
      signature_key: Optional[str] = None,
      previous_calibration_result: Optional[_CalibrationResult] = None,
  ) -> _CalibrationResult:
    """Calibrates the float model (required by static range quantization).

    Args:
      calibration_data: Calibration data for a model signature.
      signature_key: The signature key to be used for invoking the models. If
        the model doesn't have a signature key, this can be set to None.
      previous_calibration_result: Previous calibration result to be loaded. The
        calibration process will be resumed from the previous result.

    Returns:
      Calibration result ({tensor_name: tensor QSVs (e.g.,min/max)}).
    """
    if not self.need_calibration:
      return {}

    calib = calibrator.Calibrator(self.float_model)
    if previous_calibration_result is not None:
      calib.load_model_qsvs(previous_calibration_result)
    calib.calibrate(calibration_data, self._recipe_manager, signature_key)
    return calib.get_model_qsvs()

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

    if not self.get_quantization_recipe():
      raise RuntimeError('Can not quantize without a quantization recipe.')
    quant_params = self._get_quantization_params(calibration_result)
    quantized_model = self._get_quantized_model(quant_params)
    self._result = QuantizationResult(
        self.get_quantization_recipe(), quantized_model
    )
    return self._result

  # TODO: b/337299171 - generate a readable quantization report.
  def compare(
      self,
      signature_test_data: Optional[Iterable[_SignatureInput]] = None,
      error_metrics: str = 'mse',
      signature_key: Optional[str] = None,
  ) -> dict[str, float]:
    """Compares the quantized model with the float model for a model signature.

    Side by side numerical comparison will be performed on all tensors in the
    quantized model against ones from the float model. If no test data is
    provided, random normal distributed data will be used. This test is intended
    to be SANITY check for the quality of the quantized model. End to end task
    specific test should be performed as the golden standard of the quantized
    model quality. The comparison result will be saved in json format if
    json_save_path is provided.

    Args:
      signature_test_data: Test data to be used for comparison for a model
        signature.
      error_metrics: Error metrics to be used for comparison.
      signature_key: the signature key to be used for invoking the models. If
        the model doesn't have a signature key, this can be set to None.

    Returns:
      A dictionary containing the comparison result.
    """
    if signature_test_data is None:
      test_data = test_utils.create_random_normal_input_data(
          self.float_model
      )
      if signature_key is not None:
        signature_test_data = test_data[signature_key]
      else:
        if len(test_data) != 1:
          raise ValueError(
              'The model has multiple signatures but no signature key is'
              ' provided for comparison.'
          )
        signature_test_data = list(test_data.values())[0]  # single signature.

    comparison_result = model_validator.compare_model(
        self.float_model,
        self._result.quantized_model,
        signature_test_data,
        compare_fn=validation_utils.get_validation_func(error_metrics),
        signature_key=signature_key,
    )
    return comparison_result

  def save_comparison_result(
      self,
      comparison_result: dict[str, float],
      json_save_path: str,
      color_threshold: list[float],
  ) -> None:
    """Saves the comparison result in json format to be visualized through Model Explorer.

    Args:
      comparison_result: A dictionary containing the comparison result.
      json_save_path: Path to save the comparison result in json format to be
        visualized through Model Explorer.
      color_threshold: Thresholds for color coding the comparison result when
        visualize through Model Explorer.
    """
    json_object = model_validator.create_json_for_model_explorer(
        comparison_result,
        threshold=color_threshold,
    )
    with gfile.GFile(json_save_path, 'w') as output_file_handle:
      output_file_handle.write(json_object)

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
    model_modifier_instance = model_modifier.ModelModifier(
        self.float_model
    )
    return model_modifier_instance.modify_model(quant_params)
