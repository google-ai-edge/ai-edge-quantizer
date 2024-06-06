"""AI Edge Quantizer API."""

from collections.abc import Iterable
import dataclasses
import json
import os
from typing import Any, Optional
from ai_edge_quantizer import algorithm_manager
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
    float_model_path: Path to the float tflite model.
  """

  def __init__(
      self,
      float_model_path: str,
      quantization_recipe_path: Optional[str] = None,
  ):
    """Initializes the quantizer.

    Args:
      float_model_path: Path to the float tflite model.
      quantization_recipe_path: Path to the quantization recipe (.json).
    """
    self.float_model_path: str = float_model_path
    self._recipe_manager: recipe_manager.RecipeManager = (
        recipe_manager.RecipeManager()
    )
    if quantization_recipe_path is not None:
      self.load_quantization_recipe(quantization_recipe_path)
    self._result: QuantizationResult = QuantizationResult([{}], None)

  def load_quantization_recipe(self, recipe_path: str) -> None:
    """Loads a quantization recipe.

    The existing recipe will be overwritten.

    Args:
      recipe_path: Path to the quantization recipe (.json).
    """
    with gfile.Open(recipe_path) as json_file:
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
      override_algorithm: bool = True,
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
      override_algorithm: Flag to check if this rule overrides the previously
        matched rule with different algorithm key.
    """
    self._recipe_manager.add_quantization_config(
        regex, operation_name, op_config, algorithm_key, override_algorithm
    )

  def quantize(self) -> QuantizationResult:
    """Quantizes the float model.

    Returns:
      Quantization result.

    Raises:
      RuntimeError: If no quantization recipe is loaded.
    """
    if not self.get_quantization_recipe():
      raise RuntimeError('Can not quantize without a quantization recipe.')
    quant_params = self._get_quantization_params()
    quantized_model = self._get_quantized_model(quant_params)
    self._result = QuantizationResult(
        self.get_quantization_recipe(), quantized_model
    )
    return self._result

  # TODO: b/337299171 - generate a readable quantization report.
  def compare(
      self,
      test_data: Optional[Iterable[Any]] = None,
      error_metrics: str = 'mse',
  ) -> dict[str, float]:
    """Compares the quantized model with the float model.

    Side by side numerical comparison will be performed on all tensors in the
    quantized model against ones from the float model. If no test data is
    provided, random normal distributed data will be used. This test is intended
    to be SANITY check for the quality of the quantized model. End to end task
    specific test should be performed as the golden standard of the quantized
    model quality. The comparison result will be saved in json format if
    json_save_path is provided.

    Args:
      test_data: Test data to be used for comparison.
      error_metrics: Error metrics to be used for comparison.

    Returns:
      A dictionary containing the comparison result.
    """
    if test_data is None:
      test_data = test_utils.create_random_normal_input_data(
          self.float_model_path
      )
    # TODO: b/337296308 - directly pass the quantized model for comparison.
    quantized_model_path = '/tmp/quantized_model.tflite'
    with gfile.GFile(quantized_model_path, 'wb') as output_file_handle:
      output_file_handle.write(self._result.quantized_model)

    comparison_result = model_validator.compare_model(
        self.float_model_path,
        quantized_model_path,
        test_data,
        quantize_target_input=False,  # will be removed later.
        compare_fn=validation_utils.get_validation_func(error_metrics),
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
      self,
  ) -> _TensorTransformationParams:
    """Gets the quantization parameters.

    Returns:
      A dictionary containing the quantization parameters.
    """
    params_generator_instance = params_generator.ParamsGenerator(
        self.float_model_path
    )
    return params_generator_instance.generate_quantization_parameters(
        self._recipe_manager
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
        self.float_model_path
    )
    return model_modifier_instance.modify_model(quant_params)
