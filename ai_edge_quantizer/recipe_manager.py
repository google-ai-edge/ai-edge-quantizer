"""Manages model quantization recipe (configuration) for the quantizer."""

import collections
import dataclasses
import re
from typing import Any, Optional
from absl import logging
from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import qtyping

# A collection of quantization configuration.
# Key: scope regex.
# Value: list of OpQuantizationRecipe in dictionary format.
ModelQuantizationRecipe = list[dict[str, Any]]
# Expose algorithm names to users.
AlgorithmName = algorithm_manager.AlgorithmName

_TFLOpName = qtyping.TFLOperationName
_OpQuantizationConfig = qtyping.OpQuantizationConfig
_TensorQuantizationConfig = qtyping.TensorQuantizationConfig


@dataclasses.dataclass
class OpQuantizationRecipe:
  """Dataclass for quantization configuration under a scope."""

  # Regular expression for scope name matching.
  regex: str

  # Target TFL operation. * for any supported TFL operation.
  operation: _TFLOpName

  # Algorithm key to be applied.
  algorithm_key: str

  # Quantization configuration to be applied for the op.
  op_config: _OpQuantizationConfig = dataclasses.field(
      default_factory=_OpQuantizationConfig
  )

  # Flag to check if this rule overrides the previous matched rule with
  # different algorithm key. Used when the algorithm keys of previous matched
  # config and the current config are different. When set to true, the
  # previously matched config is ignored; otherwise, the current matched config
  # is ignored.
  # When the algorithm keys of both configs are the same, then this flag does
  # not have any effect; the op_config of previously matched config is updated
  # using the op_config of this one.
  override_algorithm: bool = True


class RecipeManager:
  """Sets the quantization recipe for target model.

  Very similar design as mojax/flax_quantizer/configurator.py
  """

  def __init__(self):
    """Scope name config.

    Key: scope regex. ".*" for all scopes.
    Value: list of operator quantization settings under the scope.
    The priority between rules are determined by the order they entered: later
    one has higher priority.
    """
    self._scope_configs: collections.OrderedDict[
        str, list[OpQuantizationRecipe]
    ] = collections.OrderedDict()

  # TODO: b/335254997 - Check if an op quantization config is supported.
  def add_quantization_config(
      self,
      regex: str,
      operation_name: _TFLOpName,
      op_config: Optional[_OpQuantizationConfig] = None,
      algorithm_key: str = algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      override_algorithm: bool = True,
  ) -> None:
    """Adds a quantization configuration.

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
    if op_config is None:
      op_config = _OpQuantizationConfig()

    config = OpQuantizationRecipe(
        regex, operation_name, algorithm_key, op_config, override_algorithm
    )
    # Special care if trying to set all ops to some config.
    if config.operation == _TFLOpName.ALL_SUPPORTED:
      logging.warning(
          'Reset all op configs under scope_regex %s with %s.',
          regex,
          config,
      )
      self._scope_configs[regex] = [config]
      return

    if algorithm_key != AlgorithmName.NO_QUANTIZE:
      algorithm_manager.check_op_quantization_config(
          algorithm_key, operation_name, op_config
      )

    if regex not in self._scope_configs:
      self._scope_configs[regex] = [config]
    else:
      # Reiterate configs to avoid duplication on op settings.
      configs = []
      is_new_op = True
      for existing_config in self._scope_configs[regex]:
        if existing_config.operation == config.operation:
          is_new_op = False
          op_config = config
          logging.warning(
              'Overwrite operation %s config under scope_regex %s with %s.',
              existing_config.operation,
              regex,
              config,
          )
        else:
          op_config = existing_config
        configs.append(op_config)
      if is_new_op:
        configs.append(config)
      self._scope_configs[regex] = configs

  # TODO: b/348469513 - Remove the override_algorithm flag.
  def get_quantization_configs(
      self,
      target_op_name: _TFLOpName,
      scope_name: str,
  ) -> tuple[str, _OpQuantizationConfig]:
    """Gets the algorithm key and quantization configuration for an op.

    We respect the latest valid config and fall back to no quantization.
    Specifically, we search the quantization configuration in the order of the
    scope configs. If there are two or more matching rules, if the same
    quantization algorithms are assigned for both rules, then we will overwrite
    the quantization config with the later one (if it is valid). If the assigned
    algorithms are different,override_algorithm flag is used to see which
    algorithm will be used. If the flag is True, the latter is used. If the flag
    is False, the latter is ignored. We will fall to no quantization if no
    matching rule is found or all matched configs are invalid.


    Args:
      target_op_name: Target TFLite operation. * for all supported TFLite
        operation.
      scope_name: Name of the target scope.

    Returns:
       A tuple of quantization algorithm, and quantization configuration.
    """
    result_key, result_config, selected_recipe = (
        AlgorithmName.NO_QUANTIZE,
        _OpQuantizationConfig(),
        None,
    )
    for scope_regex, recipes in self._scope_configs.items():
      if re.search(scope_regex, scope_name):
        for recipe in recipes:
          if (
              recipe.operation != _TFLOpName.ALL_SUPPORTED
              and recipe.operation != target_op_name
          ):
            continue
          if (
              result_key != recipe.algorithm_key
              and not recipe.override_algorithm
          ):
            continue
          selected_recipe = recipe
          # The selected recipe must contain a supported config.
          try:
            algorithm_manager.check_op_quantization_config(
                recipe.algorithm_key, target_op_name, recipe.op_config
            )
          except ValueError:
            continue
          result_config = selected_recipe.op_config
          result_key = selected_recipe.algorithm_key

    if (
        selected_recipe is not None
        and selected_recipe.operation == _TFLOpName.ALL_SUPPORTED
        and result_config != selected_recipe.op_config
    ):
      logging.warning(
          'Ignored operation %s with config %s under scope_regex %s. Since the'
          ' specified quantization config is not supported at the moment.'
          ' (Triggered by quantizing ALL_SUPPORTED ops under a scope.)',
          target_op_name,
          selected_recipe.op_config,
          selected_recipe.regex,
      )
    return result_key, result_config

  def get_quantization_recipe(self) -> ModelQuantizationRecipe:
    """Gets the full quantization recipe from the manager.

    Returns:
      A list of quantization configs in the recipe.
    """
    ret = []
    for _, scope_config in self._scope_configs.items():
      for quant_config in scope_config:
        config = dict()
        config['regex'] = quant_config.regex
        config['operation'] = quant_config.operation
        config['algorithm_key'] = quant_config.algorithm_key
        config['op_config'] = quant_config.op_config.to_dict()
        config['override_algorithm'] = quant_config.override_algorithm
        ret.append(config)
    return ret

  def load_quantization_recipe(
      self, quantization_recipe: ModelQuantizationRecipe
  ) -> None:
    """Loads the quantization recipe to the manager.

    Args:
      quantization_recipe: A configuration dictionary which is generated by
        get_full_config.
    """
    self._scope_configs = collections.OrderedDict()
    for config in quantization_recipe:
      self.add_quantization_config(
          config['regex'],
          config['operation'],
          _OpQuantizationConfig.from_dict(config['op_config']),
          config['algorithm_key'],
          config['override_algorithm'],
      )

  def need_calibration(self) -> bool:
    """Check if the recipe requires calibration."""
    # At the moment, only SRQ requires calibration.
    for op_quant_config in self.get_quantization_recipe():
      if (
          op_quant_config['op_config']['execution_mode']
          == qtyping.OpExecutionMode.SRQ
      ):
        return True
    return False
