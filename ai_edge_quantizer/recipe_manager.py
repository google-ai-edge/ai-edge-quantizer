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
  """Dataclass for quantization configuration under a scope.

  This class is the main entry point to a recipe schema. There could be a single
  instance of this associated with a model (with `regex=.*` if the full model is
  to be quantized with the same spec), or multiple instances targeting different
  `regex` or `operation`.

  Attributes:
    regex: Regular expression for scope name matching. Any op that matches
      `regex` will be quantized according to this instance. The narrowest scope
      would be the full output tensor name of an op. The widest scope would be
      '.*' which applies to the full model.
    operation: Target TFL operation. * for any supported TFLite operation.
    algorithm_key: Algorithm key to be applied. This can be any one of the
      strings as enumerated in `AlgorithmName`.
    op_config: Quantization configuration to be applied for the op.
  """

  regex: str
  operation: _TFLOpName
  algorithm_key: str
  op_config: _OpQuantizationConfig = dataclasses.field(
      default_factory=_OpQuantizationConfig
  )


class RecipeManager:
  """Sets the quantization recipe for target model.

  This class is internal to Quantizer to help manage loading recipes and
  resolving conflicts between recipes input by the user.
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
    """
    if op_config is None:
      op_config = _OpQuantizationConfig()

    config = OpQuantizationRecipe(
        regex, operation_name, algorithm_key, op_config
    )
    # Special care if trying to set all ops to some config.
    if config.operation == _TFLOpName.ALL_SUPPORTED:
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

  def get_quantization_configs(
      self,
      target_op_name: _TFLOpName,
      scope_name: str,
  ) -> tuple[str, _OpQuantizationConfig]:
    """Gets the algorithm key and quantization configuration for an op.

    We respect the latest valid config and fall back to no quantization.
    Specifically, we search the quantization configuration in the order of the
    scope configs. If there are two or more matching settings, the latest one
    will be used.


    Args:
      target_op_name: Target TFLite operation. * for all supported TFLite
        operation.
      scope_name: Name of the target scope.

    Returns:
       A tuple of quantization algorithm, and quantization configuration.
    """
    result_key, result_config = (
        AlgorithmName.NO_QUANTIZE,
        _OpQuantizationConfig(),
    )
    for scope_regex, recipes in self._scope_configs.items():
      if re.search(scope_regex, scope_name):
        for recipe in recipes:
          if (
              recipe.operation != _TFLOpName.ALL_SUPPORTED
              and recipe.operation != target_op_name
          ):
            continue
          selected_recipe = recipe
          if selected_recipe.algorithm_key != AlgorithmName.NO_QUANTIZE:
            # The selected recipe must contain a supported config.
            try:
              algorithm_manager.check_op_quantization_config(
                  recipe.algorithm_key, target_op_name, recipe.op_config
              )
            except ValueError:
              continue  # Skip the recipe if it is not supported.
          result_config = selected_recipe.op_config
          result_key = selected_recipe.algorithm_key

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
          _OpQuantizationConfig.from_dict(config['op_config'])
          if config['algorithm_key'] != AlgorithmName.NO_QUANTIZE
          else None,
          config['algorithm_key'],
      )

  def need_calibration(self) -> bool:
    """Check if the recipe requires calibration."""
    # At the moment, only SRQ requires calibration.
    for op_quant_config in self.get_quantization_recipe():
      if (
          op_quant_config['op_config']['compute_precision']
          == qtyping.ComputePrecision.INTEGER
          and 'activation_tensor_config' in op_quant_config['op_config']
      ):
        return True
    return False
