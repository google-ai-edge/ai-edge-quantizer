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

"""The Python API for Algorithm Manager of Quantizer."""

from collections.abc import Callable
import dataclasses
import functools
from typing import Any, Optional
from ai_edge_quantizer import qtyping


@dataclasses.dataclass
class QuantizedOperationInfo:
  """Stores all quantization functions for a given op."""

  tfl_op_key: qtyping.TFLOperationName
  init_qsv_func: Callable[..., Any]
  calibration_func: Callable[..., Any]
  materialize_func: Callable[..., Any]


@dataclasses.dataclass
class QuantizationAlgorithmInfo:
  quantization_algorithm: str
  quantized_ops: dict[qtyping.TFLOperationName, QuantizedOperationInfo]


class AlgorithmManagerApi:
  """Quantizer API client to manage quantization configs and functions."""

  def __init__(self):
    self._algorithm_registry = dict()
    # Check if an op quantization config is supported for a given algorithm.
    self._config_check_registry = dict()
    # Policy to check if an op quantization config is supported for a given
    # algorithm.
    self._config_check_policy_registry = dict()

  def register_op_quant_config_validation_func(
      self,
      algorithm_key: str,
      config_check_func: Callable[..., Any],
  ):
    """Register functions to check if an op quantization config is supported."""
    self._config_check_registry[algorithm_key] = config_check_func

  def register_quantized_op(
      self,
      algorithm_key: str,
      tfl_op_name: qtyping.TFLOperationName,
      init_qsv_func: Callable[..., Any],
      calibration_func: Callable[..., Any],
      materialize_func: Callable[..., Any],
  ):
    """Register functions to support a quantization operation.

    This function registers the relevant information to support the quantized
    version of given tfl_operation, for the algorithm specified by
    quantization_algorithm.

    Args:
      algorithm_key: Quantization algorithm keyword for which the quantized
        operation is for.
      tfl_op_name: TFLite op name.
      init_qsv_func: QSV init function to be called.
      calibration_func: Quantized operation to be called during calibration.
      materialize_func: Quantized operation to be called during materialization.
    """
    quantized_algorithm_info = self._algorithm_registry.setdefault(
        algorithm_key, QuantizationAlgorithmInfo(algorithm_key, dict())
    )

    quantized_algorithm_info.quantized_ops[tfl_op_name] = (
        QuantizedOperationInfo(
            tfl_op_name,
            init_qsv_func,
            calibration_func,
            materialize_func,
        )
    )

  def is_op_registered(
      self,
      quantization_algorithm: str,
      tfl_op_name: qtyping.TFLOperationName,
  ) -> bool:
    """Check if the given key for quantization is valid.

    Args:
      quantization_algorithm: Target quantization algorithm.
      tfl_op_name: TFL operation name.

    Returns:
      True if the given op is registered for the given algorithm, false
      otherwise.
    """
    if not self.is_algorithm_registered(quantization_algorithm):
      return False

    return (
        tfl_op_name
        in self._algorithm_registry[quantization_algorithm].quantized_ops
    )

  def is_algorithm_registered(self, quantization_algorithm: str) -> bool:
    """Check if the given algorithm is registered.

    Args:
      quantization_algorithm: Target quantization algorithm.

    Returns:
      True if the given algorithm is registered, false otherwise.
    """
    return quantization_algorithm in self._algorithm_registry

  def check_op_quantization_config(
      self,
      quantization_algorithm: str,
      tfl_op_name: qtyping.TFLOperationName,
      op_quantization_config: qtyping.OpQuantizationConfig,
  ) -> None:
    """Checks if the given op quantization config is valid.

    Args:
      quantization_algorithm: Target quantization algorithm.
      tfl_op_name: TFL operation name.
      op_quantization_config: Op quantization config to be checked.

    Raises:
      ValueError if the given op is not registered for the given algorithm, or
      the given algorithm is not registered.
    """
    if op_quantization_config.skip_checks:
      return
    if not self.is_op_registered(quantization_algorithm, tfl_op_name):
      raise ValueError(
          f"Unsupported operation {tfl_op_name} for Algorithm:"
          f" {quantization_algorithm}."
      )
    if quantization_algorithm not in self._config_check_registry:
      raise ValueError(
          f"Config checking function for  algorithm {quantization_algorithm} is"
          " not registered. Please use"
          " `register_op_quant_config_validation_func` to register the"
          " validation function."
      )
    self._config_check_registry[quantization_algorithm](
        tfl_op_name,
        op_quantization_config,
        self._config_check_policy_registry[quantization_algorithm],
    )

  def get_supported_ops(self, alg_key: str) -> list[qtyping.TFLOperationName]:
    """Returns the list of supported ops for the given algorithm.

    Args:
      alg_key: Algorithm key.

    Returns:
      The list of supported JAX operations.

    Raises:
      ValueError if the alg_key is not registered.
    """
    if alg_key not in self._algorithm_registry:
      raise ValueError(f"Unregistered algorithm: {alg_key}")

    return list(self._algorithm_registry[alg_key].quantized_ops.keys())

  def get_quantization_func(
      self,
      algorithm_key: str,
      tfl_op_name: qtyping.TFLOperationName,
      quantize_mode: qtyping.QuantizeMode,
  ) -> Callable[..., Any]:
    """Gets the quantization function.

    Args:
      algorithm_key: Target quantization algorithm key (e.g.,
        AlgorithmName.MIN_MAX_UNIFORM_QUANT).
      tfl_op_name: TFLite op name.
      quantize_mode: Quantization mode to be used.

    Returns:
      A quantized operation (function) corresponds to the requested algorithm
      for the TFL op.
    """
    if not self.is_op_registered(algorithm_key, tfl_op_name):
      raise ValueError(
          f"Unsupported operation {tfl_op_name} for Algorithm: {algorithm_key}."
          f" Supported ops for algorithm {algorithm_key}:"
          f" {self.get_supported_ops(algorithm_key)}"
      )

    quantized_algorithm_info = self._algorithm_registry[algorithm_key]
    quantized_op_info = quantized_algorithm_info.quantized_ops
    quantized_func = self._get_target_func(
        quantized_op_info, tfl_op_name, quantize_mode
    )
    if quantized_func is None:
      raise ValueError(
          "Cannot retrieve appropriate quantization function for"
          f" {tfl_op_name} for algorithm {algorithm_key} under quantization"
          f" mode {quantize_mode}. Check if the op is registed in"
          " algorithm_manager."
      )

    return quantized_func

  def get_init_qsv_func(
      self,
      algorithm_key: str,
      tfl_op_name: qtyping.TFLOperationName,
  ) -> functools.partial:
    """Gets the initial Quantization Statistics Variable function for a given op.

    Args:
      algorithm_key: Quantization algorithm to search.
      tfl_op_name: Target TFL operation.

    Returns:
      A function for qsv initialization.
    """

    if not self.is_op_registered(algorithm_key, tfl_op_name):
      raise ValueError(
          f"Unsupported operation {tfl_op_name} for Algorithm: {algorithm_key}."
          f" Supported ops for algorithm {algorithm_key}:"
          f" {self.get_supported_ops(algorithm_key)}"
      )
    quantized_algorithm_info = self._algorithm_registry[algorithm_key]
    quantized_op_info = quantized_algorithm_info.quantized_ops

    return quantized_op_info[tfl_op_name].init_qsv_func

  def _get_target_func(
      self,
      quantized_op_info,
      tfl_op_name: qtyping.TFLOperationName,
      quantize_mode: qtyping.QuantizeMode,
  ):
    """Gets the function corresponding to the given JAX quantization phase and op."""
    if quantize_mode == qtyping.QuantizeMode.CALIBRATE:
      return quantized_op_info[tfl_op_name].calibration_func
    elif quantize_mode == qtyping.QuantizeMode.MATERIALIZE:
      return quantized_op_info[tfl_op_name].materialize_func
    return None

  # TODO: b/53780772 - Merge this function with
  # register_op_quant_config_validation_func after full transition to new check
  # mechanism.
  def register_config_check_policy(
      self,
      algorithm_key: str,
      config_check_policy: Optional[qtyping.ConfigCheckPolicyDict],
  ):
    """Registers a policy to check the op quantization config."""
    self._config_check_policy_registry[algorithm_key] = config_check_policy
