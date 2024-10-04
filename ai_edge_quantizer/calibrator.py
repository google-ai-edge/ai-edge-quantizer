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

"""Quantization Calibration."""

from collections.abc import Callable, Iterable
import copy
from typing import Any, Optional, Union

from absl import logging
import numpy as np

from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe_manager
from ai_edge_quantizer.utils import calibration_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_SignatureInput = dict[str, Any]  # input_argument_name -> tensor_value.
_SignatureOutput = dict[
    str, np.ndarray
]  # output_argument_name -> tensor_value.


class Calibrator:
  """Calibrator for TFLite model."""

  def __init__(
      self,
      float_tflite: Union[str, bytes],
  ):
    self._flatbuffer_model = tfl_flatbuffer_utils.read_model(float_tflite)

    if not tfl_flatbuffer_utils.is_float_model(self._flatbuffer_model):
      raise ValueError(
          "The input model for calibration is not a float model. Please check"
          " the model (e.g., if it is already quantized)."
      )
    self._tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        float_tflite
    )
    # Tensor name to tensor content.
    self._tensor_content_map: dict[str, Any] = {}
    # QSV of all the tensors in the model.
    self._model_qsvs: dict[str, qtyping.QSV] = {}
    # Cached output of the model.
    self._cached_output: list[_SignatureOutput] = []

  # TODO(b/330740605)- Collect multiple QSVs in one run to save compute.
  def calibrate(
      self,
      calibration_dataset: Iterable[_SignatureInput],
      model_recipe_manager: recipe_manager.RecipeManager,
      signature_key: Optional[str] = None,
      cache_output: bool = False,
      qsv_update_func: Callable[
          [qtyping.QSV, qtyping.QSV],
          qtyping.QSV,
      ] = calibration_utils.moving_average_update,
  ) -> None:
    """Calibrates the model using the given dataset for a model signature.

    The process is
    0. Initialize quantization statistics values (QSVs) using the initialization
    function (from AlgorithmManager) for the op if needed.
    1. Invoke TFL interpreter on the calibration data.
    2. Go through each op, ask RecipeManager for the quantization setting
    of the op.
    3. Ask AlgorithmManager for the calibration function of the op given the
    quantization setting.
    4. Apply the function to the op to obtain quantization statistics (qsvs) for
    the tensors associated with the op.
    5. Update the global qsv dictionary
    6. Start another round of calibration.

    Args:
      calibration_dataset: A list of input data for calibration for the given
        model signature.
      model_recipe_manager: A RecipeManager object that contains the
        quantization recipe.
      signature_key: The signature key to be used for invoking the models. If
        the model doesn't have a signature key (or only has one ), this can be
        set to None.
      cache_output: Whether to cache the output of the model during the
        calibration process. This is useful if there are dependencies between
        signatures/models (e.g., decode requires encode output).
      qsv_update_func: The function to update the QSVs.
    """
    op_codes = self._flatbuffer_model.operatorCodes
    if not self._model_qsvs:
      self._initialize_model_qsvs(model_recipe_manager)
    else:
      logging.warning(
          "Calibrator contains non-empty model qsvs, and the current"
          " calibration process will start on top of this state (i.e., update"
          " the existing qsvs). If this is an unintended behavior please call"
          " reset_model_qsvs to reset model qsvs."
      )

    # TODO: b/329322226 - Enable parrallel calibration.
    for data in calibration_dataset:
      # Initialize tensor names that are updated in this round of calibration.
      updated_tensor_names = set()

      # Step1: run tfl interpreter to get tensor content.
      signature_output = tfl_interpreter_utils.invoke_interpreter_signature(
          self._tfl_interpreter, data, signature_key
      )
      if cache_output:
        self._cached_output.append(signature_output)
      self._tensor_content_map = (
          tfl_interpreter_utils.get_tensor_name_to_content_map(
              self._tfl_interpreter
          )
      )
      # Step2: go through each op to update quantization statistic values.
      for subgraph in self._flatbuffer_model.subgraphs:
        graph_info = qtyping.GraphInfo(
            subgraph.tensors, self._flatbuffer_model.buffers
        )
        # Add input/output operators to the subgraph.
        subgraph.operators += (
            tfl_flatbuffer_utils.get_subgraph_input_output_operators(subgraph)
        )
        for op in subgraph.operators:
          if isinstance(op, qtyping.IOOperator):
            op_key = op.op_key
          else:
            op_code = op_codes[op.opcodeIndex].builtinCode
            if op_code not in tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME:
              continue
            op_key = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME[op_code]
          # Step2.1: query the quantization_recipe to get op quantization
          # settings.
          op_scope = self._get_op_scope(op, subgraph.tensors)
          algorithm_name, _ = model_recipe_manager.get_quantization_configs(
              op_key, op_scope
          )
          if algorithm_name == algorithm_manager.AlgorithmName.NO_QUANTIZE:
            continue
          # Step2.2: query algorithm_manager to get/call the related calibration
          # function.
          calibrate_func = algorithm_manager.get_quantization_func(
              algorithm_name, op_key, qtyping.QuantizeMode.CALIBRATE
          )
          op_qsvs = calibrate_func(op, graph_info, self._tensor_content_map)
          # Step3: Update tensor qsvs with the new values. Ignore the tensor
          # names that are already updated in this round of calibration.
          op_updated_tensor_name = self._update_qsvs(
              op_qsvs, updated_tensor_names, qsv_update_func
          )
          updated_tensor_names.update(op_updated_tensor_name)
      # Reset interpreter after one round of calibration.
      self._tfl_interpreter.reset_all_variables()

  def get_model_qsvs(self) -> dict[str, qtyping.QSV]:
    """Get the model qsvs.

    Returns:
      A dictionary of tensor name to QSV.
    """
    return self._model_qsvs

  def get_cached_output(self) -> list[_SignatureOutput]:
    """Get the cached output of the model."""
    return self._cached_output

  def clear_cached_output(self) -> None:
    """Clear the cached output of the model."""
    self._cached_output = []

  def reset_model_qsvs(self) -> None:
    """Reset the model qsvs."""
    self._model_qsvs = {}

  def load_model_qsvs(self, model_qsvs: dict[str, qtyping.QSV]) -> None:
    """Load the model qsvs.

    Args:
      model_qsvs: A dictionary of tensor name to QSV.
    """
    self._model_qsvs = copy.deepcopy(model_qsvs)

  def _update_qsvs(
      self,
      op_qsvs: dict[str, qtyping.QSV],
      ignore_tensor_names: set[str],
      qsv_update_func: Callable[[qtyping.QSV, qtyping.QSV], qtyping.QSV],
  ) -> set[str]:
    """Update the model qsvs with the new values.

    Args:
      op_qsvs: A dictionary of tensor name to QSV.
      ignore_tensor_names: A set of tensor names to ignore.
      qsv_update_func: The function to update the QSVs.

    Returns:
      A set of tensor names that are updated.
    """
    updated_tensor_names = set()
    for tensor_name, qsv in op_qsvs.items():
      if tensor_name in ignore_tensor_names:
        continue
      if tensor_name not in self._model_qsvs:
        self._model_qsvs[tensor_name] = qsv
      else:
        updated_qsv = qsv_update_func(self._model_qsvs[tensor_name], qsv)
        self._model_qsvs[tensor_name] = updated_qsv
      updated_tensor_names.add(tensor_name)
    return updated_tensor_names

  def _get_op_scope(self, op, subgraph_tensors) -> str:
    """Get the scope of the op.

    The scope is the name of the output tensor of the op.

    Args:
      op: The op to get the scope.
      subgraph_tensors: The tensors in the subgraph.

    Returns:
      The scope of the op.
    """
    scope = ""
    for output_tensor_idx in op.outputs:
      if output_tensor_idx != -1:
        output_tensor = subgraph_tensors[output_tensor_idx]
        scope += tfl_flatbuffer_utils.get_tensor_name(output_tensor)
    return scope

  # TODO: b/354224138 - Remove code duplication between calibrate and
  # _initialize_model_qsvs.
  def _initialize_model_qsvs(
      self, model_recipe_manager: recipe_manager.RecipeManager
  ) -> None:
    """Initialize the model qsvs.

    Args:
      model_recipe_manager: A RecipeManager object that contains the
        quantization recipe.
    """
    op_codes = self._flatbuffer_model.operatorCodes
    for subgraph in self._flatbuffer_model.subgraphs:
      graph_info = qtyping.GraphInfo(
          subgraph.tensors, self._flatbuffer_model.buffers
      )
      for subgraph_op_id, op in enumerate(subgraph.operators):
        op_code = op_codes[op.opcodeIndex].builtinCode
        if op_code not in tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME:
          continue
        op_key = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME[op_code]
        # Step1: query the quantization_recipe to get op quantization
        # settings.
        op_scope = self._get_op_scope(op, subgraph.tensors)
        algorithm_name, op_quant_config = (
            model_recipe_manager.get_quantization_configs(op_key, op_scope)
        )
        if algorithm_name == algorithm_manager.AlgorithmName.NO_QUANTIZE:
          continue
        # Step2: query algorithm_manager to get/call the related qsv init
        # function.
        qsv_init_func = algorithm_manager.get_init_qsv_func(
            algorithm_name, op_key
        )
        op_info = qtyping.OpInfo(op, op_key, subgraph_op_id, op_quant_config)
        op_qsvs = qsv_init_func(op_info, graph_info)
        # Step3: initialize tensor qsvs.
        for tensor_name, qsv in op_qsvs.items():
          if tensor_name not in self._model_qsvs:
            self._model_qsvs[tensor_name] = qsv
