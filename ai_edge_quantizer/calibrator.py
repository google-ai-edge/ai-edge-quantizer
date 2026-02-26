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
import enum
import json
from typing import Any, Union

from absl import logging
import numpy as np

import os
from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import default_policy as policy
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe
from ai_edge_quantizer import recipe_manager
from ai_edge_quantizer.utils import calibration_utils
from ai_edge_quantizer.utils import progress_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils


class CalibrationMode(enum.Enum):
  INFERENCE = 1
  CALIBRATION = 2


_SignatureInput = dict[str, Any]  # input_argument_name -> tensor_value.
_SignatureOutput = dict[
    str, np.ndarray
]  # output_argument_name -> tensor_value.


class CalibrationInterpreter:
  """A TFL interpreter-like interface for model calibration.

  This is a wrapper around Calibrator that replaces the TFL Interpreter to
  enable calibration. If mode is CALIBRATION, it runs calibration, otherwise it
  acts as a regular TFL interpreter for inference. When in CALIBRATION mode,
  each invocation of a signature runner will update the calibration statistics
  in self._calibrator. Calibrator is needed in both modes because it contains
  tfl interpreter instance to run the model.
  """

  def __init__(
      self,
      model_path: str,
      mode: CalibrationMode = CalibrationMode.INFERENCE,
  ):
    """Initializes the CalibrationInterpreter.

    Args:
      model_path: The path to the TFLite model.
      mode: The mode of the interpreter. If CALIBRATION, the interpreter will
        preserve all tensors for calibration purposes.
    """
    self._calibrator = Calibrator(
        model_path,
        interpreter_preserve_all_tensors=(mode == CalibrationMode.CALIBRATION),
    )
    self._mode = mode

  def get_signature_runner(self, signature_key: str | None = None):
    """Returns the signature runner."""
    return CalibrationSignatureRunner(
        self._calibrator, signature_key, self._mode
    )

  def get_calibration_results(self):
    """Returns the calibration results."""
    if self._mode == CalibrationMode.INFERENCE:
      raise ValueError(
          "Calibration results are not available in INFERENCE mode."
      )
    return self._calibrator.get_model_qsvs()

  def save_calibration_result(self, output_path: str):
    """Saves the calibration results."""
    if self._mode == CalibrationMode.INFERENCE:
      raise ValueError(
          "Calibration results are not available in INFERENCE mode."
      )
    self._calibrator.save_calibration_result(output_path)

  def get_signature_list(self) -> list[str]:
    """Returns the signature list."""
    return self._calibrator.get_signature_list()


class CalibrationSignatureRunner:
  """Wrapper around TFL signature runner to enable calibration."""

  def __init__(
      self,
      calibrator_obj: "Calibrator",
      signature_key: str | None = None,
      mode: CalibrationMode = CalibrationMode.INFERENCE,
      quantization_recipe: recipe_manager.ModelQuantizationRecipe = recipe.static_wi8_ai8(),
  ):
    """Initializes the CalibrationSignatureRunner.

    Args:
      calibrator_obj: The Calibrator instance.
      signature_key: The key of the signature to run. If None, the default
        signature is used.
      mode: The mode of the runner. If CALIBRATION, invoking the runner will
        update 'calibrator_obj' with new quantization statistics values. If
        INFERENCE, the runner behaves like a standard signature runner.
      quantization_recipe: The quantization recipe to use for calibration.
        Defaults to static_wi8_ai8.
    """
    self._calibrator = calibrator_obj
    self._signature_key = signature_key
    self._mode = mode
    self._recipe_manager = recipe_manager.RecipeManager()
    self._recipe_manager.load_quantization_recipe(quantization_recipe)
    self._signature_runner = (
        self._calibrator._tfl_interpreter.get_signature_runner(
            self._signature_key
        )
    )

  def __call__(self, **kwargs):
    if self._mode == CalibrationMode.INFERENCE:
      return self._signature_runner(**kwargs)
    self._calibrator.calibrate(
        calibration_dataset={self._signature_key: [kwargs]},
        model_recipe_manager=self._recipe_manager,
        cache_output=True,
    )
    outputs = self._calibrator.get_cached_output()
    assert len(outputs) == 1
    self._calibrator.clear_cached_output()
    return outputs[0]

  def get_input_details(self):
    """Returns the input details of the model."""
    return self._signature_runner.get_input_details()

  def get_output_details(self):
    """Returns the output details of the model."""
    return self._signature_runner.get_output_details()


class Calibrator:
  """Calibrator for TFLite model."""

  def __init__(
      self,
      float_tflite: Union[str, bytes],
      num_threads: int = 16,
      interpreter_preserve_all_tensors: bool = True,
  ):
    self._flatbuffer_model = tfl_flatbuffer_utils.read_model(float_tflite)

    self._tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        float_tflite,
        use_xnnpack=True,
        num_threads=num_threads,
        preserve_all_tensors=interpreter_preserve_all_tensors,
    )
    # Tensor name to tensor content.
    self._tensor_content_map: dict[str, Any] = {}
    # QSV of all the tensors in the model.
    self._model_qsvs: dict[str, qtyping.QSV] = {}
    # Cached output of the model.
    self._cached_output: list[_SignatureOutput] = []

  def _get_total_operations(self, calibration_dataset) -> int:
    """Get the total number of OPs to go through while calibrating the model."""
    data_sizes = {
        key: len(list(dataset)) for key, dataset in calibration_dataset.items()
    }
    total_ops = 0
    for key, value in data_sizes.items():
      subgraph_idx = tfl_interpreter_utils.get_signature_main_subgraph_index(
          self._tfl_interpreter, key
      )
      subgraph_inds = [subgraph_idx]
      total_operators = 0
      while subgraph_inds:
        subgraph_idx = subgraph_inds.pop()
        subgraph = self._flatbuffer_model.subgraphs[subgraph_idx]
        subgraph_operators = subgraph.operators
        if not any(
            isinstance(op, qtyping.IOOperator) for op in subgraph.operators
        ):
          subgraph_operators += (
              tfl_flatbuffer_utils.get_subgraph_input_output_operators(subgraph)
          )
        total_operators += len(subgraph_operators)
        for op in subgraph_operators:
          subgraph_inds.extend(
              tfl_flatbuffer_utils.get_op_side_effect_subgraphs(op)
          )
      total_ops += value * total_operators
    return total_ops

  # TODO(b/330740605)- Collect multiple QSVs in one run to save compute.
  def calibrate(
      self,
      calibration_dataset: dict[str, Iterable[_SignatureInput]],
      model_recipe_manager: recipe_manager.RecipeManager,
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
      calibration_dataset: A dictionary of input data for calibration for the
        given model signature.
      model_recipe_manager: A RecipeManager object that contains the
        quantization recipe.
      cache_output: Whether to cache the output of the model during the
        calibration process. This is useful if there are dependencies between
        signatures/models (e.g., decode requires encode output).
      qsv_update_func: The function to update the QSVs.
    """
    op_codes = self._flatbuffer_model.operatorCodes
    if self._model_qsvs:
      logging.warning(
          "Calibrator contains non-empty model qsvs, and the current"
          " calibration process will start on top of this state (i.e., update"
          " the existing qsvs). If this is an unintended behavior please call"
          " reset_model_qsvs to reset model qsvs."
      )

    total_ops = self._get_total_operations(calibration_dataset)
    with progress_utils.ProgressBar(
        total_steps=total_ops,
        description="Running Calibration:",
        disappear_on_finish=True,
        disable=total_ops
        < 1000,  # We skip the progress bar for small models and small datasets.
    ) as pbar:
      # TODO: b/329322226 - Enable parallel calibration.
      for signature_key, dataset in calibration_dataset.items():
        # Step0: get subgraph index.
        subgraph_idx = tfl_interpreter_utils.get_signature_main_subgraph_index(
            self._tfl_interpreter, signature_key
        )

        for data in dataset:
          # Initialize tensor names updated in this round of calibration.
          updated_tensor_names = set()

          # Step1: run tfl interpreter on subgraph to get tensor content.
          signature_output = tfl_interpreter_utils.invoke_interpreter_signature(
              self._tfl_interpreter, data, signature_key
          )
          if cache_output:
            self._cached_output.append(signature_output)

          # Step2: go through each op in subgraph to update quantization
          # statistic values.
          subgraphs_inds = [subgraph_idx]
          while subgraphs_inds:
            subgraph_ind = subgraphs_inds.pop()
            self._tensor_content_map.update(
                tfl_interpreter_utils.get_tensor_name_to_content_map(
                    self._tfl_interpreter, subgraph_ind
                )
            )
            subgraph = self._flatbuffer_model.subgraphs[subgraph_ind]
            graph_info = qtyping.GraphInfo(
                subgraph.tensors, self._flatbuffer_model.buffers
            )
            # Add input/output operators if they are not in the subgraph.
            if not any(
                isinstance(op, qtyping.IOOperator) for op in subgraph.operators
            ):
              subgraph.operators += (
                  tfl_flatbuffer_utils.get_subgraph_input_output_operators(
                      subgraph
                  )
              )
            for op in subgraph.operators:
              pbar.update_single_step()
              if isinstance(op, qtyping.IOOperator):
                op_key = op.op_key
              else:
                op_code = op_codes[op.opcodeIndex].builtinCode
                if op_code not in tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME:
                  continue
                op_key = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME[op_code]
              # Step2.1: query the quantization_recipe to get op quantization
              # settings.
              op_scope = tfl_flatbuffer_utils.get_op_scope(op, subgraph.tensors)
              algorithm_name, _ = model_recipe_manager.get_quantization_configs(
                  op_key, op_scope
              )
              if algorithm_name == algorithm_manager.AlgorithmName.NO_QUANTIZE:
                continue
              if policy.is_non_quantizable_composite_op(op):
                continue

              # Step2.2: query algorithm_manager to get/call the related
              # calibration function.
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

              # Step4: Invoke any subgraphs invoked as a side effect of the op.
              subgraphs_inds.extend(
                  tfl_flatbuffer_utils.get_op_side_effect_subgraphs(op)
              )

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

  def load_model_qsvs(
      self, model_qsvs: Union[str, dict[str, qtyping.QSV]]
  ) -> None:
    """Load the model qsvs.

    Args:
      model_qsvs: A dictionary of tensor name to QSV or a path to a JSON file
        that contains the model qsvs (i.e., from save_calibration_result).
    """

    if isinstance(model_qsvs, str):
      self._model_qsvs = calibration_utils.load_calibration_results(model_qsvs)
    else:
      self._model_qsvs = copy.deepcopy(model_qsvs)

  def save_calibration_result(self, file_path: str) -> None:
    """Saves the calibration result to a json file."""
    with open(file_path, "w") as f:
      json.dump(self._model_qsvs, f, cls=calibration_utils.NumpyEncoder)

  def get_signature_list(self) -> list[str]:
    """Get the signature list of the model."""
    return self._tfl_interpreter.get_signature_list()

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
