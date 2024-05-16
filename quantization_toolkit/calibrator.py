"""Quantization Calibration."""

from collections.abc import Iterable
from typing import Any

from absl import logging

from quantization_toolkit import algorithm_manager
from quantization_toolkit import qtyping
from quantization_toolkit import recipe_manager
from quantization_toolkit.utils import tfl_flatbuffer_utils
from quantization_toolkit.utils import tfl_interpreter_utils


class Calibrator:
  """Calibrator for TFLite model."""

  def __init__(
      self,
      float_tflite_path: str,
  ):
    self._flatbuffer_model = tfl_flatbuffer_utils.read_model(float_tflite_path)
    self._model_buffer: bytearray = tfl_flatbuffer_utils.get_model_buffer(
        float_tflite_path
    )
    self._tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        float_tflite_path
    )
    # Tensor name to tensor content.
    self._tensor_content_map: dict[str, Any] = {}
    # QSV of all the tensors in the model.
    self._model_qsvs: dict[str, qtyping.QSV] = {}

  # TODO(b/330740605)- Collect multiple QSVs in one run to save compute.
  def calibrate(
      self,
      calibration_dataset: Iterable[Any],
      model_recipe_manager: recipe_manager.RecipeManager,
  ) -> None:
    """Calibrates the model with the given dataset.

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
      calibration_dataset: A list of input data for calibration.
      model_recipe_manager: A RecipeManager object that contains the
        quantization recipe.
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
      # Step1: run tfl interpreter to get tensor content.
      tfl_interpreter_utils.invoke_interpreter_once(self._tfl_interpreter, data)
      self._tensor_content_map = (
          tfl_interpreter_utils.get_tensor_name_to_content_map(
              self._tfl_interpreter
          )
      )
      # Step2: go through each op to update quantization statistic values.
      for subgraph in self._flatbuffer_model.subgraphs:
        graph_info = qtyping.GraphInfo(
            subgraph.tensors, self._flatbuffer_model.buffers, self._model_buffer
        )
        for op in subgraph.operators:
          op_code = op_codes[op.opcodeIndex].builtinCode
          if op_code not in tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME:
            raise ValueError(
                "Full integer calibration requires all ops in the model to be"
                " supported. Encounter unsupported op code: %s. Please add the"
                " op to Algorithm Manager." % op_code
            )
          op_key = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME[op_code]
          # Step2.1: query the quantization_recipe to get op quantization
          # settings.
          op_scope = self._get_op_scope(op, subgraph.tensors)
          algorithm_name, _ = model_recipe_manager.get_quantization_configs(
              op_key, op_scope
          )
          if algorithm_name == algorithm_manager.NO_QUANT:
            continue
          # Step2.2: query algorithm_manager to get/call the related calibration
          # function.
          calibrate_func = algorithm_manager.get_quantization_func(
              algorithm_name, op_key, qtyping.QuantizeMode.CALIBRATE
          )
          op_qsvs = calibrate_func(op, graph_info, self._tensor_content_map)
          # Step3: Update model qsvs with the new values.
          self._update_qsvs(op_qsvs)
      # Reset interpreter after one round of calibration.
      self._tfl_interpreter.reset_all_variables()

  def get_model_qsvs(self) -> dict[str, qtyping.QSV]:
    """Get the model qsvs.

    Returns:
      A dictionary of tensor name to QSV.
    """
    return self._model_qsvs

  def reset_model_qsvs(self) -> None:
    """Reset the model qsvs."""
    self._model_qsvs = {}

  def load_model_qsvs(self, model_qsvs: dict[str, qtyping.QSV]) -> None:
    """Load the model qsvs.

    Args:
      model_qsvs: A dictionary of tensor name to QSV.
    """
    self._model_qsvs = model_qsvs

  def _update_qsvs(self, op_qsvs: dict[str, qtyping.QSV]):
    """Update the model qsvs with the new values.

    Args:
      op_qsvs: A dictionary of tensor name to QSV.
    """
    for tensor_name, qsv in op_qsvs.items():
      if tensor_name not in self._model_qsvs:
        self._model_qsvs[tensor_name] = qsv
      else:
        previous_qsv = self._model_qsvs[tensor_name]
        self._model_qsvs[tensor_name] = (
            algorithm_manager.moving_average_update_qsv(previous_qsv, qsv)
        )

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
          subgraph.tensors, self._flatbuffer_model.buffers, self._model_buffer
      )
      for subgraph_op_id, op in enumerate(subgraph.operators):
        op_code = op_codes[op.opcodeIndex].builtinCode
        if op_code not in tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME:
          raise ValueError(
              "Full integer calibration requires all ops in the model to be"
              " supported. Encounter unsupported op code: %s. Please add the"
              " op to Algorithm Manager." % op_code
          )
        op_key = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME[op_code]
        # Step1: query the quantization_recipe to get op quantization
        # settings.
        op_scope = self._get_op_scope(op, subgraph.tensors)
        algorithm_name, op_quant_config = (
            model_recipe_manager.get_quantization_configs(op_key, op_scope)
        )
        if algorithm_name == algorithm_manager.NO_QUANT:
          continue
        # Step2: query algorithm_manager to get/call the related qsv init
        # function.
        qsv_init_func = algorithm_manager.get_init_qsv_func(
            algorithm_name, op_key
        )
        op_info = qtyping.OpInfo(op, op_key, subgraph_op_id, op_quant_config)
        op_qsvs = qsv_init_func(op_info, graph_info)
        # Step3: update tensor qsvs.
        for tensor_name, qsv in op_qsvs.items():
          self._model_qsvs[tensor_name] = qsv
