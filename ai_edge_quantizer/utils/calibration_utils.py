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

"""Utilities for model calibration."""

import copy
from typing import Any, Union

import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.utils import common_utils
from ai_edge_quantizer.utils import constrained_ops_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils
from tensorflow.lite.tools import flatbuffer_utils  # pylint: disable=g-direct-tensorflow-import


_SignatureInput = dict[str, Any]
_OpQuantConstraint = common_utils.OpQuantConstraint
_SignatureData = dict[
    str, list[str]
]  # signature_key -> list of signature_names.


def _update_moving_average(
    smoothing_factor: Union[np.ndarray, float],
    w: np.ndarray,
    update: np.ndarray,
) -> np.ndarray:
  """Updates weight w with moving average.

  Args:
    smoothing_factor: Smoothing factor used to update w.
    w: Weights to be updated.
    update: Value used for update.

  Returns:
    Weighted sum of w and update.
  """
  return smoothing_factor * w + (1.0 - smoothing_factor) * update


def moving_average_update(
    qsv: qtyping.QSV, new_qsv: qtyping.QSV, smoothing_factor: float = 0.95
) -> qtyping.QSV:
  """Update the QSV (i.e., min/max) using moving average.

  Args:
    qsv: The quantization statistical value of the tensor (min/max) that need to
      be updated.
    new_qsv: The new QSVs (e.g., from new round of calibration).
    smoothing_factor: The weight of moving average.

  Returns:
    The updated QSV for the tensor.
  """
  if not qsv:
    return new_qsv

  updated_qsv = {}
  updated_qsv["min"] = _update_moving_average(
      smoothing_factor, qsv["min"], new_qsv["min"]
  )

  updated_qsv["max"] = _update_moving_average(
      smoothing_factor, qsv["max"], new_qsv["max"]
  )
  return updated_qsv


def min_max_update(qsv: qtyping.QSV, new_qsv: qtyping.QSV) -> qtyping.QSV:
  """Update the QSV with minimum min values and maximum max values.

  Args:
    qsv: The quantization statistical value of the tensor (min/max) that need to
      be updated.
    new_qsv: The new QSVs (e.g., from new round of calibration).

  Returns:
    The updated QSV for the tensor.
  """
  if not qsv:
    return new_qsv

  updated_qsv = {}
  updated_qsv["min"] = np.minimum(qsv["min"], new_qsv["min"])
  updated_qsv["max"] = np.maximum(qsv["max"], new_qsv["max"])
  return updated_qsv


def _find_overall_min_max(
    qsv: qtyping.QSV, tensor_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
  """Finds the overall minimum and maximum values for the given tensors.

  Args:
    qsv: The quantization statistical value of the tensor (min/max).
    tensor_names: The list of tensor names to find the minimum and maximum
      values.

  Returns:
    The minimum and maximum values for the given tensors.
  """
  min_value = np.inf
  max_value = -np.inf
  for tensor_name in tensor_names:
    min_value = min(min_value, qsv[tensor_name]["min"])
    max_value = max(max_value, qsv[tensor_name]["max"])
  return min_value, max_value


class CalibrationQsvAlignmentUtils:
  """Calibration utils for alignment of QSVs.

  This class is used to align QSVs for a given model. It builds a list of ops
  that need to be constrained to the same scale as the input. Based on this
  list, it finds the corresponding tensor names for a given signature data.
  """

  def __init__(self, model_path: str):
    self._same_as_input_scale_ops = (
        constrained_ops_utils.get_constrained_op_list(
            _OpQuantConstraint.SAME_AS_INPUT_SCALE
        )
    )

    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(model_path)
    self._flatbuffer_object = tfl_flatbuffer_utils.read_model(model_path)

    signature_keys = list(tfl_interpreter.get_signature_list().keys())

    # Build a dict of signature runners.
    self._signature_runners = {}
    for signature_key in signature_keys:
      signature_runner = tfl_interpreter.get_signature_runner(signature_key)
      self._signature_runners[signature_key] = signature_runner

  def _search_tensor_by_signature_name(
      self, signature_key: str, signature_input_output_name: str, verbose=False
  ) -> list[str]:
    """Searches for a tensor name for a given signature by signature input or output name.

    Args:
      signature_key: Name of the signature.
      signature_input_output_name: Name of the input or output in the signature.
      verbose: Flag to enable verbose output.

    Returns:
      The list with one or two tensor names. The first one is the input tensor
      name, and the second one is the output tensor name.
    """

    if verbose:
      print("Searching tensor by signature name.")

    tensor_names = []

    # Search among inputs.
    input_details = self._signature_runners[signature_key].get_input_details()
    if signature_input_output_name in input_details.keys():
      tensor_names.append(input_details[signature_input_output_name]["name"])

    # Search among outputs.
    output_details = self._signature_runners[signature_key].get_output_details()
    if signature_input_output_name not in output_details:
      if not tensor_names:
        raise ValueError(
            f"Signature {signature_key} does not have input or output"
            f" `{signature_input_output_name}`"
        )
      return tensor_names

    output_tensor_name = output_details[signature_input_output_name]["name"]
    if verbose:
      print(
          ">> Starting recursive search for the output tensor name:"
          f" {output_tensor_name}"
      )

    idx = self._signature_runners[signature_key]._subgraph_index  # pylint: disable=protected-access
    subgraph = self._flatbuffer_object.subgraphs[idx]
    graph_info = qtyping.GraphInfo(
        subgraph.tensors, self._flatbuffer_object.buffers
    )

    # Recursively search the graph for the output tensor name since it may be
    # `SAME_AS_INPUT` constrainted.
    operators = copy.deepcopy(subgraph.operators)
    tensor_name = self._search_reverse_order_recursively(
        graph_info, operators, output_tensor_name, indent="  ", verbose=verbose
    )
    tensor_names.append(tensor_name)

    if verbose:
      print(f"\n\nFound tensor name: {tensor_name}")

    return tensor_names

  def _search_reverse_order_recursively(
      self,
      graph_info: qtyping.GraphInfo,
      operators: list[Any],
      output_tensor_name: str,
      indent: str,
      verbose: bool = False,
  ):
    """Searches for a tensor name in reverse order recursively.

    Stop criteria is when the tensor belongs to an operator that is not
    `SAME_AS_INPUT` constrainted.

    Args:
      graph_info: Graph information.
      operators: List of operators.
      output_tensor_name: Name of the output tensor to search for.
      indent: Indentation string for debug output.
      verbose: Flag to enable verbose output.

    Returns:
      The name of the tensor found, or None if not found.
    """
    op_codes = self._flatbuffer_object.operatorCodes

    while operators:
      op = operators.pop()
      op_code = op_codes[op.opcodeIndex].builtinCode
      op_name = flatbuffer_utils.opcode_to_name(
          self._flatbuffer_object, op.opcodeIndex
      )
      if op_code not in tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME:
        continue
      for output_idx in op.outputs:
        if output_tensor_name == tfl_flatbuffer_utils.get_tensor_name(
            graph_info.subgraph_tensors[output_idx]
        ):
          dbg_str = (
              f"{indent}>> Found `{op_name}`, output tensor"
              f" '{output_tensor_name}'"
          )

          if op_name not in self._same_as_input_scale_ops:
            if verbose:
              print(f"{dbg_str}, returning...")
            return output_tensor_name

          if verbose:
            print(f"{dbg_str}, with SAME_AS_INPUT, search recursively among:")
          for input_idx in op.inputs:
            input_tensor_name = graph_info.subgraph_tensors[
                input_idx
            ].name.decode("utf-8")

            if verbose:
              print(f"{indent}    Input: {input_tensor_name}")

            return self._search_reverse_order_recursively(
                graph_info,
                operators,
                input_tensor_name,
                indent=f"{indent}  ",
                verbose=verbose,
            )
    return output_tensor_name

  def align_quant_stats(
      self, qsv: dict[str, Any], signature_data: _SignatureData
  ) -> tuple[np.ndarray, np.ndarray]:
    """Aligns quantization statistics for a given signature data.

    This function takes quantization statistics and signature data as input,
    identifies the tensors associated with the signature data, and aligns
    the quantization statistics of these tensors by setting their minimum
    and maximum values to the same value. This ensures that the tensors
    have the same quantization parameters.

    Args:
      qsv: Quantization statistics.
      signature_data: Signature data.

    Returns:
      Tuple of min and max values.
    """
    # Go over all signature info and find the corresponding tensor names.
    tensor_names = []
    for signature_key, signature_names in signature_data.items():
      for signature_name in signature_names:
        tensor_name = self._search_tensor_by_signature_name(
            signature_key, signature_name
        )
        tensor_names.extend(tensor_name)

    # Find min and max values accross all tensors.
    min_value, max_value = _find_overall_min_max(qsv, tensor_names)

    # Overwrite the min and max values in the QSV.
    for tensor_name in tensor_names:
      qsv[tensor_name]["min"] = min_value
      qsv[tensor_name]["max"] = max_value

    return min_value, max_value

  def update_quant_stats(
      self,
      qsv: dict[str, Any],
      signature_data: _SignatureData,
      min_value: np.ndarray,
      max_value: np.ndarray,
  ):
    """Updates quantization statistics for a given signature data.

    This function updates the quantization statistics with the provided min, max
    values for the tensors specified in the signature data.

    Args:
      qsv: Quantization statistics.
      signature_data: Signature data.
      min_value: Minimum value to update.
      max_value: Maximum value to update.

    Returns:
      Updated quantization statistics.
    """
    # Go over all signature info and find the corresponding tensor names.
    tensor_names = []
    for signature_key, signature_names in signature_data.items():
      for signature_name in signature_names:
        tensor_name = self._search_tensor_by_signature_name(
            signature_key, signature_name
        )
        tensor_names.extend(tensor_name)

    # Overwrite the min and max values in the QSV.
    for tensor_name in tensor_names:
      qsv[tensor_name]["min"] = min_value
      qsv[tensor_name]["max"] = max_value
