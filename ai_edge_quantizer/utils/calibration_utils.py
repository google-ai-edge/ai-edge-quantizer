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
import json
from typing import Any

from absl import logging
import numpy as np

import os
import io
from ai_edge_litert.tools import flatbuffer_utils
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.utils import common_utils
from ai_edge_quantizer.utils import constrained_ops_utils
from ai_edge_quantizer.utils import histogram_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_SignatureInput = dict[str, Any]
_OpQuantConstraint = common_utils.OpQuantConstraint
_SignatureData = dict[
    str, list[str]
]  # signature_key -> list of signature_names.


class NumpyEncoder(json.JSONEncoder):
  """JSON Encoder for Numpy types."""

  def default(self, o):
    if isinstance(o, np.integer):
      return int(o)
    elif isinstance(o, np.floating):
      return float(o)
    elif isinstance(o, np.ndarray):
      return o.tolist()
    return super().default(o)


def _convert_to_arrays(qsv: dict[str, Any]) -> None:
  """Helper to convert lists to numpy arrays in QSV dict."""
  if "min" in qsv:
    qsv["min"] = np.array(qsv["min"])
  if "max" in qsv:
    qsv["max"] = np.array(qsv["max"])

  if "channels" in qsv:
    for channel_qsv in qsv["channels"]:
      if "hist_counts" in channel_qsv:
        channel_qsv["hist_counts"] = np.array(channel_qsv["hist_counts"])
      if "bin_edges" in channel_qsv:
        channel_qsv["bin_edges"] = np.array(channel_qsv["bin_edges"])
      if "min" in channel_qsv:
        channel_qsv["min"] = np.array(channel_qsv["min"])
      if "max" in channel_qsv:
        channel_qsv["max"] = np.array(channel_qsv["max"])


def load_calibration_results(
    file_path: str,
) -> tuple[dict[str, qtyping.QSV], dict[str, Any]]:
  """Loads calibration results from a file.

  Args:
    file_path: Path to the calibration results file.

  Returns:
    A tuple of (model_qsvs, metadata).
  """
  with open(file_path) as json_file:
    data = json.load(json_file)

  metadata = {}
  if "model_qsvs" in data:
    model_qsvs = data["model_qsvs"]
    metadata = data.get("metadata", {})
  else:
    model_qsvs = data

  # Convert lists back to numpy arrays
  for _, qsv in model_qsvs.items():
    _convert_to_arrays(qsv)

  return model_qsvs, metadata


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
    # Use np.min/np.max to handle potential per-channel arrays
    min_value = min(min_value, np.min(qsv[tensor_name]["min"]))
    max_value = max(max_value, np.max(qsv[tensor_name]["max"]))
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
    """Searches for a tensor name by signature input or output name.

    This function searches for a tensor name for a given signature by signature
    input or output name.

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
    """Aligns quantization statistics (QSVs) across tensors in the given signatures.

    This function identifies all tensors associated with the provided
    signatures, finds the global minimum and maximum values across those tensors
    in the `qsv` dictionary, and updates the dictionary so all these tensors
    share these same min/max values. This ensures consistent quantization
    parameters.

    Example:
      To align the KV cache tensors 'kv_cache_k_0' and 'kv_slice_k_0' across
      'prefill' and 'decode' signatures:

      ```
      alignment_utils = CalibrationQsvAlignmentUtils(model_path)
      signature_data = {
          'prefill': ['kv_cache_k_0', 'kv_slice_k_0'],
          'decode': ['kv_cache_k_0', 'kv_slice_k_0'],
      }
      min_val, max_val = alignment_utils.align_quant_stats(
          qsv, signature_data
      )
      ```

      This will update `qsv` in place so that both 'kv_cache_k_0' and
      'kv_slice_k_0' have the same min and max values found across all
      tensors in the 'prefill' and 'decode' signatures, and returns those
      values. In other words, kv_cache_k_0 and kv_slice_k_0 will have the
      same quantization parameters across prefill and decode signatures.

    Args:
      qsv: Dictionary mapping tensor names to their quantization scale values
        (min and max). This dictionary is modified in place.
      signature_data: Mapping from signature keys to lists of signature names.

    Returns:
      A tuple (min_value, max_value), where min_value is the global minimum
      value and max_value is the global maximum value used for alignment.
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
  ) -> None:
    """Updates quantization statistics for tensors associated with the given signatures.

    This function identifies all tensors associated with the provided
    signatures,
    and updates their min/max values in the `qsv` dictionary with the provided
    `min_value` and `max_value`. This is typically used to apply aligned
    quantization parameters across different models (e.g., applying parameters
    aligned on a main model to an auxiliary model).

    Example:
      To apply pre-computed min and max values to KV cache tensors
      'kv_cache_k_0'
      and 'kv_slice_k_0' across 'prefill' and 'decode' signatures:

      ```
      alignment_utils = CalibrationQsvAlignmentUtils(aux_model_path)
      signature_data = {
          'prefill': ['kv_cache_k_0', 'kv_slice_k_0'],
          'decode': ['kv_cache_k_0', 'kv_slice_k_0'],
      }
      alignment_utils.update_quant_stats(
          aux_qsv, signature_data, min_value, max_value
      )
      ```

    Args:
      qsv: Dictionary mapping tensor names to their quantization scale values
        (min and max). This dictionary is modified in place.
      signature_data: Mapping from signature keys to lists of signature names.
      min_value: The global minimum value to apply.
      max_value: The global maximum value to apply.
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


def merge_qsvs(
    qsvs1: dict[str, qtyping.QSV], qsvs2: dict[str, qtyping.QSV]
) -> dict[str, qtyping.QSV]:
  """Merges two QSV dictionaries."""
  merged_qsvs = {}
  all_tensors = set(qsvs1.keys()) | set(qsvs2.keys())

  for tensor_name in all_tensors:
    if tensor_name not in qsvs1:
      merged_qsvs[tensor_name] = copy.deepcopy(qsvs2[tensor_name])
      continue
    if tensor_name not in qsvs2:
      merged_qsvs[tensor_name] = copy.deepcopy(qsvs1[tensor_name])
      continue

    qsv1 = qsvs1[tensor_name]
    qsv2 = qsvs2[tensor_name]

    # Check if both have histograms (unified format has 'channels')
    has_hist1 = "channels" in qsv1
    has_hist2 = "channels" in qsv2

    if has_hist1 and has_hist2:
      hist1 = histogram_utils.DynamicHistogram.from_dict(qsv1)
      hist2 = histogram_utils.DynamicHistogram.from_dict(qsv2)
      hist1.merge(hist2)
      merged_qsv = hist1.to_dict()
    else:
      if has_hist1 or has_hist2:
        logging.warning(
            "Only one QSV has histogram for tensor %s. Falling back to"
            " min/max.",
            tensor_name,
        )
      min1 = np.atleast_1d(qsv1["min"])
      min2 = np.atleast_1d(qsv2["min"])
      max1 = np.atleast_1d(qsv1["max"])
      max2 = np.atleast_1d(qsv2["max"])
      merged_qsv = {
          "min": np.minimum(min1, min2),
          "max": np.maximum(max1, max2),
      }

    merged_qsvs[tensor_name] = merged_qsv

  return merged_qsvs


def merge_calibration_results(
    results1: tuple[dict[str, qtyping.QSV], dict[str, Any]],
    results2: tuple[dict[str, qtyping.QSV], dict[str, Any]],
) -> tuple[dict[str, qtyping.QSV], dict[str, Any]]:
  """Merges two calibration results (qsvs and metadata)."""
  qsvs1, meta1 = results1
  qsvs2, meta2 = results2

  merged_qsvs = merge_qsvs(qsvs1, qsvs2)

  merged_meta = {}
  num_samples1 = meta1.get("num_samples_calibrated", 0)
  num_samples2 = meta2.get("num_samples_calibrated", 0)
  if num_samples1 or num_samples2:
    merged_meta["num_samples_calibrated"] = num_samples1 + num_samples2

  for k, v in meta1.items():
    if k != "num_samples_calibrated":
      merged_meta[k] = v
  for k, v in meta2.items():
    if k != "num_samples_calibrated":
      if k in merged_meta and merged_meta[k] != v:
        logging.warning(
            "Overwriting metadata key %s: '%s' with '%s'", k, merged_meta[k], v
        )
      merged_meta[k] = v

  return merged_qsvs, merged_meta
