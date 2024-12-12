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

"""function for validating output models."""

from collections.abc import Callable, Iterable
import dataclasses
import json
import math
import os
from typing import Any, Optional, Union

import numpy as np

from ai_edge_quantizer.utils import tfl_interpreter_utils as utils
from tensorflow.python.platform import gfile  # pylint: disable=g-direct-tensorflow-import


_DEFAULT_SIGNATURE_KEY = utils.DEFAULT_SIGNATURE_KEY


@dataclasses.dataclass(frozen=True)
class SingleSignatureComparisonResult:
  """Comparison result for a single signature.

  Attributes:
    error_metric: The name of the error metric used for comparison.
    input_tensors: A dictionary of input tensor name and its value.
    output_tensors: A dictionary of output tensor name and its value.
    constant_tensors: A dictionary of constant tensor name and its value.
    intermediate_tensors: A dictionary of intermediate tensor name and its
      value.
  """

  error_metric: str
  input_tensors: dict[str, Any]
  output_tensors: dict[str, Any]
  constant_tensors: dict[str, Any]
  intermediate_tensors: dict[str, Any]


class ComparisonResult:
  """Comparison result for a model.

  Attributes:
    comparison_results: A dictionary of signature key and its comparison result.
  """

  def __init__(self, reference_model: bytes, target_model: bytes):
    """Initialize the ComparisonResult object.

    Args:
      reference_model: Model which will be used as the reference.
      target_model: Target model which will be compared against the reference.
        We expect target_model and reference_model to have the same graph
        structure.
    """
    self._reference_model = reference_model
    self._target_model = target_model
    self._comparison_results: dict[str, SingleSignatureComparisonResult] = {}

  def get_signature_comparison_result(
      self, signature_key: str = _DEFAULT_SIGNATURE_KEY
  ) -> SingleSignatureComparisonResult:
    """Get the comparison result for a signature.

    Args:
      signature_key: The signature key to be used for invoking the models.

    Returns:
      A SingleSignatureComparisonResult object.
    """
    if signature_key not in self._comparison_results:
      raise ValueError(
          f'{signature_key} is not in the comparison_results. Available'
          f' signature keys are: {self.available_signature_keys()}'
      )
    return self._comparison_results[signature_key]

  def available_signature_keys(self) -> list[str]:
    """Get the available signature keys in the comparison result."""
    return list(self._comparison_results.keys())

  def add_new_signature_results(
      self,
      error_metric: str,
      comparison_result: dict[str, float],
      signature_key: str = _DEFAULT_SIGNATURE_KEY,
  ):
    """Add a new signature result to the comparison result.

    Args:
      error_metric: The name of the error metric used for comparison.
      comparison_result: A dictionary of tensor name and its value.
      signature_key: The model signature that the comparison_result belongs to.

    Raises:
      ValueError: If the signature_key is already in the comparison_results.
    """
    if signature_key in self._comparison_results:
      raise ValueError(f'{signature_key} is already in the comparison_results.')

    result = {key: float(value) for key, value in comparison_result.items()}

    input_tensor_results = {}
    for name in utils.get_input_tensor_names(
        self._reference_model, signature_key
    ):
      input_tensor_results[name] = result.pop(name)

    output_tensor_results = {}
    for name in utils.get_output_tensor_names(
        self._reference_model, signature_key
    ):
      output_tensor_results[name] = result.pop(name)

    constant_tensor_results = {}
    # Only get constant tensors from the main subgraph of the signature.
    subgraph_index = utils.get_signature_main_subgraph_index(
        utils.create_tfl_interpreter(self._reference_model),
        signature_key,
    )
    for name in utils.get_constant_tensor_names(
        self._reference_model,
        subgraph_index,
    ):
      constant_tensor_results[name] = result.pop(name)

    self._comparison_results[signature_key] = SingleSignatureComparisonResult(
        error_metric=error_metric,
        input_tensors=input_tensor_results,
        output_tensors=output_tensor_results,
        constant_tensors=constant_tensor_results,
        intermediate_tensors=result,
    )

  def get_all_tensor_results(self) -> dict[str, Any]:
    """Get all the tensor results in a single dictionary.

    Returns:
      A dictionary of tensor name and its value.
    """
    result = {}
    for _, signature_comparison_result in self._comparison_results.items():
      result.update(signature_comparison_result.input_tensors)
      result.update(signature_comparison_result.output_tensors)
      result.update(signature_comparison_result.constant_tensors)
      result.update(signature_comparison_result.intermediate_tensors)
    return result

  def save(self, save_folder: str, model_name: str) -> None:
    """Saves the model comparison result.

    Args:
      save_folder: Path to the folder to save the comparison result.
      model_name: Name of the model.

    Raises:
      RuntimeError: If no quantized model is available.
    """
    reduced_model_size = len(self._reference_model) - len(self._target_model)
    reduction_ratio = reduced_model_size / len(self._reference_model) * 100
    result = {
        'reduced_size_bytes': reduced_model_size,
        'reduced_size_percentage': reduction_ratio,
    }
    for signature, comparison_result in self._comparison_results.items():
      result[str(signature)] = {
          'error_metric': comparison_result.error_metric,
          'input_tensors': comparison_result.input_tensors,
          'output_tensors': comparison_result.output_tensors,
          'constant_tensors': comparison_result.constant_tensors,
          'intermediate_tensors': comparison_result.intermediate_tensors,
      }
    result_save_path = os.path.join(
        save_folder, model_name + '_comparison_result.json'
    )
    with gfile.GFile(result_save_path, 'w') as output_file_handle:
      output_file_handle.write(json.dumps(result))

    # TODO: b/365578554 - Remove after ME is updated to use the new json format.
    color_threshold = [0.05, 0.1, 0.2, 0.4, 1, 10, 100]
    json_object = create_json_for_model_explorer(
        self,
        threshold=color_threshold,
    )
    json_save_path = os.path.join(
        save_folder, model_name + '_comparison_result_me_input.json'
    )
    with gfile.GFile(json_save_path, 'w') as output_file_handle:
      output_file_handle.write(json_object)


def _setup_validation_interpreter(
    model: bytes,
    signature_input: dict[str, Any],
    signature_key: Optional[str],
    use_xnnpack: bool,
    num_threads: int,
) -> tuple[Any, int, dict[str, Any]]:
  """Setup the interpreter for validation given a signature key.

  Args:
    model: The model to be validated.
    signature_input: A dictionary of input tensor name and its value.
    signature_key: The signature key to be used for invoking the models. If the
      model only has one signature, this can be set to None.
    use_xnnpack: Whether to use xnnpack for the interpreter.
    num_threads: The number of threads to use for the interpreter.

  Returns:
    A tuple of interpreter, subgraph_index and tensor_name_to_details.
  """

  interpreter = utils.create_tfl_interpreter(
      tflite_model=model, use_xnnpack=use_xnnpack, num_threads=num_threads
  )
  utils.invoke_interpreter_signature(
      interpreter, signature_input, signature_key
  )
  # Only validate tensors from the main subgraph of the signature.
  subgraph_index = utils.get_signature_main_subgraph_index(
      interpreter, signature_key
  )
  tensor_name_to_details = utils.get_tensor_name_to_details_map(
      interpreter,
      subgraph_index,
  )
  return interpreter, subgraph_index, tensor_name_to_details


# TODO: b/330797129 - Enable multi-threaded evaluation.
def compare_model(
    reference_model: bytes,
    target_model: bytes,
    test_data: dict[str, Iterable[dict[str, Any]]],
    error_metric: str,
    compare_fn: Callable[[Any, Any], float],
    use_xnnpack: bool = True,
    num_threads: int = 16,
) -> ComparisonResult:
  """Compares model tensors over a model signature using the compare_fn.

  This function will run the model signature on the provided dataset over and
  compare all the tensors (cached) using the compare_fn (e.g., mean square
  error).

  Args:
    reference_model: Model which will be used as the reference
    target_model: Target model which will be compared against the reference. We
      expect reference_model and target_model have the inputs and outputs
      signature.
    test_data: A dictionary of signature key and its correspending test input
      data that will be used for comparison.
    error_metric: The name of the error metric used for comparison.
    compare_fn: a comparison function to be used for calculating the statistics,
      this function must be taking in two ArrayLike strcuture and output a
      single float value.
    use_xnnpack: Whether to use xnnpack for the interpreter.
    num_threads: The number of threads to use for the interpreter.

  Returns:
    A ComparisonResult object.
  """
  model_comparion_result = ComparisonResult(reference_model, target_model)
  for signature_key, signature_inputs in test_data.items():
    comparison_results = {}
    for signature_input in signature_inputs:
      # Invoke the signature on both interpreters.
      ref_interpreter, ref_subgraph_index, ref_tensor_name_to_details = (
          _setup_validation_interpreter(
              reference_model,
              signature_input,
              signature_key,
              use_xnnpack,
              num_threads,
          )
      )
      targ_interpreter, targ_subgraph_index, targ_tensor_name_to_details = (
          _setup_validation_interpreter(
              target_model,
              signature_input,
              signature_key,
              use_xnnpack,
              num_threads,
          )
      )
      # Compare the cached tensor values.
      for tensor_name, detail in ref_tensor_name_to_details.items():
        if detail['dtype'] == np.object_:
          continue
        if tensor_name in targ_tensor_name_to_details:
          if tensor_name not in comparison_results:
            comparison_results[tensor_name] = []

          reference_data = utils.get_tensor_data(
              ref_interpreter, detail, ref_subgraph_index
          )
          target_data = utils.get_tensor_data(
              targ_interpreter,
              targ_tensor_name_to_details[tensor_name],
              targ_subgraph_index,
          )
          comparison_results[tensor_name].append(
              compare_fn(target_data, reference_data)
          )

    agregated_results = {}
    for tensor_name in comparison_results:
      agregated_results[tensor_name] = np.mean(comparison_results[tensor_name])
    model_comparion_result.add_new_signature_results(
        error_metric,
        agregated_results,
        signature_key,
    )
  return model_comparion_result


def create_json_for_model_explorer(
    data: ComparisonResult, threshold: list[Union[int, float]]
) -> str:
  """create a dict type that can be exported as json for model_explorer to use.

  Args:
    data: Output from compare_model function
    threshold: A list of numbers representing thresholds for model_exlorer to
      display different colors

  Returns:
    A string represents the json format accepted by model_explorer
  """
  data = data.get_all_tensor_results()
  color_scheme = []
  results = {name: {'value': float(value)} for name, value in data.items()}
  if threshold:
    green = 255
    gradient = math.floor(255 / len(threshold))
    for val in threshold:
      color_scheme.append({'value': val, 'bgColor': f'rgb(200, {green}, 0)'})
      green = max(0, green - gradient)

  return json.dumps({
      'results': results,
      'thresholds': color_scheme,
  })
