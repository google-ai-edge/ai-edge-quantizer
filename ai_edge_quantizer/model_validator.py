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


@dataclasses.dataclass(frozen=True, init=False)
class ComparisonResult:
  """A class for storing the comparison result of two models.

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

  def __init__(
      self,
      source_model: Union[str, bytearray],
      error_metric: str,
      comparison_result: dict[str, float],
  ):
    """Initializes the ComparisonResult object.

    Args:
      source_model: The model that is used to generate the comparison result.
      error_metric: The name of the error metric used for comparison.
      comparison_result: The comparison result from compare_model.
    """
    # Construct a new comparison_result dictionary with float values (needed for
    # json serialization).
    result = {key: float(value) for key, value in comparison_result.items()}

    input_tensor_results = {}
    for name in utils.get_input_tensor_names(source_model):
      input_tensor_results[name] = result.pop(name)

    output_tensor_results = {}
    for name in utils.get_output_tensor_names(source_model):
      output_tensor_results[name] = result.pop(name)

    constant_tensor_results = {}
    for name in utils.get_constant_tensor_names(source_model):
      constant_tensor_results[name] = result.pop(name)

    object.__setattr__(self, 'error_metric', error_metric)
    object.__setattr__(self, 'input_tensors', input_tensor_results)
    object.__setattr__(self, 'output_tensors', output_tensor_results)
    object.__setattr__(self, 'constant_tensors', constant_tensor_results)
    object.__setattr__(self, 'intermediate_tensors', result)

  def get_all_tensor_results(self) -> dict[str, Any]:
    """Get all the tensor results in a single dictionary.

    Returns:
      A dictionary of tensor name and its value.
    """
    result = self.input_tensors.copy()
    result.update(self.output_tensors)
    result.update(self.constant_tensors)
    result.update(self.intermediate_tensors)
    return result

  def save(self, save_folder: str, model_name: str) -> None:
    """Saves the model comparison result.

    Args:
      save_folder: Path to the folder to save the comparison result.
      model_name: Name of the model.

    Raises:
      RuntimeError: If no quantized model is available.
    """
    result = {
        'error_metric': self.error_metric,
        'input_tensors': self.input_tensors,
        'output_tensors': self.output_tensors,
        'constant_tensors': self.constant_tensors,
        'intermediate_tensors': self.intermediate_tensors,
    }
    result_save_path = os.path.join(
        save_folder, model_name + '_comparison_result.json'
    )
    with gfile.GFile(result_save_path, 'w') as output_file_handle:
      output_file_handle.write(json.dumps(result))

    # TODO: b/358122753 - Automatically generate the threshold.
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


# TODO: b/331655892 - have this function automatically detect the input tensor
# type
def compare_model(
    reference_model: Union[str, bytearray],
    target_model: Union[str, bytearray],
    signature_dataset: Iterable[dict[str, Any]],
    error_metric: str,
    compare_fn: Callable[[Any, Any], float],
    signature_key: Optional[str] = None,
    use_reference_kernel: bool = False,
) -> ComparisonResult:
  """Compares model tensors over a model signature using the compare_fn.

  This function will run the model signature on the provided dataset over and
  compare all the tensors (cached) using the compare_fn (e.g., mean square
  error).

  Args:
    reference_model: Model which will be used as the reference
    target_model: Target model which will be compared against the reference. We
      expect reference_model and target_model have the inputs and outputs
    signature_dataset: A list of inputs of the signature to be run on reference
      and target models.
    error_metric: The name of the error metric used for comparison.
    compare_fn: a comparison function to be used for calculating the statistics,
      this function must be taking in two ArrayLike strcuture and output a
      single float value.
    signature_key: the signature key to be used for invoking the models. If the
      model doesn't have a signature key, this can be set to None.
    use_reference_kernel: Whether to use the reference kernel for the
      interpreter.

  Returns:
    A ComparisonResult object.
  """
  reference_interpreter = utils.create_tfl_interpreter(
      tflite_model=reference_model, use_reference_kernel=use_reference_kernel
  )
  target_interpreter = utils.create_tfl_interpreter(
      tflite_model=target_model, use_reference_kernel=use_reference_kernel
  )
  comparison_results = {}

  # TODO: b/330797129 - enable multi-threaded evaluation.
  for signature_input in signature_dataset:
    utils.invoke_interpreter_signature(
        reference_interpreter, signature_input, signature_key
    )
    utils.invoke_interpreter_signature(
        target_interpreter,
        signature_input,
        signature_key,
    )

    reference_name_to_details = utils.get_tensor_name_to_details_map(
        reference_interpreter
    )
    target_name_to_details = utils.get_tensor_name_to_details_map(
        target_interpreter
    )

    for tensor_name, detail in reference_name_to_details.items():
      if detail['dtype'] == np.object_:
        continue
      if tensor_name in target_name_to_details:
        if tensor_name not in comparison_results:
          comparison_results[tensor_name] = []
        reference_data = utils.get_tensor_data(reference_interpreter, detail)
        target_data = utils.get_tensor_data(
            target_interpreter, target_name_to_details[tensor_name]
        )
        comparison_results[tensor_name].append(
            compare_fn(target_data, reference_data)
        )

  agregated_results = {}
  for tensor_name in comparison_results:
    agregated_results[tensor_name] = np.mean(comparison_results[tensor_name])

  return ComparisonResult(
      reference_model,
      error_metric,
      agregated_results,
  )


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
