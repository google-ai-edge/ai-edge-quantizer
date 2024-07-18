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
import json
import math
from typing import Any, Optional, Union

import numpy as np

from ai_edge_quantizer.utils import tfl_interpreter_utils


# TODO: b/331655892 - have this function automatically detect the input tensor
# type
def compare_model(
    reference_model: Union[str, bytearray],
    target_model: Union[str, bytearray],
    signature_dataset: Iterable[dict[str, Any]],
    compare_fn: Callable[[Any, Any], float],
    signature_key: Optional[str] = None,
    quantize_target_input: bool = True,
) -> dict[str, float]:
  """Compares model tensors over a model signature using the compare_fn.

  This function will run the model signature on the provided dataset over and
  compare all the tensors (cached) using the compare_fn (e.g., mean square
  error).

  Args:
    reference_model: Model which will be used as the reference
    target_model: Target model which will be compared against the reference.
      We expect reference_model and target_model have the inputs and outputs
    signature_dataset: A list of inputs of the signature to be run on reference
      and target models.
    compare_fn: a comparison function to be used for calculating the statistics,
      this function must be taking in two ArrayLike strcuture and output a
      single float value.
    signature_key: the signature key to be used for invoking the models. If the
      model doesn't have a signature key, this can be set to None.
    quantize_target_input: indicating whether the target requires quantized
      input.

  Returns:
    a dictionary of tensor name and a single float value representing
    the differences
  """
  reference_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
      reference_model
  )
  target_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
      target_model
  )
  comparison_results = {}

  # TODO: b/330797129 - enable multi-threaded evaluation.
  for signature_input in signature_dataset:
    tfl_interpreter_utils.invoke_interpreter_signature(
        reference_interpreter, signature_input, signature_key
    )
    tfl_interpreter_utils.invoke_interpreter_signature(
        target_interpreter,
        signature_input,
        signature_key,
        quantize_input=quantize_target_input,
    )

    reference_name_to_details = (
        tfl_interpreter_utils.get_tensor_name_to_details_map(
            reference_interpreter
        )
    )
    target_name_to_details = (
        tfl_interpreter_utils.get_tensor_name_to_details_map(target_interpreter)
    )

    for tensor_name, detail in reference_name_to_details.items():
      if tensor_name in target_name_to_details:
        if tensor_name not in comparison_results:
          comparison_results[tensor_name] = []
        reference_data = tfl_interpreter_utils.get_tensor_data(
            reference_interpreter, detail
        )
        target_data = tfl_interpreter_utils.get_tensor_data(
            target_interpreter, target_name_to_details[tensor_name]
        )
        comparison_results[tensor_name].append(
            compare_fn(target_data, reference_data)
        )

  agregated_results = {}
  for tensor_name in comparison_results:
    agregated_results[tensor_name] = np.mean(comparison_results[tensor_name])

  return agregated_results


def create_json_for_model_explorer(
    data: dict[str, float], threshold: list[Union[int, float]]
) -> str:
  """create a dict type that can be exported as json for model_explorer to use.

  Args:
    data: output from compare_model function
    threshold: a list of numbers representing thresholds for model_exlorer to
      display different colors

  Returns:
    a string represents the json format accepted by model_explorer
  """
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
