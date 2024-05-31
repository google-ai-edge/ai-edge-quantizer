"""function for validating output models."""

from collections.abc import Callable, Iterable
import json
import math
from typing import Any, Union

import numpy as np

from ai_edge_quantizer.utils import tfl_interpreter_utils


# TODO(b/331655892): have this function automatically detect the input tensor
# type
def compare_model(
    reference_model_path: str,
    target_model_path: str,
    dataset: Iterable[Any],
    quantize_target_input: bool,
    compare_fn: Callable[[Any, Any], float],
) -> dict[str, float]:
  """Produces comparison of all intermediate tensors given 2 models and a compare_fn.

  This function returns a per-tensor mean difference comparison across all
  inputs in the dataset, which will be returned at the end of this function

  Args:
    reference_model_path: path to the model which will be used as the reference
    target_model_path: path to the model which we're interested in the output,
      we expect reference_model and target_model have the inputs and outputs
    dataset: A list of input dataset to be run on reference and target models
    quantize_target_input: indicating whether the target requires quantized
      input
    compare_fn: a comparison function to be used for calculating the statistics,
      this function must be taking in two ArrayLike strcuture and output a
      single float value

  Returns:
    a dictionary of tensor name and a single float value representing
    the differences
  """
  reference_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
      reference_model_path
  )
  target_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
      target_model_path
  )
  comparison_results = {}

  # TODO(b/330797129): enable multi-threaded evaluation
  for data in dataset:
    tfl_interpreter_utils.invoke_interpreter_once(
        reference_interpreter, data, False
    )
    tfl_interpreter_utils.invoke_interpreter_once(
        target_interpreter, data, quantize_target_input
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
            compare_fn(reference_data, target_data)
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
