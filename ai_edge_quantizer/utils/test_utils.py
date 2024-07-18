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

"""Utils for tests."""

import inspect as _inspect
import os.path as _os_path
import sys as _sys
from typing import Any, Union

import numpy as np

from ai_edge_quantizer.utils import tfl_interpreter_utils


def get_path_to_datafile(path):
  """Get the path to the specified file in the data dependencies.

  The path is relative to the file calling the function.

  Args:
    path: a string resource path relative to the calling file.

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  data_files_path = _os_path.dirname(_inspect.getfile(_sys._getframe(1)))  # pylint: disable=protected-access
  path = _os_path.join(data_files_path, path)
  path = _os_path.normpath(path)
  return path


def create_random_normal_dataset(
    input_details: dict[str, Any],
    num_samples: int,
    random_seed: Union[int, np._typing.ArrayLike],
) -> list[dict[str, Any]]:
  """create random dataset following random distribution.

  Args:
    input_details: list of dict created by
      tensorflow.lite.interpreter.get_input_details() for generating dataset
    num_samples: number of input samples to be generated
    random_seed: random seed to be used for function

  Returns:
    a list of inputs to the given interpreter, for a single interpreter we may
    have multiple input tensors so each set of inputs is also represented as
    list
  """
  rng = np.random.default_rng(random_seed)
  dataset = []
  for _ in range(num_samples):
    input_data = {}
    for arg_name, input_tensor in input_details.items():
      new_data = rng.normal(size=input_tensor['shape']).astype(
          input_tensor['dtype']
      )
      input_data[arg_name] = new_data
    dataset.append(input_data)
  return dataset


def create_random_normal_input_data(
    tflite_model: Union[str, bytearray],
    num_samples: int = 4,
    random_seed: int = 666,
) -> dict[str, list[dict[str, Any]]]:
  """create random dataset following random distribution for signature runner.

  Args:
    tflite_model: TFLite model path or bytearray
    num_samples: number of input samples to be generated
    random_seed: random seed to be used for function

  Returns:
    a list of inputs to the given interpreter, for a single interpreter we may
    have multiple signatures so each set of inputs is also represented as
    list
  """
  tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(tflite_model)
  signature_defs = tfl_interpreter.get_signature_list()
  signature_keys = list(signature_defs.keys())
  test_data = {}
  for signature_key in signature_keys:
    signature_runner = tfl_interpreter.get_signature_runner(signature_key)
    input_details = signature_runner.get_input_details()
    test_data[signature_key] = create_random_normal_dataset(
        input_details, num_samples, random_seed
    )
  return test_data
