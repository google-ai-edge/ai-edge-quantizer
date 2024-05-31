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
    input_details: list[Any],
    num_samples: int,
    random_seed: Union[int, np._typing.ArrayLike],
) -> list[list[np._typing.ArrayLike]]:
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
    input_data = []
    for input_tensor in input_details:
      input_data.append(
          rng.normal(size=input_tensor['shape']).astype(input_tensor['dtype'])
      )
    dataset.append(input_data)
  return dataset


def create_random_normal_input_data(
    tfl_model_path: str, num_samples: int = 8, random_seed: int = 666
) -> list[list[np._typing.ArrayLike]]:
  """create random dataset for a TFLite model following normal distribution.

  Args:
    tfl_model_path: path to the tflite model
    num_samples: number of input samples to be generated
    random_seed: random seed to be used for function

  Returns:
    a list of inputs to the given interpreter, for a single interpreter we may
    have multiple input tensors so each set of inputs is also represented as
    list
  """
  tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(tfl_model_path)
  return create_random_normal_dataset(
      tfl_interpreter.get_input_details(), num_samples, random_seed
  )
