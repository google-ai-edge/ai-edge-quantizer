"""A library of comparator function to be used by validation function."""

from collections.abc import Callable
from typing import Any, Tuple
import numpy as np


def get_validation_func(
    func_name: str,
) -> Callable[[np._typing.ArrayLike, np._typing.ArrayLike], Any]:
  """Returns a validation function based on the function name.

  Args:
    func_name: name of the validation function

  Returns:
    a validation function

  Raises:
    Value error if the function name is not supported
  """
  if func_name == "mse":
    return mean_squared_difference
  elif func_name == "median_diff_ratio":
    return median_diff_ratio
  else:
    raise ValueError(f"Validation function {func_name} not supported")


def mean_squared_difference(
    data1: np._typing.ArrayLike, data2: np._typing.ArrayLike
) -> float:
  """Calculates the mean squared difference between data1 & data2.

  ref: https://en.wikipedia.org/wiki/Mean_squared_error

  Args:
    data1: input data to be used for comparison
    data2: input data to be used for comparison, data1 & 2 must be of the same
      shape

  Returns:
    a float value representing the MSD between data1 & 2

  Raises:
    Value error if the two inputs don't have the same number of elements
  """
  data1, data2 = _preprocess_same_size_arrays(data1, data2)
  # special handling for tensor of size 0
  if data1.size == 0:
    return float(0)
  return float(np.square(np.subtract(data1, data2)).mean())


def median_diff_ratio(
    data1: np._typing.ArrayLike,
    data2: np._typing.ArrayLike,
    tolerance_threshold=1e-6,
) -> float:
  """Calculates the median absolute diff ratio between data1 & data2.

  mdr = median(abs(data1 - data2) / data2)

  Args:
    data1: input data to be used for comparison
    data2: input data to be used for comparison, data1 & 2 must be of the same
      shape
    tolerance_threshold: a float value to be used as a threshold to avoid
      division by zero

  Returns:
    a float value representing the median diff ratio between data1 & 2

  Raises:
    Value error if the two inputs don't have the same number of elements
  """
  data1, data2 = _preprocess_same_size_arrays(data1, data2)
  # special handling for tensor of size 0
  if data1.size == 0:
    return float(0)
  diff = abs(data1 - data2)
  demoninator = abs(data2) + tolerance_threshold
  median_ratio = np.median(diff / demoninator)
  return median_ratio


def _preprocess_same_size_arrays(
    data1: np._typing.ArrayLike, data2: np._typing.ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
  """Flattens and removes the nan, inf, and -inf values from the input data.

  Args:
    data1: input data to be used for comparison
    data2: input data to be used for comparison, data1 & 2 must be of the same
      shape

  Returns:
    a tuple of the preprocessed data1 & 2

  Raises:
    Value error if the two inputs don't have the same number of elements
  """
  data1 = np.array(data1, dtype=np.float32).flatten()
  data2 = np.array(data2, dtype=np.float32).flatten()
  if np.shape(data1) != np.shape(data2):
    raise ValueError("data1 & data2 must be of the same size")
  data1 = np.nan_to_num(data1, nan=1e-9, neginf=-1e9, posinf=1e9)
  data2 = np.nan_to_num(data2, nan=1e-9, neginf=-1e9, posinf=1e9)

  return data1, data2
