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

from typing import Union
import numpy as np
from ai_edge_quantizer import qtyping


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
