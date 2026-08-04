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

"""Implements OSCAR calibration: activation-aware optimal clipping for weight tensors.

OSCAR (Optimal Scaling and Channel ARrangement) minimizes the
activation-weighted weight error E||(Q(W) - W) x||^2 under a diagonal
activation model, instead of plain weight range (min/max) or weight MSE.

This module (`oscar.py`) is responsible for collecting the
per-input-channel second moment mu2_j = E[x_j^2] -- exactly the diagonal
of the Hessian that GPTQ collects, at O(d) memory instead of O(d^2) --
which is needed by the clipping bounds search.

The OSCAR algorithm is implemented across 3 files:
1. `oscar.py` (this file): Collects activation statistics (mu2).
2. `oscar_quant_params.py`: Computes the optimal clipping bounds based on the
stats.
3. `quantization/oscar_conditioner.py`: The second part of OSCAR --
per-input-channel
scales folded into upstream producers (SmoothQuant-style equivalent transform)
--
which is a whole-graph conditioning pass that runs *before* quantization.

Scope: FULLY_CONNECTED only. FC dominates LLM parameter counts, and keeping a
single op keeps the layout, the fold rules in the conditioner, and the test
surface small.
"""

from collections.abc import MutableMapping, Sequence
import dataclasses
from typing import Any
from absl import logging
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import common_quantize
from ai_edge_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from ai_edge_quantizer.algorithms.utils import common_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

ALGORITHM_KEY = "OSCAR"
_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation


def calibrate(
    tfl_op: qtyping.OperatorT,
    graph_info: qtyping.GraphInfo,
    tensor_content_map: MutableMapping[str, np.ndarray],
    inputs_to_ignore: Sequence[int] | None = None,
    outputs_to_ignore: Sequence[int] | None = None,
    valid_range: tuple[float, float] = (-3e38, 3e38),
) -> dict[str, qtyping.QSV]:
  """Collects min/max plus per-channel activation second moments (mu2).

  Mirrors gptq.calibrate, storing only the Hessian diagonal:
  mu2 = mean(x*x) over all rows, per trailing-axis channel.

  Args:
    tfl_op: The tfl operation.
    graph_info: Graph information needed to perform quantization for the op.
    tensor_content_map: A map of tensor name to tensor content.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.
    valid_range: The valid range for tensor content for min/max collection.

  Returns:
    A dictionary mapping tensor names to the collected QSVs.
  """
  op_qsvs = {}
  min_val, max_val = valid_range

  tensor_ids = common_quantize.get_tensor_indices_requiring_calibration(
      tfl_op, graph_info, inputs_to_ignore, outputs_to_ignore
  )
  for tensor_idx in tensor_ids:
    result = common_quantize.collect_activation_tensor_statistics(
        tensor_idx,
        graph_info,
        tensor_content_map,
        valid_float_range_min=min_val,
        valid_float_range_max=max_val,
    )
    if result is None:
      continue

    tensor_name, tensor_content, tensor_qsvs = result

    # Per-channel second moment over the trailing (channel) axis. This is
    # the diagonal of GPTQ's Hessian (up to its factor 2) at O(d) memory.
    x = np.asarray(tensor_content, np.float64).reshape(
        [-1, tensor_content.shape[-1]]
    )
    tensor_qsvs["mu2"] = np.mean(x * x, axis=0)
    tensor_qsvs["num_samples"] = x.shape[0]
    op_qsvs[tensor_name] = tensor_qsvs

  return op_qsvs


_EPS = 1e-12


def _floor_positive(mu2: np.ndarray) -> np.ndarray:
  """Guards against zero/negative masses (dead channels)."""
  mu2 = np.asarray(mu2, np.float64)
  return np.maximum(mu2, float(np.max(mu2)) * 1e-8 + _EPS)


def _optimal_group_clip(a: np.ndarray, m: np.ndarray, qmax: int) -> np.ndarray:
  """Exact per-row optimal clip bound via breakpoint scan.

  Minimizes, per row, E(c) = c^2 * M / (12 qmax^2)
                            + sum_j max(a_j - c, 0)^2 * m_j
  where M = sum_j m_j. Between consecutive sorted magnitudes the objective
  is quadratic in c, so each segment has a closed-form minimizer and the
  scan over segments is exact.

  Args:
    a: (n, g) weight magnitudes within one group.
    m: (g,) per-column masses (activation second moments).
    qmax: The largest quantized level, 2^(bits-1) - 1.

  Returns:
    (n,) optimal clip bound per row.
  """
  n, _ = a.shape
  order = np.argsort(-a, axis=1)
  a_s = np.take_along_axis(a, order, 1)  # descending
  m_s = m[order]
  mass = float(m.sum()) + _EPS

  s_m = np.cumsum(m_s, 1)
  s_am = np.cumsum(a_s * m_s, 1)
  s_a2m = np.cumsum(a_s * a_s * m_s, 1)

  # With exactly k clipped channels, dE/dc = 0 gives:
  c_k = 2.0 * s_am / (mass / (6.0 * qmax * qmax) + 2.0 * s_m)
  lower = np.concatenate([a_s[:, 1:], np.zeros((n, 1))], 1)
  c_k = np.clip(c_k, lower, a_s)  # keep each candidate inside its segment
  e_k = (
      (c_k**2) * (mass / (12.0 * qmax * qmax))
      + s_a2m
      - 2.0 * c_k * s_am
      + (c_k**2) * s_m
  )
  c0 = a_s[:, :1]  # no-clip candidate
  e0 = (c0**2) * (mass / (12.0 * qmax * qmax))

  cand_c = np.concatenate([c0, c_k], 1)
  cand_e = np.concatenate([e0, e_k], 1)
  return cand_c[np.arange(n), np.argmin(cand_e, 1)]


def _weight_layout(
    op_name: _TFLOpName,
    tensor_content: np.ndarray,
    mu2: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
  """Views the FC weight as (rows, cols) and maps mu2 onto the columns.
  
  OSCAR delibrately supports FULLY_CONNECTED only: FC dominates LLM parameter
  counts, and is the only op whose input-channel scales have the simple
  deployment story the conditioner relies on.

  Args:
    op_name: The TFL op that consumes the weight.
    tensor_content: The weight tensor.
    mu2: Per-input-channel activation second moments, or None for uniform
      weighting (weight-only fallback: exact weight-MSE clipping).

  Returns:
    (matrix, col_mu2, channelwise_param_shape) where matrix rows are output
    channels, columns are the reduction (input-channel) axis, col_mu2 holds one
    mass per column, and channelwise_param_shape is the keepdims shape that
    channelwise scale/zero_point must have (matching
    common_quantize.init_tensor_min_max).
  """
  if op_name != _TFLOpName.FULLY_CONNECTED:
    raise ValueError(
        f"OSCAR supports FULLY_CONNECTED only, got: {op_name}"
    )
  w = np.asarray(tensor_content, np.float64)
  if w.ndim != 2:
    raise ValueError(
        f"OSCAR expects 2-D weights for {op_name}, got {w.shape}"
    )
  out_ch, in_ch = w.shape
  if mu2 is None:
    col_mu2 = np.ones(in_ch)
  else:
    col_mu2 = np.asarray(mu2, np.float64).ravel()
    if col_mu2.size != in_ch:
      raise ValueError(
          f"OSCAR: activation mu2 has {col_mu2.size} channels but"
          f" {op_name} weights of shape {w.shape} expect {in_ch}."
          " The calibration statistics do not match this tensor."
      )
  return w, col_mu2, (out_ch, 1)


def get_clip_bounds(
    op_name: _TFLOpName,
    tensor_content: np.ndarray,
    mu2: np.ndarray | None,
    num_bits: int,
    granularity: qtyping.QuantGranularity,
) -> np.ndarray:
  """Computes activation-weighted optimal symmetric clip bounds.

  The returned array has exactly the shape that min/max QSVs have for this
  op and granularity (see common_quantize.init_tensor_min_max), so it can be
  fed as min=-bounds, max=+bounds into tensor_zp_scale_from_min_max.

  Args:
    op_name: The TFL op that consumes the weight.
    tensor_content: The weight tensor.
    mu2: Per-input-channel activation second moments, or None for uniform
      weighting (weight-only fallback: exact weight-MSE clipping).
    num_bits: Bit width of the target integer type.
    granularity: The quantization granularity.

  Returns:
    Clip bounds: scalar keepdims shape for TENSORWISE, keepdims channel
    shape for CHANNELWISE, (rows, cols // block_size) for BLOCKWISE.
  """
  qmax = 2 ** (num_bits - 1) - 1
  matrix, col_mu2, param_shape = _weight_layout(op_name, tensor_content, mu2)
  col_mu2 = _floor_positive(col_mu2)
  n, d = matrix.shape
  a = np.abs(matrix)

  if granularity == qtyping.QuantGranularity.TENSORWISE:
    # One shared bound: solve the same problem on a single flattened row.
    bound = _optimal_group_clip(a.reshape(1, n * d), np.tile(col_mu2, n), qmax)
    return bound.reshape((1,) * tensor_content.ndim)

  if granularity == qtyping.QuantGranularity.CHANNELWISE:
    bounds = _optimal_group_clip(a, col_mu2, qmax)
    return bounds.reshape(param_shape)

  if uniform_quantize_tensor.is_blockwise(granularity):
    block_size = uniform_quantize_tensor.extract_block_size_from_granularity(
        granularity
    )
    if (
        op_name
        not in tfl_flatbuffer_utils.TFL_OP_TO_BLOCKWISE_WEIGHT_QUANTIZED_DIM
    ):
      raise ValueError(
          f"Blockwise granularity is not supported for op: {op_name}"
      )
    if d % block_size:
      raise ValueError(
          f"Block size {block_size} must divide the reduction dimension"
          f" {d} of {op_name} weights with shape {tensor_content.shape}."
      )
    n_blocks = d // block_size
    bounds = np.empty((n, n_blocks))
    for block_idx in range(n_blocks):
      cols = slice(block_idx * block_size, (block_idx + 1) * block_size)
      bounds[:, block_idx] = _optimal_group_clip(
          a[:, cols], col_mu2[cols], qmax
      )
    return bounds

  raise ValueError(f"Unsupported granularity: {granularity}")


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: np.ndarray | None = None,
    tensor_qsv: dict[str, Any] | None = None,
) -> qtyping.UniformQuantParams:
  """Returns the quantization parameters for a tensor.

  For weight tensors, replaces the max-abs bound with OSCAR's
  activation-weighted optimal clip bound and derives scale/zero_point and
  quantized data through the standard min/max machinery -- so the result is
  standard-representable by construction. Non-weight tensors fall back to
  naive_min_max_quantize.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor. None means the tensor is not
      a weight tensor.
    tensor_qsv: A dictionary containing the tensor QSVs. It may contain
      "activation_tensor_qsv" with the "mu2" statistic collected by
      calibrate(); if absent, OSCAR degrades gracefully to uniform
      weighting (exact weight-MSE clipping) with a warning.

  Raises:
    ValueError: If asymmetric weight quantization is requested, or blockwise
      is requested for an unsupported op, or the mu2 statistic exists but
      does not match the weight shape.
  """
  # Fallback to naive_min_max_quantize for non-weight tensors.
  if tensor_content is None:
    return naive_min_max_quantize.get_tensor_quant_params(
        op_info, tensor_quant_config, tensor_content, tensor_qsv
    )

  if not tensor_quant_config.symmetric:
    raise ValueError(
        "OSCAR supports symmetric weight quantization only, got asymmetric"
        f" config for op {op_info.op_name}."
    )

  mu2 = None
  if tensor_qsv:
    activation_tensor_qsv = tensor_qsv.get("activation_tensor_qsv")
    if activation_tensor_qsv:
      mu2 = activation_tensor_qsv.get("mu2")
  if mu2 is None:
    logging.warning(
        "OSCAR: no activation second moments (mu2) found for op %s (index"
        " %d). Was the model calibrated with the OSCAR algorithm? Falling"
        " back to uniform weighting (weight-MSE clipping).",
        op_info.op_name,
        op_info.subgraph_op_index,
    )

  granularity = tensor_quant_config.granularity
  is_blockwise = uniform_quantize_tensor.is_blockwise(granularity)

  bounds = get_clip_bounds(
      op_info.op_name,
      tensor_content,
      mu2,
      tensor_quant_config.num_bits,
      granularity,
  )
  zp, scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
      -bounds,
      bounds,
      tensor_quant_config.num_bits,
      tensor_quant_config.symmetric,
      granularity,
      None,
  )
  quantized_dim = common_utils.get_weight_quantized_dim(
      op_info, tensor_content, granularity
  )
  quant_params = qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=quantized_dim,
      block_size=uniform_quantize_tensor.extract_block_size_from_granularity(
          granularity
      ),
  )
  quantized_vars = uniform_quantize_tensor.uniform_quantize(
      tensor_content, quant_params, is_blockwise
  )
  return dataclasses.replace(quant_params, quantized_data=quantized_vars)
