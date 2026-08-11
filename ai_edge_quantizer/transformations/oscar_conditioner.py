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

"""OSCAR scale conditioning pass: deploy activation-aware scales by graph rewrite.

This module implements the whole-graph scale conditioning pass of OSCAR (the
per-tensor half is in algorithms/uniform_quantize/oscar.py). For every
quantization site -- a float activation tensor feeding FULLY_CONNECTED weights
(the only op OSCAR supports) -- it computes per-input-channel scales s from
calibrated activation second moments and rewrites the model equivalently:

    W_consumer <- W_consumer * s        (per input channel; all consumers)
    x_site     <- x_site * (1/s)        (deployed upstream)
"""

from collections.abc import Iterable
import dataclasses
import enum
from typing import Any, Optional, Union

import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_OP = qtyping.BuiltinOperator
_EPS = 1e-12
_SCALE_CLAMP = (1e-4, 1e4)

_CONSUMER_OPS = {
    _OP.FULLY_CONNECTED: (1, 1, 2),
}


class ConditioningError(Exception):
  """A rewrite would change model semantics; the pass aborts."""


class SiteStatus(enum.Enum):
  FOLDED = "folded"
  MUL_INSERTED = "mul_inserted"
  SKIPPED = "skipped"


@dataclasses.dataclass
class SiteResult:
  """Outcome for one conditioning site."""

  subgraph_index: int
  tensor_name: str
  consumer_names: list[str]
  num_params: int
  status: SiteStatus
  reason: str = ""
  objective_ratio: float = 1.0


@dataclasses.dataclass
class ConditioningReport:
  """Per-site outcomes plus aggregate statistics."""

  sites: list[SiteResult] = dataclasses.field(default_factory=list)
  in_place_patched: bool = False

  def _by_status(self, status: SiteStatus) -> list[SiteResult]:
    return [s for s in self.sites if s.status == status]

  @property
  def num_folded(self) -> int:
    return len(self._by_status(SiteStatus.FOLDED))

  @property
  def num_mul_inserted(self) -> int:
    return len(self._by_status(SiteStatus.MUL_INSERTED))

  @property
  def num_skipped(self) -> int:
    return len(self._by_status(SiteStatus.SKIPPED))

  def summary(self) -> str:
    """Human-readable per-site report."""
    lines = []
    total_params = sum(s.num_params for s in self.sites) or 1
    conditioned_params = sum(
        s.num_params for s in self.sites if s.status != SiteStatus.SKIPPED
    )
    lines.append(
        f"OSCAR conditioning: {self.num_folded} folded,"
        f" {self.num_mul_inserted} MUL-inserted, {self.num_skipped} skipped"
        f" ({100.0 * conditioned_params / total_params:.1f}% of site"
        " parameters conditioned)"
    )
    for s in self.sites:
      tag = {
          SiteStatus.FOLDED: "fold",
          SiteStatus.MUL_INSERTED: "mul ",
          SiteStatus.SKIPPED: "skip",
      }[s.status]
      detail = f"obj {s.objective_ratio:5.2f}x" if s.objective_ratio else ""
      reason = f" ({s.reason})" if s.reason else ""
      lines.append(
          f"  {tag} sg{s.subgraph_index} {s.tensor_name[:60]:<60}"
          f" {detail}{reason}"
      )
    return "\n".join(lines)


# ============================================================================
# Graph helpers
# ============================================================================


def _builtin_code(model: qtyping.ModelT, op: qtyping.OperatorT) -> int:
  return model.operatorCodes[op.opcodeIndex].builtinCode


def _tensor_data(model, subgraph, tensor_idx) -> Optional[np.ndarray]:
  return tfl_flatbuffer_utils.get_tensor_data(
      subgraph.tensors[tensor_idx], model.buffers
  )


def _is_float_const(model, subgraph, tensor_idx) -> bool:
  tensor = subgraph.tensors[tensor_idx]
  return (
      tensor.type == qtyping.TensorType.FLOAT32
      and _tensor_data(model, subgraph, tensor_idx) is not None
  )


def _write_const(
    model, subgraph, tensor_idx, array: np.ndarray, patches, sg_idx
) -> None:
  data = np.ravel(np.ascontiguousarray(array, dtype=np.float32).view(np.uint8))
  model.buffers[subgraph.tensors[tensor_idx].buffer].data = data
  patches[(sg_idx, tensor_idx)] = data


def _tensor_name(subgraph, tensor_idx) -> str:
  return tfl_flatbuffer_utils.get_tensor_name(subgraph.tensors[tensor_idx])


# ============================================================================
# Site-level scale solver
# ============================================================================


def _floor_positive(mu2: np.ndarray) -> np.ndarray:
  mu2 = np.asarray(mu2, np.float64)
  return np.maximum(mu2, float(np.max(mu2)) * 1e-8 + _EPS)


def _site_objective(views, s: np.ndarray, mu2: np.ndarray, block_size: int):
  m = mu2 / (s * s)
  total = 0.0
  for matrix in views:
    a = np.abs(matrix) * s
    d = matrix.shape[1]
    g = block_size if (block_size and d % block_size == 0) else d
    for gi in range(d // g):
      cols = slice(gi * g, (gi + 1) * g)
      mx = a[:, cols].max(1)
      total += float((mx * mx).sum()) * float(m[cols].sum())
  return total


def _compute_site_scales(
    views, mu2: np.ndarray, block_size: int, iters: int = 3
) -> tuple[Optional[np.ndarray], float]:
  """Computes optimal per-input-channel scales s for a quantization site."""
  in_ch = mu2.size
  mu2 = _floor_positive(mu2)
  mu = np.sqrt(mu2)

  def normalized(v):
    v = v / np.exp(np.mean(np.log(v)))
    return np.clip(v, *_SCALE_CLAMP)

  a_base = sum((matrix * matrix).sum(0) for matrix in views) + _EPS

  identity_loss = _site_objective(views, np.ones(in_ch), mu2, block_size)
  s = normalized(np.sqrt(mu / np.sqrt(a_base)))
  best = (_site_objective(views, s, mu2, block_size), s)

  for _ in range(iters):
    a_eff = np.zeros(in_ch)
    for matrix in views:
      d = matrix.shape[1]
      g = block_size if (block_size and d % block_size == 0) else d
      ws_abs = np.abs(matrix) * s
      rows = np.arange(matrix.shape[0])
      for gi in range(d // g):
        j_star = gi * g + np.argmax(ws_abs[:, gi * g : (gi + 1) * g], 1)
        np.add.at(a_eff, j_star, matrix[rows, j_star] ** 2)
    a_eff = np.maximum(a_eff, 0.25 * a_base)
    s_cand = normalized(np.sqrt(mu / np.sqrt(a_eff)))
    s = normalized(np.sqrt(s * s_cand))
    loss = _site_objective(views, s, mu2, block_size)
    if loss < best[0]:
      best = (loss, s)

  if best[0] >= identity_loss:
    return None, 1.0
  return best[1], identity_loss / max(best[0], _EPS)


@dataclasses.dataclass
class _Site:
  subgraph_index: int
  tensor_idx: int
  consumer_op_indices: list[int]
  weight_tensor_indices: list[int]
  in_channels: int


def _find_sites(model, subgraph, subgraph_index, min_input_channels):
  """Discovers candidate quantization sites with supported weight tensors."""
  sites = {}
  for op_idx, op in enumerate(subgraph.operators):
    code = _builtin_code(model, op)
    if code not in _CONSUMER_OPS:
      continue
    weight_input_idx, in_ch_axis, weight_rank = _CONSUMER_OPS[code]
    if op.inputs is None or len(op.inputs) <= weight_input_idx:
      continue
    weight_idx = int(op.inputs[weight_input_idx])
    input_idx = int(op.inputs[0])
    if weight_idx < 0 or input_idx < 0:
      continue
    if not _is_float_const(model, subgraph, weight_idx):
      continue
    weight = _tensor_data(model, subgraph, weight_idx)
    if weight is None or weight.ndim != weight_rank:
      continue
    in_ch = weight.shape[in_ch_axis]
    if in_ch < min_input_channels:
      continue
    if subgraph.tensors[input_idx].type != qtyping.TensorType.FLOAT32:
      continue
    site = sites.setdefault(
        input_idx,
        _Site(subgraph_index, input_idx, [], [], in_ch),
    )
    if site.in_channels != in_ch:
      continue
    site.consumer_op_indices.append(op_idx)
    site.weight_tensor_indices.append(weight_idx)
  return sites


def _weight_views(model, subgraph, site):
  views = []
  for weight_idx in site.weight_tensor_indices:
    data = _tensor_data(model, subgraph, weight_idx)
    assert data is not None
    views.append(data.astype(np.float64))
  return views


def _scale_consumer_weights(
    model, subgraph, site, s: np.ndarray, patches
) -> None:
  for weight_idx in site.weight_tensor_indices:
    data = _tensor_data(model, subgraph, weight_idx)
    assert data is not None
    w = data.astype(np.float64)
    _write_const(
        model,
        subgraph,
        weight_idx,
        w * s[None, :],
        patches,
        site.subgraph_index,
    )


def condition_model(
    float_model: Union[str, bytes, bytearray],
    calibration_result: dict[str, qtyping.QSV],
    *,
    num_bits: int = 4,
    block_size: int = 0,
    iters: int = 3,
    min_input_channels: int = 4,
) -> tuple[bytearray, ConditioningReport]:
  """Applies OSCAR scale conditioning (data structures & scale calculation)."""
  del num_bits
  if isinstance(float_model, (bytes, bytearray)):
    model_bytes = bytearray(float_model)
  else:
    model_bytes = bytearray(tfl_flatbuffer_utils.get_model_content(float_model))
  model = tfl_flatbuffer_utils.read_model(model_bytes)
  report = ConditioningReport()
  patches: dict[tuple[int, int], np.ndarray] = {}

  for sg_idx, subgraph in enumerate(model.subgraphs):
    sites = _find_sites(model, subgraph, sg_idx, min_input_channels)

    for tensor_idx in sorted(sites):
      site = sites[tensor_idx]
      views = _weight_views(model, subgraph, site)
      site_result = SiteResult(
          subgraph_index=sg_idx,
          tensor_name=_tensor_name(subgraph, tensor_idx),
          consumer_names=[
              _tensor_name(subgraph, subgraph.operators[i].outputs[0])
              for i in site.consumer_op_indices
          ],
          num_params=int(sum(m.size for m in views)),
          status=SiteStatus.SKIPPED,
      )
      report.sites.append(site_result)

      qsv = calibration_result.get(site_result.tensor_name)
      mu2 = qsv.get("mu2") if qsv else None
      if mu2 is None:
        site_result.reason = "no mu2 in calibration result"
        continue
      mu2 = np.asarray(mu2, np.float64).ravel()
      if mu2.size != site.in_channels:
        site_result.reason = (
            f"mu2 has {mu2.size} channels, site expects {site.in_channels}"
        )
        continue

      s, ratio = _compute_site_scales(views, mu2, block_size, iters)
      if s is None:
        site_result.reason = "scaling not beneficial at this site"
        continue
      site_result.objective_ratio = ratio
      _scale_consumer_weights(model, subgraph, site, s, patches)
      site_result.status = SiteStatus.FOLDED

  packed_model = qtyping.Model.GetRootAs(model_bytes)
  for (sg_idx, tensor_idx), new_data in patches.items():
    packed_tensor = packed_model.Subgraphs(sg_idx).Tensors(tensor_idx)
    packed_buffer = packed_model.Buffers(packed_tensor.Buffer())
    view = packed_buffer.DataAsNumpy()
    if isinstance(view, np.ndarray) and view.size == new_data.size:
      view[:] = new_data
  report.in_place_patched = True
  return model_bytes, report


def verify_conditioned_model(
    original_model: Union[str, bytes, bytearray],
    conditioned_model: Union[str, bytes, bytearray],
    signature_data: dict[str, Iterable[dict[str, Any]]],
    rel_tolerance: float = 1e-3,
) -> float:
  """Checks the conditioned float model against the original float model."""

  def to_bytes(m):
    return m if isinstance(m, (bytes, bytearray)) else str(m)

  worst = 0.0
  original = tfl_interpreter_utils.create_tfl_interpreter(
      to_bytes(original_model)
  )
  conditioned = tfl_interpreter_utils.create_tfl_interpreter(
      to_bytes(conditioned_model)
  )
  for signature_key, samples in signature_data.items():
    for sample in samples:
      out_a = tfl_interpreter_utils.invoke_interpreter_signature(
          original, sample, signature_key
      )
      out_b = tfl_interpreter_utils.invoke_interpreter_signature(
          conditioned, sample, signature_key
      )
      for name, a in out_a.items():
        b = out_b[name]
        denom = max(float(np.abs(a).max()), _EPS)
        worst = max(worst, float(np.abs(a - b).max()) / denom)
  if worst > rel_tolerance:
    raise ConditioningError(
        f"conditioned model deviates from the original (max relative"
        f" deviation {worst:.3e} > {rel_tolerance:.1e})"
    )
  return worst
