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

_PASS_THROUGH_OPS = frozenset([
    _OP.RESHAPE,
    _OP.SQUEEZE,
    _OP.EXPAND_DIMS,
])

_CONSUMER_OPS = {
    _OP.FULLY_CONNECTED: (1, 1, 2),
}


class ConditioningError(Exception):
  """A rewrite would change model semantics; the pass aborts."""


class _FoldError(Exception):
  """No exact fold path exists for a site (site-local, non-fatal)."""


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


def _build_maps(subgraph):
  """producer[tensor_idx] = op_idx; readers[tensor_idx] = [op_idx, ...]."""
  producer, readers = {}, {}
  for op_idx, op in enumerate(subgraph.operators):
    for t in op.outputs or []:
      if t >= 0:
        producer[int(t)] = op_idx
    for t in op.inputs or []:
      if t >= 0:
        readers.setdefault(int(t), []).append(op_idx)
  return producer, readers


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


# ============================================================================
# Buffer claims & Recursive Upstream Fold Planner
# ============================================================================

_COMPOSABLE_KINDS = (frozenset({"consumer_weights", "out_channels"}),)


def _can_claim(claimed_kinds: set[str], kind: str) -> bool:
  if kind in claimed_kinds:
    return False
  return all(
      frozenset({kind, existing}) in _COMPOSABLE_KINDS
      for existing in claimed_kinds
  )


def _add_claim(claimed: dict[int, set[str]], buffer_idx: int, kind: str):
  claimed.setdefault(buffer_idx, set()).add(kind)


def _check_sole_reader(readers, subgraph, tensor_idx, allowed_op_idx):
  extra = [r for r in readers.get(tensor_idx, []) if r != allowed_op_idx]
  if extra:
    raise _FoldError(
        f"tensor {tensor_idx} has other readers (ops {extra}); the scale"
        " would leak into them"
    )
  if tensor_idx in (subgraph.outputs or []):
    raise _FoldError(f"tensor {tensor_idx} is a subgraph output")


def _last_dim(subgraph, tensor_idx) -> int:
  shape = subgraph.tensors[tensor_idx].shape
  if shape is None or len(shape) == 0:
    return -1
  return int(shape[-1])


def _plan_fold(
    model,
    subgraph,
    producer,
    readers,
    tensor_idx: int,
    length: int,
    edits: dict[int, tuple[Any, ...]],
    claimed: dict[int, set[str]],
) -> None:
  """Recursively plans upstream folding edits for a given site tensor."""
  if tensor_idx not in producer:
    raise _FoldError("site input is a graph input or a constant")
  op_idx = producer[tensor_idx]
  op = subgraph.operators[op_idx]
  code = _builtin_code(model, op)

  def claim(buffer_idx, spec):
    if buffer_idx in edits or not _can_claim(
        claimed.get(buffer_idx, set()), spec[0]
    ):
      raise _FoldError(
          f"buffer {buffer_idx} already claimed by an incompatible edit"
      )
    edits[buffer_idx] = spec

  if code == _OP.FULLY_CONNECTED:
    weight_idx = int(op.inputs[1]) if len(op.inputs) > 1 else -1
    if weight_idx < 0 or not _is_float_const(model, subgraph, weight_idx):
      raise _FoldError("producer weight is not a float constant")
    weight = _tensor_data(model, subgraph, weight_idx)
    if weight is None or weight.ndim != 2 or weight.shape[0] != length:
      ch = weight.shape[0] if weight is not None and weight.ndim > 0 else -1
      raise _FoldError(
          f"producer output channels {ch} != segment length {length}"
      )
    bias_idx = (
        int(op.inputs[2]) if len(op.inputs) > 2 and op.inputs[2] >= 0 else -1
    )
    if bias_idx >= 0 and not _is_float_const(model, subgraph, bias_idx):
      raise _FoldError("producer bias is not a float constant")
    claim(
        subgraph.tensors[weight_idx].buffer,
        ("out_channels", weight_idx, bias_idx),
    )
    return

  if code == _OP.MUL:
    a, b = int(op.inputs[0]), int(op.inputs[1])
    for const_t, _ in ((a, b), (b, a)):
      arr = (
          _tensor_data(model, subgraph, const_t)
          if _is_float_const(model, subgraph, const_t)
          else None
      )
      if (
          arr is not None
          and arr.size == length
          and (arr.ndim == 1 or arr.shape[-1] == length)
      ):
        claim(subgraph.tensors[const_t].buffer, ("const", const_t))
        return
    last_err = None
    for branch in (a, b):
      if _is_float_const(model, subgraph, branch):
        continue
      snapshot = dict(edits)
      try:
        _check_sole_reader(readers, subgraph, branch, op_idx)
        if _last_dim(subgraph, branch) != length:
          raise _FoldError("MUL branch does not carry the channel axis")
        _plan_fold(
            model,
            subgraph,
            producer,
            readers,
            branch,
            length,
            edits,
            claimed,
        )
        return
      except _FoldError as e:
        edits.clear()
        edits.update(snapshot)
        last_err = e
    raise _FoldError(f"MUL: no foldable branch ({last_err})")

  if code in (_OP.ADD, _OP.SUB):
    a, b = int(op.inputs[0]), int(op.inputs[1])
    const_t = a if _is_float_const(model, subgraph, a) else None
    if const_t is None:
      const_t = b if _is_float_const(model, subgraph, b) else None
    if const_t is None:
      raise _FoldError("ADD/SUB of two dynamic tensors (residual join)")
    dyn_t = b if const_t == a else a
    arr = _tensor_data(model, subgraph, const_t)
    if (
        arr is not None
        and arr.size == length
        and (arr.ndim == 1 or arr.shape[-1] == length)
    ):
      claim(subgraph.tensors[const_t].buffer, ("const", const_t))
    elif arr is not None and arr.size == 1 and float(np.abs(arr).max()) == 0.0:
      pass
    else:
      raise _FoldError("ADD/SUB constant is not per-channel or zero")
    _check_sole_reader(readers, subgraph, dyn_t, op_idx)
    if _last_dim(subgraph, dyn_t) != length:
      raise _FoldError("ADD/SUB operand does not carry the channel axis")
    _plan_fold(
        model,
        subgraph,
        producer,
        readers,
        dyn_t,
        length,
        edits,
        claimed,
    )
    return

  if code in _PASS_THROUGH_OPS:
    src = int(op.inputs[0])
    if (
        _last_dim(subgraph, src) != length
        or _last_dim(subgraph, tensor_idx) != length
    ):
      raise _FoldError("pass-through op does not preserve the channel axis")
    _check_sole_reader(readers, subgraph, src, op_idx)
    _plan_fold(
        model,
        subgraph,
        producer,
        readers,
        src,
        length,
        edits,
        claimed,
    )
    return

  raise _FoldError(f"unfoldable producer op (builtin code {code})")


def _apply_fold_edits(
    model, subgraph, edits, inv_s: np.ndarray, patches, sg_idx
) -> None:
  """Applies planned upstream fold edits by rescaling constant buffers."""
  for spec in edits.values():
    if spec[0] == "const":
      _, tensor_idx = spec
      data = _tensor_data(model, subgraph, tensor_idx)
      assert data is not None
      arr = data.astype(np.float64)
      _write_const(
          model,
          subgraph,
          tensor_idx,
          arr * inv_s.reshape((1,) * (arr.ndim - 1) + (inv_s.size,)),
          patches,
          sg_idx,
      )
    else:
      _, weight_idx, bias_idx = spec
      w_data = _tensor_data(model, subgraph, weight_idx)
      assert w_data is not None
      weight = w_data.astype(np.float64)
      _write_const(
          model,
          subgraph,
          weight_idx,
          weight * inv_s[:, None],
          patches,
          sg_idx,
      )
      if bias_idx >= 0:
        bias = _tensor_data(model, subgraph, bias_idx)
        if bias is not None:
          _write_const(
              model,
              subgraph,
              bias_idx,
              bias.astype(np.float64) * inv_s,
              patches,
              sg_idx,
          )


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


def _check_cross_subgraph_consistency(
    model, claimed: dict[int, set[str]]
) -> None:
  """Verifies shared buffers are not used inconsistently across subgraphs."""
  allowed = {
      "out_channels": (_OP.FULLY_CONNECTED,),
      "out_channels_bias": (_OP.FULLY_CONNECTED,),
      "const": (_OP.MUL, _OP.ADD, _OP.SUB),
      "consumer_weights": (_OP.FULLY_CONNECTED,),
  }
  problems = []
  for sg_idx, subgraph in enumerate(model.subgraphs):
    for op in subgraph.operators:
      code = _builtin_code(model, op)
      for input_t in op.inputs or []:
        if input_t < 0:
          continue
        buffer_idx = subgraph.tensors[input_t].buffer
        for kind in claimed.get(buffer_idx, ()):
          if code not in allowed[kind]:
            problems.append(
                (sg_idx, _tensor_name(subgraph, input_t), kind, code)
            )
  if problems:
    raise ConditioningError(
        "conditioned buffers are used with a different op structure in"
        f" other subgraphs: {problems[:5]}"
    )


def _check_site_readers(readers, subgraph, site) -> None:
  """Ensures site tensor is read only by consumer ops and not graph outputs."""
  extra = [
      r
      for r in readers.get(site.tensor_idx, [])
      if r not in site.consumer_op_indices
  ]
  if extra:
    raise _FoldError(
        f"site tensor has non-consumer readers (ops {extra}); the scale"
        " would leak into them"
    )
  if site.tensor_idx in (subgraph.outputs or []):
    raise _FoldError("site tensor is a subgraph output")


def condition_model(
    float_model: Union[str, bytes, bytearray],
    calibration_result: dict[str, qtyping.QSV],
    *,
    num_bits: int = 4,
    block_size: int = 0,
    iters: int = 3,
    min_input_channels: int = 4,
) -> tuple[bytearray, ConditioningReport]:
  """Applies OSCAR scale conditioning with recursive upstream folding."""
  del num_bits
  if isinstance(float_model, (bytes, bytearray)):
    model_bytes = bytearray(float_model)
  else:
    model_bytes = bytearray(tfl_flatbuffer_utils.get_model_content(float_model))
  model = tfl_flatbuffer_utils.read_model(model_bytes)
  report = ConditioningReport()
  claimed: dict[int, set[str]] = {}
  patches: dict[tuple[int, int], np.ndarray] = {}
  shared_sites: dict[frozenset[int], tuple[SiteStatus, Any, Any]] = {}

  for sg_idx, subgraph in enumerate(model.subgraphs):
    producer, readers = _build_maps(subgraph)
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

      weight_buffers = frozenset([
          subgraph.tensors[w].buffer for w in site.weight_tensor_indices
      ])
      if weight_buffers in shared_sites:
        status, _, _ = shared_sites[weight_buffers]
        if status == SiteStatus.FOLDED:
          site_result.status = SiteStatus.FOLDED
          site_result.reason = (
              "shared weight buffer: upstream fold already applied globally"
          )
        else:
          site_result.reason = "shared weight buffer: upstream was skipped"
        continue
      elif any(
          not _can_claim(claimed.get(b, set()), "consumer_weights")
          for b in weight_buffers
      ):
        site_result.reason = (
            "weights shared with an already-conditioned site"
        )
        continue

      tensor_name = site_result.tensor_name
      qsv = calibration_result.get(tensor_name)
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

      inv_s = (1.0 / s).astype(np.float64)
      edits: dict[int, tuple[Any, ...]] = {}
      fold_error = None
      try:
        _check_site_readers(readers, subgraph, site)
      except _FoldError as e:
        fold_error = e
      else:
        try:
          _plan_fold(
              model,
              subgraph,
              producer,
              readers,
              tensor_idx,
              site.in_channels,
              edits,
              claimed,
          )
        except _FoldError as e:
          fold_error = e

      if fold_error is not None:
        site_result.reason = f"fold failed: {fold_error}"

      if fold_error is None:
        _scale_consumer_weights(model, subgraph, site, s, patches)
        _apply_fold_edits(model, subgraph, edits, inv_s, patches, sg_idx)
        for buffer_idx, spec in edits.items():
          _add_claim(claimed, buffer_idx, spec[0])
          if spec[0] == "out_channels" and spec[2] >= 0:
            _add_claim(
                claimed, subgraph.tensors[spec[2]].buffer, "out_channels_bias"
            )
        for buffer_idx in weight_buffers:
          _add_claim(claimed, buffer_idx, "consumer_weights")
        shared_sites[weight_buffers] = (SiteStatus.FOLDED, inv_s, s)
        site_result.status = SiteStatus.FOLDED
      else:
        shared_sites[weight_buffers] = (SiteStatus.SKIPPED, inv_s, s)
        site_result.reason = f"fold failed: {fold_error}"

  _check_cross_subgraph_consistency(model, claimed)

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
