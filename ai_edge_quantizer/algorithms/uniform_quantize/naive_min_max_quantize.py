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

"""Performs naive min/max uniform quantization."""

from collections.abc import Sequence
from typing import Any, Optional
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from ai_edge_quantizer.algorithms.utils import common_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

ALGORITHM_KEY = "min_max_uniform_quantize"
_TFLOpName = qtyping.TFLOperationName
_QuantTransformation = qtyping.QuantTransformation
_IntType = uniform_quantize_tensor.IntType


def _init_tensor_min_max(
    tensor_data: Optional[np.ndarray],
    op_info: qtyping.OpInfo,
) -> qtyping.QSV:
  """Initialize the min/max for a tensor."""
  if tensor_data is None:
    return {}
  else:
    weight_tensor_config = op_info.op_quant_config.weight_tensor_config
    quantized_dim = None
    if weight_tensor_config is not None and (
        weight_tensor_config.granularity == qtyping.QuantGranularity.CHANNELWISE
        or weight_tensor_config.granularity
        == qtyping.QuantGranularity.BLOCKWISE
    ):
      quantized_dim = common_utils.get_weight_quantized_dim(
          op_info, tensor_data
      )
    if (
        weight_tensor_config is not None
        and weight_tensor_config.granularity
        == qtyping.QuantGranularity.BLOCKWISE
    ):
      reshaped_data, reduce_dims = _reshape_data_for_blockwise(
          tensor_data,
          quantized_dim,
          weight_tensor_config.block_size,
      )
      return {
          "min": np.min(reshaped_data, axis=reduce_dims, keepdims=False),
          "max": np.max(reshaped_data, axis=reduce_dims, keepdims=False),
      }

    else:
      reduce_dims = common_utils.get_reduce_dims(
          quantized_dim, tensor_data.shape
      )
      return {
          "min": np.min(tensor_data, axis=reduce_dims, keepdims=True),
          "max": np.max(tensor_data, axis=reduce_dims, keepdims=True),
      }


def _get_tensor_shape_for_blockwise(
    tensor_shape: Sequence[int], quantized_dim: int, block_size: int
) -> list[int]:
  """Get the tensor shape for blockwise quantization.

  This function splits the quantize dimension of the tensor into blocks and the
  dim/blocks. Hence, min/max of the tensor can be calculated for each block
  using existing functions.

  Args:
    tensor_shape: The original shape of the tensor.
    quantized_dim: The dimension to be quantized blockwise.
    block_size: The size of the block.

  Returns:
    The new tensor shape for calculating scale and zp for blockwise
    quantization.
  """
  new_shape = []
  for index, val in enumerate(tensor_shape):
    if index == quantized_dim:
      new_shape.append(int(val / block_size))
      new_shape.append(block_size)
    else:
      new_shape.append(val)
  return new_shape


def _reshape_data_for_blockwise(
    tensor_data: np.ndarray, quantized_dim: int, block_size: int
) -> tuple[np.ndarray, int]:
  """Reshapes data for blockwise quantization.

  Args:
    tensor_data: The original tensor data.
    quantized_dim: The dimension to be quantized blockwise.
    block_size: The size of the block.

  Returns:
    A tuple containing the reshaped tensor data and the new reduce dimension.
  """
  new_shape = _get_tensor_shape_for_blockwise(
      tensor_data.shape, quantized_dim, block_size
  )
  reshaped_data = tensor_data.reshape(new_shape)
  return reshaped_data, quantized_dim + 1


def _broadcast_scale_zp_for_blockwise(
    tensor_content: np.ndarray,
    quant_params: qtyping.UniformQuantParams,
) -> qtyping.UniformQuantParams:
  """Broadcasts scale and zp for blockwise quantization.

  Args:
    tensor_content: The original tensor data.
    quant_params: The quantization parameters.

  Returns:
    The updated quantization parameters with broadcasted scale and zp for
    correct constant quantization.
  """
  if quant_params.quantized_dimension is None:
    raise ValueError("Quantized dimension must be specified.")
  if quant_params.block_size is None or quant_params.block_size <= 0:
    raise ValueError("Block size must be specified and positive.")
  quantized_dim = quant_params.quantized_dimension
  expanded_tensor_shape = _get_tensor_shape_for_blockwise(
      tensor_content.shape, quantized_dim, quant_params.block_size
  )
  expanded_scale = np.reshape(
      np.broadcast_to(
          np.expand_dims(quant_params.scale, quantized_dim + 1),
          expanded_tensor_shape,
      ),
      tensor_content.shape,
  )
  expanded_zp = np.reshape(
      np.broadcast_to(
          np.expand_dims(quant_params.zero_point, quantized_dim + 1),
          expanded_tensor_shape,
      ),
      tensor_content.shape,
  )
  return qtyping.UniformQuantParams(
      scale=expanded_scale,
      zero_point=expanded_zp,
      num_bits=quant_params.num_bits,
      symmetric=quant_params.symmetric,
      quantized_dimension=quantized_dim,
      block_size=quant_params.block_size,
  )


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: Optional[np.ndarray] = None,
    tensor_qsv: Optional[dict[str, Any]] = None,
) -> qtyping.UniformQuantParams:
  """Get the quantization parameters for a tensor.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor.
    tensor_qsv: A dictionary containingthe min/max of the tensor.

  Returns:
    The quantization parameters for the tensor.
  """
  # Get quant params.
  if tensor_qsv is None:
    if tensor_content is not None:
      # We need min/max to calculate quantization parameters, which
      # should be collected during the calibration process. However,
      # weight-only and DRQ do not require calibration, thus it is
      # possible that this information is missing here. In that case we
      # collect min/max on the spot.
      tensor_min_max = _init_tensor_min_max(
          tensor_content,
          op_info,
      )
    else:
      raise ValueError(
          f"{op_info.op_name}(index: {op_info.subgraph_op_index}) not found in"
          " tensor_name_to_qsv. Check if the correct calibration results are"
          " passed into the ParamsGenerator."
      )
  else:
    tensor_min_max = tensor_qsv

  if "min" not in tensor_min_max or "max" not in tensor_min_max:
    raise ValueError(
        "min and max must be provided to produce tensor quantization"
        " parameters. Check if the correct calibration results are passed into"
        " the ParamsGenerator."
    )
  zp, scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
      tensor_min_max["min"],
      tensor_min_max["max"],
      tensor_quant_config.num_bits,
      tensor_quant_config.symmetric,
  )
  quantized_dim = None
  if (
      tensor_quant_config.granularity == qtyping.QuantGranularity.CHANNELWISE
      or tensor_quant_config.granularity == qtyping.QuantGranularity.BLOCKWISE
  ):
    quantized_dim = common_utils.get_weight_quantized_dim(
        op_info, tensor_content
    )
  quant_params = qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=quantized_dim,
      block_size=tensor_quant_config.block_size,
  )
  if tensor_content is None:
    return quant_params

  # The reshaping for blockwise quantization is unique hence we do this here
  # to avoid unexpected broadcast behavior downstream.
  if tensor_quant_config.granularity == qtyping.QuantGranularity.BLOCKWISE:
    quant_params = _broadcast_scale_zp_for_blockwise(
        tensor_content, quant_params
    )

  quantized_vars = uniform_quantize_tensor.uniform_quantize(
      tensor_content, quant_params
  )
  # Update with quantized values.
  return qtyping.UniformQuantParams(
      scale=scale,
      zero_point=zp,
      num_bits=tensor_quant_config.num_bits,
      symmetric=tensor_quant_config.symmetric,
      quantized_dimension=quantized_dim,
      quantized_data=quantized_vars,
      block_size=tensor_quant_config.block_size,
  )


# TODO: b/333731147 - Use named tuple to store min/max.
def init_qsvs(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    inputs_to_ignore: Optional[list[int]] = None,
    outputs_to_ignore: Optional[list[int]] = None,
) -> qtyping.QSV:
  """Initialize the QSVs.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    inputs_to_ignore: Operand indices to ignore.
    outputs_to_ignore: Result indices to ignore.

  Returns:
    QSVs.
  """
  op_qsvs = {}

  inputs_to_ignore = inputs_to_ignore or []
  outputs_to_ignore = outputs_to_ignore or []
  for opr_idx, tensor_idx in enumerate(op_info.op.inputs):
    if tensor_idx != -1 and opr_idx not in inputs_to_ignore:
      tensor = graph_info.subgraph_tensors[tensor_idx]
      tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
      tensor_data = tfl_flatbuffer_utils.get_tensor_data(
          tensor, graph_info.buffers
      )
      op_qsvs[tensor_name] = _init_tensor_min_max(
          tensor_data,
          op_info,
      )
  for res_idx, tensor_idx in enumerate(op_info.op.outputs):
    if tensor_idx != -1 and res_idx not in outputs_to_ignore:
      tensor = graph_info.subgraph_tensors[tensor_idx]
      tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
      tensor_data = tfl_flatbuffer_utils.get_tensor_data(
          tensor, graph_info.buffers
      )
      op_qsvs[tensor_name] = _init_tensor_min_max(
          tensor_data,
          op_info,
      )
  return op_qsvs


def min_max_calibrate(
    tfl_op: Any,
    graph_info: qtyping.GraphInfo,
    tensor_content_map: dict[str, np.ndarray],
    inputs_to_ignore: Optional[list[int]] = None,
    outputs_to_ignore: Optional[list[int]] = None,
) -> dict[str, qtyping.QSV]:
  """Collect quantization statistics variable (QSV, e.g., min/max) for the op.

  Args:
    tfl_op: The tfl operation.
    graph_info: Graph information needed to perform quantization for the op.
    tensor_content_map: A map of tensor name to tensor content.
    inputs_to_ignore: Input tensor indices to ignore.
    outputs_to_ignore: Output tensor indices to ignore.

  Returns:
    A dictionary with key as tensor name and value as the collected QSV.
  """
  op_qsvs = {}

  def _collect_activation_tensor_min_max(tensor_idx):
    tensor = graph_info.subgraph_tensors[tensor_idx]
    tensor_data = tfl_flatbuffer_utils.get_tensor_data(
        tensor, graph_info.buffers
    )
    # Skip constant tensors.
    if tensor_data is not None:
      return
    tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
    tensor_content = tensor_content_map[tensor_name]
    op_qsvs[tensor_name] = {
        "min": np.min(tensor_content, axis=None, keepdims=True),
        "max": np.max(tensor_content, axis=None, keepdims=True),
    }

  inputs_to_ignore = inputs_to_ignore or []
  outputs_to_ignore = outputs_to_ignore or []
  for i, tensor_idx in enumerate(tfl_op.inputs):
    if tensor_idx != -1 and i not in inputs_to_ignore:
      _collect_activation_tensor_min_max(tensor_idx)
  for i, tensor_idx in enumerate(tfl_op.outputs):
    if tensor_idx != -1 and i not in outputs_to_ignore:
      _collect_activation_tensor_min_max(tensor_idx)

  return op_qsvs
