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

"""Implements the Hadamard Rotation quantization."""

from typing import Any, Optional
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import octav
from ai_edge_quantizer.algorithms.utils import common_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils


ALGORITHM_KEY = "HADAMARD_ROTATION"


def _make_hadamard_matrix(size: int) -> np.ndarray:
  """Generates a Hadamard matrix of the given size.

  Args:
    size: The size of the Hadamard matrix. Should be a power of 2. This
      represents a single dimension. E.g. if size is 4, then the Hadamard matrix
      is a 4x4 matrix.

  Returns:
    The Hadamard matrix.
  """
  h = h2 = np.array([[1, 1], [1, -1]])
  current_size = 2
  while current_size < size:
    h = np.kron(h, h2)
    current_size *= 2
  return h / np.sqrt(size)


def _rotate_with_diagonal_hadamard(
    tensor_content: np.ndarray,
    axis: int,
):
  """Quantizes the given float array using the diagonal Hadamard algorithm.

  Args:
    tensor_content: The float array to quantize.
    axis: The axis of the tensor to quantize.

  Returns:
    A tuple containing the quantized array and the recovered array.

  Raises:
    ValueError: If the axis is not 1. To support other axes, please add
      support to the matrix multiplication.
  """
  if axis != 1:
    raise ValueError(
        "Hadamard rotation is only supported for 2D tensors with quantized"
        " dimension 0."
    )

  # Use the largest power of 2 that is a factor of the dimension and then
  # tile this Hadamard matrix along the diagonal. 1024^3 is just a large power
  # of 2 to calculate this factor.
  hadamard_size = np.gcd(tensor_content.shape[axis], 1024 * 1024 * 1024)
  diagonal_size = tensor_content.shape[axis] // hadamard_size
  random_vector = np.ones(hadamard_size, dtype=np.int8)

  # Use a canonical Hadamard matrix.
  hadamard = _make_hadamard_matrix(hadamard_size)
  hadamard_diagonal = np.kron(np.eye(diagonal_size), hadamard)
  w_rotated = np.einsum("ij,aj->ai", hadamard_diagonal, tensor_content)
  return w_rotated, hadamard_size, random_vector


def get_tensor_quant_params(
    op_info: qtyping.OpInfo,
    tensor_quant_config: qtyping.TensorQuantizationConfig,
    tensor_content: Optional[np.ndarray] = None,
    tensor_qsv: Optional[dict[str, Any]] = None,
) -> qtyping.UniformQuantParams:
  """Returns the quantization parameters for a tensor.

  This function will rotate the tensor with a Hadamard matrix and then
  quantize it with OCTAV.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    tensor_quant_config: The quantization config for the tensor.
    tensor_content: The content of the tensor. When None, it means the tensor is
      not a weight tensor (e.g. static quantization).
    tensor_qsv: A dictionary containing the min/max of the tensor.

  Raises:
    ValueError: If the blockwise quantization is requested.
    ValueError: If the asymmetric quantization is requested.
    ValueError: `tensor_qsv` must contain min/max values, or `tensor_content`
      must be provided so that they can be inferred.
  """
  if tensor_content is None:
    raise ValueError("Hadamard rotation is only supported for weight tensors.")

  if tensor_qsv is not None:
    raise ValueError(
        "Hadamard rotation is not supported for static quantization."
    )

  if tensor_content.ndim != 2:
    raise ValueError("Hadamard rotation is only supported for 2D tensors.")

  if tensor_quant_config.granularity != qtyping.QuantGranularity.CHANNELWISE:
    raise ValueError(
        "Hadamard rotation is not supported for"
        f" {tensor_quant_config.granularity} granularity."
    )

  quantized_dim = common_utils.get_weight_quantized_dim(op_info, tensor_content)
  if quantized_dim != 0:
    raise ValueError(
        f"Unsupported quantized dimension: {quantized_dim}. Only 0 is"
        " supported."
    )

  # Reduction axis is the non-quantized dimension. Since we only support 2D
  # tensors and quantized_dim of 0, the reduction axis is 1.
  reduce_axis = 1

  # Rotate the tensor with a Hadamard matrix.
  w_rotated, hadamard_size, random_vector = _rotate_with_diagonal_hadamard(
      tensor_content, axis=reduce_axis
  )

  # Get the quantized values of the rotated tensor.
  qparams = octav.get_tensor_quant_params(
      op_info, tensor_quant_config, w_rotated, tensor_qsv
  )

  return qtyping.UniformQuantParams(
      quantized_dimension=qparams.quantized_dimension,
      num_bits=qparams.num_bits,
      scale=qparams.scale,
      zero_point=qparams.zero_point,
      symmetric=qparams.symmetric,
      quantized_data=qparams.quantized_data,
      block_size=qparams.block_size,
      hadamard=qtyping.UniformQuantParams.HadamardRotationParams(
          random_binary_vector=random_vector,
          hadamard_size=hadamard_size,
      ),
  )


def materialize_fully_connected(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,  # pylint: disable=unused-argument
) -> list[qtyping.TensorTransformationParams]:
  """Materialize the fully_connected op.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  op_tensor_params = []

  # Materialize weight.
  weight_tensor_index = 1
  weight_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[weight_tensor_index]
  ]
  tensor_data = tfl_flatbuffer_utils.get_tensor_data(
      weight_tensor, graph_info.buffers
  )
  # quant_params contains the rotated and quantized weights done by
  # get_tensor_quant_params().
  quant_params = get_tensor_quant_params(
      op_info,
      op_info.op_quant_config.weight_tensor_config,
      tensor_data,
      None,
  )
  transformations = [qtyping.QuantTransformation.QUANTIZE_TENSOR]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  weight_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(weight_tensor),
      consumers=[op2tensor_params],
  )

  # Materialize input. A hadamard rotation op should be inserted on the input
  # tensor to do the inverse of the weight's transformation.
  input_tensor_index = 0
  input_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[input_tensor_index]
  ]
  transformations = [
      qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION,
  ]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  input_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(input_tensor),
      consumers=[op2tensor_params],
  )
  op_tensor_params.append(input_transformation_params)
  op_tensor_params.append(weight_transformation_params)

  # Materialize bias. Since static quantization is not supported, we do not
  # quantize the bias tensor.
  bias_tensor_index = 2
  bias_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[bias_tensor_index]
  ]
  no_quant_tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
  )
  bias_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(bias_tensor),
      consumers=[no_quant_tensor_params],
  )
  op_tensor_params.append(bias_transformation_params)

  # Materialize output. Since static quantization is not supported, we do not
  # quantize the output tensor.
  output_tensor_index = 0
  output_tensor = graph_info.subgraph_tensors[
      op_info.op.outputs[output_tensor_index]
  ]
  no_quant_tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
  )
  output_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(output_tensor),
      producer=no_quant_tensor_params,
  )
  op_tensor_params.append(output_transformation_params)

  return op_tensor_params


def materialize_embedding_lookup(
    op_info: qtyping.OpInfo,
    graph_info: qtyping.GraphInfo,
    tensor_name_to_qsv: Optional[dict[str, Any]] = None,  # pylint: disable=unused-argument
) -> list[qtyping.TensorTransformationParams]:
  """Materialize the embedding_lookup op.

  Args:
    op_info: Aggregated information about the op (e.g., quantization config).
    graph_info: Graph information needed to perform quantization for the op.
    tensor_name_to_qsv: A map of tensor name to quantization parameters.

  Returns:
    Quantization configuration for the tensors associated with the op (e.g.,
    weights, bias).
  """
  op_tensor_params = []

  # Materialize lookup.
  lookup_tensor_index = 0
  lookup_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[lookup_tensor_index]
  ]
  transformations = [
      qtyping.QuantTransformation.NO_QUANTIZE,
  ]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=None,
      transformations=transformations,
  )
  lookup_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(lookup_tensor),
      consumers=[op2tensor_params],
  )
  op_tensor_params.append(lookup_transformation_params)

  # Materialize embedding. The embedding table should be rotated and then
  # quantized.
  embedding_tensor_index = 1
  embedding_tensor = graph_info.subgraph_tensors[
      op_info.op.inputs[embedding_tensor_index]
  ]
  tensor_data = tfl_flatbuffer_utils.get_tensor_data(
      embedding_tensor, graph_info.buffers
  )
  quant_params = get_tensor_quant_params(
      op_info,
      op_info.op_quant_config.weight_tensor_config,
      tensor_data,
      None,
  )
  transformations = [qtyping.QuantTransformation.QUANTIZE_TENSOR]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  weight_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(embedding_tensor),
      consumers=[op2tensor_params],
  )
  op_tensor_params.append(weight_transformation_params)

  # Materialize output. A hadamard rotation op should be inserted on the output
  # tensor to do the inverse of the embedding's transformation.
  output_tensor_index = 0
  output_tensor = graph_info.subgraph_tensors[
      op_info.op.outputs[output_tensor_index]
  ]
  transformations = [
      qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION,
  ]
  op2tensor_params = qtyping.OpToTensorParams(
      subgraph_op_id=op_info.subgraph_op_index,
      parameters=quant_params,
      transformations=transformations,
  )
  output_transformation_params = qtyping.TensorTransformationParams(
      tensor_name=tfl_flatbuffer_utils.get_tensor_name(output_tensor),
      producer=op2tensor_params,
  )
  op_tensor_params.append(output_transformation_params)

  return op_tensor_params
