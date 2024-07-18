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

"""Test utils for naive min max quantize."""

import dataclasses
from typing import Any

from absl.testing import parameterized
import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

_TFLOpName = qtyping.TFLOperationName
_OpExecutionMode = qtyping.OpExecutionMode
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation


@dataclasses.dataclass
class OpTestInfo:
  """Aggregate op test information.

  Attributes:
    test_model: The test model.
    op_tensor_names: A map of op tensor names with keys for input(s) and output.
      Note for multiple inputs, the keys will follow the pattern of "input",
      "input2", "input3" etc.
    input_range: A tuple of input min and max.
    output_range: A tuple of output min and max.
    quantized_dimension: Quantized dimension.
  """

  test_model: Any
  op_tensor_names: dict[str, str]
  input_range: tuple[np.ndarray, np.ndarray]
  output_range: tuple[np.ndarray, np.ndarray]
  quantized_dimension: int = 0


DEFAULT_ACTIVATION_QUANT_SETTING = _TensorQuantConfig(
    num_bits=8,
    symmetric=False,
    channel_wise=False,
)
DEFAULT_WEIGHT_QUANT_SETTING = _TensorQuantConfig(
    num_bits=8,
    symmetric=True,
    channel_wise=True,
)


class NaiveMinMaxQuantizeTest(parameterized.TestCase):
  """Base test class for naive min max quantize.

  This class provides test utils for naive min max quantize.
  """

  def setUp(self):
    super().setUp()
    self._tensor_name_to_qsv = {}

  def _setup_op_test_config(
      self,
      execution_mode,
      op_test_info,
      num_inputs=1,
  ):
    """Helper to set up qsv for op test."""
    # SRQ requires QSVs (min/max).
    input_min, input_max = op_test_info.input_range
    output_min, output_max = op_test_info.output_range
    if execution_mode == _OpExecutionMode.SRQ:
      input_qsv = {
          "min": input_min,
          "max": input_max,
      }
      output_qsv = {
          "min": output_min,
          "max": output_max,
      }
      self._tensor_name_to_qsv = {
          op_test_info.op_tensor_names["output"]: output_qsv,
      }
      for i in range(num_inputs):
        input_name = "input"
        if i > 0:
          input_name = f"input{i+1}"
        self._tensor_name_to_qsv[op_test_info.op_tensor_names[input_name]] = (
            input_qsv
        )

  def _test_single_input_output_ops(
      self,
      op_info,
      graph_info,
      op_test_info,
      materialization_func,
      same_input_output_params=False,
  ):
    """Tests ops with single input and output.

    Args:
      op_info: OpInfo object.
      graph_info: GraphInfo object.
      op_test_info: OpTestInfo object.
      materialization_func: Function to materialize tensor transformation
        parameters.
      same_input_output_params: Whether the input and output tensor
        transformation parameters are the same.
    """
    self._setup_op_test_config(
        execution_mode=_OpExecutionMode.SRQ,
        op_test_info=op_test_info,
    )

    tensor_quant_params = materialization_func(
        op_info, graph_info, self._tensor_name_to_qsv
    )
    self.assertLen(tensor_quant_params, 2)

    # Test input tensor settings.
    op_quant_config = op_info.op_quant_config
    self._test_tensor_transformation_params(
        op_test_info.op_tensor_names["input"],
        op_info.subgraph_op_index,
        is_inbounding_tensor=True,
        tensor_quant_config=op_quant_config.activation_tensor_config,
        transformation_params=tensor_quant_params[0],
        desired_transformations=[_QuantTransformation.ADD_QUANTIZE],
    )
    # Test output tensor settings
    self._test_tensor_transformation_params(
        op_test_info.op_tensor_names["output"],
        op_info.subgraph_op_index,
        is_inbounding_tensor=False,
        tensor_quant_config=op_quant_config.activation_tensor_config,
        transformation_params=tensor_quant_params[1],
        desired_transformations=[_QuantTransformation.ADD_DEQUANTIZE],
    )
    if same_input_output_params:
      input_tensor_quant_params = tensor_quant_params[0].consumers[0].parameters  # pytype: disable=attribute-error
      output_tensor_quant_params = tensor_quant_params[1].producer.parameters  # pytype: disable=attribute-error
      self.assertEqual(input_tensor_quant_params, output_tensor_quant_params)

  def _test_two_input_one_output_ops(
      self,
      op_info,
      graph_info,
      op_test_info,
      materialization_func,
  ):
    """Tests ops with two inputs and single output.

    Can be used for ops such as ADD, MUL, SUB.

    Args:
      op_info: OpInfo object.
      graph_info: GraphInfo object.
      op_test_info: OpTestInfo object.
      materialization_func: Function to materialize tensor transformation
        parameters.
    """
    op_quant_config = op_info.op_quant_config
    self._setup_op_test_config(
        execution_mode=op_quant_config.execution_mode,
        op_test_info=op_test_info,
        num_inputs=2,
    )
    tensor_quant_params = materialization_func(
        op_info, graph_info, self._tensor_name_to_qsv
    )
    self.assertLen(tensor_quant_params, 3)

    # Test input tensor settings.
    transformations = [_QuantTransformation.NO_QUANTIZE]
    if op_quant_config.execution_mode == _OpExecutionMode.SRQ:
      transformations = [_QuantTransformation.ADD_QUANTIZE]
    self._test_tensor_transformation_params(
        op_test_info.op_tensor_names["input"],
        op_info.subgraph_op_index,
        is_inbounding_tensor=True,
        tensor_quant_config=op_quant_config.activation_tensor_config,
        transformation_params=tensor_quant_params[0],
        desired_transformations=transformations,
    )
    self._test_tensor_transformation_params(
        op_test_info.op_tensor_names["input2"],
        op_info.subgraph_op_index,
        is_inbounding_tensor=True,
        tensor_quant_config=op_quant_config.activation_tensor_config,
        transformation_params=tensor_quant_params[1],
        desired_transformations=transformations,
    )
    # Test output tensor settings
    transformations = [_QuantTransformation.NO_QUANTIZE]
    if op_quant_config.execution_mode == _OpExecutionMode.SRQ:
      transformations = [_QuantTransformation.ADD_DEQUANTIZE]
    self._test_tensor_transformation_params(
        op_test_info.op_tensor_names["output"],
        op_info.subgraph_op_index,
        is_inbounding_tensor=False,
        tensor_quant_config=op_quant_config.activation_tensor_config,
        transformation_params=tensor_quant_params[2],
        desired_transformations=transformations,
    )

  def _test_fc_bmm_conv(
      self,
      op_info,
      graph_info,
      op_test_info,
      materialization_func,
  ):
    """Tests fully connected, batch matmul and conv ops.

    Args:
      op_info: OpInfo object.
      graph_info: GraphInfo object.
      op_test_info: OpTestInfo object.
      materialization_func: Function to materialize tensor transformation
        parameters.
    """
    op_quant_config = op_info.op_quant_config
    self._setup_op_test_config(
        execution_mode=op_quant_config.execution_mode,
        op_test_info=op_test_info,
    )
    tensor_quant_params = materialization_func(
        op_info, graph_info, self._tensor_name_to_qsv
    )

    _, weight_tensor, bias_tensor, _ = (
        tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
            op_info.op,
            graph_info.subgraph_tensors,
        )
    )
    num_configs = 4 if bias_tensor is not None else 3
    self.assertLen(tensor_quant_params, num_configs)
    bias_tensor_data = None
    if bias_tensor is not None:
      bias_tensor_data = tfl_flatbuffer_utils.get_tensor_data(
          bias_tensor,
          op_test_info.test_model.buffers,
      )

    # Test input tensor settings
    transformations = [_QuantTransformation.NO_QUANTIZE]
    if op_quant_config.execution_mode == _OpExecutionMode.SRQ:
      transformations = [_QuantTransformation.ADD_QUANTIZE]
    self._test_tensor_transformation_params(
        op_test_info.op_tensor_names["input"],
        op_info.subgraph_op_index,
        is_inbounding_tensor=True,
        tensor_quant_config=op_quant_config.activation_tensor_config,
        transformation_params=tensor_quant_params[0],
        desired_transformations=transformations,
    )

    # Test weight tensor settings.
    transformations = [_QuantTransformation.QUANTIZE_TENSOR]
    if op_quant_config.execution_mode == _OpExecutionMode.WEIGHT_ONLY:
      transformations = [
          _QuantTransformation.ADD_DEQUANTIZE,
      ]
    weight_tensor_data = tfl_flatbuffer_utils.get_tensor_data(
        weight_tensor,
        op_test_info.test_model.buffers,
    )
    self._test_tensor_transformation_params(
        op_test_info.op_tensor_names["weight"],
        op_info.subgraph_op_index,
        is_inbounding_tensor=True,
        tensor_quant_config=op_quant_config.weight_tensor_config,
        transformation_params=tensor_quant_params[1],
        desired_transformations=transformations,
        tensor_data=weight_tensor_data,
        quantized_dimension=op_test_info.quantized_dimension,
    )

    # Test output tensor settings.
    transformations = [_QuantTransformation.NO_QUANTIZE]
    if op_quant_config.execution_mode == _OpExecutionMode.SRQ:
      transformations = [_QuantTransformation.ADD_DEQUANTIZE]
    self._test_tensor_transformation_params(
        op_test_info.op_tensor_names["output"],
        op_info.subgraph_op_index,
        is_inbounding_tensor=False,
        tensor_quant_config=op_quant_config.activation_tensor_config,
        transformation_params=tensor_quant_params[2],
        desired_transformations=transformations,
    )

    # Test bias tensor settings.
    if bias_tensor is not None:
      bias_bit_width = 32
      transformations = [_QuantTransformation.NO_QUANTIZE]
      if op_quant_config.execution_mode == _OpExecutionMode.SRQ:
        transformations = [_QuantTransformation.QUANTIZE_TENSOR]
        if op_quant_config.activation_tensor_config.num_bits == 16:  # pytype: disable=attribute-error
          bias_bit_width = 64
      bias_config = qtyping.TensorQuantizationConfig(
          num_bits=bias_bit_width,
          symmetric=True,
          channel_wise=op_quant_config.weight_tensor_config.channel_wise,
      )
      self._test_tensor_transformation_params(
          op_test_info.op_tensor_names["bias"],
          op_info.subgraph_op_index,
          is_inbounding_tensor=True,
          tensor_quant_config=bias_config,
          transformation_params=tensor_quant_params[3],
          desired_transformations=transformations,
          tensor_data=bias_tensor_data,
      )

  def _test_tensor_transformation_params(
      self,
      tensor_name,
      subgraph_op_id,
      is_inbounding_tensor,
      tensor_quant_config,
      transformation_params,
      desired_transformations,
      tensor_data=None,
      expected_tensor_max=None,
      quantized_dimension=0,
  ):
    """Tests tensor transformation parameters.

    Args:
      tensor_name: Tensor name.
      subgraph_op_id: Subgraph op id.
      is_inbounding_tensor: Whether the tensor is an inbounding tensor.
      tensor_quant_config: Tensor quantization config.
      transformation_params: Tensor transformation parameters.
      desired_transformations: Desired transformations.
      tensor_data: Tensor data.
      expected_tensor_max: Expected tensor max.
      quantized_dimension: Quantized dimension.
    """
    self.assertEqual(transformation_params.tensor_name, tensor_name)
    if is_inbounding_tensor:
      self.assertIsNone(transformation_params.producer)
      self.assertLen(transformation_params.consumers, 1)
      op_params = transformation_params.consumers[0]
    else:
      self.assertIsNone(transformation_params.consumers)
      op_params = transformation_params.producer
      self.assertIsNotNone(op_params)
    self.assertEqual(op_params.subgraph_op_id, subgraph_op_id)
    self.assertSequenceEqual(op_params.transformations, desired_transformations)
    quantization_params = op_params.parameters
    if desired_transformations == [_QuantTransformation.NO_QUANTIZE]:
      self.assertIsNone(quantization_params)
    else:
      self.assertIsNotNone(quantization_params)
      if tensor_quant_config.channel_wise:
        self.assertEqual(
            quantization_params.quantized_dimension, quantized_dimension
        )
      else:
        self.assertIsNone(quantization_params.quantized_dimension)
      self.assertEqual(
          quantization_params.num_bits, tensor_quant_config.num_bits
      )
      # Test correctness of quantization parameters
      if tensor_data is not None:
        expected_quantized_data = uniform_quantize_tensor.uniform_quantize(
            tensor_data, quantization_params
        )
        self.assertSequenceEqual(
            list(expected_quantized_data.flatten()),
            list(quantization_params.quantized_data.flatten()),  # pytype: disable=attribute-error
        )
      elif expected_tensor_max:
        max_q = 2**tensor_quant_config.num_bits / 2 - 1
        calculated_tensor_max = quantization_params.scale[0] * (
            max_q - quantization_params.zero_point[0]
        )
        self.assertAlmostEqual(
            calculated_tensor_max, expected_tensor_max, delta=5e-2
        )
