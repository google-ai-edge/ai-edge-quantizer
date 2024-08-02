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

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from ai_edge_quantizer.algorithms.uniform_quantize.naive_min_max_quantize_op_tests import test_utils as naive_min_max_test_utils
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

_TFLOpName = qtyping.TFLOperationName
_OpExecutionMode = qtyping.OpExecutionMode
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation
_OpTestInfo = naive_min_max_test_utils.OpTestInfo

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile(
    "../../../tests/models"
)


class SubTest(naive_min_max_test_utils.NaiveMinMaxQuantizeTest):

  def _custom_setup(self, test_model_file: str):
    np.random.seed(666)
    self._test_model_path = os.path.join(
        _TEST_DATA_PREFIX_PATH, test_model_file
    )
    self._op_test_info = _OpTestInfo(
        test_model=tfl_flatbuffer_utils.read_model(self._test_model_path),
        op_tensor_names={},
        input_range=(np.array([[-10]]), np.array([[8]])),
        output_range=(np.array([[10]]), np.array([[88]])),
    )
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._op_test_info.test_model.subgraphs[0].tensors,
        buffers=self._op_test_info.test_model.buffers,
    )

  @parameterized.named_parameters(
      ("int8_nonsymmetric", 8, False),
      ("int16_symmetric", 16, True),
  )
  def test_materialize_srq_sub_succeeds(
      self,
      activation_num_bits: int,
      activation_symmetric: bool,
  ):
    self._custom_setup("single_sub.tflite")
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["input"] = "serving_default_input_1:0"
    op_tensor_names["input2"] = "serving_default_input_2:0"
    op_tensor_names["output"] = "PartitionedCall:0"
    self._op_test_info.op_tensor_names = op_tensor_names

    activation_tensor_config = _TensorQuantConfig(
        num_bits=activation_num_bits,
        symmetric=activation_symmetric,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.SUB,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=activation_tensor_config,
            execution_mode=_OpExecutionMode.SRQ,
        ),
    )
    self._test_two_input_one_output_ops(
        op_info,
        self._graph_info,
        self._op_test_info,
        naive_min_max_quantize.materialize_sub,
    )

  @parameterized.named_parameters(
      ("int8_nonsymmetric", 8, False),
      ("int16_symmetric", 16, True),
  )
  def test_materialize_srq_sub1_constant_input_succeeds(
      self,
      activation_num_bits: int,
      activation_symmetric: bool,
  ):
    """Tests the case where one of the SUB inputs is a constant tensor."""
    self._custom_setup("single_sub1_constant_input.tflite")
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["input"] = "serving_default_input_1:0"
    op_tensor_names["weight"] = "model/subtract/ExpandDims"
    op_tensor_names["output"] = "PartitionedCall:0"
    self._op_test_info.op_tensor_names = op_tensor_names

    activation_tensor_config = _TensorQuantConfig(
        num_bits=activation_num_bits,
        symmetric=activation_symmetric,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.SUB,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=activation_tensor_config,
            execution_mode=_OpExecutionMode.SRQ,
        ),
    )
    # We re-use the fc_bmm_conv helper test function here because the constant
    # tensor is treated as a weight tensor.
    self._test_fc_bmm_conv(
        op_info,
        self._graph_info,
        self._op_test_info,
        naive_min_max_quantize.materialize_sub,
    )


if __name__ == "__main__":
  googletest.main()
