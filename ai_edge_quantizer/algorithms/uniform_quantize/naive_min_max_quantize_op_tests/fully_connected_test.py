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
_ComputePrecision = qtyping.ComputePrecision
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation
_OpTestInfo = naive_min_max_test_utils.OpTestInfo

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile(
    "../../../tests/models"
)
_DEFAULT_ACTIVATION_QUANT_SETTING = (
    naive_min_max_test_utils.DEFAULT_ACTIVATION_QUANT_SETTING
)


class FullyConnectedTest(naive_min_max_test_utils.NaiveMinMaxQuantizeTest):

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = os.path.join(
        _TEST_DATA_PREFIX_PATH, "conv_fc_mnist.tflite"
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
    self._set_op_tensor_names()

  def _set_op_tensor_names(self):
    op_tensor_names = {}
    op_tensor_names["weight"] = "arith.constant1"
    op_tensor_names["bias"] = "arith.constant2"
    op_tensor_names["input"] = "sequential/flatten/Reshape"
    op_tensor_names["output"] = (
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    )
    self._op_test_info.op_tensor_names = op_tensor_names

  # TODO(rewu): add int16 tests.
  @parameterized.product(
      num_bits_weight=(4, 8),
      symmetric_weight=(True, False),
      granularity=(
          qtyping.QuantGranularity.CHANNELWISE,
          qtyping.QuantGranularity.TENSORWISE,
      ),
      test_case=(
          # Tuple holds compute precision and whether to use srq and explicit
          # dequantize.
          (_ComputePrecision.FLOAT, False, True),
          (_ComputePrecision.INTEGER, False, False),
          (_ComputePrecision.INTEGER, True, False),
      ),
  )
  def test_materialize_fully_connected_succeeds(
      self,
      num_bits_weight,
      symmetric_weight,
      granularity,
      test_case,
  ):
    compute_precision, is_srq, explicit_dequantize = test_case

    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 3
    fc_op = subgraph0.operators[subgraph_op_id]
    activation_tensor_config = None
    # Check if SRQ.
    if compute_precision == _ComputePrecision.INTEGER and is_srq:
      activation_tensor_config = _DEFAULT_ACTIVATION_QUANT_SETTING
    op_info = qtyping.OpInfo(
        op=fc_op,
        op_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=num_bits_weight,
                symmetric=symmetric_weight,
                granularity=granularity,
            ),
            compute_precision=compute_precision,
            explicit_dequantize=explicit_dequantize,
        ),
    )

    self._test_fc_bmm_conv(
        op_info,
        self._graph_info,
        self._op_test_info,
        naive_min_max_quantize.materialize_fc_conv,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="weights_are_not_quantized",
          min_weight_elements=1000000,
          expect_weights_quantized=False,
      ),
      dict(
          testcase_name="weights_are_quantized_for_min_weight_elements_0",
          min_weight_elements=0,
          expect_weights_quantized=True,
      ),
      dict(
          testcase_name="weights_are_quantized_for_min_weight_elements_1",
          min_weight_elements=1,
          expect_weights_quantized=True,
      ),
  )
  def test_materialize_fully_connected_quantizes_weights_larger_than_min_weight_elements_for_w8_afp32(
      self, min_weight_elements, expect_weights_quantized
  ):
    self._test_materialize_fn_quantizes_weights_larger_than_min_weight_elements_for_w8_afp32(
        op_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        subgraph_op_id=3,
        min_weight_elements=min_weight_elements,
        graph_info=self._graph_info,
        op_test_info=self._op_test_info,
        materialization_func=naive_min_max_quantize.materialize_fc_conv,
        expect_weights_quantized=expect_weights_quantized,
    )


if __name__ == "__main__":
  googletest.main()
