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

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile(
    "../../../tests/models"
)
_TFLOpName = qtyping.TFLOperationName
_ComputePrecision = qtyping.ComputePrecision
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation
_OpTestInfo = naive_min_max_test_utils.OpTestInfo
_DEFAULT_ACTIVATION_QUANT_SETTING = (
    naive_min_max_test_utils.DEFAULT_ACTIVATION_QUANT_SETTING
)


class BatchMatmulTest(naive_min_max_test_utils.NaiveMinMaxQuantizeTest):
  """Tests bmm op where both inputs are non-constant tensors."""

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = os.path.join(_TEST_DATA_PREFIX_PATH, "bmm.tflite")
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
      ("activation_int8_asymmetric_srq", 8, False),
      ("activation_int16_symmetric", 16, True),
  )
  def test_batch_matmul_adjy_false_srq_succeeds(
      self,
      num_bits_activation,
      symmetric_activation,
  ):
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["input"] = "input1"
    op_tensor_names["input2"] = "input2"
    op_tensor_names["output"] = (
        "BatchMatMulV3;jax2tf_export_func_/PartitionedCall/BatchMatMulV3"
    )
    self._op_test_info.op_tensor_names = op_tensor_names
    self._op_test_info.quantized_dimension = 2

    activation_tensor_config = _TensorQuantConfig(
        num_bits=num_bits_activation,
        symmetric=symmetric_activation,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )

    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.BATCH_MATMUL,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            compute_precision=_ComputePrecision.INTEGER,  # SRQ.
        ),
    )
    self._test_no_weights_op(
        op_info,
        self._graph_info,
        self._op_test_info,
        naive_min_max_quantize.materialize_batch_matmul,
    )

  @parameterized.named_parameters(
      ("activation_int8_asymmetric", 8, False),
      ("activation_int16_symmetric", 16, True),
  )
  def test_batch_matmul_adjy_true_srq_succeeds(
      self,
      num_bits_activation,
      symmetric_activation,
  ):
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 1
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["input"] = (
        "BatchMatMulV3;jax2tf_export_func_/PartitionedCall/BatchMatMulV3"
    )
    op_tensor_names["input2"] = "input2"
    op_tensor_names["output"] = "Identity_1"
    self._op_test_info.op_tensor_names = op_tensor_names
    self._op_test_info.quantized_dimension = 1

    activation_tensor_config = _TensorQuantConfig(
        num_bits=num_bits_activation,
        symmetric=symmetric_activation,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.BATCH_MATMUL,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            compute_precision=_ComputePrecision.INTEGER,  # SRQ.
        ),
    )
    self._test_no_weights_op(
        op_info,
        self._graph_info,
        self._op_test_info,
        naive_min_max_quantize.materialize_batch_matmul,
    )


class BatchMatmulConstantInputTest(
    naive_min_max_test_utils.NaiveMinMaxQuantizeTest
):
  """Tests bmm op where one the inputs is a constant tensor."""

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = os.path.join(
        _TEST_DATA_PREFIX_PATH, "bmm_constant_input.tflite"
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

  @parameterized.product(
      num_bits_weight=(4, 8),
      symmetric_weight=(True, False),
      test_case=(
          # Tuple holds compute precision, whether to use srq and whether
          # to use explicit dequantize.
          (_ComputePrecision.INTEGER, True, False),
          (_ComputePrecision.INTEGER, False, False),
          (_ComputePrecision.FLOAT, False, True),
      ),
  )
  def test_batch_matmul_adjy_false_succeeds(
      self,
      num_bits_weight,
      symmetric_weight,
      test_case,
  ):
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["weight"] = "arith.constant1"
    op_tensor_names["input"] = "inputs"
    op_tensor_names["output"] = (
        "BatchMatMulV3;jax2tf_export_func_/PartitionedCall/BatchMatMulV3"
    )
    compute_precision, is_srq, explicit_dequantize = test_case
    self._op_test_info.op_tensor_names = op_tensor_names
    self._op_test_info.quantized_dimension = 2

    activation_tensor_config = None
    # Check if SRQ.
    if compute_precision == _ComputePrecision.INTEGER and is_srq:
      activation_tensor_config = _DEFAULT_ACTIVATION_QUANT_SETTING

    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.BATCH_MATMUL,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=num_bits_weight,
                symmetric=symmetric_weight,
                granularity=qtyping.QuantGranularity.TENSORWISE,
            ),
            compute_precision=compute_precision,
            explicit_dequantize=explicit_dequantize,
        ),
    )
    self._test_fc_bmm_conv(
        op_info,
        self._graph_info,
        self._op_test_info,
        naive_min_max_quantize.materialize_batch_matmul,
    )

  @parameterized.product(
      num_bits_weight=(4, 8),
      symmetric_weight=(True, False),
      test_case=(
          # Tuple holds compute precision, whether to use srq and whether
          # to use explicit dequantize.
          (_ComputePrecision.INTEGER, True, False),
          (_ComputePrecision.INTEGER, False, False),
          (_ComputePrecision.FLOAT, False, True),
      ),
  )
  def test_batch_matmul_adjy_true_succeeds(
      self,
      num_bits_weight,
      symmetric_weight,
      test_case,
  ):
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 1
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["weight"] = "arith.constant"
    op_tensor_names["input"] = (
        "BatchMatMulV3;jax2tf_export_func_/PartitionedCall/BatchMatMulV3"
    )
    op_tensor_names["output"] = "Identity_1"
    compute_precision, is_srq, explicit_dequantize = test_case
    self._op_test_info.op_tensor_names = op_tensor_names
    self._op_test_info.quantized_dimension = 1

    activation_tensor_config = None
    # Check if SRQ.
    if compute_precision == _ComputePrecision.INTEGER and is_srq:
      activation_tensor_config = _DEFAULT_ACTIVATION_QUANT_SETTING

    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.BATCH_MATMUL,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=num_bits_weight,
                symmetric=symmetric_weight,
                granularity=qtyping.QuantGranularity.TENSORWISE,
            ),
            compute_precision=compute_precision,
            explicit_dequantize=explicit_dequantize,
        ),
    )
    self._test_fc_bmm_conv(
        op_info,
        self._graph_info,
        self._op_test_info,
        naive_min_max_quantize.materialize_batch_matmul,
    )


if __name__ == "__main__":
  googletest.main()
