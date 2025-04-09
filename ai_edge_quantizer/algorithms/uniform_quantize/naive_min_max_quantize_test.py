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
from typing import cast

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../../tests/models")
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


class NaiveMinMaxQuantizeTest(parameterized.TestCase):
  """Tests for general functions innaive min-max quantize algorithm.

  See naive_min_max_quantize_op_tests for op specific tests.
  """

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = os.path.join(
        _TEST_DATA_PREFIX_PATH, "conv_fc_mnist.tflite"
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._tensor_name_to_qsv = {}

  @parameterized.parameters(
      (qtyping.QuantGranularity.TENSORWISE),
      (qtyping.QuantGranularity.CHANNELWISE),
  )
  def test_init_qsvs(self, granularity):
    # Read from Model Explorer.
    subgraph0 = self._test_model.subgraphs[0]
    subgraph_op_index = 3
    fc_op = subgraph0.operators[subgraph_op_index]
    op_info = qtyping.OpInfo(
        op=fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                8,
                symmetric=True,
                granularity=granularity,
            ),
        ),
    )

    initial_qsvs = naive_min_max_quantize.init_qsvs(
        op_info,
        self._graph_info,
    )
    self.assertIn("sequential/flatten/Reshape", initial_qsvs)
    input_tensor_qsv = initial_qsvs["sequential/flatten/Reshape"]
    self.assertEmpty(input_tensor_qsv)
    self.assertIn(
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd",
        initial_qsvs,
    )
    output_tensor_qsv = initial_qsvs[
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    ]
    self.assertEmpty(output_tensor_qsv)

    self.assertIn("arith.constant1", initial_qsvs)
    weight_tensor_qsv = initial_qsvs["arith.constant1"]
    if granularity is qtyping.QuantGranularity.CHANNELWISE:
      mins_maxs_shape = (32, 1)
    else:
      mins_maxs_shape = (1, 1)
    self.assertTupleEqual(weight_tensor_qsv["min"].shape, mins_maxs_shape)
    self.assertTupleEqual(weight_tensor_qsv["max"].shape, mins_maxs_shape)

    self.assertIn("arith.constant2", initial_qsvs)
    bias_tensor_qsv = initial_qsvs["arith.constant2"]
    if granularity is qtyping.QuantGranularity.CHANNELWISE:
      mins_maxs_shape = (32,)
    else:
      mins_maxs_shape = (1,)
    self.assertTupleEqual(bias_tensor_qsv["min"].shape, mins_maxs_shape)
    self.assertTupleEqual(bias_tensor_qsv["max"].shape, mins_maxs_shape)

    initial_qsvs = naive_min_max_quantize.init_qsvs(
        op_info,
        self._graph_info,
        inputs_to_ignore=[0],
        outputs_to_ignore=[0],
    )
    self.assertNotIn("sequential/flatten/Reshape", initial_qsvs)
    self.assertNotIn(
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd",
        initial_qsvs,
    )

  def test_min_max_calibrate(self):
    # Sample input/output data for the fc op.
    tensor_content_map = {
        "sequential/flatten/Reshape": np.array([[1, 2, 3, 4, 5]]),
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd": np.array(
            [[6, 7, 8, 9, 10]]
        ),
    }
    # Read from Model Explorer.
    subgraph0 = self._test_model.subgraphs[0]
    fc_op = subgraph0.operators[3]
    # ignore 1(weight), and 2(bias) in inputs.
    op_qsvs = naive_min_max_quantize.min_max_calibrate(
        fc_op,
        self._graph_info,
        tensor_content_map,
        [1, 2],
        [],
    )
    self.assertIn("sequential/flatten/Reshape", op_qsvs)
    input_tensor_qsv = op_qsvs["sequential/flatten/Reshape"]
    self.assertTupleEqual(input_tensor_qsv["min"].shape, (1, 1))
    self.assertEqual(input_tensor_qsv["min"], np.array([[1]]))
    self.assertTupleEqual(input_tensor_qsv["max"].shape, (1, 1))
    self.assertEqual(input_tensor_qsv["max"], np.array([[5]]))
    self.assertIn(
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd",
        op_qsvs,
    )
    output_tensor_qsv = op_qsvs[
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    ]
    self.assertTupleEqual(output_tensor_qsv["min"].shape, (1, 1))
    self.assertEqual(output_tensor_qsv["min"], np.array([[6]]))
    self.assertTupleEqual(output_tensor_qsv["max"].shape, (1, 1))
    self.assertEqual(output_tensor_qsv["max"], np.array([[10]]))
    # weight and bias are excluded.
    self.assertNotIn("arith.constant1", op_qsvs)
    self.assertNotIn("arith.constant2", op_qsvs)

  def test_get_tensor_quant_params_for_blockwise_weight(self):
    subgraph0 = self._test_model.subgraphs[0]
    subgraph_op_index = 3
    fc_op = subgraph0.operators[subgraph_op_index]
    weight_tensor_config = _TensorQuantConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.BLOCKWISE,
        block_size=2,
    )
    op_info = qtyping.OpInfo(
        op=fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=weight_tensor_config,
        ),
    )
    test_data = np.array([[-7, 7], [4, -4], [4, -4], [7, 7]])
    quant_params = naive_min_max_quantize.get_tensor_quant_params(
        op_info=op_info,
        tensor_quant_config=weight_tensor_config,
        tensor_content=test_data,
    )
    scale = quant_params.scale
    zp = quant_params.zero_point
    expected_scale = np.array([
        [1],
        [0.5703125],
        [0.5703125],
        [1],
    ])
    expected_zp = np.zeros([4, 1])
    self.assertTrue(np.array_equal(zp, expected_zp))
    self.assertTrue(np.array_equal(scale, expected_scale))
    self.assertIsNotNone(quant_params.quantized_data)
    self.assertTupleEqual(
        cast(np.ndarray, quant_params.quantized_data).shape, test_data.shape
    )
    self.assertEqual(quant_params.block_size, 2)
    self.assertEqual(quant_params.quantized_dimension, 1)


if __name__ == "__main__":
  googletest.main()
