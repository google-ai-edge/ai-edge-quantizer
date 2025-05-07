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

"""Test Hadamard rotation materialization."""

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import hadamard_rotation
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../../tests/models")
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


class HadamardRotationFullyConnectedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(888)
    self._test_model_path = os.path.join(
        _TEST_DATA_PREFIX_PATH, "conv_fc_mnist.tflite"
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._tensor_name_to_qsv = None
    self._subgraph = self._test_model.subgraphs[0]
    fc_subgraph_op_index = 3
    self._fc_op = self._subgraph.operators[fc_subgraph_op_index]
    self._fc_buffer_id = self._subgraph.tensors[self._fc_op.inputs[1]].buffer
    self._op_info = qtyping.OpInfo(
        op=self._fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=fc_subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=True,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
            ),
        ),
    )

  def test_materialize_fully_connected_basic(self):
    params = hadamard_rotation.materialize_fully_connected(
        self._op_info, self._graph_info, self._tensor_name_to_qsv
    )
    fc_input = params[0]
    weight = params[1]
    bias = params[2]
    output = params[3]

    self.assertLen(params, 4)
    self.assertIsNone(fc_input.producer)
    self.assertIsNotNone(fc_input.consumers)
    self.assertIsNone(weight.producer)
    self.assertIsNotNone(weight.consumers)
    self.assertIsNone(bias.producer)
    self.assertIsNotNone(bias.consumers)
    self.assertIsNotNone(output.producer)
    self.assertIsNone(output.consumers)
    self.assertEqual(
        fc_input.consumers[0].transformations,
        [qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION],
    )
    self.assertEqual(
        weight.consumers[0].transformations,
        [qtyping.QuantTransformation.QUANTIZE_TENSOR],
    )
    self.assertEqual(
        bias.consumers[0].transformations,
        [qtyping.QuantTransformation.NO_QUANTIZE],
    )
    if output.producer is not None:
      self.assertEqual(
          output.producer.transformations,
          [qtyping.QuantTransformation.NO_QUANTIZE],
      )

  def test_get_tensor_quant_params_basic(self):
    input_tensor = self._subgraph.tensors[self._fc_op.inputs[1]]
    buffer = self._graph_info.buffers[self._fc_buffer_id]
    np_buffer = np.frombuffer(buffer.data, dtype=np.float32).reshape(
        input_tensor.shape
    )
    qparams = hadamard_rotation.get_tensor_quant_params(
        self._op_info,
        self._op_info.op_quant_config.weight_tensor_config,
        np_buffer,
        self._tensor_name_to_qsv,
    )
    self.assertEqual(qparams.num_bits, 8)
    self.assertEqual(qparams.zero_point.all(), 0)
    self.assertEqual(qparams.symmetric, True)
    self.assertIsNotNone(qparams.quantized_data)
    self.assertEqual(qparams.block_size, 0)
    self.assertIsNotNone(qparams.hadamard)
    if qparams.hadamard is not None:
      self.assertEqual(qparams.hadamard.hadamard_size, 32)

  def test_raise_missing_tensor_content(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: "weight tensor" in str(err)
    ):
      hadamard_rotation.get_tensor_quant_params(
          self._op_info,
          self._op_info.op_quant_config.weight_tensor_config,
          None,
          self._tensor_name_to_qsv,
      )

  def test_raise_qsv_set(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: "static quantization" in str(err)
    ):
      hadamard_rotation.get_tensor_quant_params(
          self._op_info,
          self._op_info.op_quant_config.weight_tensor_config,
          self._graph_info.buffers[self._fc_buffer_id],
          self._graph_info.buffers[self._fc_buffer_id],
      )

  def test_raise_non_2d_constant(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: "2D tensors" in str(err)
    ):
      hadamard_rotation.get_tensor_quant_params(
          self._op_info,
          self._op_info.op_quant_config.weight_tensor_config,
          np.array([1.0, 2.0, 3.0]),
          self._tensor_name_to_qsv,
      )


class HadamardRotationEmbeddingLookupTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(888)
    self._test_model_path = os.path.join(
        _TEST_DATA_PREFIX_PATH, "embedding_lookup.tflite"
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._tensor_name_to_qsv = None

  def test_materialize_embedding_lookup_basic(self):
    subgraph = self._test_model.subgraphs[0]
    embedding_subgraph_op_index = 0
    embedding_op = subgraph.operators[embedding_subgraph_op_index]
    op_info = qtyping.OpInfo(
        op=embedding_op,
        op_name=_TFLOpName.EMBEDDING_LOOKUP,
        subgraph_op_index=embedding_subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=True,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
            ),
        ),
    )
    params = hadamard_rotation.materialize_embedding_lookup(
        op_info, self._graph_info, self._tensor_name_to_qsv
    )
    self.assertLen(params, 3)
    lookup = params[0]
    value = params[1]
    output = params[2]
    self.assertIsNone(lookup.producer)
    self.assertIsNotNone(lookup.consumers)
    self.assertIsNone(value.producer)
    self.assertIsNotNone(value.consumers)
    self.assertIsNotNone(output.producer)
    self.assertIsNone(output.consumers)
    self.assertEqual(
        lookup.consumers[0].transformations,
        [qtyping.QuantTransformation.NO_QUANTIZE],
    )
    self.assertEqual(
        value.consumers[0].transformations,
        [qtyping.QuantTransformation.QUANTIZE_TENSOR],
    )
    if output.producer is not None:
      self.assertEqual(
          output.producer.transformations,
          [qtyping.QuantTransformation.INSERT_HADAMARD_ROTATION],
      )


if __name__ == "__main__":
  googletest.main()
