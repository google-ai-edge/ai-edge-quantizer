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

"""Tests for emulated_subchannel."""

import os
import numpy as np
from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import emulated_subchannel
from ai_edge_quantizer.transformations import transformation_utils
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from tensorflow.lite.python import schema_py_generated  # pylint: disable=g-direct-tensorflow-import

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("..")


class EmulatedSubchannelTest(googletest.TestCase):
  """Tests for emulated_subchannel."""

  def setUp(self):
    super().setUp()
    self.params = qtyping.UniformQuantParams(
        num_bits=8,
        quantized_dimension=None,
        scale=np.ones([1, 1, 1, 4], dtype=np.float32),
        zero_point=np.zeros([1, 1, 1, 4], dtype=np.int64),
        symmetric=True,
        quantized_data=np.ones([1, 4, 2, 4], dtype=np.int8),
    )

  def test_emulate_subchannel_without_bias_succeeds(self):
    """Tests the emulated_subchannel function."""
    self._model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/single_fc_no_bias.tflite"
    )
    self._model = tfl_flatbuffer_utils.read_model(self._model_path)
    subgraph = self._model.subgraphs[0]
    model = self._model
    ret = emulated_subchannel.emulated_subchannel(
        transformation_utils.TransformationInput(
            tensor_id=1,
            op_codes=model.operatorCodes,
            buffers=model.buffers,
            subgraph=subgraph,
            producer=-1,
            consumers=[0],
            quant_params=self.params,
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 4)
    self.assertEqual(ret.output_tensor_id, 2)
    self.assertEqual(
        model.operatorCodes[subgraph.operators[0].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.RESHAPE,
    )
    self.assertEqual(
        model.operatorCodes[subgraph.operators[1].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.BATCH_MATMUL,
    )
    self.assertEqual(
        model.operatorCodes[subgraph.operators[2].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.MUL,
    )
    self.assertEqual(
        model.operatorCodes[subgraph.operators[3].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.SUM,
    )
    self.assertEqual(
        model.operatorCodes[subgraph.operators[4].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.RESHAPE,
    )
    self.assertEqual(
        subgraph.tensors[subgraph.operators[2].inputs[1]].name,
        b"arith.constant_scale",
    )
    self.assertListEqual(
        np.frombuffer(
            model.buffers[
                subgraph.tensors[subgraph.operators[2].inputs[1]].buffer
            ].data,
            dtype=np.float32,
        ).tolist(),
        np.ones([1, 1, 1, 4]).flatten().tolist(),
    )

  def test_emulate_subchannel_with_bias_succeeds(self):
    """Tests the emulated_subchannel function."""
    self._model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/single_fc_bias.tflite"
    )
    self._model = tfl_flatbuffer_utils.read_model(self._model_path)
    subgraph = self._model.subgraphs[0]
    model = self._model
    ret = emulated_subchannel.emulated_subchannel(
        transformation_utils.TransformationInput(
            tensor_id=1,
            op_codes=model.operatorCodes,
            buffers=model.buffers,
            subgraph=subgraph,
            producer=-1,
            consumers=[0],
            quant_params=self.params,
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 5)
    self.assertEqual(ret.output_tensor_id, 3)
    self.assertEqual(
        model.operatorCodes[subgraph.operators[0].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.RESHAPE,
    )
    self.assertEqual(
        model.operatorCodes[subgraph.operators[1].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.BATCH_MATMUL,
    )
    self.assertEqual(
        model.operatorCodes[subgraph.operators[2].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.MUL,
    )
    self.assertEqual(
        model.operatorCodes[subgraph.operators[3].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.SUM,
    )
    self.assertEqual(
        model.operatorCodes[subgraph.operators[4].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.RESHAPE,
    )
    self.assertEqual(
        model.operatorCodes[subgraph.operators[5].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.ADD,
    )
    self.assertEqual(
        subgraph.tensors[subgraph.operators[2].inputs[1]].name,
        b"arith.constant_scale",
    )
    self.assertListEqual(
        np.frombuffer(
            model.buffers[
                subgraph.tensors[subgraph.operators[2].inputs[1]].buffer
            ].data,
            dtype=np.float32,
        ).tolist(),
        np.ones([1, 1, 1, 4]).flatten().tolist(),
    )

  def test_emulated_subchannel_with_fused_relu_succeeds(self):
    """Tests the emulated_subchannel function with fused relu."""
    self._model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/single_fc_bias_relu.tflite"
    )
    self._model = tfl_flatbuffer_utils.read_model(self._model_path)
    self._model = tfl_flatbuffer_utils.read_model(self._model_path)
    subgraph = self._model.subgraphs[0]
    model = self._model
    ret = emulated_subchannel.emulated_subchannel(
        transformation_utils.TransformationInput(
            tensor_id=1,
            op_codes=model.operatorCodes,
            buffers=model.buffers,
            subgraph=subgraph,
            producer=-1,
            consumers=[0],
            quant_params=self.params,
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 6)
    self.assertEqual(ret.output_tensor_id, 3)
    self.assertEqual(
        model.operatorCodes[subgraph.operators[6].opcodeIndex].builtinCode,
        schema_py_generated.BuiltinOperator.RELU,
    )

  def test_emulated_subchannel_raises_when_unsupported_activation(self):
    """Tests the emulated_subchannel function with unsupported activation."""
    self._model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/single_fc_bias_relu6.tflite"
    )
    self._model = tfl_flatbuffer_utils.read_model(self._model_path)
    subgraph = self._model.subgraphs[0]
    model = self._model
    with self.assertRaises(ValueError):
      emulated_subchannel.emulated_subchannel(
          transformation_utils.TransformationInput(
              tensor_id=1,
              op_codes=model.operatorCodes,
              buffers=model.buffers,
              subgraph=subgraph,
              producer=-1,
              consumers=[0],
              quant_params=self.params,
          )
      )


if __name__ == "__main__":
  googletest.main()
