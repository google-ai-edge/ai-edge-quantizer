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

"""test for quantize tensor."""

import os
import numpy as np
from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import quantize_tensor
from ai_edge_quantizer.transformations import transformation_utils
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_litert import schema_py_generated  # pylint: disable=g-direct-tensorflow-import

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("..")


class QuantizeTensorTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self._orig_test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/insert_dequant_test.tflite"
    )
    self._model = tfl_flatbuffer_utils.read_model(self._orig_test_model_path)

  def test_quantize_constant_tensor(self):
    """test quantizing a constant tensor."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    data = np.ones([1, 112, 112, 3], dtype=np.int8)
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            7,
            model.operatorCodes,
            model.buffers,
            subgraph,
            -1,
            [4],
            qtyping.UniformQuantParams(
                8, None, np.ones(1), np.ones(1), True, data
            ),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    self.assertListEqual(
        np.array(model.buffers[8].data).tolist(), data.flatten().tolist()
    )
    quant_param = subgraph.tensors[7].quantization
    self.assertListEqual(np.array(quant_param.scale).tolist(), [1])
    self.assertEqual(np.array(quant_param.zeroPoint).tolist(), [1])
    self.assertEqual(quant_param.quantizedDimension, 0)

  def test_quantize_activation_tensor(self):
    """test quantizing an activation tensor."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            4,
            model.operatorCodes,
            model.buffers,
            subgraph,
            1,
            [3],
            qtyping.UniformQuantParams(
                8, None, np.array([22]), np.array([127])
            ),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    quant_param = subgraph.tensors[4].quantization
    self.assertListEqual(np.array(quant_param.scale).tolist(), [22])
    self.assertListEqual(np.array(quant_param.zeroPoint).tolist(), [127])
    self.assertEqual(quant_param.quantizedDimension, 0)

  def test_quantize_tensor_with_per_channel_quantization(self):
    """test quantizing an activation tensor."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            4,
            model.operatorCodes,
            model.buffers,
            subgraph,
            1,
            [3],
            qtyping.UniformQuantParams(8, 3, np.ones([22]), np.zeros([22])),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    quant_param = subgraph.tensors[4].quantization
    self.assertListEqual(
        np.array(quant_param.scale).tolist(), np.ones([22]).tolist()
    )
    self.assertListEqual(
        np.array(quant_param.zeroPoint).tolist(), np.zeros([22]).tolist()
    )
    self.assertEqual(quant_param.quantizedDimension, 3)

  def test_quantize_tensor_with_nonlinear_quantization(self):
    """test quantizing an activation tensor with non-linear quantization."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            4,
            model.operatorCodes,
            model.buffers,
            subgraph,
            1,
            [3],
            qtyping.NonLinearQuantParams(16, None),
        )
    )
    self.assertEqual(
        subgraph.tensors[4].type, schema_py_generated.TensorType.FLOAT16
    )

  def test_int4_constant_packed_correctly(self):
    subgraph = self._model.subgraphs[0]
    model = self._model
    data = np.array(
        [
            0x0,
            0x1,
            0x2,
            0x3,
            0x4,
            0x5,
            0x6,
            0x7,
            0x8,
            0x9,
            0xA,
            0xB,
            0xC,
            0xD,
            0xE,
        ],
        dtype=np.int8,
    )
    expected = np.array([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0x0E])
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=7,
            op_codes=model.operatorCodes,
            buffers=model.buffers,
            subgraph=subgraph,
            producer=-1,
            consumers=[4],
            quant_params=qtyping.UniformQuantParams(
                4, None, np.ones(1), np.ones(1), True, data
            ),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    np.testing.assert_array_equal(model.buffers[8].data, expected)
    quant_param = subgraph.tensors[7].quantization
    np.testing.assert_array_equal(quant_param.scale, [1])
    np.testing.assert_array_equal(quant_param.zeroPoint, [1])
    self.assertEqual(quant_param.quantizedDimension, 0)

  def test_int5_constant_not_packed(self):
    subgraph = self._model.subgraphs[0]
    model = self._model
    data = np.array(
        [
            0x0,
            0x1,
            0x2,
            0x3,
            0x4,
            0x5,
            0x6,
            0x7,
        ],
        dtype=np.int8,
    )
    expected = np.array([0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7])
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=7,
            op_codes=model.operatorCodes,
            buffers=model.buffers,
            subgraph=subgraph,
            producer=-1,
            consumers=[4],
            quant_params=qtyping.UniformQuantParams(
                5, None, np.ones(1), np.ones(1), True, data
            ),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    np.testing.assert_array_equal(model.buffers[8].data, expected)
    quant_param = subgraph.tensors[7].quantization
    np.testing.assert_array_equal(quant_param.scale, [1])
    np.testing.assert_array_equal(quant_param.zeroPoint, [1])
    self.assertEqual(quant_param.quantizedDimension, 0)


if __name__ == "__main__":
  googletest.main()
