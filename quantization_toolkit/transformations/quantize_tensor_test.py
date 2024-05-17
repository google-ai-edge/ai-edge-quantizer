"""test for quantize tensor."""

import os
import numpy as np
from tensorflow.python.platform import googletest
from quantization_toolkit import qtyping
from quantization_toolkit.transformations import quantize_tensor
from quantization_toolkit.transformations import transformation_utils
from quantization_toolkit.utils import test_utils
from quantization_toolkit.utils import tfl_flatbuffer_utils
from tensorflow.lite.python import schema_py_generated  # pylint: disable=g-direct-tensorflow-import

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("..")


class QuantizeTensorTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self._orig_test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "test_models/insert_dequant_test.tflite"
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


if __name__ == "__main__":
  googletest.main()
