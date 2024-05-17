"""Tests for tfl_flatbuffer_utils.py."""

import os
import numpy as np
from tensorflow.python.platform import googletest
from quantization_toolkit.utils import test_utils
from quantization_toolkit.utils import tfl_flatbuffer_utils


TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../test_models")


# TODO: b/328830092 - Add test cases for model require buffer offset.
class FlatbufferUtilsTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "conv_fc_mnist.tflite"
    )

    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)

  def test_get_model_buffer(self):
    model_buffer = tfl_flatbuffer_utils.get_model_buffer(self._test_model_path)
    file_stats = os.stat(self._test_model_path)
    self.assertLen(model_buffer, file_stats.st_size)

  def test_parse_op_tensors(self):
    subgraph0 = self._test_model.subgraphs[0]
    conv2d_op = subgraph0.operators[0]
    op_tensors = tfl_flatbuffer_utils.parse_op_tensors(
        conv2d_op, subgraph0.tensors
    )
    # conv2d have three inputs and one output
    self.assertLen(op_tensors, 4)

    average_pool_op = subgraph0.operators[1]
    op_tensors = tfl_flatbuffer_utils.parse_op_tensors(
        average_pool_op, subgraph0.tensors
    )
    # averagepool have one input and one output
    self.assertLen(op_tensors, 2)

  def test_parse_fc_bmm_conv_tensors(self):
    subgraph0 = self._test_model.subgraphs[0]
    conv2d_op = subgraph0.operators[0]
    inputs, weight, bias, output = (
        tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
            conv2d_op, subgraph0.tensors
        )
    )
    self.assertEqual(tuple(inputs.shape), (1, 28, 28, 1))
    self.assertEqual(tuple(weight.shape), (8, 3, 3, 1))
    self.assertEqual(tuple(bias.shape), (8,))
    self.assertEqual(tuple(output.shape), (1, 28, 28, 8))

    fc_with_bias = subgraph0.operators[3]
    inputs, weight, bias, output = (
        tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
            fc_with_bias,
            subgraph0.tensors,
        )
    )
    self.assertEqual(tuple(inputs.shape), (1, 1568))
    self.assertEqual(tuple(weight.shape), (32, 1568))
    self.assertEqual(tuple(bias.shape), (32,))
    self.assertEqual(tuple(output.shape), (1, 32))

    fc_no_bias = subgraph0.operators[4]
    inputs, weight, bias, output = (
        tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
            fc_no_bias,
            subgraph0.tensors,
        )
    )
    self.assertEqual(tuple(inputs.shape), (1, 32))
    self.assertEqual(tuple(weight.shape), (10, 32))
    self.assertIsNone(bias)
    self.assertEqual(tuple(output.shape), (1, 10))

  def test_buffer_to_tensors(self):
    buffer_to_tensor_map = tfl_flatbuffer_utils.buffer_to_tensors(
        self._test_model
    )
    # Read from Netron/Model Explorer
    tensors = buffer_to_tensor_map[6]
    self.assertLen(tensors, 1)
    conv2d_filter_tensor = tensors[0]
    self.assertEqual(tuple(conv2d_filter_tensor.shape), (8, 3, 3, 1))

  def test_get_tensor_name(self):
    subgraph0 = self._test_model.subgraphs[0]
    subgraph_tensors = subgraph0.tensors
    conv2d_op = subgraph0.operators[0]
    weight_tensor = subgraph_tensors[conv2d_op.inputs[1]]
    weight_tensor_name = tfl_flatbuffer_utils.get_tensor_name(weight_tensor)
    self.assertEqual(weight_tensor_name, "sequential/conv2d/Conv2D")

  # TODO(b/325123193): test tensor with data outside of flatbuffer.
  def test_get_tensor_data(self):
    model_buffer = tfl_flatbuffer_utils.get_model_buffer(self._test_model_path)
    subgraph0 = self._test_model.subgraphs[0]
    subgraph_tensors = subgraph0.tensors
    conv2d_op = subgraph0.operators[0]
    # Check tensor with data
    weight_tensor = subgraph_tensors[conv2d_op.inputs[1]]
    weight_tensor_data = tfl_flatbuffer_utils.get_tensor_data(
        weight_tensor, self._test_model.buffers, model_buffer
    )
    self.assertEqual(
        tuple(weight_tensor.shape), tuple(weight_tensor_data.shape)  # pytype: disable=attribute-error
    )
    self.assertAlmostEqual(weight_tensor_data[0][0][0][0], -0.12941549718379974)

    # Check tensor with no data
    input_tensor = subgraph_tensors[conv2d_op.inputs[0]]
    input_tensor_data = tfl_flatbuffer_utils.get_tensor_data(
        input_tensor, self._test_model.buffers, model_buffer
    )
    self.assertIsNone(input_tensor_data)

  def test_has_same_quantization_succeeds(self):
    tensor0, tensor1 = self._test_model.subgraphs[0].tensors[:2]
    tensor0.quantization.scale = np.array([1, 2, 3]).astype(np.float32)
    tensor0.quantization.zeroPoint = np.array([3, 2, 1]).astype(np.int32)
    tensor1.quantization.scale = np.array([1, 2, 3]).astype(np.float32)
    tensor1.quantization.zeroPoint = np.array([3, 2, 1]).astype(np.int32)
    self.assertTrue(
        tfl_flatbuffer_utils.has_same_quantization(tensor0, tensor1)
    )

  def test_has_same_quantization_succeds_not_quantized(self):
    tensor0, tensor1 = self._test_model.subgraphs[0].tensors[:2]
    tensor0.type = 10
    self.assertTrue(
        tfl_flatbuffer_utils.has_same_quantization(tensor0, tensor1)
    )

  def test_has_same_quantization_fails_different_scale(self):
    tensor0, tensor1 = self._test_model.subgraphs[0].tensors[:2]
    tensor1.quantization.scale = np.array([1, 2, 3]).astype(np.float32)
    self.assertFalse(
        tfl_flatbuffer_utils.has_same_quantization(tensor0, tensor1)
    )

  def test_has_same_quantization_fails_different_zp(self):
    tensor0, tensor1 = self._test_model.subgraphs[0].tensors[:2]
    tensor0.quantization.scale = np.array([1, 2, 3]).astype(np.float32)
    tensor0.quantization.zeroPoint = np.array([3, 2, 1]).astype(np.int32)
    tensor1.quantization.scale = np.array([1, 2, 3]).astype(np.float32)
    tensor1.quantization.zeroPoint = np.array([1, 2, 3]).astype(np.int32)
    self.assertFalse(
        tfl_flatbuffer_utils.has_same_quantization(tensor0, tensor1)
    )


if __name__ == "__main__":
  googletest.main()