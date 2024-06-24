"""Tests for tfl_interpreter_utils."""

import os
import numpy as np
from tensorflow.python.platform import googletest
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils


TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../tests/models")


# TODO(rewu): add quantized model to tests.
class TflUtilsTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    self._test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "conv_fc_mnist.tflite"
    )
    self._input_data = np.random.rand(1, 28, 28, 1).astype(np.float32)

  def test_create_tfl_interpreter(self):
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    self.assertIsNotNone(tfl_interpreter)

  def test_invoke_interpreter_once(self):
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    tfl_interpreter_utils.invoke_interpreter_once(
        tfl_interpreter, [self._input_data]
    )
    output_details = tfl_interpreter.get_output_details()[0]
    output_data = tfl_interpreter.get_tensor(output_details["index"])
    self.assertIsNotNone(output_data)
    self.assertEqual(tuple(output_data.shape), (1, 10))
    self.assertAlmostEqual(output_data[0][0], 0.0031010755)

  def test_get_tensor_data(self):
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    tfl_interpreter_utils.invoke_interpreter_once(
        tfl_interpreter, [self._input_data]
    )
    output_details = tfl_interpreter.get_output_details()[0]
    output_data = tfl_interpreter_utils.get_tensor_data(
        tfl_interpreter, output_details
    )
    self.assertEqual(tuple(output_data.shape), (1, 10))

  def test_get_tensor_name_to_content_map(self):
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    tfl_interpreter_utils.invoke_interpreter_once(
        tfl_interpreter, [self._input_data]
    )

    tensor_name_to_content_map = (
        tfl_interpreter_utils.get_tensor_name_to_content_map(tfl_interpreter)
    )
    input_content = tensor_name_to_content_map["serving_default_conv2d_input:0"]
    self.assertSequenceAlmostEqual(
        self._input_data.flatten(), input_content.flatten()
    )
    weight_content = tensor_name_to_content_map["sequential/conv2d/Conv2D"]
    self.assertEqual(tuple(weight_content.shape), (8, 3, 3, 1))

    self.assertIn(
        "sequential/average_pooling2d/AvgPool", tensor_name_to_content_map
    )
    average_pool_res = tensor_name_to_content_map[
        "sequential/average_pooling2d/AvgPool"
    ]
    self.assertEqual(tuple(average_pool_res.shape), (1, 14, 14, 8))

  def test_is_tensor_quantized(self):
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    input_details = tfl_interpreter.get_input_details()[0]
    self.assertFalse(tfl_interpreter_utils._is_tensor_quantized(input_details))


if __name__ == "__main__":
  googletest.main()
