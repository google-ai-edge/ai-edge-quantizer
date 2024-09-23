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
import numpy as np
from tensorflow.python.platform import googletest
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils


TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../tests/models")


class TflUtilsSingleSignatureModelTest(googletest.TestCase):

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
    self.assertFalse(tfl_interpreter_utils.is_tensor_quantized(input_details))

  def test_get_input_tensor_names(self):
    input_tensor_names = tfl_interpreter_utils.get_input_tensor_names(
        self._test_model_path
    )
    self.assertEqual(
        input_tensor_names,
        ["serving_default_conv2d_input:0"],
    )

  def test_get_output_tensor_names(self):
    output_tensor_names = tfl_interpreter_utils.get_output_tensor_names(
        self._test_model_path
    )
    self.assertEqual(
        output_tensor_names,
        ["StatefulPartitionedCall:0"],
    )

  def test_get_constant_tensor_names(self):
    const_tensor_names = tfl_interpreter_utils.get_constant_tensor_names(
        self._test_model_path
    )
    self.assertEqual(
        set(const_tensor_names),
        set([
            "sequential/conv2d/Conv2D",
            "sequential/conv2d/Relu;sequential/conv2d/BiasAdd;sequential/conv2d/Conv2D;sequential/conv2d/BiasAdd/ReadVariableOp",
            "arith.constant",
            "arith.constant1",
            "arith.constant2",
            "arith.constant3",
        ]),
    )


class TflUtilsQuantizedModelTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    self._test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "conv_fc_mnist_srq_a8w8.tflite"
    )
    self._signature_input_data = {
        "conv2d_input": np.random.rand(1, 28, 28, 1).astype(np.float32)
    }

  def test_is_tensor_quantized(self):
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    input_details = tfl_interpreter.get_input_details()[0]
    self.assertTrue(tfl_interpreter_utils.is_tensor_quantized(input_details))

  def test_invoke_interpreter_signature(self):
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    signature_output = tfl_interpreter_utils.invoke_interpreter_signature(
        tfl_interpreter, self._signature_input_data
    )
    print(signature_output)
    self.assertEqual(tuple(signature_output["dense_1"].shape), (1, 10))

    # Assert the input data is not modified in-place b/353340272.
    self.assertEqual(
        self._signature_input_data["conv2d_input"].dtype, np.float32
    )


class TflUtilsMultiSignatureModelTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    self._test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "two_signatures.tflite"
    )
    self._signature_input_data = {"x": np.array([2.0]).astype(np.float32)}

  def test_create_tfl_interpreter(self):
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    self.assertIsNotNone(tfl_interpreter)

  def test_get_input_tensor_names(self):
    signature_name = "add"
    input_tensor_names = tfl_interpreter_utils.get_input_tensor_names(
        self._test_model_path, signature_name
    )
    self.assertEqual(
        input_tensor_names,
        ["add_x:0"],
    )

    signature_name = "multiply"
    input_tensor_names = tfl_interpreter_utils.get_input_tensor_names(
        self._test_model_path, signature_name
    )
    self.assertEqual(
        input_tensor_names,
        ["multiply_x:0"],
    )

  def test_get_output_tensor_names(self):
    signature_name = "add"
    input_tensor_names = tfl_interpreter_utils.get_output_tensor_names(
        self._test_model_path, signature_name
    )
    self.assertEqual(
        input_tensor_names,
        ["PartitionedCall:0"],
    )

    signature_name = "multiply"
    input_tensor_names = tfl_interpreter_utils.get_output_tensor_names(
        self._test_model_path, signature_name
    )
    self.assertEqual(
        input_tensor_names,
        ["PartitionedCall_1:0"],
    )

  def test_get_constant_tensor_names(self):
    subgraph0_const_tensor_names = (
        tfl_interpreter_utils.get_constant_tensor_names(
            self._test_model_path, 0
        )
    )
    self.assertEqual(subgraph0_const_tensor_names, ["Add/y"])

    subgraph1_const_tensor_names = (
        tfl_interpreter_utils.get_constant_tensor_names(
            self._test_model_path, 1
        )
    )
    self.assertEqual(subgraph1_const_tensor_names, ["Mul/y"])

  def test_get_signature_main_subgraph_index(self):
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    add_subgraph_index = (
        tfl_interpreter_utils.get_signature_main_subgraph_index(
            tfl_interpreter, "add"
        )
    )
    self.assertEqual(add_subgraph_index, 0)
    multiply_subgraph_index = (
        tfl_interpreter_utils.get_signature_main_subgraph_index(
            tfl_interpreter, "multiply"
        )
    )
    self.assertEqual(multiply_subgraph_index, 1)

  def test_get_tensor_data(self):
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    # Invoke the ADD signature.
    tfl_interpreter_utils.invoke_interpreter_signature(
        tfl_interpreter, self._signature_input_data, "add"
    )
    output_details = {"index": 2, "quantization_parameters": {"scales": []}}
    output_data = tfl_interpreter_utils.get_tensor_data(
        tfl_interpreter, output_details, subgraph_index=0
    )  # The ADD signature is in the first subgraph.
    self.assertEqual(output_data, [12.0])  # 10 + 2

    # Invoke the MULTIPLY signature.
    tfl_interpreter_utils.invoke_interpreter_signature(
        tfl_interpreter, self._signature_input_data, "multiply"
    )
    output_data = tfl_interpreter_utils.get_tensor_data(
        tfl_interpreter, output_details, subgraph_index=1
    )  # The Multiply signature is in the second subgraph.
    self.assertEqual(output_data, [20.0])  # 10 * 2

  def test_get_tensor_name_to_content_map(self):
    tfl_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
        self._test_model_path
    )
    # Invoke all signatures.
    tfl_interpreter_utils.invoke_interpreter_signature(
        tfl_interpreter, self._signature_input_data, "multiply"
    )
    tfl_interpreter_utils.invoke_interpreter_signature(
        tfl_interpreter, self._signature_input_data, "add"
    )

    # Test tensors belonging to the ADD signature.
    add_subgraph_index = (
        tfl_interpreter_utils.get_signature_main_subgraph_index(
            tfl_interpreter, "add"
        )
    )
    add_tensor_content = tfl_interpreter_utils.get_tensor_name_to_content_map(
        tfl_interpreter, add_subgraph_index
    )

    add_input_content = add_tensor_content["add_x:0"]
    self.assertSequenceAlmostEqual(
        self._signature_input_data["x"].flatten(), add_input_content.flatten()
    )
    weight_content = add_tensor_content["Add/y"]
    self.assertEqual(weight_content, 10)
    add_output_content = add_tensor_content["PartitionedCall:0"]
    self.assertEqual(add_output_content, [12.0])

    # Test tensors belonging to the MULTIPLY signature.
    multiply_subgraph_index = (
        tfl_interpreter_utils.get_signature_main_subgraph_index(
            tfl_interpreter, "multiply"
        )
    )
    mul_tensor_content = tfl_interpreter_utils.get_tensor_name_to_content_map(
        tfl_interpreter, multiply_subgraph_index
    )
    multiply_input_content = mul_tensor_content["multiply_x:0"]
    self.assertSequenceAlmostEqual(
        self._signature_input_data["x"].flatten(),
        multiply_input_content.flatten(),
    )
    weight_content = mul_tensor_content["Mul/y"]
    self.assertEqual(weight_content, 10)
    multiply_output_content = mul_tensor_content["PartitionedCall_1:0"]
    self.assertEqual(multiply_output_content, [20.0])


if __name__ == "__main__":
  googletest.main()
