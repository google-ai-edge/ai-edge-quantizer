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

"""Tests for transformation_utils."""

import os
import numpy as np
from tensorflow.python.platform import googletest
from absl.testing import parameterized
from ai_edge_quantizer.transformations import transformation_utils
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_litert import schema_py_generated  # pylint: disable=g-direct-tensorflow-import

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../tests/models")


class TransformationUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "single_fc_bias.tflite"
    )
    self.model = tfl_flatbuffer_utils.read_model(self.model_path)

  @parameterized.named_parameters(
      dict(
          testcase_name="add_new_op_code",
          op_code=schema_py_generated.BuiltinOperator.LOGISTIC,
          expected=1,
          custom_op_name=None,
      ),
      dict(
          testcase_name="add_existing_op_code",
          op_code=schema_py_generated.BuiltinOperator.FULLY_CONNECTED,
          expected=0,
          custom_op_name=None,
      ),
      dict(
          testcase_name="add_new_custom_op_code",
          op_code=schema_py_generated.BuiltinOperator.CUSTOM,
          expected=1,
          custom_op_name="random_new_custom_op",
      ),
  )
  def test_add_op_code(self, op_code, expected, custom_op_name):
    """Tests if the op code is added to the model."""
    got = transformation_utils.add_op_code(
        op_code=op_code,
        model_op_codes=self.model.operatorCodes,
        custom_op_name=custom_op_name,
    )
    self.assertEqual(expected, got)
    if custom_op_name is not None:
      self.assertEqual(self.model.operatorCodes[got].customCode, custom_op_name)

  def test_add_custom_op_code_without_op_string_raises_error(self):
    with self.assertRaisesRegex(ValueError, "Custom string is required"):
      transformation_utils.add_op_code(
          op_code=schema_py_generated.BuiltinOperator.CUSTOM,
          model_op_codes=self.model.operatorCodes,
          custom_op_name=None,
      )

  def test_add_two_custom_op_codes(self):
    custom_op_name = "random_new_custom_op"
    added_index = transformation_utils.add_op_code(
        op_code=schema_py_generated.BuiltinOperator.CUSTOM,
        model_op_codes=self.model.operatorCodes,
        custom_op_name=custom_op_name,
    )
    self.assertEqual(1, added_index)
    self.assertEqual(
        self.model.operatorCodes[added_index].customCode, custom_op_name
    )

    custom_op_name_2 = "random_new_custom_op_2"
    added_index = transformation_utils.add_op_code(
        op_code=schema_py_generated.BuiltinOperator.CUSTOM,
        model_op_codes=self.model.operatorCodes,
        custom_op_name=custom_op_name_2,
    )
    self.assertEqual(2, added_index)
    self.assertEqual(
        self.model.operatorCodes[added_index].customCode, custom_op_name_2
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="float32",
          data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
      ),
      dict(
          testcase_name="int8",
          data=np.array([[1, 2], [3, 4]], dtype=np.int8),
      ),
  )
  def test_add_new_constant_buffer(self, data):
    """Tests if the constant buffer is added to the model."""
    prev_num_buffers = len(self.model.buffers) - 1
    new_buffer_idx = transformation_utils.get_constant_buffer(
        data=data,
        buffers=self.model.buffers,
    )
    self.assertEqual(new_buffer_idx, prev_num_buffers + 1)

    expected_buffer_data = (
        np.frombuffer(
            data.tobytes(),
            dtype=np.uint8,
        )
        .flatten()
        .tolist()
    )
    self.assertEqual(
        self.model.buffers[new_buffer_idx].data.tolist(),
        expected_buffer_data,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="float32",
          tensor_data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
          tensor_type=schema_py_generated.TensorType.FLOAT32,
          expected_type=schema_py_generated.TensorType.FLOAT32,
          expected_shape=(4,),
          expected_buffer_data=np.frombuffer(
              np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes(),
              dtype=np.uint8,
          ).flatten(),
      ),
      dict(
          testcase_name="int8",
          tensor_data=np.array([[1, 2], [3, 4]], dtype=np.int8),
          tensor_type=schema_py_generated.TensorType.INT8,
          expected_type=schema_py_generated.TensorType.INT8,
          expected_shape=(2, 2),
          expected_buffer_data=np.frombuffer(
              np.array([[1, 2], [3, 4]], dtype=np.int8).tobytes(),
              dtype=np.uint8,
          ).flatten(),
      ),
  )
  def test_add_new_constant_tensor(
      self,
      tensor_data,
      tensor_type,
      expected_type,
      expected_shape,
      expected_buffer_data,
  ):
    """Tests if the constant tensor is added to the model."""
    ret = transformation_utils.add_new_constant_tensor(
        tensor_name="test_tensor",
        data=tensor_data,
        tensor_type=tensor_type,
        subgraph=self.model.subgraphs[0],
        buffers=self.model.buffers,
    )
    self.assertEqual(ret, len(self.model.subgraphs[0].tensors) - 1)
    self.assertEqual(
        str(self.model.subgraphs[0].tensors[-1].name), "test_tensor"
    )
    self.assertEqual(
        expected_type,
        self.model.subgraphs[0].tensors[-1].type,
    )
    self.assertEqual(
        expected_shape,
        self.model.subgraphs[0].tensors[-1].shape,
    )
    self.assertListEqual(
        expected_buffer_data.tolist(),
        self.model.buffers[
            self.model.subgraphs[0].tensors[-1].buffer
        ].data.tolist(),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="float32",
          tensor_type=schema_py_generated.TensorType.FLOAT32,
          tensor_shape=[1, 1, 1, 1],
          expected_shape=[1, 1, 1, 1],
          expected_type=schema_py_generated.TensorType.FLOAT32,
      ),
      dict(
          testcase_name="int8",
          tensor_type=schema_py_generated.TensorType.INT8,
          tensor_shape=[1, 224, 224, 1],
          expected_shape=[1, 224, 224, 1],
          expected_type=schema_py_generated.TensorType.INT8,
      ),
  )
  def test_add_new_activation_tensor_to_subgraph(
      self,
      tensor_type,
      tensor_shape,
      expected_shape,
      expected_type,
  ):
    """Tests if the activation tensor is added to the subgraph."""
    ret = transformation_utils.add_new_activation_tensor(
        tensor_name="test_tensor",
        shape=tensor_shape,
        tensor_type=tensor_type,
        subgraph=self.model.subgraphs[0],
    )
    self.assertEqual(ret, len(self.model.subgraphs[0].tensors) - 1)
    self.assertEqual(
        str(self.model.subgraphs[0].tensors[-1].name), "test_tensor"
    )
    self.assertEqual(
        expected_type,
        self.model.subgraphs[0].tensors[-1].type,
    )
    self.assertEqual(
        expected_shape,
        self.model.subgraphs[0].tensors[-1].shape,
    )

  def test_add_new_activation_tensor_with_dynamic_shape(self):
    """Tests adding an activation tensor with dynamic shape."""
    subgraph = self.model.subgraphs[0]
    new_id = transformation_utils.add_new_activation_tensor(
        tensor_name="test_tensor",
        shape=[1, -1, -1, 1],
        tensor_type=schema_py_generated.TensorType.FLOAT32,
        subgraph=subgraph,
    )
    # Originally had 4 tensors, new tensor is added at index 4.
    self.assertEqual(new_id, 4)
    self.assertLen(subgraph.tensors, 5)
    self.assertEqual(subgraph.tensors[-1].name, "test_tensor")
    self.assertEqual(
        subgraph.tensors[-1].type, schema_py_generated.TensorType.FLOAT32
    )
    self.assertEqual(subgraph.tensors[-1].shape, [1, 1, 1, 1])
    self.assertEqual(subgraph.tensors[-1].shapeSignature, [1, -1, -1, 1])


if __name__ == "__main__":
  googletest.main()
