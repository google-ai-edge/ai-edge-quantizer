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

"""Tests for transformation_performer."""

import os

import numpy as np

from tensorflow.python.platform import googletest
from absl.testing import parameterized
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import transformation_performer
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile(".")


class TransformationPerformerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._transformation_performer = (
        transformation_performer.TransformationPerformer()
    )
    self._test_model = tfl_flatbuffer_utils.read_model(
        os.path.join(TEST_DATA_PREFIX_PATH, "tests/models/conv_fc_mnist.tflite")
    )

  def test_apply_single_insert_dequant(self):
    """test for _apply_transformation."""
    self._transformation_performer._create_op_id_map(self._test_model)
    instructions = qtyping.TensorTransformationInsts(
        tensor_name="sequential/conv2d/Relu;sequential/conv2d/BiasAdd;"
        + "sequential/conv2d/Conv2D;sequential/conv2d/BiasAdd/ReadVariableOp1",
        subgraph_id=0,
        instructions=[
            qtyping.TransformationInst(
                transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                tensor_id=7,
                producer=0,
                consumers=[1],
                parameters=qtyping.UniformQuantParams(
                    8, None, np.array([1]), np.array([0])
                ),
            ),
            qtyping.TransformationInst(
                transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                tensor_id=7,
                producer=0,
                consumers=[1],
                parameters=qtyping.UniformQuantParams(
                    8, None, np.array([1]), np.array([0])
                ),
            ),
        ],
    )
    self._transformation_performer._apply_single_transformation(
        instructions, 0, self._test_model
    )
    subgraph = self._test_model.subgraphs[0]
    self.assertIn(b"_dequant", subgraph.tensors[13].name)
    self.assertEqual(
        subgraph.operators[1].opcodeIndex,
        len(self._test_model.operatorCodes) - 1,
    )
    self.assertEqual(subgraph.operators[2].inputs[0], 13)

  def test_create_op_id_map(self):
    """test for _create_op_id_map."""
    self._transformation_performer._create_op_id_map(self._test_model)
    op_id_map = self._transformation_performer._original_op_id_map
    self.assertLen(op_id_map, 1)
    self.assertLen(op_id_map[0], 6)
    for index, op_id in enumerate(op_id_map[0]):
      self.assertEqual(op_id, index)

  def test_update_op_id_map_changing_value(self):
    """test for _update_op_id_map."""
    self._transformation_performer._create_op_id_map(self._test_model)
    self._transformation_performer._update_op_id_map(0, 1, 6)
    op_id_map = self._transformation_performer._original_op_id_map
    self.assertLen(op_id_map, 1)
    self.assertLen(op_id_map[0], 6)
    for index in range(1, len(op_id_map[0])):
      self.assertEqual(op_id_map[0][index], index + 6)

  def test_update_op_id_map_not_changing_value(self):
    """test for _update_op_id_map."""
    self._transformation_performer._create_op_id_map(self._test_model)
    self._transformation_performer._update_op_id_map(0, 0, 0)
    op_id_map = self._transformation_performer._original_op_id_map
    self.assertLen(op_id_map, 1)
    self.assertLen(op_id_map[0], 6)
    for index, op_id in enumerate(op_id_map[0]):
      self.assertEqual(op_id, index)

  @parameterized.named_parameters(
      dict(
          testcase_name="test_no_update",
          prev_trans_idx=0,
          instructions=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=0,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=0,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
          trans_info=qtyping.TransformationInfo(0, 0, 0),
          expected_instructions=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=0,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=0,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
          expected_added_op_id_map=[[]],
      ),
      dict(
          testcase_name="test_no_matching_consumer",
          prev_trans_idx=0,
          instructions=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=0,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=0,
                  producer=0,
                  consumers=[2],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
          trans_info=qtyping.TransformationInfo(2, 2, 13),
          expected_instructions=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=0,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=0,
                  producer=0,
                  consumers=[2],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
          expected_added_op_id_map=[[3]],
      ),
      dict(
          testcase_name="test_insert_one_op",
          prev_trans_idx=0,
          instructions=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=0,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                  tensor_id=0,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
          trans_info=qtyping.TransformationInfo(1, 1, 13),
          expected_instructions=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=0,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                  tensor_id=13,
                  producer=6,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
          expected_added_op_id_map=[[1]],
      ),
  )
  def test_update_instructions(
      self,
      prev_trans_idx,
      instructions,
      trans_info,
      expected_instructions,
      expected_added_op_id_map,
  ):
    """test for _update_instructions."""
    self._transformation_performer._create_op_id_map(self._test_model)
    self._transformation_performer._update_instructions(
        prev_trans_idx, instructions, 0, trans_info
    )
    self.assertSequenceEqual(instructions, expected_instructions)
    self.assertListEqual(
        self._transformation_performer._added_op_id_map,
        expected_added_op_id_map,
    )

  def test_transform_graph(self):
    """test for transform_graph."""
    instructions = {
        "sequential/conv2d/Relu;sequential/conv2d/BiasAdd;"
        + "sequential/conv2d/Conv2D;sequential/conv2d/BiasAdd/ReadVariableOp1": qtyping.TensorTransformationInsts(
            tensor_name="sequential/conv2d/Relu;sequential/conv2d/BiasAdd;"
            + "sequential/conv2d/Conv2D;sequential/conv2d/BiasAdd/ReadVariableOp1",
            subgraph_id=0,
            instructions=[
                qtyping.TransformationInst(
                    transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                    tensor_id=7,
                    producer=0,
                    consumers=[1],
                    parameters=qtyping.UniformQuantParams(
                        8, None, np.array([1]), np.array([0])
                    ),
                ),
                qtyping.TransformationInst(
                    transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                    tensor_id=7,
                    producer=0,
                    consumers=[1],
                    parameters=qtyping.UniformQuantParams(
                        8, None, np.array([1]), np.array([0])
                    ),
                ),
            ],
        ),
        "sequential/average_pooling2d/AvgPool": qtyping.TensorTransformationInsts(
            tensor_name="sequential/average_pooling2d/AvgPool",
            subgraph_id=0,
            instructions=[
                qtyping.TransformationInst(
                    transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                    tensor_id=8,
                    producer=1,
                    consumers=[2],
                    parameters=qtyping.UniformQuantParams(
                        8, None, np.array([1]), np.array([0])
                    ),
                ),
                qtyping.TransformationInst(
                    transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                    tensor_id=8,
                    producer=1,
                    consumers=[2],
                    parameters=qtyping.UniformQuantParams(
                        8, None, np.array([1]), np.array([0])
                    ),
                ),
            ],
        ),
    }
    self._transformation_performer.transform_graph(
        instructions, self._test_model
    )
    self.assertLen(self._test_model.subgraphs, 1)
    self.assertLen(self._test_model.subgraphs[0].operators, 10)
    self.assertLen(self._test_model.subgraphs[0].tensors, 17)
    self.assertEqual(
        self._test_model.subgraphs[0].operators[1].opcodeIndex,
        len(self._test_model.operatorCodes) - 1,
    )
    self.assertEqual(self._test_model.subgraphs[0].operators[2].inputs[0], 13)
    self.assertEqual(self._test_model.subgraphs[0].operators[2].outputs[0], 14)
    self.assertEqual(
        self._test_model.subgraphs[0].operators[2].outputs[0],
        self._test_model.subgraphs[0].operators[3].inputs[0],
    )
    self.assertEqual(self._test_model.subgraphs[0].operators[3].outputs[0], 8)
    self.assertEqual(self._test_model.subgraphs[0].operators[4].outputs[0], 15)

if __name__ == "__main__":
  googletest.main()
