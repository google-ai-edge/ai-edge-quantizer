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

"""Tests for instruction_generator."""

import os

import numpy as np

from tensorflow.python.platform import googletest
from absl.testing import parameterized
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import transformation_instruction_generator as instruction_generator
from ai_edge_quantizer.utils import test_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile(".")


class InstructionGeneratorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="second_index_test",
          param1=qtyping.OpToTensorParams(
              0,
              [
                  qtyping.QuantTransformation.ADD_QUANTIZE,
                  qtyping.QuantTransformation.ADD_DEQUANTIZE,
              ],
              qtyping.UniformQuantParams(8, None, np.array([1]), np.array([0])),
          ),
          param2=qtyping.OpToTensorParams(
              2,
              [
                  qtyping.QuantTransformation.ADD_QUANTIZE,
                  qtyping.QuantTransformation.ADD_DEQUANTIZE,
              ],
              qtyping.UniformQuantParams(8, None, np.array([1]), np.array([0])),
          ),
          index=1,
          expected=True,
      ),
      dict(
          testcase_name="different_trans_length_test",
          param1=qtyping.OpToTensorParams(
              0,
              [
                  qtyping.QuantTransformation.ADD_QUANTIZE,
              ],
              qtyping.UniformQuantParams(8, None, np.array([1]), np.array([0])),
          ),
          param2=qtyping.OpToTensorParams(
              2,
              [
                  qtyping.QuantTransformation.ADD_QUANTIZE,
                  qtyping.QuantTransformation.ADD_DEQUANTIZE,
              ],
              qtyping.UniformQuantParams(8, None, np.array([1]), np.array([0])),
          ),
          index=1,
          expected=False,
      ),
      dict(
          testcase_name="different_trans_length_test2",
          param1=qtyping.OpToTensorParams(
              0,
              [
                  qtyping.QuantTransformation.ADD_QUANTIZE,
                  qtyping.QuantTransformation.ADD_DEQUANTIZE,
              ],
              qtyping.UniformQuantParams(8, None, np.array([1]), np.array([0])),
          ),
          param2=qtyping.OpToTensorParams(
              2,
              [
                  qtyping.QuantTransformation.ADD_QUANTIZE,
              ],
              qtyping.UniformQuantParams(8, None, np.array([1]), np.array([0])),
          ),
          index=1,
          expected=False,
      ),
      dict(
          testcase_name="test_unmatched_transforamtions",
          param1=qtyping.OpToTensorParams(
              0,
              [
                  qtyping.QuantTransformation.ADD_QUANTIZE,
              ],
              qtyping.UniformQuantParams(8, None, np.array([1]), np.array([0])),
          ),
          param2=qtyping.OpToTensorParams(
              2,
              [
                  qtyping.QuantTransformation.ADD_QUANTIZE,
                  qtyping.QuantTransformation.ADD_DEQUANTIZE,
              ],
              qtyping.UniformQuantParams(
                  16, None, np.array([1]), np.array([0])
              ),
          ),
          index=0,
          expected=False,
      ),
  )
  def test_check_horizontal_optimization(self, param1, param2, index, expected):
    got = instruction_generator.check_horizontal_optimization(
        param1=param1, param2=param2, index=index
    )
    self.assertEqual(expected, got)

  @parameterized.named_parameters(
      dict(
          testcase_name="test_success",
          producer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          expected=True,
      ),
      dict(
          testcase_name="test_wrong_transformation",
          producer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          expected=False,
      ),
      dict(
          testcase_name="test_wrong_parameters",
          producer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  16, None, np.array([1]), np.array([0])
              ),
          ),
          expected=False,
      ),
  )
  def test_check_dq_q_elimination(self, producer_inst, consumer_inst, expected):
    got = instruction_generator.check_dq_q_elimination(
        producer_inst=producer_inst, consumer_inst=consumer_inst
    )
    self.assertEqual(expected, got)

  @parameterized.named_parameters(
      dict(
          testcase_name="test_success",
          producer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  16, None, np.array([1]), np.array([0])
              ),
          ),
          expected=True,
      ),
      dict(
          testcase_name="test_wrong_transformation",
          producer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          expected=False,
      ),
      dict(
          testcase_name="test_wrong_parameters",
          producer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          expected=False,
      ),
  )
  def test_check_replace_dq_q_with_rq(
      self, producer_inst, consumer_inst, expected
  ):
    got = instruction_generator.check_replace_dq_q_with_rq(
        producer_inst=producer_inst, consumer_inst=consumer_inst
    )
    self.assertEqual(expected, got)

  @parameterized.named_parameters(
      dict(
          testcase_name="test_elimination_success",
          producer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.NO_QUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          expected=True,
      ),
      dict(
          testcase_name="test_wrong_transformation1",
          producer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          expected=False,
      ),
      dict(
          testcase_name="test_wrong_transformation2",
          producer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_inst=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.NO_QUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          expected=False,
      ),
  )
  def test_check_dq_no_quant_elimination(
      self, producer_inst, consumer_inst, expected
  ):
    got = instruction_generator.check_dq_no_quant_elimination(
        producer_inst, consumer_inst
    )
    self.assertEqual(expected, got)

  @parameterized.named_parameters(
      dict(
          testcase_name="test_empty_consumer",
          producer_trans_rule=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_trans_rule=[],
          expected=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[2],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              )
          ],
      ),
      dict(
          testcase_name="test_no_vertical_trans",
          producer_trans_rule=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[0, np.array([1]), 2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_trans_rule=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[0],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
          expected=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[0, np.array([1]), 2],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[0],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
      ),
      dict(
          testcase_name="test_vertical_trans_with_mix_output",
          producer_trans_rule=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[0, np.array([1]), 2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_trans_rule=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[0],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      16, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
          expected=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[2],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.QUANTIZE_TENSOR,
                  tensor_id=1,
                  producer=0,
                  consumers=[0],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.QUANTIZE_TENSOR,
                  tensor_id=1,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[1],
                  parameters=qtyping.UniformQuantParams(
                      16, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
      ),
      dict(
          testcase_name="test_multi_match",
          producer_trans_rule=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[0, 1, 2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_trans_rule=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[0, 1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
          expected=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[2],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.QUANTIZE_TENSOR,
                  tensor_id=1,
                  producer=0,
                  consumers=[0, 1],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
      ),
      dict(
          testcase_name="test_dequant_no_quant_elimination_succeeds",
          producer_trans_rule=qtyping.TransformationInst(
              transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
              tensor_id=1,
              producer=0,
              consumers=[0, 1, 2],
              parameters=qtyping.UniformQuantParams(
                  8, None, np.array([1]), np.array([0])
              ),
          ),
          consumer_trans_rule=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.NO_QUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[0, 1, 2],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
          expected=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[0, 1, 2],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
      ),
  )
  def test_apply_vertical_optimization(
      self, producer_trans_rule, consumer_trans_rule, expected
  ):
    ins_gen = instruction_generator.TransformationInstructionsGenerator(
        os.path.join(
            TEST_DATA_PREFIX_PATH, "tests/models/single_fc_bias.tflite"
        )
    )
    got = ins_gen._apply_vertical_optimization(
        producer_trans_rule, consumer_trans_rule
    )
    self.assertEqual(expected, got)

  @parameterized.named_parameters(
      dict(testcase_name="test_empty_consumer", param={}, expected=[]),
      dict(
          testcase_name="test_multi_level_grouping",
          param=qtyping.TensorTransformationParams(
              "tfl.quantize",
              qtyping.OpToTensorParams(
                  subgraph_op_id=0,
                  transformations=[qtyping.QuantTransformation.ADD_DEQUANTIZE],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              [
                  qtyping.OpToTensorParams(
                      subgraph_op_id=1,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=2,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=3,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=4,
                      transformations=[
                          qtyping.QuantTransformation.NO_QUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
              ],
          ),
          expected=[
              [{0, 1, 2, 3}],
              [{0, 1, 2}, {3}],
              [{1, 2}],
          ],
      ),
  )
  def test_group_consumer_transformations(self, param, expected):
    ins_gen = instruction_generator.TransformationInstructionsGenerator(
        os.path.join(
            TEST_DATA_PREFIX_PATH, "tests/models/single_fc_bias.tflite"
        )
    )
    got = ins_gen._group_consumer_transformations(param)
    self.assertEqual(expected, got)

  @parameterized.named_parameters(
      dict(
          testcase_name="test_empty_input",
          consumer_group=[],
          param=qtyping.TensorTransformationParams(
              "arg0",
              None,
              [],
          ),
          expected=[],
      ),
      dict(
          testcase_name="test_multi_level_grouping",
          consumer_group=[
              [{0, 1, 2, 3}],
              [{0, 1, 2}, {3}],
              [{1, 2}],
          ],
          param=qtyping.TensorTransformationParams(
              "tfl.quantize",
              qtyping.OpToTensorParams(
                  subgraph_op_id=0,
                  transformations=[qtyping.QuantTransformation.ADD_DEQUANTIZE],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              [
                  qtyping.OpToTensorParams(
                      subgraph_op_id=1,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=2,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=3,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=4,
                      transformations=[
                          qtyping.QuantTransformation.NO_QUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
              ],
          ),
          expected=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[1, 2, 3],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.NO_QUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[4],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
      ),
  )
  def test_produce_transformation_for_vertical_opt(
      self, consumer_group, param, expected
  ):
    ins_gen = instruction_generator.TransformationInstructionsGenerator(
        os.path.join(
            TEST_DATA_PREFIX_PATH, "tests/models/insert_dequant_test.tflite"
        )
    )
    got = ins_gen._produce_transformation_for_vertical_opt(
        consumer_group, param
    )
    self.assertEqual(expected, got)

  @parameterized.named_parameters(
      dict(
          testcase_name="test_empty_input",
          consumer_group=[],
          param=qtyping.TensorTransformationParams(
              "arg0",
              None,
              [],
          ),
          expected=[],
      ),
      dict(
          testcase_name="test_multi_level_grouping",
          consumer_group=[
              [{0, 1, 2, 3}],
              [{0, 1, 2}, {3}],
              [{1, 2}],
          ],
          param=qtyping.TensorTransformationParams(
              "tfl.quantize",
              qtyping.OpToTensorParams(
                  subgraph_op_id=0,
                  transformations=[qtyping.QuantTransformation.ADD_DEQUANTIZE],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              [
                  qtyping.OpToTensorParams(
                      subgraph_op_id=1,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=2,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=3,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=4,
                      transformations=[
                          qtyping.QuantTransformation.NO_QUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
              ],
          ),
          expected=[
              qtyping.TransformationInst(
                  transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                  tensor_id=1,
                  producer=0,
                  consumers=[2, 3],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
          ],
      ),
  )
  def test_produce_customer_transformations_unavailable_for_vertical_opt(
      self, consumer_group, param, expected
  ):
    ins_gen = instruction_generator.TransformationInstructionsGenerator(
        os.path.join(
            TEST_DATA_PREFIX_PATH, "tests/models/insert_dequant_test.tflite"
        )
    )
    got = (
        ins_gen._produce_consumer_transformations_unavailable_for_vertical_opt(
            consumer_group, param
        )
    )
    self.assertEqual(expected, got)

  def test_empty_param(self):
    """test the capability to handle empty params."""
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/single_fc_bias.tflite"
    )
    quant_parameters = {}
    ins_gen = instruction_generator.TransformationInstructionsGenerator(
        test_model_path
    )
    instructions = ins_gen.quant_params_to_transformation_insts(
        quant_parameters
    )
    self.assertEmpty(instructions)

  def test_generate_instruction_for_single_fc_bias(self):
    """test the capability to run multiple tensor infos."""
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/single_fc_bias.tflite"
    )
    quant_parameters = {}
    quant_parameters["serving_default_input_2:0"] = (
        qtyping.TensorTransformationParams(
            "serving_default_input_2:0",
            None,
            [
                qtyping.OpToTensorParams(
                    0,
                    [qtyping.QuantTransformation.ADD_QUANTIZE],
                    qtyping.UniformQuantParams(
                        8, None, np.array([1]), np.array([0])
                    ),
                )
            ],
        )
    )

    quant_parameters["StatefulPartitionedCall:0"] = (
        qtyping.TensorTransformationParams(
            "StatefulPartitionedCall:0",
            qtyping.OpToTensorParams(
                0,
                [
                    qtyping.QuantTransformation.ADD_DEQUANTIZE,
                    qtyping.QuantTransformation.ADD_QUANTIZE,
                ],
                qtyping.UniformQuantParams(
                    8, None, np.array([1]), np.array([0])
                ),
            ),
            [],
        )
    )

    ins_gen = instruction_generator.TransformationInstructionsGenerator(
        test_model_path
    )
    instructions = ins_gen.quant_params_to_transformation_insts(
        quant_parameters
    )
    input_transformation = qtyping.TensorTransformationInsts(
        tensor_name="serving_default_input_2:0",
        subgraph_id=0,
        instructions=[
            qtyping.TransformationInst(
                transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                tensor_id=0,
                producer=-1,  # input tensor is the subgraph input
                consumers=[0],  # consumed by node 0
                parameters=qtyping.UniformQuantParams(
                    8, None, np.array([1]), np.array([0])
                ),
            )
        ],
    )
    output_transformation = qtyping.TensorTransformationInsts(
        tensor_name="StatefulPartitionedCall:0",
        subgraph_id=0,
        instructions=[
            qtyping.TransformationInst(
                transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                tensor_id=3,
                producer=0,
                consumers=[-1],
                parameters=qtyping.UniformQuantParams(
                    8, None, np.array([1]), np.array([0])
                ),
            ),
            qtyping.TransformationInst(
                transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                tensor_id=3,
                producer=0,
                consumers=[-1],
                parameters=qtyping.UniformQuantParams(
                    8, None, np.array([1]), np.array([0])
                ),
            ),
        ],
    )
    self.assertLen(instructions, 2)
    self.assertEqual(
        instructions["serving_default_input_2:0"], input_transformation
    )
    self.assertEqual(
        instructions["StatefulPartitionedCall:0"], output_transformation
    )

  def test_raise_error_on_op_replacement_transformation_is_not_unique(self):
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/insert_dequant_test.tflite"
    )
    quant_parameters = {}
    quant_parameters["tfl.quantize"] = qtyping.TensorTransformationParams(
        "tfl.quantize",
        qtyping.OpToTensorParams(
            subgraph_op_id=0,
            transformations=[
                qtyping.QuantTransformation.ADD_DEQUANTIZE,
                qtyping.QuantTransformation.EMULATED_SUBCHANNEL,
            ],
            parameters=qtyping.UniformQuantParams(
                8, None, np.array([1]), np.array([0])
            ),
        ),
        [],
    )
    ins_gen = instruction_generator.TransformationInstructionsGenerator(
        test_model_path
    )
    with self.assertRaisesRegex(
        ValueError, "op replacement transformation can not be combined"
    ):
      ins_gen.quant_params_to_transformation_insts(quant_parameters)

  def test_raise_error_on_no_quant_conflict(self):
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/insert_dequant_test.tflite"
    )
    quant_parameters = {}
    quant_parameters["tfl.quantize"] = qtyping.TensorTransformationParams(
        "tfl.quantize",
        None,
        [
            qtyping.OpToTensorParams(
                subgraph_op_id=1,
                transformations=[qtyping.QuantTransformation.QUANTIZE_TENSOR],
                parameters=qtyping.UniformQuantParams(
                    8, None, np.array([1]), np.array([0])
                ),
            ),
            qtyping.OpToTensorParams(
                subgraph_op_id=2,
                transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
                parameters=None,
            ),
        ],
    )
    ins_gen = instruction_generator.TransformationInstructionsGenerator(
        test_model_path
    )
    with self.assertRaisesRegex(
        ValueError, "can not be both quantized and unquantized"
    ):
      ins_gen.quant_params_to_transformation_insts(quant_parameters)

  def test_generate_instruction_for_branching(self):
    """test horizontal and vertial optimization on a graph with multi branch."""
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, "tests/models/insert_dequant_test.tflite"
    )
    quant_parameters = {}
    quant_parameters["tfl.quantize"] = qtyping.TensorTransformationParams(
        "tfl.quantize",
        qtyping.OpToTensorParams(
            subgraph_op_id=0,
            transformations=[qtyping.QuantTransformation.ADD_DEQUANTIZE],
            parameters=qtyping.UniformQuantParams(
                8, None, np.array([1]), np.array([0])
            ),
        ),
        [
            qtyping.OpToTensorParams(
                subgraph_op_id=1,
                transformations=[qtyping.QuantTransformation.ADD_QUANTIZE],
                parameters=qtyping.UniformQuantParams(
                    8, None, np.array([1]), np.array([0])
                ),
            ),
            qtyping.OpToTensorParams(
                subgraph_op_id=2,
                transformations=[
                    qtyping.QuantTransformation.ADD_QUANTIZE,
                    qtyping.QuantTransformation.ADD_DEQUANTIZE,
                ],
                parameters=qtyping.UniformQuantParams(
                    8, None, np.array([1]), np.array([0])
                ),
            ),
        ],
    )
    ins_gen = instruction_generator.TransformationInstructionsGenerator(
        test_model_path
    )
    instructions = ins_gen.quant_params_to_transformation_insts(
        quant_parameters
    )
    expected_instructions = qtyping.TensorTransformationInsts(
        tensor_name="tfl.quantize",
        subgraph_id=0,
        instructions=[
            qtyping.TransformationInst(
                transformation=qtyping.QuantTransformation.QUANTIZE_TENSOR,
                tensor_id=1,
                producer=0,
                consumers=[1, 2],
                parameters=qtyping.UniformQuantParams(
                    8, None, np.array([1]), np.array([0])
                ),
            ),
            qtyping.TransformationInst(
                transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                tensor_id=1,
                producer=0,
                consumers=[2],
                parameters=qtyping.UniformQuantParams(
                    8, None, np.array([1]), np.array([0])
                ),
            ),
        ],
    )
    self.assertLen(instructions, 1)
    self.assertEqual(instructions["tfl.quantize"], expected_instructions)


if __name__ == "__main__":
  googletest.main()
