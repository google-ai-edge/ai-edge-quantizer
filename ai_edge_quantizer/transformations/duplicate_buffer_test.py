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
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import duplicate_buffer
from ai_edge_quantizer.transformations import transformation_utils
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('..')


class DuplicateBufferTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/weight_sharing_fcs.tflite'
    )
    self.model = tfl_flatbuffer_utils.read_model(model_path)

  def _get_transformation_input(
      self, subgraph_idx: int, tensor_idx: int
  ) -> transformation_utils.TransformationInput:
    return transformation_utils.TransformationInput(
        tensor_id=tensor_idx,
        buffers=self.model.buffers,
        # Dummy params below.
        op_codes=self.model.operatorCodes,
        subgraph=self.model.subgraphs[subgraph_idx],
        producer=-1,
        consumers=[],
        quant_params=qtyping.UniformQuantParams(
            num_bits=8,
            quantized_dimension=None,
            scale=np.ones(1),
            zero_point=np.zeros(1),
        ),
    )

  def test_constant_buffer_is_correctly_duplicated(self):
    # Duplicate the FC weight tensor in the second subgraph.
    subgraph_idx = 1
    subgraph = self.model.subgraphs[subgraph_idx]
    weight_tensor_idx = 1
    prev_buffer_id = subgraph.tensors[weight_tensor_idx].buffer
    prev_num_buffers = len(self.model.buffers)
    transformation_input = self._get_transformation_input(
        subgraph_idx, weight_tensor_idx
    )
    transformation_info = duplicate_buffer.duplicate_buffer(
        transformation_input
    )
    self.assertEqual(transformation_info.op_id, 0)
    self.assertEqual(transformation_info.num_ops_added, 0)
    self.assertEqual(transformation_info.output_tensor_id, 1)
    # Check that a new buffer was added.
    self.assertLen(self.model.buffers, prev_num_buffers + 1)
    # Check that the new buffer is used by the weight tensor.
    new_buffer_id = len(self.model.buffers) - 1
    self.assertEqual(subgraph.tensors[weight_tensor_idx].buffer, new_buffer_id)
    # Check that the new buffer has the same data as the original one.
    self.assertTrue(
        np.all(
            np.frombuffer(
                self.model.buffers[new_buffer_id].data,
                dtype=np.float32,
            )
            == np.frombuffer(
                self.model.buffers[prev_buffer_id].data,
                dtype=np.float32,
            )
        )
    )

  def test_duplicate_buffer_raises_error_when_tensor_is_not_constant(self):
    # Duplicate the FC input tensor in the second subgraph.
    subgraph_idx = 1
    weight_tensor_idx = 0
    transformation_input = self._get_transformation_input(
        subgraph_idx, weight_tensor_idx
    )
    with self.assertRaisesRegex(
        ValueError,
        'Duplicate Buffer transformation supports only constant tensors.',
    ):
      duplicate_buffer.duplicate_buffer(transformation_input)


if __name__ == '__main__':
  googletest.main()
