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

  def test_constant_buffer_is_duplicated(self):
    model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/weight_sharing_fcs.tflite'
    )
    model = tfl_flatbuffer_utils.read_model(model_path)
    # Duplicate the FC weight tensor in the second subgraph.
    subgraph = model.subgraphs[1]
    weight_tensor_id = 1
    prev_buffer_id = subgraph.tensors[weight_tensor_id].buffer
    prev_num_buffers = len(model.buffers)
    ret = duplicate_buffer.duplicate_buffer(
        transformation_utils.TransformationInput(
            tensor_id=weight_tensor_id,
            op_codes=model.operatorCodes,
            buffers=model.buffers,
            subgraph=subgraph,
            producer=-1,
            consumers=[0],
            quant_params=qtyping.UniformQuantParams(  # Dummy params.
                num_bits=8,
                quantized_dimension=None,
                scale=np.ones([1, 1, 1, 4], dtype=np.float32),
                zero_point=np.zeros([1, 1, 1, 4], dtype=np.int64),
                symmetric=True,
                quantized_data=np.ones([1, 4, 2, 4], dtype=np.int8),
            ),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    self.assertEqual(ret.output_tensor_id, 1)
    # Check that a new buffer was added.
    self.assertLen(model.buffers, prev_num_buffers + 1)
    # Check that the new buffer is used by the weight tensor.
    new_buffer_id = len(model.buffers) - 1
    self.assertEqual(subgraph.tensors[weight_tensor_id].buffer, new_buffer_id)
    # Check that the new buffer has the same data as the original one.
    self.assertTrue(
        np.all(
            np.frombuffer(
                model.buffers[new_buffer_id].data,
                dtype=np.float32,
            )
            == np.frombuffer(
                model.buffers[prev_buffer_id].data,
                dtype=np.float32,
            )
        )
    )


if __name__ == '__main__':
  googletest.main()
