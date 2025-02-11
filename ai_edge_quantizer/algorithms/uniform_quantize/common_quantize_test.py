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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import default_policy
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import common_quantize
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../../tests/models")
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


class CommonQuantizeTest(parameterized.TestCase):
  """Tests for general quantize functions.
  """

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = os.path.join(
        _TEST_DATA_PREFIX_PATH, "conv_fc_mnist.tflite"
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._tensor_name_to_qsv = {}

  def test_check_op_quantization_config_with_negative_min_weight_elements_raises_error(
      self,
  ):
    op_quant_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,  # DRQ.
        min_weight_elements=-1,
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        lambda err: "min_weight_elements must be non-negative" in str(err),
    ):
      common_quantize.check_op_quantization_config(
          _TFLOpName.FULLY_CONNECTED,
          op_quant_config,
          default_policy.DEFAULT_CONFIG_CHECK_POLICY,
      )


if __name__ == "__main__":
  googletest.main()
