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

"""E2E tests for the quantizer on Mixture-of-Experts (MoE) models."""

import json

from absl.testing import absltest
import numpy as np

from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils


class MoeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.float_model_path = test_utils.get_path_to_datafile(
        '../models/single_moe_multiaxis.tflite'
    )
    self._quantizer = quantizer.Quantizer(self.float_model_path)
    recipe_content = [{
        'regex': '.*',
        'operation': 'CUSTOM_OP',
        'algorithm_key': 'min_max_uniform_quantize',
        'op_config': {
            'weight_tensor_config': {
                'num_bits': 8,
                'symmetric': True,
                'dtype': 'INT',
                'granularity': 'CHANNELWISE',
                'quantized_dimensions': [0, 1],
            },
            'compute_precision': 'INTEGER',
            'explicit_dequantize': False,
            'skip_checks': False,
            'min_weight_elements': 0,
        },
    }]

    self.temp_recipe = self.create_tempfile(
        content=json.dumps(recipe_content), file_path='recipe.json'
    )

  def test_moe_multiaxis(self):
    self._quantizer.load_quantization_recipe(self.temp_recipe.full_path)
    quant_result = self._quantizer.quantize()
    flatbuffer_model = tfl_flatbuffer_utils.read_model(
        quant_result.quantized_model
    )
    self.assertFalse(tfl_flatbuffer_utils.is_float_model(flatbuffer_model))

    for subgraph in flatbuffer_model.subgraphs:
      for op in subgraph.operators:
        opcode = flatbuffer_model.operatorCodes[op.opcodeIndex]
        if opcode.customCode != b'moe':
          continue

        # 1. The float MoE operator (7 inputs) is expanded to 10 inputs.
        with self.subTest('moe_inputs_expanded'):
          self.assertLen(op.inputs, 10)

        # 2. Quantized Weight Buffers (Zero-Loss Quantization Check):
        # The test model weights were constructed as (base_int * target_scale)
        # with max magnitude 127. Under symmetric int8 quantization, dividing
        # by target_scale recovers the exact integer patterns without loss.
        expected_gate_ff1 = np.tile(
            np.array([-127, -64, 0, 127], dtype=np.int8), (6, 1)
        )
        gate_w = tfl_flatbuffer_utils.get_tensor_data(
            subgraph.tensors[op.inputs[3]], flatbuffer_model.buffers
        )
        ff1_w = tfl_flatbuffer_utils.get_tensor_data(
            subgraph.tensors[op.inputs[5]], flatbuffer_model.buffers
        )
        with self.subTest('gate_w_quantized_correctly'):
          np.testing.assert_array_equal(
              gate_w.reshape(-1, 4), expected_gate_ff1
          )
        with self.subTest('ff1_w_quantized_correctly'):
          np.testing.assert_array_equal(ff1_w.reshape(-1, 4), expected_gate_ff1)

        expected_linear = np.tile(
            np.array([-127, 0, 127], dtype=np.int8), (8, 1)
        )
        linear_w = tfl_flatbuffer_utils.get_tensor_data(
            subgraph.tensors[op.inputs[7]], flatbuffer_model.buffers
        )
        with self.subTest('linear_w_quantized_correctly'):
          np.testing.assert_array_equal(
              linear_w.reshape(-1, 3), expected_linear
          )

        # 3. Multi-Axis Scale Tensor Verification:
        # Quantization along axes [0, 1] (experts, channels) calculates
        # max(|W|)/127 per expert and channel. We verify the exact scales that
        # were used to generate the float weights.
        gate_scale = tfl_flatbuffer_utils.get_tensor_data(
            subgraph.tensors[op.inputs[4]], flatbuffer_model.buffers
        )
        with self.subTest('gate_scale_correct'):
          np.testing.assert_allclose(
              gate_scale.squeeze().T,  # Transpose for easier visual comparison.
              [
                  [1.0, 2.0, 3.0],  # Expert 0.
                  [4.0, 5.0, 6.0],  # Expert 1.
              ],
              rtol=1e-5,
          )

        ff1_scale = tfl_flatbuffer_utils.get_tensor_data(
            subgraph.tensors[op.inputs[6]], flatbuffer_model.buffers
        )
        with self.subTest('ff1_scale_correct'):
          np.testing.assert_allclose(
              ff1_scale.squeeze().T,  # Transpose for easier visual comparison.
              [
                  [10.0, 20.0, 30.0],  # Expert 0.
                  [40.0, 50.0, 60.0],  # Expert 1.
              ],
              rtol=1e-5,
          )

        linear_scale = tfl_flatbuffer_utils.get_tensor_data(
            subgraph.tensors[op.inputs[8]], flatbuffer_model.buffers
        )
        with self.subTest('linear_scale_correct'):
          np.testing.assert_allclose(
              linear_scale.squeeze().T,  # Transpose for visual comparison.
              [
                  [100.0, 200.0, 300.0, 400.0],  # Expert 0.
                  [500.0, 600.0, 700.0, 800.0],  # Expert 1.
              ],
              rtol=1e-5,
          )


if __name__ == '__main__':
  absltest.main()
