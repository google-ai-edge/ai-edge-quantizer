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

from absl.testing import parameterized
from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils

_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig


class SharedBufferTest(parameterized.TestCase):

  def _get_fc_recipe_entry(self, regex: str, num_bits: int):
    return {
        'regex': regex,
        'operation': 'FULLY_CONNECTED',
        'algorithm_key': 'min_max_uniform_quantize',
        'op_config': {
            'weight_tensor_config': {
                'num_bits': num_bits,
                'symmetric': True,
                'granularity': 'CHANNELWISE',
                'dtype': 'INT',
                'block_size': 0,
            },
            'compute_precision': 'INTEGER',
            'explicit_dequantize': False,
            'skip_checks': False,
            'min_weight_elements': 0,
        },
    }

  @parameterized.named_parameters(
      dict(
          testcase_name='fc_1_quant_fc_2_no_quant',
          fc_1_num_bits=8,
          fc_2_num_bits=None,
      ),
      dict(
          testcase_name='fc_1_no_quant_fc_2_quant',
          fc_1_num_bits=None,
          fc_2_num_bits=8,
      ),
      dict(
          testcase_name='fc_1_int8_fc_2_int4',
          fc_1_num_bits=8,
          fc_2_num_bits=4,
      ),
  )
  def test_quantization_fails_for_constant_tensors_with_shared_buffer_and_different_quantization_params(
      self, fc_1_num_bits, fc_2_num_bits
  ):
    # This model has two FC layers with the same weights.
    float_model_path = test_utils.get_path_to_datafile(
        'models/weight_sharing_fcs.tflite'
    )
    qt = quantizer.Quantizer(float_model_path)

    fc_1_regex = '.*PartitionedCall:0'
    fc_2_regex = '.*PartitionedCall_1:0'
    recipe = []
    if fc_1_num_bits is not None:
      recipe.append(self._get_fc_recipe_entry(fc_1_regex, fc_1_num_bits))
    if fc_2_num_bits is not None:
      recipe.append(self._get_fc_recipe_entry(fc_2_regex, fc_2_num_bits))
    qt.load_quantization_recipe(recipe)
    with self.assertRaisesRegex(
        expected_exception=RuntimeError,
        expected_regex=(
            'The tensors .* do not have the same quantization parameters'
        ),
    ):
      qt.quantize()


if __name__ == '__main__':
  googletest.main()
