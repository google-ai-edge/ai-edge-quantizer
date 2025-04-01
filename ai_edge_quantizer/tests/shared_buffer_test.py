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

from collections.abc import Sequence
from absl.testing import parameterized
from tensorflow.python.platform import googletest
from ai_edge_quantizer import model_validator
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig


class SharedBufferTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # This two-signature model demonstrates both constant tensor and constant
    # buffer sharing. Each signature has two consecutive fully connected (FC)
    # layers that share a weight tensor: the first signature uses
    # `arith.constant` and the second uses `arith.constant1`, demonstrating
    # constant tensor sharing. Furthermore, 'arith.constant' and
    # 'arith.constant1' share a buffer, demonstrating constant buffer sharing.
    self.float_model_path = test_utils.get_path_to_datafile(
        'models/constant_tensor_and_buffer_only_sharing_weight_fcs.tflite'
    )
    self.sig1_output_tensor_name = 'PartitionedCall:0'
    self.sig2_output_tensor_name = 'PartitionedCall_1:0'

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

  def _check_comparison_result(
      self,
      output_tensor_names: Sequence[str],
      comparison_result: model_validator.ComparisonResult,
      output_tolerance: float,
  ):
    tensors_results = comparison_result.get_all_tensor_results()
    for output_tensor_name in output_tensor_names:
      self.assertLess(
          tensors_results[output_tensor_name],
          output_tolerance,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='sig1_fcs_quant_sig2_fcs_no_quant',
          sig1_num_bits=8,
          sig2_num_bits=None,
          output_tolerance=1e-2,
      ),
      dict(
          testcase_name='sig1_fcs_no_quant_sig2_fcs_quant',
          sig1_num_bits=None,
          sig2_num_bits=8,
          output_tolerance=1e-2,
      ),
      dict(
          testcase_name='sig1_fcs_quant_sig2_fcs_quant_different_params',
          sig1_num_bits=8,
          sig2_num_bits=4,
          output_tolerance=0.5,
      ),
  )
  def test_quantization_succeeds_for_distinct_constant_tensors_with_shared_buffer_and_different_quantization_params(
      self, sig1_num_bits, sig2_num_bits, output_tolerance
  ):
    # This test checks a constant buffer shared by tensors `arith.constant`
    # and `arith.constant1` from FCs from two different signatures.
    qt = quantizer.Quantizer(self.float_model_path)

    sig1_fc1_regex = 'BatchMatMulV3;'
    sig1_fc2_regex = f'{self.sig1_output_tensor_name};'
    sig2_fc1_regex = 'BatchMatMulV31;'
    sig2_fc2_regex = f'{self.sig2_output_tensor_name};'
    recipe = []
    # Use same quantization params for FCs within each signature to ensure we
    # don't get a tensor sharing error.
    if sig1_num_bits is not None:
      recipe.append(self._get_fc_recipe_entry(sig1_fc1_regex, sig1_num_bits))
      recipe.append(self._get_fc_recipe_entry(sig1_fc2_regex, sig1_num_bits))
    if sig2_num_bits is not None:
      recipe.append(self._get_fc_recipe_entry(sig2_fc1_regex, sig2_num_bits))
      recipe.append(self._get_fc_recipe_entry(sig2_fc2_regex, sig2_num_bits))
    qt.load_quantization_recipe(recipe)
    quantized_model = qt.quantize().quantized_model
    self.assertIsNotNone(quantized_model)

    test_data = tfl_interpreter_utils.create_random_normal_input_data(
        quantized_model, num_samples=4
    )
    comparison_result = qt.validate(error_metrics='mse', test_data=test_data)
    self._check_comparison_result(
        output_tensor_names=[
            self.sig1_output_tensor_name,
            self.sig2_output_tensor_name,
        ],
        comparison_result=comparison_result,
        output_tolerance=output_tolerance,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='fc1_quant_fc2_no_quant',
          fc1_num_bits=8,
          fc2_num_bits=None,
          output_tolerance=1e-3,
      ),
      dict(
          testcase_name='fc1_no_quant_fc2_quant',
          fc1_num_bits=None,
          fc2_num_bits=8,
          output_tolerance=1e-2,
      ),
      dict(
          testcase_name='fc1_quant_fc2_quant_different_params',
          fc1_num_bits=8,
          fc2_num_bits=4,
          output_tolerance=0.25,
      ),
  )
  def test_quantization_succeeds_for_a_constant_tensor_with_different_quantization_params(
      self, fc1_num_bits, fc2_num_bits, output_tolerance
  ):
    # This test checks a constant tensor `arith.constant` shared by FCs in the
    # first signature.
    qt = quantizer.Quantizer(self.float_model_path)

    sig1_fc1_regex = 'BatchMatMulV3;'
    sig1_fc2_regex = 'PartitionedCall:0;'

    recipe = []
    if fc1_num_bits is not None:
      recipe.append(self._get_fc_recipe_entry(sig1_fc1_regex, fc1_num_bits))
    if fc2_num_bits is not None:
      recipe.append(self._get_fc_recipe_entry(sig1_fc2_regex, fc2_num_bits))
    qt.load_quantization_recipe(recipe)

    quantized_model = qt.quantize().quantized_model
    self.assertIsNotNone(quantized_model)

    test_data = tfl_interpreter_utils.create_random_normal_input_data(
        quantized_model, num_samples=4
    )
    comparison_result = qt.validate(error_metrics='mse', test_data=test_data)
    self._check_comparison_result(
        output_tensor_names=[
            self.sig1_output_tensor_name,
            self.sig2_output_tensor_name,
        ],
        comparison_result=comparison_result,
        output_tolerance=output_tolerance,
    )


if __name__ == '__main__':
  googletest.main()
