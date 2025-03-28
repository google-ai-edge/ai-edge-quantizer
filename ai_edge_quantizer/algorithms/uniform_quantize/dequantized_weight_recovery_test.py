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
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import dequantized_weight_recovery

_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


class DequantizedWeightRecoveryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._dummy_quantized_weights = np.array([
        [1, -2, 3, 4],
        [6, 7, -6, 5],
        [2, -6, -7, -4],
    ])
    self._dummy_op_info = qtyping.OpInfo(
        op=None,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=0,
        op_quant_config=qtyping.OpQuantizationConfig(),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="per-tensor-recovery",
          quantized_dimension=None,
          scale=np.array([0.1875]).reshape(1, 1),
      ),
      dict(
          testcase_name="channel0-recovery",
          quantized_dimension=0,
          scale=np.array([0.1875, 1e-4, 12.3]).reshape(3, 1),
      ),
      dict(
          testcase_name="channel1-recovery",
          quantized_dimension=1,
          scale=np.array([0.003, 1.234, 12.65, 2.24e-4]).reshape(1, 4),
      ),
  )
  def test_tensor_zp_scale_from_2d_dequantized_symmetric_weights_success(
      self, quantized_dimension, scale
  ):
    dequant_vals = scale * self._dummy_quantized_weights
    zp, recovered_scale = (
        dequantized_weight_recovery.get_zp_scale_from_dequantized_symmetric_weights(
            dequant_vals, quantized_dimension
        )
    )
    self.assertEqual(recovered_scale.shape, scale.shape)
    self.assertSequenceAlmostEqual(recovered_scale.flatten(), scale.flatten())
    # Zero point should be zero for symmetric quantization.
    self.assertEqual(np.sum(zp), 0)
    self.assertEqual(zp.shape, scale.shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="per-tensor-recovery",
          quantized_dimension=None,
          scale=np.array([0.1875]).reshape(1, 1),
      ),
      dict(
          testcase_name="channel0-recovery",
          quantized_dimension=0,
          scale=np.array([0.1875, 1e-4, 12.3]).reshape(3, 1, 1),
      ),
      dict(
          testcase_name="channel1-recovery",
          quantized_dimension=1,
          scale=np.array([0.003, 1.234]).reshape(1, 2, 1),
      ),
  )
  def test_tensor_zp_scale_from_3d_dequantized_symmetric_weights_success(
      self, quantized_dimension, scale
  ):
    dequant_vals = scale * self._dummy_quantized_weights.reshape(3, 2, 2)
    zp, recovered_scale = (
        dequantized_weight_recovery.get_zp_scale_from_dequantized_symmetric_weights(
            dequant_vals, quantized_dimension
        )
    )
    with self.subTest("shapes_match"):
      self.assertEqual(recovered_scale.shape, scale.shape)
      self.assertEqual(zp.shape, scale.shape)
    with self.subTest("scale_value_match"):
      self.assertSequenceAlmostEqual(recovered_scale.flatten(), scale.flatten())
    with self.subTest("zp_is_zero"):
      # Zero point should be zero for symmetric quantization.
      self.assertEqual(np.sum(zp), 0)

  @parameterized.named_parameters(
      dict(testcase_name="negative_dimension", quantized_dimension=-1),
      dict(testcase_name="too_large_dimension", quantized_dimension=2),
  )
  def test_tensor_zp_scale_from_2d_dequantized_symmetric_weights_raises_error_for_invalid_quantized_dimension(
      self, quantized_dimension
  ):
    dequant_vals = self._dummy_quantized_weights * 1.02
    with self.assertRaisesRegex(
        ValueError, "quantized_dimension must be 0, 1, or None. Got"
    ):
      dequantized_weight_recovery.get_zp_scale_from_dequantized_symmetric_weights(
          dequant_vals, quantized_dimension
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="tensor-recovery-tensor-quant",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              granularity=qtyping.QuantGranularity.TENSORWISE,
          ),
          scale=np.array([0.1875]).reshape(1, 1),
      ),
      dict(
          testcase_name="channel-recovery-channel-quant",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          scale=np.array([0.1875, 1e-4, 12.3]).reshape(3, 1),
      ),
      dict(
          testcase_name="channel-recovery-excessive-bits",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=8,  # int4 is enough for the sample weights.
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          scale=np.array([0.1875, 1e-4, 12.3]).reshape(3, 1),
      ),
  )
  def test_get_tensor_quant_params_success_with_dequantized_weights(
      self, tensor_quant_config, scale
  ):
    dequant_vals = scale * self._dummy_quantized_weights
    tensor_quant_params = dequantized_weight_recovery.get_tensor_quant_params(
        self._dummy_op_info, tensor_quant_config, dequant_vals
    )

    if tensor_quant_config.granularity is qtyping.QuantGranularity.TENSORWISE:
      self.assertIsNone(tensor_quant_params.quantized_dimension)
    else:
      self.assertEqual(tensor_quant_params.quantized_dimension, 0)

    recovered_scale = tensor_quant_params.scale
    self.assertEqual(recovered_scale.shape, scale.shape)
    self.assertSequenceAlmostEqual(recovered_scale.flatten(), scale.flatten())

    # Zero point should be zero for symmetric quantization.
    recovered_zp = tensor_quant_params.zero_point
    self.assertEqual(np.sum(recovered_zp), 0)
    self.assertEqual(recovered_zp.shape, scale.shape)

  def test_get_tensor_quant_params_success_with_qsv(self):
    # Fall back to naive_min_max_quantize.py for non-weight tensors.
    tensor_quant_params = dequantized_weight_recovery.get_tensor_quant_params(
        self._dummy_op_info,
        tensor_quant_config=qtyping.TensorQuantizationConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        tensor_qsv={
            "min": np.array([-1]),
            "max": np.array([1]),
        },
    )

    self.assertIsNone(tensor_quant_params.quantized_dimension)
    recovered_scale = tensor_quant_params.scale
    self.assertEqual(recovered_scale.shape, (1,))
    self.assertSequenceAlmostEqual(recovered_scale.flatten(), [1 / 127])

    # Zero point should be zero for symmetric quantization.
    recovered_zp = tensor_quant_params.zero_point
    self.assertEqual(np.sum(recovered_zp), 0)
    self.assertEqual(recovered_zp.shape, (1,))

  @parameterized.named_parameters(
      dict(
          testcase_name="recovery_on_wrong_dimension",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          scale=np.array([0.003, 1.234, 12.65, 2.24e-4]).reshape(1, 4),
      ),
      dict(
          testcase_name="tensor_recovery_for_channel_quantization",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              granularity=qtyping.QuantGranularity.TENSORWISE,
          ),
          scale=np.array([0.1875, 1e-2, 12.3]).reshape(3, 1),
      ),
      dict(
          testcase_name="insufficient_bits",
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=2,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          scale=np.array([0.1875, 1e-2, 12.3]).reshape(3, 1),
      ),
  )
  def test_get_tensor_quant_params_raises_error_big_recovery_error(
      self, tensor_quant_config, scale
  ):
    dequant_vals = scale * self._dummy_quantized_weights
    with self.assertRaisesRegex(
        RuntimeError,
        "Failed to recover the original quantized values from dequantized"
        " values. Max diff between recovered and original values: ",
    ):
      dequantized_weight_recovery.get_tensor_quant_params(
          self._dummy_op_info, tensor_quant_config, dequant_vals
      )


if __name__ == "__main__":
  googletest.main()
