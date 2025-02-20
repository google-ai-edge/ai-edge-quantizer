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
from ai_edge_quantizer.utils import test_utils

_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


class DequantizedWeightRecoveryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._dummy_quantized_weights = np.array([
        [1, -2, 3, 4],
        [6, 7, -8, 5],
        [-1, 8, -7, -4],
    ])

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
        dequantized_weight_recovery.get_zp_scale_from_2d_dequantized_symmetric_weights(
            dequant_vals, quantized_dimension
        )
    )
    self.assertEqual(recovered_scale.shape, scale.shape)
    self.assertSequenceAlmostEqual(recovered_scale.flatten(), scale.flatten())
    # Zero point should be zero for symmetric quantization.
    self.assertEqual(np.sum(zp), 0)
    self.assertEqual(zp.shape, scale.shape)

  def test_tensor_zp_scale_from_2d_dequantized_symmetric_weights_raises_error_for_non_2d_weights(
      self,
  ):
    weights_3d = self._dummy_quantized_weights.reshape(1, 3, 4)
    weights_3d = weights_3d * 1.02
    with self.assertRaisesRegex(
        ValueError, "Only 2D weights are supported. Got 3 dimensions."
    ):
      dequantized_weight_recovery.get_zp_scale_from_2d_dequantized_symmetric_weights(
          weights_3d, quantized_dimension=None
      )

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
      dequantized_weight_recovery.get_zp_scale_from_2d_dequantized_symmetric_weights(
          dequant_vals, quantized_dimension
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="recovery_on_wrong_dimension",
          quantized_dimension=1,  # should be 0.
          scale=np.array([0.1875, 1e-4, 12.3]).reshape(3, 1),
      ),
      dict(
          testcase_name="tensor_recovery_for_channel_quantization",
          quantized_dimension=None,  # should be 0.
          scale=np.array([0.003, 1.234, 12.65, 2.24e-4]).reshape(1, 4),
      ),
  )
  def test_tensor_zp_scale_from_2d_dequantized_symmetric_weights_raises_error_big_recovery_error(
      self, quantized_dimension, scale
  ):
    dequant_vals = scale * self._dummy_quantized_weights
    with self.assertRaisesRegex(
        RuntimeError,
        "Failed to recover the original quantized values from dequantized"
        " values. Max diff between recovered and original values: ",
    ):
      dequantized_weight_recovery.get_zp_scale_from_2d_dequantized_symmetric_weights(
          dequant_vals, quantized_dimension
      )


if __name__ == "__main__":
  googletest.main()
