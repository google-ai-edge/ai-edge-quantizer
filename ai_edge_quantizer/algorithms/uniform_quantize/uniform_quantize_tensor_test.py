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

"""Tests for tensor_utils."""

from absl.testing import parameterized
import numpy as np
from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor

_IntType = uniform_quantize_tensor.IntType


class TensorUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_data = np.array([[1, 2], [4, 5]])
    self._expected_quantized_result = np.array(
        [[9, 19], [22, 27]], dtype=np.int8
    )

  @parameterized.parameters(
      (_IntType(8, True), (-128, 127)),
      (_IntType(8, False), (0, 255)),
      (_IntType(4, True), (-8, 7)),
      (_IntType(4, False), (0, 15)),
      (_IntType(2, True), (-2, 1)),
      (_IntType(2, False), (0, 3)),
  )
  def test_get_quantized_range(self, qtype, expected_range):
    self.assertEqual(
        expected_range, uniform_quantize_tensor.get_quantized_range(qtype)
    )

  @parameterized.parameters(
      (_IntType(8, True), np.int8),
      (_IntType(16, False), np.uint16),
      (_IntType(32, True), np.int32),
      (_IntType(64, True), np.int64),
  )
  def test_assign_quantized_type(self, qtype, expected_type):
    sample_input = np.array([2.0, 2.0])
    result = uniform_quantize_tensor.assign_quantized_type(sample_input, qtype)
    self.assertEqual(expected_type, result.dtype)

  @parameterized.named_parameters(
      dict(
          testcase_name="scalar_tensor_1d_params",
          scale=np.array([0.1]),
          zero_point=np.array([-1], dtype=np.int8),
          tensor_data=np.array(6.66),
          quantized_dimension=None,
      ),
      dict(
          testcase_name="1d_tensor_scalar_params",
          scale=np.array(0.1),
          zero_point=np.array(-1, dtype=np.int8),
          tensor_data=np.array([6.66, 8.88]),
          quantized_dimension=None,
      ),
      dict(
          testcase_name="2d_tensor_1d_params",
          scale=np.array([0.1, 0.2]),
          zero_point=np.array([-1, 2], dtype=np.int8),
          tensor_data=np.array([[1, 2], [4, 5]]),
          quantized_dimension=0,
      ),
  )
  def test_fix_quantization_params_rank_succeed(
      self, scale, zero_point, tensor_data, quantized_dimension
  ):
    quant_params = qtyping.UniformQuantParams(
        num_bits=8,
        quantized_dimension=quantized_dimension,
        scale=scale,
        zero_point=zero_point,
        symmetric=False,
    )
    fixed_quant_params = uniform_quantize_tensor.fix_quantization_params_rank(
        tensor_data, quant_params
    )
    self.assertEqual(fixed_quant_params.scale.ndim, tensor_data.ndim)
    self.assertEqual(fixed_quant_params.zero_point.ndim, tensor_data.ndim)

  def test_fix_quantization_params_rank_scalar_tensor_1d_params_raise(self):
    quant_params = qtyping.UniformQuantParams(
        num_bits=8,
        quantized_dimension=None,
        scale=np.array([0.1, 0.5]),
        zero_point=np.array([-1, -2], dtype=np.int8),
        symmetric=False,
    )
    error_message = (
        "Scale and zero_point must contain single element for scalar tensor."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      uniform_quantize_tensor.fix_quantization_params_rank(
          np.array(6.66), quant_params
      )

  @parameterized.parameters(
      (
          [-3.0, 1.3, 2.4, 16.0],
          [0.12598425],
          [0],
          8,
          False,
          [-24, 10, 19, 127],
      ),
      (
          [-3.0, 1.3, 2.4, 16.0],
          [1.2666667],
          [-6],
          4,
          False,
          [-8, -5, -4, 7],
      ),
      (
          [-3.0, 1.3, 2.4, 16.0],
          [1.2666667],
          [-6],
          4,
          True,
          [-7, -5, -4, 7],
      ),
  )
  def test_uniform_quantize(
      self, tensor, scale, zero_points, num_bits, symmetric, expected_tensor
  ):
    quant_params = qtyping.UniformQuantParams(
        quantized_dimension=0,
        num_bits=num_bits,
        scale=np.array(scale),
        zero_point=np.array(zero_points),
        symmetric=symmetric,
    )

    quantized_tensor = uniform_quantize_tensor.uniform_quantize(
        np.array(tensor), quant_params
    )

    self.assertSequenceAlmostEqual(expected_tensor, quantized_tensor)

  def test_uniform_quantize_wrong_shape(self):
    tensor = [-3.0, 1.3, 2.4, 16.0]

    error_message = "scale and zero_point must have the same shape."
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      uniform_quantize_tensor.uniform_quantize(
          np.array(tensor),
          qtyping.UniformQuantParams(
              quantized_dimension=0,
              num_bits=4,
              scale=np.array([[[1.2666667]]]),
              zero_point=np.array([[-6]]),
              symmetric=True,
          ),
      )

    error_message = "Ranks of scales"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      uniform_quantize_tensor.uniform_quantize(
          np.array(tensor),
          qtyping.UniformQuantParams(
              quantized_dimension=0,
              num_bits=4,
              scale=np.array([[1.2666667]]),
              zero_point=np.array([[-6]]),
              symmetric=True,
          ),
      )

  @parameterized.parameters(
      (
          8,
          [-24, 10, 19, 127],
          [0.12598425],
          [0],
          [-3.023622, 1.2598425, 2.3937008, 16.0],
      ),
      (
          4,
          [-8, -5, -4, 7],
          [1.2666667],
          [-6],
          [-2.5333335, 1.2666668, 2.5333335, 16.466667],
      ),
  )
  def test_uniform_dequantize(
      self,
      num_bits,
      quantized_tensor,
      scale,
      zero_points,
      expected_output_tensor,
  ):
    quant_params = qtyping.UniformQuantParams(
        quantized_dimension=0,
        num_bits=num_bits,
        scale=np.array(scale),
        zero_point=np.array(zero_points),
        symmetric=False,
    )

    dequantized_tensor = uniform_quantize_tensor.uniform_dequantize(
        np.array(quantized_tensor), quant_params
    )

    self.assertSequenceAlmostEqual(
        expected_output_tensor, dequantized_tensor, places=4
    )

  def test_uniform_dequantize_wrong_shape(self):
    tensor = [-3.0, 1.3, 2.4, 16.0]

    error_message = "scale and zero_point must have the same shape."
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      uniform_quantize_tensor.uniform_dequantize(
          np.array(tensor),
          qtyping.UniformQuantParams(
              quantized_dimension=0,
              num_bits=4,
              scale=np.array([[[1.2666667]]]),
              zero_point=np.array([[-6]]),
              symmetric=True,
          ),
      )

    error_message = "Ranks of scales"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      uniform_quantize_tensor.uniform_dequantize(
          np.array(tensor),
          qtyping.UniformQuantParams(
              quantized_dimension=0,
              num_bits=4,
              scale=np.array([[1.2666667]]),
              zero_point=np.array([[-6]]),
              symmetric=True,
          ),
      )

  @parameterized.parameters(
      (8, 8, True, True), (8, 4, False, True), (16, 8, True, False)
  )
  def test_quantize_bias_tensor(
      self,
      activation_num_bits,
      weight_num_bits,
      symmetric_weights,
      channelwise_weight,
  ):
    input_quant_config = qtyping.UniformQuantParams(
        scale=np.array([0.8]),
        zero_point=np.array([10]),
        num_bits=activation_num_bits,
        symmetric=False,
        quantized_dimension=None,
    )
    weight_scale, weight_zp = [0.1], [-1]
    num_channels, quantized_dimension = 1, None
    if channelwise_weight:
      num_channels = 2
      weight_scale = weight_scale * num_channels
      weight_zp = weight_zp * num_channels
      quantized_dimension = 0

    weight_quant_config = qtyping.UniformQuantParams(
        quantized_dimension=0,
        num_bits=weight_num_bits,
        scale=np.array(weight_scale, dtype=np.float32),
        zero_point=np.array(weight_zp, dtype=np.int8),
        symmetric=symmetric_weights,
    )
    bias_tensor_data = np.array([66.0, 88.0])

    bias_quant_config = uniform_quantize_tensor.symmetric_quantize_bias_tensor(
        bias_tensor_data,
        input_quant_config,
        weight_quant_config,
    )
    bias_num_bits = 32 if activation_num_bits == 8 else 64
    self.assertEqual(bias_quant_config.num_bits, bias_num_bits)
    # Alwasys a 1D array
    self.assertLen(bias_quant_config.scale.shape, 1)
    self.assertLen(bias_quant_config.zero_point.shape, 1)

    self.assertLen(bias_quant_config.scale, num_channels)
    effective_scale = input_quant_config.scale[0] * weight_quant_config.scale[0]
    self.assertEqual(bias_quant_config.scale[0], effective_scale)
    self.assertEqual(bias_quant_config.zero_point[0], 0)  # Always symmetric
    self.assertEqual(bias_quant_config.symmetric, True)
    self.assertEqual(bias_quant_config.quantized_dimension, quantized_dimension)

    # Check quantized content
    dequantized_bias = uniform_quantize_tensor.uniform_dequantize(
        bias_quant_config.quantized_data, bias_quant_config
    )
    self.assertSequenceAlmostEqual(
        list(dequantized_bias.flatten()), list(bias_tensor_data), places=5
    )
    expected_quantized_data = uniform_quantize_tensor.uniform_quantize(
        bias_tensor_data, bias_quant_config
    )
    self.assertSequenceEqual(
        list(expected_quantized_data.flatten()),
        list(bias_quant_config.quantized_data.flatten()),  # pytype: disable=attribute-error
    )

  @parameterized.parameters((8, True), (16, False))
  def test_tensor_zp_scale_from_min_max(self, num_bits, symmetric):
    min_val = np.min(self._test_data, keepdims=True)
    max_val = np.max(self._test_data, keepdims=True)

    zp, scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
        min_val, max_val, num_bits, symmetric
    )
    self.assertEqual(zp.shape, scale.shape)
    max_q = 2**num_bits / 2 - 1
    calculated_max = scale[0] * (max_q - zp[0])
    self.assertAlmostEqual(calculated_max, max_val, delta=1e-3)
    min_q = -(2**num_bits) / 2
    if symmetric:
      min_q += 1
    calculated_min = scale[0] * (min_q - zp[0])
    if symmetric:
      self.assertAlmostEqual(calculated_min, -max_val, delta=1e-3)
    else:
      # Range has to be extended to include zero.
      self.assertEqual(calculated_min, 0)

  @parameterized.parameters(
      # number of bits, is_symmetric, max bound of the quantized range.
      (4, True, 7),
      (8, False, 255),
    )
  def test_tensor_zp_scale_from_min_max_with_clipping(
      self, num_bits, symmetric, quantized_bound
  ):
    min_val = np.array([[1.0]])
    max_val = np.array([[5.0]])
    clipping_values = np.array([4.0])
    zp, scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
        min_val, max_val, num_bits, symmetric, clipping_values
    )
    expected_scale = clipping_values / quantized_bound

    with self.subTest(name="CheckShapes"):
      self.assertEqual(zp.shape, scale.shape)
      self.assertEqual(zp.shape, (1, 1))

    if symmetric:
      with self.subTest(name="CheckSymmetricZpValue"):
        self.assertEqual(zp[0], 0)

    with self.subTest(name="CheckScaleValue"):
      self.assertEqual(scale[0], expected_scale)


if __name__ == "__main__":
  googletest.main()
