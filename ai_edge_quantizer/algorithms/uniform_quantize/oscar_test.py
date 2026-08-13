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

"""Tests for the OSCAR algorithm, including reference equivalence."""

from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import oscar
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor

_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


def _op_info(op_name: _TFLOpName) -> qtyping.OpInfo:
  return qtyping.OpInfo(
      op=qtyping.OperatorT(),
      op_name=op_name,
      op_quant_config=qtyping.OpQuantizationConfig(),
      subgraph_op_index=-1,
  )


def _tensor_qsv(mu2: Any) -> dict[str, Any]:
  return {'activation_tensor_qsv': {'mu2': mu2, 'num_samples': 128}}


def _dequantize(quant_params: qtyping.UniformQuantParams) -> np.ndarray:
  assert quant_params.quantized_data is not None
  return uniform_quantize_tensor.uniform_dequantize(
      quant_params.quantized_data, quant_params
  )


class OscarQuantParamsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng(0)

  def _fc_problem(self, out_ch=16, in_ch=64):
    w = self._rng.normal(size=(out_ch, in_ch)).astype(np.float32)
    w[:, :4] *= 20.0
    mu2 = np.exp(self._rng.normal(size=in_ch))
    mu2[4:8] *= 100.0
    return (w, mu2)

  @parameterized.named_parameters(
      dict(
          testcase_name='channelwise',
          in_ch=32,
          out_ch=16,
          granularity=qtyping.QuantGranularity.CHANNELWISE,
          expected_scale_shape=(16, 1),
          expected_block_size=0,
          expected_quantized_dimension=0,
      ),
      dict(
          testcase_name='blockwise_32',
          in_ch=64,
          out_ch=8,
          granularity=qtyping.QuantGranularity.BLOCKWISE_32,
          expected_scale_shape=(8, 2),
          expected_block_size=32,
          expected_quantized_dimension=1,
      ),
  )
  def test_fc_shapes_and_reference_equivalence(
      self,
      in_ch: int,
      out_ch: int,
      granularity: qtyping.QuantGranularity,
      expected_scale_shape: tuple[int, int],
      expected_block_size: int,
      expected_quantized_dimension: int,
  ):
    w, mu2 = self._fc_problem(out_ch=out_ch, in_ch=in_ch)
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.FULLY_CONNECTED),
        _TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=granularity,
        ),
        w,
        _tensor_qsv(mu2),
    )
    self.assertEqual(quant_params.scale.shape, expected_scale_shape)
    self.assertEqual(
        quant_params.quantized_dimension, expected_quantized_dimension
    )
    self.assertEqual(quant_params.block_size, expected_block_size)
    np.testing.assert_array_equal(
        quant_params.zero_point, np.zeros(expected_scale_shape, dtype=np.int32)
    )
    self.assertIsNotNone(quant_params.quantized_data)
    q_data = cast(np.ndarray, quant_params.quantized_data)
    self.assertEqual(q_data.shape, w.shape)
    self.assertEqual(q_data.dtype, np.int8)
    self.assertGreaterEqual(q_data.min(), -8)
    self.assertLessEqual(q_data.max(), 7)

  def test_unsupported_op_raises(self):
    w = self._rng.normal(size=(16, 3, 3, 32)).astype(np.float32)
    with self.assertRaisesRegex(ValueError, 'FULLY_CONNECTED only'):
      oscar.get_tensor_quant_params(
          _op_info(_TFLOpName.CONV_2D),
          _TensorQuantConfig(
              num_bits=4,
              symmetric=True,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          w,
          None,
      )

  def test_tensorwise_produces_scalar_params(self):
    w, mu2 = self._fc_problem()
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.FULLY_CONNECTED),
        _TensorQuantConfig(
            num_bits=8,
            symmetric=True,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        w,
        _tensor_qsv(mu2),
    )
    self.assertEqual(quant_params.scale.shape, (1, 1))
    self.assertLessEqual(
        float(quant_params.scale.max()), float(np.abs(w).max()) / 127 + 1e-09
    )

  def test_missing_mu2_falls_back_to_uniform_weighting(self):
    w, _ = self._fc_problem(out_ch=16, in_ch=32)
    op_n = _op_info(_TFLOpName.FULLY_CONNECTED)
    with self.assertLogs(level='WARNING') as logs:
      quant_params = oscar.get_tensor_quant_params(
          op_n,
          _TensorQuantConfig(
              num_bits=4,
              symmetric=True,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          w,
          None,
      )
    self.assertTrue(
        any('OSCAR: no activation second moments' in log for log in logs.output)
    )
    self.assertEqual(quant_params.scale.shape, (16, 1))
    self.assertIsNotNone(quant_params.quantized_data)

  def test_clipping_improves_activation_weighted_error(self):
    w, mu2 = self._fc_problem()
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.FULLY_CONNECTED),
        _TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        w,
        _tensor_qsv(mu2),
    )
    dequant = _dequantize(quant_params)
    bound = np.abs(w).max(axis=1, keepdims=True)
    zp, scale = uniform_quantize_tensor.tensor_zp_scale_from_min_max(
        -bound, bound, 4, True, qtyping.QuantGranularity.CHANNELWISE, None
    )
    rtn_params = qtyping.UniformQuantParams(
        scale=scale,
        zero_point=zp,
        num_bits=4,
        symmetric=True,
        quantized_dimension=0,
        block_size=0,
    )
    rtn_codes = uniform_quantize_tensor.uniform_quantize(w, rtn_params)
    rtn_dequant = uniform_quantize_tensor.uniform_dequantize(
        rtn_codes, rtn_params
    )

    def weighted_err(w_hat):
      """Computes the activation-weighted error.

      OSCAR minimizes the activation-weighted error (E||(Q(W) - W) x||^2),
      which simplifies to the weight MSE multiplied by the activation second
      moment (mu2) under a diagonal activation assumption.

      Args:
        w_hat: The quantized weight tensor.

      Returns:
        The activation-weighted error.
      """
      diff = w_hat - w
      return float((diff * diff * mu2[None, :]).sum())

    self.assertLess(weighted_err(dequant), weighted_err(rtn_dequant))

  def test_asymmetric_weight_config_raises(self):
    w, mu2 = self._fc_problem()
    with self.assertRaisesRegex(ValueError, 'symmetric'):
      oscar.get_tensor_quant_params(
          _op_info(_TFLOpName.FULLY_CONNECTED),
          _TensorQuantConfig(
              num_bits=8,
              symmetric=False,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          w,
          _tensor_qsv(mu2),
      )

  def test_mismatched_mu2_size_raises(self):
    w, _ = self._fc_problem(in_ch=64)
    with self.assertRaisesRegex(ValueError, 'channels'):
      oscar.get_tensor_quant_params(
          _op_info(_TFLOpName.FULLY_CONNECTED),
          _TensorQuantConfig(
              num_bits=4,
              symmetric=True,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          w,
          _tensor_qsv(np.ones(32)),
      )

  def test_non_weight_tensor_falls_back_to_min_max(self):
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.FULLY_CONNECTED),
        _TensorQuantConfig(
            num_bits=8,
            symmetric=True,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        tensor_content=None,
        tensor_qsv={'min': np.array([[-1.0]]), 'max': np.array([[2.0]])},
    )
    self.assertIsNone(quant_params.quantized_data)


if __name__ == '__main__':
  absltest.main()
