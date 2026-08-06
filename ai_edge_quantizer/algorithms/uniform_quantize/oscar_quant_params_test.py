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

from collections.abc import Sequence
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import oscar_quant_params as oscar
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from ai_edge_quantizer.utils import tfl_interpreter_utils

_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


def _create_tensor(name: str, shape: Sequence[int]) -> qtyping.TensorT:
  tensor = qtyping.TensorT()
  tensor.name = name.encode('utf-8')
  tensor.shape = list(shape)
  tensor.quantization = None
  return tensor


def _create_op(
    inputs: Sequence[int], outputs: Sequence[int]
) -> qtyping.OperatorT:
  op = qtyping.OperatorT()
  op.inputs = list(inputs)
  op.outputs = list(outputs)
  return op


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

  def test_fc_channelwise_shapes_and_reference_equivalence(self):
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
    self.assertEqual(quant_params.scale.shape, (16, 1))
    self.assertEqual(quant_params.quantized_dimension, 0)
    self.assertEqual(quant_params.block_size, 0)
    np.testing.assert_array_equal(
        quant_params.zero_point, np.zeros((16, 1), dtype=np.int32)
    )
    assert quant_params.quantized_data is not None
    self.assertEqual(quant_params.quantized_data.dtype, np.int8)
    self.assertGreaterEqual(quant_params.quantized_data.min(), -8)
    self.assertLessEqual(quant_params.quantized_data.max(), 7)

  def test_fc_blockwise_shapes_and_reference_equivalence(self):
    w, mu2 = self._fc_problem(out_ch=8, in_ch=64)
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.FULLY_CONNECTED),
        _TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.BLOCKWISE_32,
        ),
        w,
        _tensor_qsv(mu2),
    )
    self.assertEqual(quant_params.scale.shape, (8, 2))
    self.assertEqual(quant_params.block_size, 32)
    assert quant_params.quantized_data is not None
    self.assertEqual(quant_params.quantized_data.shape, w.shape)

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
    w, _ = self._fc_problem()
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.FULLY_CONNECTED),
        _TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        w,
        None,
    )
    assert quant_params.quantized_data is not None

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


def _get_calibration_data(num_samples: int = 64):
  rng = np.random.default_rng(66)
  samples = [
      {'conv2d_input': rng.uniform(size=(1, 28, 28, 1)).astype(np.float32)}
      for _ in range(num_samples)
  ]
  return {tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: samples}


if __name__ == '__main__':
  absltest.main()
