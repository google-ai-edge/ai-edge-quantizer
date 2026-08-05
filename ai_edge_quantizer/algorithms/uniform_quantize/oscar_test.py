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
import pathlib
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.algorithms.uniform_quantize import oscar
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from ai_edge_quantizer.utils import qsv_utils
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig

# The pure-NumPy reference implementation lives at the repo root; the
# equivalence tests pin this module's numerics to it.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
try:
  import oscar_quantizer as oscar_reference  # pylint: disable=g-import-not-at-top,g-bad-import-order  # pytype: disable=import-error

  _HAS_REFERENCE = True
except ImportError:
  _HAS_REFERENCE = False


def _create_tensor(name: str, shape: Sequence[int]) -> qtyping.TensorT:
  tensor = qtyping.TensorT()
  tensor.name = name.encode("utf-8")
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


def _tensor_qsv(mu2) -> dict:
  return {"activation_tensor_qsv": {"mu2": mu2, "num_samples": 128}}


def _dequantize(quant_params: qtyping.UniformQuantParams) -> np.ndarray:
  return uniform_quantize_tensor.uniform_dequantize(
      quant_params.quantized_data, quant_params
  )


class OscarCalibrationTest(parameterized.TestCase):

  def test_calibrate_computes_correct_mu2(self):
    input_tensor = _create_tensor("input", shape=(1, 2, 3))
    output_tensor = _create_tensor("output", shape=(1, 2, 3))
    op = _create_op(inputs=[0], outputs=[1])
    graph_info = qtyping.GraphInfo(
        subgraph_tensors=[input_tensor, output_tensor], buffers=[]
    )
    x = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    tensor_content_map = {
        "input": x,
        "output": 2.0 * x,
    }
    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_name",
            side_effect=["input", "output"],
            autospec=True,
            spec_set=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_data",
            return_value=None,
            autospec=True,
            spec_set=True,
        )
    )
    qsvs = oscar.calibrate(op, graph_info, tensor_content_map)

    self.assertIn("input", qsvs)
    self.assertAlmostEqual(qsvs["input"]["min"].item(), 1.0)
    self.assertAlmostEqual(qsvs["input"]["max"].item(), 6.0)
    # mu2 = mean over rows of x^2, per channel.
    np.testing.assert_allclose(
        qsvs["input"]["mu2"], np.array([8.5, 14.5, 22.5])
    )
    self.assertEqual(qsvs["input"]["num_samples"], 2)
    np.testing.assert_allclose(
        qsvs["output"]["mu2"], 4.0 * np.array([8.5, 14.5, 22.5])
    )

  def test_qsv_merge_matches_direct_computation(self):
    rng = np.random.default_rng(0)
    batches = [rng.normal(size=(4, 5, 8)) for _ in range(3)]
    qsv = {}
    for x in batches:
      flat = x.reshape(-1, 8)
      new_qsv = {
          "min": np.min(x).reshape((1, 1, 1)),
          "max": np.max(x).reshape((1, 1, 1)),
          "mu2": np.mean(flat * flat, axis=0),
          "num_samples": flat.shape[0],
      }
      qsv = qsv_utils.oscar_and_moving_average_update(qsv, new_qsv)
    ref = np.concatenate([x.reshape(-1, 8) for x in batches])
    np.testing.assert_allclose(qsv["mu2"], (ref**2).mean(axis=0))
    self.assertEqual(qsv["num_samples"], 60)


class OscarQuantParamsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._rng = np.random.default_rng(0)

  def _fc_problem(self, out_ch=16, in_ch=64):
    w = self._rng.normal(size=(out_ch, in_ch)).astype(np.float32)
    w[:, :4] *= 20.0
    mu2 = np.exp(self._rng.normal(size=in_ch))
    mu2[4:8] *= 100.0
    return w, mu2

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

    if not _HAS_REFERENCE:
      self.skipTest("oscar_quantizer.py reference not found at repo root")
    ref = oscar_reference.oscar_quantize(
        w, "fully_connected", mu2, num_bits=4, block_size=0
    )
    np.testing.assert_allclose(
        _dequantize(quant_params), ref.w_fakequant, atol=1e-6
    )
    np.testing.assert_array_equal(quant_params.quantized_data, ref.codes)

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

    if not _HAS_REFERENCE:
      self.skipTest("oscar_quantizer.py reference not found at repo root")
    ref = oscar_reference.oscar_quantize(
        w, "fully_connected", mu2, num_bits=4, block_size=32
    )
    # The reference emulates AEQ's blockwise float16 scale rounding, so
    # scales and codes must agree exactly.
    np.testing.assert_allclose(quant_params.scale, ref.scale, rtol=1e-7)
    np.testing.assert_array_equal(quant_params.quantized_data, ref.codes)

  def test_conv2d_channelwise_shapes_and_reference_equivalence(self):
    w = self._rng.normal(size=(16, 3, 3, 32)).astype(np.float32)
    w[:, :, :, :2] *= 15.0
    mu2 = np.exp(self._rng.normal(size=32))
    mu2[2:5] *= 100.0
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.CONV_2D),
        _TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        w,
        _tensor_qsv(mu2),
    )
    self.assertEqual(quant_params.scale.shape, (16, 1, 1, 1))
    self.assertEqual(quant_params.quantized_dimension, 0)

    if not _HAS_REFERENCE:
      self.skipTest("oscar_quantizer.py reference not found at repo root")
    ref = oscar_reference.oscar_quantize(w, "conv_2d", mu2, num_bits=4)
    np.testing.assert_allclose(
        _dequantize(quant_params), ref.w_fakequant, atol=1e-6
    )

  def test_depthwise_conv2d_channelwise_shapes(self):
    w = self._rng.normal(size=(1, 3, 3, 24)).astype(np.float32)
    quant_params = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.DEPTHWISE_CONV_2D),
        _TensorQuantConfig(
            num_bits=8,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        w,
        _tensor_qsv(np.ones(24)),
    )
    self.assertEqual(quant_params.scale.shape, (1, 1, 1, 24))
    self.assertEqual(quant_params.quantized_dimension, 3)
    assert quant_params.quantized_data is not None
    self.assertEqual(quant_params.quantized_data.shape, w.shape)

  def test_embedding_lookup_ignores_meaningless_mu2(self):
    w = self._rng.normal(size=(50, 16)).astype(np.float32)
    # Embedding op input is token ids; whatever mu2 was collected for it
    # must not influence the result.
    params_with_ids_mu2 = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.EMBEDDING_LOOKUP),
        _TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        w,
        _tensor_qsv(np.array([12345.0])),  # garbage stats, wrong size too
    )
    params_without = oscar.get_tensor_quant_params(
        _op_info(_TFLOpName.EMBEDDING_LOOKUP),
        _TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        w,
        None,
    )
    assert params_with_ids_mu2.quantized_data is not None
    assert params_without.quantized_data is not None
    np.testing.assert_array_equal(
        params_with_ids_mu2.quantized_data, params_without.quantized_data
    )
    self.assertEqual(params_with_ids_mu2.scale.shape, (50, 1))

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
    # The optimal bound never exceeds the max-abs bound.
    self.assertLessEqual(
        float(quant_params.scale.max()), float(np.abs(w).max()) / 127 + 1e-9
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
    if not _HAS_REFERENCE:
      self.skipTest("oscar_quantizer.py reference not found at repo root")
    ref = oscar_reference.oscar_quantize(
        w, "fully_connected", None, num_bits=4, block_size=0
    )
    np.testing.assert_allclose(
        _dequantize(quant_params), ref.w_fakequant, atol=1e-6
    )

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

    # Plain max-abs (RTN) at the same granularity.
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
      return float(((diff * diff) * mu2[None, :]).sum())

    self.assertLess(weighted_err(dequant), weighted_err(rtn_dequant))

  def test_asymmetric_weight_config_raises(self):
    w, mu2 = self._fc_problem()
    with self.assertRaisesRegex(ValueError, "symmetric"):
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
    with self.assertRaisesRegex(ValueError, "channels"):
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
        tensor_qsv={"min": np.array([[-1.0]]), "max": np.array([[2.0]])},
    )
    self.assertIsNone(quant_params.quantized_data)


def _get_calibration_data(num_samples: int = 64):
  rng = np.random.default_rng(66)
  samples = [
      {"conv2d_input": rng.uniform(size=(1, 28, 28, 1)).astype(np.float32)}
      for _ in range(num_samples)
  ]
  return {tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: samples}


class OscarEndToEndTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.float_model_path = test_utils.get_path_to_datafile(
        "../../tests/models/conv_fc_mnist.tflite"
    )
    self._quantizer = quantizer.Quantizer(self.float_model_path)

  @parameterized.named_parameters(
      ("int8_drq", 8, qtyping.ComputePrecision.INTEGER, False),
      ("int8_weight_only", 8, qtyping.ComputePrecision.FLOAT, True),
      ("int4_drq", 4, qtyping.ComputePrecision.INTEGER, False),
  )
  def test_oscar_e2e_conv_and_fc(
      self, num_bits, compute_precision, explicit_dequantize
  ):
    for op_name in (_TFLOpName.FULLY_CONNECTED, _TFLOpName.CONV_2D):
      self._quantizer.update_quantization_recipe(
          regex=".*",
          operation_name=op_name,
          algorithm_key=quantizer.AlgorithmName.OSCAR,
          op_config=qtyping.OpQuantizationConfig(
              weight_tensor_config=_TensorQuantConfig(
                  num_bits=num_bits,
                  symmetric=True,
                  granularity=qtyping.QuantGranularity.CHANNELWISE,
              ),
              compute_precision=compute_precision,
              explicit_dequantize=explicit_dequantize,
          ),
      )
    self.assertTrue(self._quantizer.need_calibration)
    calibration_result = self._quantizer.calibrate(_get_calibration_data())
    result = self._quantizer.quantize(calibration_result)
    self.assertLess(len(result.quantized_model), 60000)

    comparison_result = self._quantizer.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE]
    )
    all_results = comparison_result.get_all_tensor_results()
    output_mse = all_results["StatefulPartitionedCall:0"]["mse"]
    self.assertLess(output_mse, 1e-2 if num_bits == 4 else 1e-4)


if __name__ == "__main__":
  absltest.main()
