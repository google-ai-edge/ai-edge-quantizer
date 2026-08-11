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

"""Unit tests for OSCAR graph conditioner (Part 3: MUL Insertion & External Buffers)."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from ai_edge_quantizer import model_modifier
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.transformations import oscar_conditioner
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


def _op_codes(model_bytes):
  model = tfl_flatbuffer_utils.read_model(bytearray(model_bytes))
  return [
      model.operatorCodes[op.opcodeIndex].builtinCode
      for op in model.subgraphs[0].operators
  ]


def _calibrate_with_oscar(model_path, target_ops, calibration_data):
  qt = quantizer.Quantizer(model_path)
  for op in target_ops:
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=op,
        algorithm_key=getattr(quantizer.AlgorithmName, "OSCAR", "OSCAR"),
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=4,
                symmetric=True,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
            ),
            compute_precision=qtyping.ComputePrecision.INTEGER,
        ),
    )
  return qt.calibrate(calibration_data)


class OscarConditionerFoldTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_utils.get_path_to_datafile(
        "../tests/models/conv_fc_mnist.tflite"
    )
    self.calibration_data = (
        tfl_interpreter_utils.create_random_normal_input_data(self.model_path)
    )
    self.calibration_result = _calibrate_with_oscar(
        self.model_path, (_OpName.FULLY_CONNECTED,), self.calibration_data
    )

  def test_folds_fc_to_fc_site_and_stays_float_equivalent(self):
    conditioned, report = oscar_conditioner.condition_model(
        self.model_path, self.calibration_result
    )
    self.assertGreaterEqual(report.num_folded, 1)
    self.assertTrue(report.in_place_patched)
    worst = oscar_conditioner.verify_conditioned_model(
        self.model_path,
        bytes(conditioned),
        self.calibration_data,
        rel_tolerance=1e-4,
    )
    self.assertLess(worst, 1e-4)

  def test_fold_rewrites_producer_and_consumer_weights(self):
    conditioned, _ = oscar_conditioner.condition_model(
        self.model_path, self.calibration_result
    )
    orig_model = tfl_flatbuffer_utils.read_model(
        tfl_flatbuffer_utils.get_model_content(self.model_path)
    )
    cond_model = tfl_flatbuffer_utils.read_model(bytearray(conditioned))

    orig_buf = orig_model.subgraphs[0].tensors[2].buffer
    cond_buf = cond_model.subgraphs[0].tensors[2].buffer
    orig_data = tfl_flatbuffer_utils.get_tensor_data(
        orig_model.subgraphs[0].tensors[2], orig_model.buffers
    )
    cond_data = tfl_flatbuffer_utils.get_tensor_data(
        cond_model.subgraphs[0].tensors[2], cond_model.buffers
    )
    self.assertFalse(np.array_equal(orig_data, cond_data))
    self.assertEqual(orig_buf, cond_buf)

  def test_fold_patches_external_buffer_model_in_place(self):
    model = tfl_flatbuffer_utils.read_model(
        tfl_flatbuffer_utils.get_model_content(self.model_path)
    )
    packed = model_modifier._PackedBufferData(model, min_size_bytes=64)
    external_bytes = bytes(
        model_modifier.ModelModifier(model)._serialize_model(model, packed)
    )
    conditioned, report = oscar_conditioner.condition_model(
        external_bytes, self.calibration_result
    )
    self.assertTrue(report.in_place_patched)
    self.assertEqual(report.num_folded, 1)
    worst = oscar_conditioner.verify_conditioned_model(
        external_bytes,
        bytes(conditioned),
        self.calibration_data,
        rel_tolerance=1e-4,
    )
    self.assertLess(worst, 1e-4)

  def test_conditioned_model_quantizes_end_to_end(self):
    conditioned, _ = oscar_conditioner.condition_model(
        self.model_path,
        self.calibration_result,
        allow_mul_insertion=True,
        mul_min_gain=1.05,
    )
    qt = quantizer.Quantizer(bytes(conditioned))
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=_OpName.FULLY_CONNECTED,
        algorithm_key=getattr(quantizer.AlgorithmName, "OSCAR", "OSCAR"),
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=4,
                symmetric=True,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
            ),
            compute_precision=qtyping.ComputePrecision.INTEGER,
        ),
    )
    result = qt.quantize(qt.calibrate(self.calibration_data))
    self.assertLess(len(result.quantized_model), 60000)
    comparison = qt.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE],
        test_data=self.calibration_data,
    )
    output_mse = comparison.get_all_tensor_results()[
        "StatefulPartitionedCall:0"
    ]["mse"]
    self.assertLess(output_mse, 1e-2)


class OscarConditionerMulInsertionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_utils.get_path_to_datafile(
        "../tests/models/single_fc_bias.tflite"
    )
    self.calibration_data = (
        tfl_interpreter_utils.create_random_normal_input_data(self.model_path)
    )
    self.calibration_result = _calibrate_with_oscar(
        self.model_path, (_OpName.FULLY_CONNECTED,), self.calibration_data
    )

  def test_graph_input_site_uses_mul_fallback(self):
    conditioned, report = oscar_conditioner.condition_model(
        self.model_path,
        self.calibration_result,
        allow_mul_insertion=True,
        mul_min_gain=1.01,
    )
    self.assertEqual(report.num_mul_inserted, 1)
    self.assertEqual(report.num_folded, 0)
    self.assertFalse(report.in_place_patched)
    codes = _op_codes(conditioned)
    self.assertEqual(
        codes,
        [qtyping.BuiltinOperator.MUL, qtyping.BuiltinOperator.FULLY_CONNECTED],
    )
    worst = oscar_conditioner.verify_conditioned_model(
        self.model_path,
        bytes(conditioned),
        self.calibration_data,
        rel_tolerance=1e-4,
    )
    self.assertLess(worst, 1e-4)

  def test_site_discovery_and_helper_validation(self):
    utils = oscar_conditioner.tfl_flatbuffer_utils
    model = utils.read_model(utils.get_model_content(self.model_path))
    subgraph = model.subgraphs[0]

    # Test _is_float_const on float const weight tensor vs invalid index
    self.assertTrue(oscar_conditioner._is_float_const(model, subgraph, 2))
    self.assertFalse(oscar_conditioner._is_float_const(model, subgraph, 0))

    # Test _find_sites filtering by min_input_channels
    sites = oscar_conditioner._find_sites(
        model, subgraph, 0, min_input_channels=4
    )
    self.assertNotEmpty(sites)
    empty_sites = oscar_conditioner._find_sites(
        model, subgraph, 0, min_input_channels=100
    )
    self.assertEmpty(empty_sites)

    # Test _weight_views
    site = list(sites.values())[0]
    views = oscar_conditioner._weight_views(model, subgraph, site)
    self.assertLen(views, 1)
    self.assertEqual(views[0].shape[1], site.in_channels)

    # Test _compute_site_scales produces valid scales
    mu2 = np.tile(np.array([100.0, 1.0], dtype=np.float32), 4)
    s, ratio = oscar_conditioner._compute_site_scales(views, mu2, block_size=0)
    self.assertIsNotNone(s)
    self.assertGreater(ratio, 1.0)

  def test_mul_fallback_disabled_leaves_model_unchanged(self):
    conditioned, report = oscar_conditioner.condition_model(
        self.model_path, self.calibration_result, allow_mul_insertion=False
    )
    self.assertEqual(report.num_skipped, 1)
    self.assertEqual(
        _op_codes(conditioned), [qtyping.BuiltinOperator.FULLY_CONNECTED]
    )
    oscar_conditioner.verify_conditioned_model(
        self.model_path,
        bytes(conditioned),
        self.calibration_data,
        rel_tolerance=1e-6,
    )

  def test_missing_mu2_skips_site(self):
    _, report = oscar_conditioner.condition_model(
        self.model_path, {}, allow_mul_insertion=True
    )
    self.assertEqual(report.num_skipped, 1)

  def test_report_summary_is_printable(self):
    _, report = oscar_conditioner.condition_model(
        self.model_path,
        self.calibration_result,
        allow_mul_insertion=True,
        mul_min_gain=1.01,
    )
    summary = report.summary()
    self.assertIn("OSCAR conditioning", summary)
    self.assertIn("1 MUL-inserted", summary)

if __name__ == "__main__":
  absltest.main()
