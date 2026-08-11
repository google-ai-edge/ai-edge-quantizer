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

"""Unit tests for OSCAR graph conditioner (Part 2: Upstream Folding & In-Place Patching)."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import oscar_conditioner
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_OpName = qtyping.TFLOperationName


def _calibrate_with_oscar():
  mu2 = np.tile(np.array([100.0, 1.0], dtype=np.float32), 16)
  return {
      "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd": {
          "mu2": mu2
      },
  }


class OscarConditionerFoldTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_utils.get_path_to_datafile(
        "../tests/models/conv_fc_mnist.tflite"
    )
    self.calibration_data = (
        tfl_interpreter_utils.create_random_normal_input_data(self.model_path)
    )
    self.calibration_result = _calibrate_with_oscar()

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

  def test_consumer_and_producer_edits_compose_on_shared_weights(self):
    conditioned, report = oscar_conditioner.condition_model(
        self.model_path, self.calibration_result
    )
    by_name = {s.tensor_name: s for s in report.sites}
    dense = by_name[
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    ]
    self.assertEqual(dense.status, oscar_conditioner.SiteStatus.FOLDED)
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
        model, subgraph, 0, min_input_channels=2000
    )
    self.assertEmpty(empty_sites)

    # Test _weight_views
    site = list(sites.values())[0]
    views = oscar_conditioner._weight_views(model, subgraph, site)
    self.assertLen(views, 1)
    self.assertEqual(views[0].shape[1], site.in_channels)

    # Test _compute_site_scales produces valid scales
    mu2 = np.tile(np.array([100.0, 1.0], dtype=np.float32), 784)
    s, ratio = oscar_conditioner._compute_site_scales(views, mu2, block_size=0)
    self.assertIsNotNone(s)
    self.assertGreater(ratio, 1.0)


if __name__ == "__main__":
  absltest.main()
