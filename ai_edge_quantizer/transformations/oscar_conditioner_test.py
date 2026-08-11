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

"""Unit tests for OSCAR graph conditioner (Part 1: Solver, Structures & Harness)."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import oscar_conditioner
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_OpName = qtyping.TFLOperationName


def _calibrate_with_oscar():
  # Alternating high/low variance ensures OSCAR scale solver finds beneficial
  # scaling
  mu2 = np.tile(np.array([100.0, 1.0], dtype=np.float32), 4)
  return {"serving_default_input_2:0": {"mu2": mu2}}


class OscarConditionerBaseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_utils.get_path_to_datafile(
        "../tests/models/single_fc_bias.tflite"
    )
    self.calibration_data = (
        tfl_interpreter_utils.create_random_normal_input_data(self.model_path)
    )
    self.calibration_result = _calibrate_with_oscar()

  def test_report_summary_is_printable_and_folds_site(self):
    _, report = oscar_conditioner.condition_model(
        self.model_path, self.calibration_result
    )
    summary = report.summary()
    self.assertIn("OSCAR conditioning", summary)
    self.assertTrue(report.in_place_patched)
    self.assertGreaterEqual(report.num_folded, 1)
    self.assertTrue(
        any(
            s.status == oscar_conditioner.SiteStatus.FOLDED
            for s in report.sites
        )
    )

  def test_verify_conditioned_model_passes_on_valid_rewrite(self):
    conditioned, _ = oscar_conditioner.condition_model(self.model_path, {})
    worst = oscar_conditioner.verify_conditioned_model(
        self.model_path,
        bytes(conditioned),
        self.calibration_data,
        rel_tolerance=1e-3,
    )
    self.assertLess(worst, 1e-3)

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
        model, subgraph, 0, min_input_channels=1000
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


if __name__ == "__main__":
  absltest.main()
