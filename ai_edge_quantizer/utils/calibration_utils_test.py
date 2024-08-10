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
from tensorflow.python.platform import googletest
from ai_edge_quantizer.utils import calibration_utils


class CalibrationUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_smoothing_factor",
          smoothing_factor=0,
          expected_vals={"min": -1000, "max": 800},
      ),
      dict(
          testcase_name="one_smoothing_factor",
          smoothing_factor=1,
          expected_vals={"min": -10, "max": 8},
      ),
      dict(
          testcase_name="normal_smoothing_factor",
          smoothing_factor=0.99,
          expected_vals={"min": -19.9, "max": 15.92},
      ),
  )
  def test_update_tensor_qsv_moving_average(
      self, smoothing_factor, expected_vals
  ):
    old_qsv = {"min": -10, "max": 8}
    # Large values to mimic outlier.
    new_qsv = {"min": -1000, "max": 800}
    updated_qsv = calibration_utils.moving_average_update(
        old_qsv, new_qsv, smoothing_factor=smoothing_factor
    )
    self.assertAlmostEqual(updated_qsv["min"], expected_vals["min"])
    self.assertAlmostEqual(updated_qsv["max"], expected_vals["max"])

  @parameterized.named_parameters(
      dict(
          testcase_name="scalar",
          old_qsv={"min": -10, "max": 8},
          new_qsv={"min": -1000, "max": 1},
          expected_qsv={"min": -1000, "max": 8},
      ),
      dict(
          testcase_name="2darray",
          old_qsv={"min": [[-19], [20]], "max": [[21], [250]]},
          new_qsv={"min": [[-1000], [25]], "max": [[33], [100]]},
          expected_qsv={"min": [[-1000], [20]], "max": [[33], [250]]},
      ),
  )
  def test_update_tensor_qsv_min_max(self, old_qsv, new_qsv, expected_qsv):
    updated_qsv = calibration_utils.min_max_update(old_qsv, new_qsv)
    if isinstance(expected_qsv["min"], list):
      self.assertListEqual(list(updated_qsv["min"]), expected_qsv["min"])
      self.assertListEqual(list(updated_qsv["max"]), expected_qsv["max"])
    else:
      self.assertEqual(updated_qsv["min"], expected_qsv["min"])
      self.assertEqual(updated_qsv["max"], expected_qsv["max"])


if __name__ == "__main__":
  googletest.main()
