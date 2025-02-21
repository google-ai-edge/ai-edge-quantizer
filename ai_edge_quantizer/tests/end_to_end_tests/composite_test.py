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

"""E2E tests for the quantizer for model with add."""

import json
import pathlib

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_RNG = np.random.default_rng(42)

_SHAPE = (
    32,
    32,
)


def _get_calibration_data(num_inputs, num_samples: int = 128):
  data = []
  for _ in range(num_samples):
    data.append({
        f'arg{i}': _RNG.uniform(size=_SHAPE).astype(np.float32)
        for i in range(num_inputs)
    })
  return {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: data,
  }


class CompositeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    model = test_utils.get_path_to_datafile('../models/simple_composite.tflite')
    self._quantizer = quantizer.Quantizer(model)

  @property
  def quantizer(self) -> quantizer.Quantizer:
    return self._quantizer

  @property
  def output_tolerance(self) -> float:
    return 1e-4

  @property
  def output_name(self) -> str:
    return 'output'

  def set_recipe(self, path: str):
    recipe_path = test_utils.get_path_to_datafile(path)
    recipe_json = json.load(pathlib.Path(recipe_path).open('r'))
    recipe_json[0]['op_config']['skip_checks'] = True
    self.quantizer.load_quantization_recipe(recipe_json)
    print(self.quantizer.get_quantization_recipe())

  @parameterized.parameters(
      '../../recipes/default_a8w8_recipe.json',
      '../../recipes/default_a16w8_recipe.json',
  )
  def test_add_model_full_integer(self, recipe_path):
    self.set_recipe(recipe_path)
    self.assertTrue(self.quantizer.need_calibration)
    calibration_result = self.quantizer.calibrate(
        _get_calibration_data(num_inputs=2)
    )
    print(calibration_result)
    result = self.quantizer.quantize(calibration_result)
    q_model = tfl_flatbuffer_utils.read_model(result.quantized_model)
    for sg in q_model.subgraphs:
      for tensor in sg.tensors:
        self.assertIsNotNone(tensor.quantization)

    comparison = self.quantizer.validate(
        error_metrics='mse', test_data=_get_calibration_data(num_inputs=2)
    )
    comparison = comparison.get_all_tensor_results()
    output_mse = comparison[self.output_name]
    self.assertLess(output_mse, self.output_tolerance)


if __name__ == '__main__':
  googletest.main()
