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

import os

from tensorflow.python.platform import googletest
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import recipe
from ai_edge_quantizer.utils import test_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('')


class RecipeTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = os.path.join(
        _TEST_DATA_PREFIX_PATH,
        'tests/models/single_conv2d_transpose_bias.tflite',
    )
    self._test_json_recipe_path = os.path.join(
        _TEST_DATA_PREFIX_PATH,
        'recipes/dynamic_wi8_afp32_recipe.json',
    )

  def test_quantization_from_dynamic_wi8_afp32_func_succeeds(self):
    qt = quantizer.Quantizer(self._test_model_path)
    qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    self.assertIsNone(qt._result.quantized_model)
    quant_result = qt.quantize()
    self.assertIsNotNone(quant_result.quantized_model)
    self.assertLess(
        len(quant_result.quantized_model),
        os.path.getsize(self._test_model_path),
    )

  def test_dynamic_wi8_afp32_func_and_json_matches(self):
    # Quantize with dynamic_wi8_afp32() from recipe module.
    qt_func = quantizer.Quantizer(self._test_model_path)
    qt_func.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    self.assertIsNone(qt_func._result.quantized_model)
    quant_result_from_func = qt_func.quantize()
    self.assertIsNotNone(quant_result_from_func.quantized_model)

    # Quantize with dynamic_wi8_afp32_recipe.json.
    qt_json = quantizer.Quantizer(self._test_model_path)
    qt_json.load_quantization_recipe(self._test_json_recipe_path)
    quant_result_from_json = qt_json.quantize()
    self.assertIsNotNone(quant_result_from_json.quantized_model)

    # Check if the recipes and quantized models match.
    self.assertEqual(
        quant_result_from_func.recipe,
        quant_result_from_json.recipe,
    )
    self.assertEqual(
        len(quant_result_from_func.quantized_model),
        len(quant_result_from_json.quantized_model),
    )


if __name__ == '__main__':
  googletest.main()
