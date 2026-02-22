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

import pathlib
import tempfile
from absl.testing import absltest
from absl.testing import flagsaver
from ai_edge_quantizer import aeq
from ai_edge_quantizer.utils import test_utils


class AeqTest(absltest.TestCase):

  def test_quantize(self):
    model_file = test_utils.get_path_to_datafile(
        "tests/models/conv_fc_mnist.tflite"
    )
    recipe_file = test_utils.get_path_to_datafile(
        "recipes/default_af32w8float_recipe.json"
    )

    with tempfile.TemporaryDirectory() as output_dir:
      with flagsaver.flagsaver(
          model_file=model_file, recipe_file=recipe_file, output_dir=output_dir
      ):
        aeq.main([])

      model_basename = pathlib.Path(model_file).stem
      recipe_basename = pathlib.Path(recipe_file).stem
      output_filename = f"{model_basename}_{recipe_basename}.tflite"
      output_path = str(pathlib.Path(output_dir) / output_filename)

      self.assertTrue(pathlib.Path(output_path).exists())

      original_size = pathlib.Path(model_file).stat().st_size
      quantized_size = pathlib.Path(output_path).stat().st_size

      # Check if file size of the quantized model is roughly 4X smaller than
      # the float model
      self.assertLess(quantized_size, original_size * 0.35)


if __name__ == "__main__":
  absltest.main()
