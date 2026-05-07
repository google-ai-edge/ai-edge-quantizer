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

import argparse
import pathlib
import tempfile

from absl.testing import absltest
from absl.testing import parameterized

from ai_edge_litert.tools import mmap_utils
from ai_edge_quantizer import aeq
from ai_edge_quantizer import model_validator
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.utils import litertlm_utils
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils
from ai_edge_quantizer.utils import validation_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile(".")
RECIPE_PREFIX_PATH = pathlib.Path(TEST_DATA_PREFIX_PATH) / "recipes"


class AeqTest(parameterized.TestCase):

  def _validate_quantized_model(
      self,
      float_model_buffer: qtyping.BufferType,
      quantized_model_buffer: qtyping.BufferType,
  ):
    # Create some random test data.
    test_data = tfl_interpreter_utils.create_random_normal_input_data(
        float_model_buffer, num_samples=10
    )

    # Run the evaluation.
    result = model_validator.compare_model(
        float_model_buffer,
        quantized_model_buffer,
        test_data,
        error_metric="mse",
        compare_fn=validation_utils.get_validation_func("mse"),
        validate_output_tensors_only=True,
    )

    # Verify that the results are indeed reasonable.
    for mse in result.get_all_tensor_results().values():
      self.assertLess(mse, 1e-4)

  @parameterized.named_parameters(
      (
          "from_recipe_file",
          str(RECIPE_PREFIX_PATH / "dynamic_wi8_afp32_recipe.json"),
      ),
      ("from_recipe_name", "dynamic_wi8_afp32"),
  )
  def test_quantize_and_validate_single_tflite_file(self, recipe_file: str):
    model_file = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / "tests/models/conv_fc_mnist.tflite"
    )

    with tempfile.TemporaryDirectory() as output_dir:
      self.assertEqual(
          aeq.main(
              argparse.Namespace(
                  model_file=model_file,
                  recipe=recipe_file,
                  output_dir=output_dir,
                  overwrite_outputs=False,
              )
          ),
          0,
      )

      model_basename = pathlib.Path(model_file).stem
      recipe_basename = pathlib.Path(recipe_file).stem
      output_filename = f"{model_basename}_{recipe_basename}.tflite"
      output_path = str(pathlib.Path(output_dir) / output_filename)

      self.assertTrue(pathlib.Path(output_path).exists())

      original_size = pathlib.Path(model_file).stat().st_size
      quantized_size = pathlib.Path(output_path).stat().st_size

      # Check whether the file size of the quantized model is roughly 4X smaller
      # than that of the float model
      self.assertLess(quantized_size, original_size * 0.35)

      # Validate the quantized model.
      self._validate_quantized_model(
          mmap_utils.get_file_contents(model_file),
          mmap_utils.get_file_contents(output_path),
      )

  @parameterized.named_parameters(
      (
          "litertlm_recipe_file",
          str(RECIPE_PREFIX_PATH / "dynamic_wi8_afp32_litertlm_recipe.json"),
          None,
      ),
      (
          "default_recipe_file",
          None,
          str(RECIPE_PREFIX_PATH / "dynamic_wi8_afp32_recipe.json"),
      ),
  )
  def test_quantize_and_validate_tflite_models_in_litertlm_file(
      self, litertlm_recipe_path: str | None, default_recipe_path: str | None
  ):
    model_file = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / "tests/models/conv_fc_mnist.litertlm"
    )

    with tempfile.TemporaryDirectory() as output_dir:
      self.assertEqual(
          aeq.main(
              argparse.Namespace(
                  model_file=model_file,
                  litertlm_recipe=litertlm_recipe_path,
                  recipe=default_recipe_path,
                  output_dir=output_dir,
                  overwrite_outputs=False,
              )
          ),
          0,
      )

      model_basename = pathlib.Path(model_file).stem
      recipe_basename = pathlib.Path(
          litertlm_recipe_path or default_recipe_path
      ).stem
      output_filename = f"{model_basename}_{recipe_basename}.litertlm"
      output_path = str(pathlib.Path(output_dir) / output_filename)

      self.assertTrue(pathlib.Path(output_path).exists())

      original_size = pathlib.Path(model_file).stat().st_size
      quantized_size = pathlib.Path(output_path).stat().st_size

      # Check whether the file size of the quantized model is roughly 4X smaller
      # than that of the float model (keep in mind that the LiteRT-LM header
      # size did not change).
      self.assertLess(quantized_size, original_size * 0.36)

      # Check whether the system and section metadata match.
      original_litertlm = litertlm_utils.LiteRTLMFile(model_file)
      quantized_litertlm = litertlm_utils.LiteRTLMFile(output_path)
      self.assertEqual(
          original_litertlm.get_system_metadata(),
          quantized_litertlm.get_system_metadata(),
      )
      self.assertEqual(
          original_litertlm.get_section_metadata(0),
          quantized_litertlm.get_section_metadata(0),
      )

      # Validate the quantized model.
      self._validate_quantized_model(
          original_litertlm.get_section_buffer(0),
          quantized_litertlm.get_section_buffer(0),
      )


if __name__ == "__main__":
  absltest.main()
