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

"""Unit tests for reading/parsing LiteRT-LM files."""

import json
import os
import pathlib
import random

from absl.testing import absltest

import os
import io
from ai_edge_litert.internal import litertlm_core
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.utils import litertlm_utils
from ai_edge_quantizer.utils import test_utils


class LitertlmUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._litertlm_path = os.path.join(
        test_utils.get_path_to_datafile("../tests/models"),
        "test_tok_tfl_llm.litertlm",
    )

  def test_gets_model_type_from_litertlm(self):
    litertlm_file = litertlm_utils.LiteRTLMFile(self._litertlm_path)
    sections = litertlm_file.sections
    self.assertLen(sections, 4)
    self.assertIsNone(litertlm_file.get_model_type(0))
    self.assertEqual(litertlm_file.get_model_type(1), "tf_lite_prefill_decode")
    self.assertIsNone(litertlm_file.get_model_type(2))
    self.assertIsNone(litertlm_file.get_model_type(3))

  def test_reads_model_from_litertlm(self):
    litertlm_file = litertlm_utils.LiteRTLMFile(self._litertlm_path)
    self.assertIsNotNone(litertlm_file.read_model(1))

  def test_serializes_with_data_overrides(self):
    # Load the original file.
    litertlm_file = litertlm_utils.LiteRTLMFile(self._litertlm_path)

    # Re-serialize the file overriding two of its sections. The section sizes
    # are chosen to _not_ be integer multiples of the LiteRT-LM block size to
    # verify that they are padded correctly.
    output_path = self.create_tempfile("modified.litertlm").full_path
    overrides = {
        0: random.randbytes(10 * litertlm_core.BLOCK_SIZE + 1),
        2: random.randbytes(10 * litertlm_core.BLOCK_SIZE - 1),
    }
    litertlm_file.serialize(output_path, overrides)

    # Read the modified LiteRT-LM file.
    modified_litertlm_file = litertlm_utils.LiteRTLMFile(output_path)

    # Compare the sections.
    self.assertEqual(
        len(litertlm_file.sections), len(modified_litertlm_file.sections)
    )
    for section_id in range(len(litertlm_file.sections)):
      # Check that the key value pairs are the same.
      self.assertEqual(
          litertlm_file.get_section_metadata(section_id),
          modified_litertlm_file.get_section_metadata(section_id),
      )

      # Check whether the buffers match the original or override data.
      if buff := overrides.get(section_id):
        self.assertEqual(
            buff, modified_litertlm_file.get_section_buffer(section_id)
        )
      else:
        self.assertEqual(
            litertlm_file.get_section_buffer(section_id),
            modified_litertlm_file.get_section_buffer(section_id),
        )

  def test_resolve_litertlm_quant_recipe_files_correctly(self):
    # Load a single quantization recipe.
    recipe_file = os.path.join(
        test_utils.get_path_to_datafile("../recipes"),
        "dynamic_wi8_afp32_recipe.json",
    )
    with open(recipe_file, "r") as f:
      orig_recipe: qtyping.ModelQuantizationRecipe = json.load(f)

    # Create a litertlm recipe consisting of:
    #  * "raw": The recipe itself,
    #  * "basename": The name of the recipe file in the same directory,
    #  * "path": The full path of the recipe.
    orig_litertlm_recipe = {
        "raw": orig_recipe,
        "basename": pathlib.Path(recipe_file).name,
        "path": recipe_file,
    }
    output_dir = self.create_tempdir()
    os.mkdir(pathlib.Path(output_dir) / "recipes")
    with open(
        pathlib.Path(output_dir) / "recipes/dynamic_wi8_afp32_recipe.json",
        "w",
    ) as f:
      json.dump(orig_recipe, f)
    litertlm_recipe_file = str(
        pathlib.Path(output_dir) / "recipes/generated_litertlm_recipe.json"
    )
    with open(litertlm_recipe_file, "w") as f:
      json.dump(orig_litertlm_recipe, f)

    # Check that the litertlm default recipe resolves to the corresponding.
    self.assertIsNotNone(
        litertlm_recipe := litertlm_utils.resolve_litertlm_recipes(
            litertlm_recipe_file
        )
    )
    for key in orig_litertlm_recipe:
      self.assertCountEqual(orig_recipe, litertlm_recipe[key])


if __name__ == "__main__":
  absltest.main()
