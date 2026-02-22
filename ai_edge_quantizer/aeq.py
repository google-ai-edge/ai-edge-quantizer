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

"""A command-line tool to quantize TFLite models using a quantization recipe."""

import pathlib

from absl import app
from absl import flags

import os
from ai_edge_quantizer import quantizer


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_file", None, "Path to the .tflite file to be quantized."
)
flags.DEFINE_string(
    "recipe_file", None, "Path to the .json file with the quantization recipe."
)
flags.DEFINE_string(
    "output_dir",
    None,
    "Path to the directory to save the quantized model(s).",
)
flags.DEFINE_bool(
    "overwrite_outputs",
    False,
    "Outputs files overwrite exisiting files without requiring user input.",
)


def quantize_model(
    model_file: str,
    recipe_file: str,
    output_dir: str,
    overwrite_outputs: bool = False,
):
  """Quantizes a TFLite model using a recipe.

  Args:
    model_file: Path to the .tflite file to be quantized.
    recipe_file: Path to the .json file with the quantization recipe.
    output_dir: Path to the directory to save the quantized model(s).
    overwrite_outputs: Outputs files overwrite exisiting files without requiring
      user input.
  """

  model_basename = pathlib.Path(model_file).stem
  recipe_basename = pathlib.Path(recipe_file).stem
  output_filename = f"{model_basename}_{recipe_basename}.tflite"

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  output_file_path = str(pathlib.Path(output_dir) / output_filename)

  overwrite = overwrite_outputs
  if os.path.exists(output_file_path) and not overwrite:
    overwrite_input = input(
        f"Output file {output_file_path} already exists. Overwrite? (y/N): "
    )
    if overwrite_input.lower() != "y":
      print("Aborting.")
      return
    overwrite = True

  qt = quantizer.Quantizer(model_file)
  qt.load_quantization_recipe(recipe_file)
  qt.quantize().export_model(output_file_path, overwrite=overwrite)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.model_file is None:
    raise app.UsageError("Flag --model_file must be specified.")
  if FLAGS.recipe_file is None:
    raise app.UsageError("Flag --recipe_file must be specified.")
  if FLAGS.output_dir is None:
    raise app.UsageError("Flag --output_dir must be specified.")

  quantize_model(
      model_file=FLAGS.model_file,
      recipe_file=FLAGS.recipe_file,
      output_dir=FLAGS.output_dir,
      overwrite_outputs=FLAGS.overwrite_outputs,
  )


if __name__ == "__main__":
  app.run(main)
