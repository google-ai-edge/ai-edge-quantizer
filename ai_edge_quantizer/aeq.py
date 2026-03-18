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

import argparse
from collections.abc import Sequence
import pathlib
import sys

import os
import io
from ai_edge_quantizer import quantizer


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
    overwrite_outputs: Output files overwrite exisiting files without requiring
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

  qt = quantizer.Quantizer(model_file)

  qt.load_quantization_recipe(recipe_file)

  qt.quantize(serialize_to_path=output_file_path)


def parse_args(args: Sequence[str]) -> argparse.Namespace:
  """Parses command-line arguments.

  Args:
    args: A list of strings to parse. If None, sys.argv is used.

  Returns:
    An argparse.Namespace containing the parsed arguments,
  """
  parser = argparse.ArgumentParser(
      description="Quantize TFLite models using a quantization recipe.",
  )
  parser.add_argument(
      "--model_file",
      required=True,
      help="Path to the .tflite file to be quantized.",
  )
  parser.add_argument(
      "--recipe_file",
      required=True,
      help="Path to the .json file with the quantization recipe.",
  )
  parser.add_argument(
      "--output_dir",
      required=True,
      help="Path to the directory in which to save the quantized model(s).",
  )
  parser.add_argument(
      "--overwrite_outputs",
      action="store_true",
      help="Overwrite exisiting output files without requesting user input.",
  )
  return parser.parse_args(args[1:])


def main(parsed_args: argparse.Namespace):
  quantize_model(
      model_file=parsed_args.model_file,
      recipe_file=parsed_args.recipe_file,
      output_dir=parsed_args.output_dir,
      overwrite_outputs=parsed_args.overwrite_outputs,
  )

# Wrapper to play nicely with uv tool install ai-edge-quantizer
def cli_main():
  main(parse_args(sys.argv))


if __name__ == "__main__":
  main(parse_args(sys.argv))
