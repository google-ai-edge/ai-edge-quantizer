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

from absl.testing import parameterized
import absl.testing.absltest as absltest
import numpy as np
import os

from ai_edge_litert.tools import flatbuffer_utils
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils
from ai_edge_litert import schema_py_generated as schema  # pylint:disable=g-direct-tensorflow-import

_RNG = np.random.default_rng(42)


# test_with_decomposition.mlir
# func @main(%arg0, %arg1) ->  {
#   %0 = composite "npu_call" %arg0, %arg1 {decomposition = @decomp}
#   return %0
# }
# func private @decomp(%arg0: , %arg1: ) ->  {
#   %0 = tfl.add %arg0, %arg1
#   return %0
# }

# test_with_kernel.mlir (subset)
# %12 = stablehlo.composite "scaled_dot_product_attention" ...


def _get_calibration_data(
    model: schema.ModelT,
    name_pref: str,
    subgraph_index: int = 0,
    num_samples: int = 2,
):
  sg: schema.SubGraphT = model.subgraphs[subgraph_index]
  inputs = [sg.tensors[i] for i in sg.inputs]

  data = []

  for _ in range(num_samples):
    samp_data = {}
    for i, inp in enumerate(inputs):
      inp: schema.TensorT = inp
      name = f'{name_pref}{i}'
      samp_data[name] = _RNG.uniform(size=inp.shape).astype(np.float32)

    data.append(samp_data)

  return {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: data,
  }


class CompositeTest(parameterized.TestCase):

  @property
  def output_tolerance(self) -> float:
    return 1e-4

  def assertNotQuantiezedType(self, tensor: schema.TensorT):
    self.assertIn(
        tensor.type, [schema.TensorType.FLOAT32, schema.TensorType.INT32]
    )

  @parameterized.parameters(
      '../../recipes/default_a8w8_recipe.json',
      '../../recipes/default_a16w8_recipe.json',
  )
  def test_composite_with_decomposition(self, recipe_path):
    model = test_utils.get_path_to_datafile('../models/simple_composite.tflite')
    qt = quantizer.Quantizer(model)

    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    with open(recipe_path, 'r') as f:
      recipe_json = json.load(f)
    qt.load_quantization_recipe(recipe_json)

    f_model = tfl_flatbuffer_utils.read_model(model)
    self.assertTrue(qt.need_calibration)
    calib_data = _get_calibration_data(f_model, 'arg')
    calibration_result = qt.calibrate(calib_data)

    result = qt.quantize(calibration_result)
    q_model = tfl_flatbuffer_utils.read_model(result.quantized_model)
    for sg in q_model.subgraphs:
      for tensor in sg.tensors:
        self.assertIsNotNone(tensor.quantization)

    test_data = _get_calibration_data(f_model, 'arg')
    comparison = qt.validate(error_metrics='mse', test_data=test_data)
    comparison = comparison.get_all_tensor_results()
    output_mse = comparison['output']
    self.assertLess(output_mse, self.output_tolerance)

  @parameterized.parameters(
      '../../recipes/default_a8w8_recipe.json',
      '../../recipes/default_a16w8_recipe.json',
  )
  def test_composite_with_kernel(self, recipe_path):
    model = test_utils.get_path_to_datafile('../models/sdpa_composite.tflite')
    qt = quantizer.Quantizer(model)

    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    with open(recipe_path, 'r') as f:
      recipe_json = json.load(f)
    qt.load_quantization_recipe(recipe_json)
    self.assertTrue(qt.need_calibration)

    f_model = tfl_flatbuffer_utils.read_model(model)
    calib_data = _get_calibration_data(f_model, 'args_')
    calibration_result = qt.calibrate(calib_data)

    result = qt.quantize(calibration_result)
    q_model = tfl_flatbuffer_utils.read_model(result.quantized_model)

    main_sg = q_model.subgraphs[0]
    for op in main_sg.operators:
      if 'COMPOSITE' in flatbuffer_utils.opcode_to_name(
          q_model, op.opcodeIndex
      ):
        for inp in op.inputs:
          tensor = main_sg.tensors[inp]
          self.assertNotQuantiezedType(tensor)
        for out in op.outputs:
          tensor = main_sg.tensors[out]
          self.assertNotQuantiezedType(tensor)

    sdpa_sg = q_model.subgraphs[1]
    for tensor in sdpa_sg.tensors:
      self.assertNotQuantiezedType(tensor)


if __name__ == '__main__':
  absltest.main()
