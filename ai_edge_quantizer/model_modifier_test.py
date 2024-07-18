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

"""Tests for model_modifier."""

import os
from tensorflow.python.platform import googletest
from absl.testing import parameterized
from ai_edge_quantizer import model_modifier
from ai_edge_quantizer import params_generator
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe_manager
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from tensorflow.lite.tools import flatbuffer_utils  # pylint: disable=g-direct-tensorflow-import

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('.')


class ModelModifierTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/conv_fc_mnist.tflite'
    )
    self._model_modifier = model_modifier.ModelModifier(self._model_path)
    self._model_buffer: bytearray = tfl_flatbuffer_utils.get_model_buffer(
        self._model_path
    )

  def test_process_constant_map(self):
    constant_size = self._model_modifier._process_constant_map(
        self._model_modifier._flatbuffer_model
    )
    self.assertEqual(constant_size, 202540)
    pass

  def test_modify_model(self):
    """Test modidfy Model."""
    recipe_manager_instance = recipe_manager.RecipeManager()
    params_generator_instance = params_generator.ParamsGenerator(
        self._model_path
    )
    global_recipe = [
        {
            'regex': '.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'weight_tensor_config': {
                    'dtype': qtyping.TensorDataType.INT,
                    'num_bits': 8,
                    'symmetric': False,
                    'channel_wise': True,
                },
                'execution_mode': qtyping.OpExecutionMode.WEIGHT_ONLY,
            },
        },
    ]
    recipe_manager_instance.load_quantization_recipe(global_recipe)
    tensor_quantization_params = (
        params_generator_instance.generate_quantization_parameters(
            recipe_manager_instance
        )
    )
    new_model_binary = self._model_modifier.modify_model(
        tensor_quantization_params
    )
    flatbuffer_utils.convert_bytearray_to_object(new_model_binary)
    self.assertLess(new_model_binary, self._model_buffer)


if __name__ == '__main__':
  googletest.main()
