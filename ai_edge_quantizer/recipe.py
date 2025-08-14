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

"""Quantization recipe module."""

from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe_manager

AlgorithmName = algorithm_manager.AlgorithmName


def dynamic_wi8_afp32(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
):
  """Returns a dynamic quantization recipe with int8 weights and float32 activation.

  All supported ops will be quantized with int8 weights and float32 activations,
  which will be dynamically quantized to int8 during inference to enable int8
  compute. The model quality may suffer due to the on-the-fly quantization. If
  quality is a concern, consider using weight-only quantization.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A dynamic quantization recipe.
  """
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_dynamic_config(
      regex='.*',
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      num_bits=8,
      algorithm_key=algorithm_key,
  )
  return rp_manager.get_quantization_recipe()


def dynamic_wi4_afp32(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
):
  """Returns a dynamic quantization recipe with int4 weights and float32 activation.

  All supported ops will be quantized with int4 weights and float32 activations,
  which will be dynamically quantized to int4 during inference to enable int4
  compute.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A dynamic quantization recipe.
  """
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_dynamic_config(
      regex='.*',
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      num_bits=4,
      algorithm_key=algorithm_key,
  )
  return rp_manager.get_quantization_recipe()


def weight_only_wi8_afp32(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
):
  """Returns a weight-only quantization recipe with int8 weights and float32 activation.

  All supported ops will be quantized with int8 weights and float32 activations.
  The weights will be explicitly dequantized before being fed into the op to
  enable float compute thus retain model quality. If latency is a concern,
  consider using dynamic range quantization.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A weight-only quantization recipe.
  """
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_weight_only_config(
      regex='.*',
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      num_bits=8,
      algorithm_key=algorithm_key,
  )
  return rp_manager.get_quantization_recipe()


def weight_only_wi4_afp32(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
):
  """Returns a weight-only quantization recipe with int4 weights and float32 activation.

  All supported ops will be quantized with int4 weights and float32 activations.
  The weights will be explicitly dequantized before being fed into the op to
  enable float compute thus retain model quality.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A weight-only quantization recipe.
  """
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_weight_only_config(
      regex='.*',
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      num_bits=4,
      algorithm_key=algorithm_key,
  )
  return rp_manager.get_quantization_recipe()


def static_wi8_ai8(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
):
  """Returns a static quantization recipe with int8 weights and int8 activations.

  All supported ops will be quantized with int8 weights and int8 activations.
  Calibration is needed to use this recipe.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A static quantization recipe.
  """
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_static_config(
      regex='.*',
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      activation_num_bits=8,
      weight_num_bits=8,
      algorithm_key=algorithm_key,
  )
  return rp_manager.get_quantization_recipe()


def static_wi8_ai16(
    algorithm_key: AlgorithmName = AlgorithmName.MIN_MAX_UNIFORM_QUANT,
):
  """Returns a static quantization recipe with int8 weights and int16 activations.

  All supported ops will be quantized with int8 weights and int16 activations.
  Calibration is needed to use this recipe.

  Args:
    algorithm_key: The algorithm to use for quantization.

  Returns:
    A static quantization recipe.
  """
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_static_config(
      regex='.*',
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      activation_num_bits=16,
      weight_num_bits=8,
      algorithm_key=algorithm_key,
  )
  return rp_manager.get_quantization_recipe()


def dynamic_legacy_wi8_afp32():
  """Returns a dynamic quantization legacy recipe with int8 weights and float32 activation.

  The difference between this and dynamic_wi8_afp32 is that this recipe sets
  min_weight_elements to 1024 to match the old quantizer behavior.
  """
  return [
      dict({
          'regex': '.*',
          'operation': '*',
          'algorithm_key': 'min_max_uniform_quantize',
          'op_config': {
              'weight_tensor_config': {
                  'num_bits': 8,
                  'symmetric': True,
                  'granularity': 'CHANNELWISE',
                  'dtype': 'INT',
                  'block_size': 0,
              },
              'compute_precision': 'INTEGER',
              'explicit_dequantize': False,
              'skip_checks': False,
              'min_weight_elements': 1024,
          },
      })
  ]
