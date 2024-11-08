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


def dynamic_wi8_afp32():
  """Returns a dynamic quantization recipe with int8 weights and float32 activation."""
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
          },
      })
  ]


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
