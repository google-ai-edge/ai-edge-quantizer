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

"""Default quantization policy."""

import collections
import copy
import json
from typing import Any
from ai_edge_quantizer import qtyping

_TFLOpName = qtyping.TFLOperationName
_OpQuantizationConfig = qtyping.OpQuantizationConfig
_TensorQuantizationConfig = qtyping.TensorQuantizationConfig
_ComputePrecision = qtyping.ComputePrecision
_Granularity = qtyping.QuantGranularity
_INT = qtyping.TensorDataType.INT

# Default config check policy in JSON format. It has two keys: "configs" and
# "ops_per_config". "configs" is a dictionary mapping from config name to config
# content. "ops_per_config" is a dictionary mapping from config name to a list
# of op names.
DEFAULT_JSON_POLICY = """
{
  "configs": {
    "dynamic_wi8_afp32": {
      "weight_tensor_config": {
        "num_bits": 8,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "dynamic_wi4_afp32": {
      "weight_tensor_config": {
        "num_bits": 4,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "static_wi8_ai16": {
      "activation_tensor_config": {
        "num_bits": 16,
        "symmetric": [true],
        "granularity": ["TENSORWISE"],
        "dtype": "INT"
      },
      "weight_tensor_config": {
        "num_bits": 8,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "static_wi4_ai16": {
      "activation_tensor_config": {
        "num_bits": 16,
        "symmetric": [true],
        "granularity": ["TENSORWISE"],
        "dtype": "INT"
      },
      "weight_tensor_config": {
        "num_bits": 4,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "static_wi8_ai8": {
      "activation_tensor_config": {
        "num_bits": 8,
        "symmetric": [true, false],
        "granularity": ["TENSORWISE"],
        "dtype": "INT"
      },
      "weight_tensor_config": {
        "num_bits": 8,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "static_wi4_ai8": {
      "activation_tensor_config": {
        "num_bits": 8,
        "symmetric": [true, false],
        "granularity": ["TENSORWISE"],
        "dtype": "INT"
      },
      "weight_tensor_config": {
        "num_bits": 4,
        "symmetric": [true],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": false,
      "compute_precision": "INTEGER"
    },
    "weightonly_wi8_afp32": {
      "weight_tensor_config": {
        "num_bits": 8,
        "symmetric": [true, false],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": true,
      "compute_precision": "FLOAT"
    },
    "weightonly_wi4_afp32": {
      "weight_tensor_config": {
        "num_bits": 4,
        "symmetric": [true, false],
        "granularity": ["CHANNELWISE", "TENSORWISE"],
        "dtype": "INT"
      },
      "explicit_dequantize": true,
      "compute_precision": "FLOAT"
    }
  },
  "ops_per_config": {
    "static_wi8_ai16": [
      "ADD",
      "AVERAGE_POOL_2D",
      "BATCH_MATMUL",
      "CONCATENATION",
      "CONV_2D",
      "CONV_2D_TRANSPOSE",
      "DEPTHWISE_CONV_2D",
      "FULLY_CONNECTED",
      "GELU",
      "LOGISTIC",
      "MEAN",
      "MUL",
      "RESHAPE",
      "RSQRT",
      "SOFTMAX",
      "SPLIT",
      "STRIDED_SLICE",
      "SUB",
      "TANH",
      "TRANSPOSE",
      "INPUT",
      "OUTPUT",
      "SLICE",
      "EMBEDDING_LOOKUP",
      "SUM",
      "SELECT_V2"
    ],
    "static_wi8_ai8": [
      "ADD",
      "AVERAGE_POOL_2D",
      "BATCH_MATMUL",
      "CONCATENATION",
      "FULLY_CONNECTED",
      "CONV_2D",
      "CONV_2D_TRANSPOSE",
      "DEPTHWISE_CONV_2D",
      "GELU",
      "LOGISTIC",
      "MEAN",
      "MUL",
      "RESHAPE",
      "RSQRT",
      "SOFTMAX",
      "SPLIT",
      "STRIDED_SLICE",
      "SUB",
      "TANH",
      "TRANSPOSE",
      "INPUT",
      "OUTPUT",
      "SLICE",
      "EMBEDDING_LOOKUP",
      "SUM",
      "SELECT_V2"
    ],
    "static_wi4_ai8": ["FULLY_CONNECTED", "CONV_2D", "INPUT", "OUTPUT"],
    "static_wi4_ai16": ["FULLY_CONNECTED", "CONV_2D", "INPUT", "OUTPUT"],
    "dynamic_wi8_afp32": [
      "BATCH_MATMUL",
      "CONV_2D",
      "CONV_2D_TRANSPOSE",
      "DEPTHWISE_CONV_2D",
      "EMBEDDING_LOOKUP",
      "FULLY_CONNECTED"
    ],
    "dynamic_wi4_afp32": ["FULLY_CONNECTED", "EMBEDDING_LOOKUP", "CONV_2D"],
    "weightonly_wi8_afp32": [
      "BATCH_MATMUL",
      "CONV_2D",
      "CONV_2D_TRANSPOSE",
      "DEPTHWISE_CONV_2D",
      "EMBEDDING_LOOKUP",
      "FULLY_CONNECTED"
    ],
    "weightonly_wi4_afp32": ["BATCH_MATMUL", "FULLY_CONNECTED", "EMBEDDING_LOOKUP", "CONV_2D"]
  }
}
"""


def _unroll_json_config(
    json_config: dict[str, Any],
) -> list[_OpQuantizationConfig]:
  """Unrolls the config.

  Args:
    json_config: JSON config to be unrolled.

  Returns:
    quant_configs: A list of unrolled configs.
  """

  # Unroll activation configs first.
  activation_configs = []
  if "activation_tensor_config" in json_config:
    for symmetric in json_config["activation_tensor_config"]["symmetric"]:
      for granularity in json_config["activation_tensor_config"]["granularity"]:
        tensor_config = {
            "num_bits": json_config["activation_tensor_config"]["num_bits"],
            "symmetric": symmetric,
            "granularity": granularity,
            "dtype": json_config["activation_tensor_config"]["dtype"],
        }
        activation_configs.append(
            qtyping.TensorQuantizationConfig.from_dict(tensor_config)
        )

  # Then unroll weight configs and turn them into quantization configs.
  quant_configs = []
  for symmetric in json_config["weight_tensor_config"]["symmetric"]:
    for granularity in json_config["weight_tensor_config"]["granularity"]:
      tensor_config = {
          "num_bits": json_config["weight_tensor_config"]["num_bits"],
          "symmetric": symmetric,
          "granularity": granularity,
          "dtype": json_config["weight_tensor_config"]["dtype"],
      }

      if activation_configs:
        for activation_config in activation_configs:
          quant_configs.append(
              qtyping.OpQuantizationConfig(
                  activation_tensor_config=activation_config,
                  weight_tensor_config=qtyping.TensorQuantizationConfig.from_dict(
                      tensor_config
                  ),
                  compute_precision=json_config["compute_precision"],
                  explicit_dequantize=json_config["explicit_dequantize"],
              )
          )
      else:
        quant_configs.append(
            qtyping.OpQuantizationConfig(
                weight_tensor_config=qtyping.TensorQuantizationConfig.from_dict(
                    tensor_config
                ),
                compute_precision=json_config["compute_precision"],
                explicit_dequantize=json_config["explicit_dequantize"],
            )
        )

  return quant_configs


def update_default_config_policy(raw_json_policy: str):
  """Updates the default config check policy."""
  json_policy_content = json.loads(raw_json_policy)

  # Unroll the json config and add the configs to the policy.
  policy = collections.OrderedDict()
  for json_policy_config in json_policy_content["ops_per_config"]:
    unrolled_configs = _unroll_json_config(
        json_policy_content["configs"][json_policy_config]
    )

    for op in json_policy_content["ops_per_config"][json_policy_config]:
      op_name = _TFLOpName(op)
      quant_configs = copy.deepcopy(unrolled_configs)
      if op in policy.keys():
        quant_configs += policy[op_name]
      policy[op_name] = quant_configs

  return policy


DEFAULT_CONFIG_CHECK_POLICY = update_default_config_policy(DEFAULT_JSON_POLICY)
