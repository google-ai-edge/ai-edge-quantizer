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

from ai_edge_quantizer import qtyping

_TFLOpName = qtyping.TFLOperationName
_OpQuantizationConfig = qtyping.OpQuantizationConfig
_TensorQuantizationConfig = qtyping.TensorQuantizationConfig
_OpExecutionMode = qtyping.OpExecutionMode
_INT = qtyping.TensorDataType.INT

DEFAULT_CONFIG_CHECK_POLICY = qtyping.ConfigCheckPolicyDict({
    _TFLOpName.FULLY_CONNECTED: [
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=4, symmetric=True, channel_wise=True, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=4, symmetric=True, channel_wise=False, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=True, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=False, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
    ],
    _TFLOpName.CONV_2D: [
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=True, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=False, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
    ],
    _TFLOpName.BATCH_MATMUL: [
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=True, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=False, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
    ],
    _TFLOpName.EMBEDDING_LOOKUP: [
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=4, symmetric=True, channel_wise=True, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=4, symmetric=True, channel_wise=False, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=True, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=False, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
    ],
    _TFLOpName.DEPTHWISE_CONV_2D: [
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=True, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=False, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
    ],
    _TFLOpName.CONV_2D_TRANSPOSE: [
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=True, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
        _OpQuantizationConfig(
            activation_tensor_config=None,
            weight_tensor_config=_TensorQuantizationConfig(
                num_bits=8, symmetric=True, channel_wise=False, dtype=_INT
            ),
            execution_mode=_OpExecutionMode.DRQ,
        ),
    ],
})
