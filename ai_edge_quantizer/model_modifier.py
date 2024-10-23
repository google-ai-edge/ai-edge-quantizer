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

"""Model Modifier class that produce the final quantized TFlite model."""

import copy

import numpy as np

from ai_edge_quantizer import qtyping
from ai_edge_quantizer import transformation_instruction_generator
from ai_edge_quantizer import transformation_performer
from ai_edge_litert import schema_py_generated  # pylint: disable=g-direct-tensorflow-import
from tensorflow.lite.tools import flatbuffer_utils  # pylint: disable=g-direct-tensorflow-import


class ModelModifier:
  """Model Modifier class that produce the final quantized TFlite model."""

  def __init__(self, float_tflite: bytes):
    """Constructor.

    Args:
      float_tflite: the original TFlite model in bytearray or file path
    """

    self._model_content = float_tflite

    self._constant_map = []
    self._transformation_instruction_generator = (
        transformation_instruction_generator.TransformationInstructionsGenerator()
    )
    self._transformation_performer = (
        transformation_performer.TransformationPerformer()
    )

  def modify_model(
      self, params: dict[str, qtyping.TensorTransformationParams]
  ) -> bytearray:
    """Modify the model.

    Args:
      params: a dictionary with tensor name and a list of tensor transformation
        params

    Returns:
      a byte buffer that represents the serialized tflite model
    """
    quantized_model = copy.deepcopy(
        flatbuffer_utils.read_model_from_bytearray(self._model_content)
    )

    instructions = self._transformation_instruction_generator.quant_params_to_transformation_insts(
        params, quantized_model
    )

    self._transformation_performer.transform_graph(
        instructions, quantized_model
    )
    constant_buffer_size = self._process_constant_map(quantized_model)
    if constant_buffer_size > 2**31 - 2**20:
      return self._serialize_large_model(quantized_model)
    else:
      return self._serialize_small_model(quantized_model)

  def _process_constant_map(
      self, quantized_model: schema_py_generated.ModelT
  ) -> int:
    """Process the constant map after all transformations are applied.

    If the resulting model is > 2GB then we would need to serialize constants
    separately, as such, we collect all the constant buffers using this
    function.

    Args:
      quantized_model: a quantized TFlite ModelT

    Returns:
      an integer representing the total size of the constant buffers
    """
    buffer_size = 0
    for buffer in quantized_model.buffers:
      if buffer.data is None:
        self._constant_map.append(buffer.data)
      elif isinstance(buffer.data, np.ndarray):
        self._constant_map.append(buffer.data.tobytes())
        buffer_size += len(buffer.data.tobytes())
      else:
        self._constant_map.append(buffer.data)
        buffer_size += len(buffer.data)
    return buffer_size

  def _pad_bytearray(self, bytearr: bytearray):
    """Pad the bytearray to 16 bytes."""
    remainder = len(bytearr) % 16
    if remainder != 0:
      padding_size = 16 - remainder
      bytearr.extend(b'\0' * padding_size)

  # TODO: b/333797307 - support > 2GB output model
  def _serialize_large_model(
      self, quantized_model: schema_py_generated.ModelT
  ) -> bytearray:
    """serialize models > 2GB.

    Args:
      quantized_model: a quantized TFlite ModelT

    Returns:
      a byte buffer that represents the serialized tflite model
    """
    # TODO: b/338244867 - we can have more efficient way to calculate the
    # buffer offsets.

    # remove all the constant from the model.
    for buffer in quantized_model.buffers:
      if buffer.data is not None:
        buffer.data = None
        buffer.offset = 1
        buffer.size = 1
    dummy_bytearray = bytearray(
        flatbuffer_utils.convert_object_to_bytearray(quantized_model)
    )
    # calculate the correct buffer size and offset
    self._pad_bytearray(dummy_bytearray)
    for buffer_idx, buffer in enumerate(quantized_model.buffers):
      buffer_data = self._constant_map[buffer_idx]
      if buffer_data is None:
        continue
      buffer.offset = len(dummy_bytearray)
      buffer.size = len(buffer_data)
      dummy_bytearray += buffer_data
      self._pad_bytearray(dummy_bytearray)
    del dummy_bytearray

    # build new tflite file with correct buffer offset
    model_bytearray = bytearray(
        flatbuffer_utils.convert_object_to_bytearray(quantized_model)
    )
    self._pad_bytearray(model_bytearray)
    for buffer_idx, _ in enumerate(quantized_model.buffers):
      buffer_data = self._constant_map[buffer_idx]
      if buffer_data is None:
        continue
      model_bytearray += buffer_data
      self._pad_bytearray(model_bytearray)
    return model_bytearray

  def _serialize_small_model(
      self, quantized_model: schema_py_generated.ModelT
  ) -> bytearray:
    """serialize models < 2GB.

    Args:
      quantized_model: a quantized TFlite ModelT

    Returns:
      a byte buffer that represents the serialized tflite model
    """
    model_bytearray = flatbuffer_utils.convert_object_to_bytearray(
        quantized_model
    )
    return model_bytearray
