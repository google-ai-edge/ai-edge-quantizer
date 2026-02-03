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
import logging

import numpy as np

from ai_edge_litert.tools import flatbuffer_utils
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import transformation_instruction_generator
from ai_edge_quantizer import transformation_performer
from ai_edge_quantizer.utils import progress_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils
from ai_edge_litert import interpreter as tfl  # pylint: disable=g-direct-tensorflow-import
from ai_edge_litert import schema_py_generated  # pylint: disable=g-direct-tensorflow-import

_DEQUANT_SUFFIX = "_dequant"
_QUANT_SUFFIX = "_quantized"


class ModelModifier:
  """Model Modifier class that produce the final quantized TFlite model."""

  def __init__(self, float_model: tfl_flatbuffer_utils.ModelT):
    """Constructor.

    Args:
      float_model: the original TFlite model.
    """
    self._model: tfl_flatbuffer_utils.ModelT = float_model

    self._constant_map = []
    self._transformation_instruction_generator = (
        transformation_instruction_generator.TransformationInstructionsGenerator()
    )
    self._transformation_performer = (
        transformation_performer.TransformationPerformer()
    )

  def _get_tensor_processing_order(
      self,
      tensor_names: set[str],
      flatbuffer_model: tfl_flatbuffer_utils.ModelT,
  ) -> list[str]:
    """Get the tensor processing order obtained from `buffer_to_tensors`.

    The processing order is used to ensure that last tensor in a buffer is
    processed the last. This is required for the correctness of buffer
    duplication, as the last tensor in a buffer won't be duplicated.

    Args:
      tensor_names: Names of the tensors that need to be processed.
      flatbuffer_model: TFlite model.

    Returns:
      A list of tensor names in the processing order.
    """
    buffer_to_tensors = tfl_flatbuffer_utils.buffer_to_tensors(flatbuffer_model)

    processing_order = []
    for buffer_tensors in buffer_to_tensors.values():
      for tensor in buffer_tensors:
        tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
        if tensor_name in tensor_names:
          processing_order.append(tensor_name)

    return processing_order

  def modify_model(
      self,
      params: dict[str, qtyping.TensorTransformationParams],
      enable_progress_bar: bool | None = None,
  ) -> bytearray:
    """Modify the model.

    Args:
      params: a dictionary with tensor name and a list of tensor transformation
        params
      enable_progress_bar: Whether to enable the progress bar. By default, it is
        disabled for smaller models and enabled for larger models.

    Returns:
      a byte buffer that represents the serialized tflite model
    """
    # Make a copy of the model, but don't duplicate the buffer data.
    buffer_data = []
    for buffer in self._model.buffers:
      buffer_data.append(buffer.data)
      buffer.data = None
    quantized_model = copy.deepcopy(self._model)
    for bid, data in enumerate(buffer_data):
      self._model.buffers[bid].data = data
      quantized_model.buffers[bid].data = data

    instructions = self._transformation_instruction_generator.quant_params_to_transformation_insts(
        params, quantized_model, enable_progress_bar
    )

    tensor_processing_order = self._get_tensor_processing_order(
        set(instructions.keys()), quantized_model
    )
    self._transformation_performer.transform_graph(
        instructions, quantized_model, tensor_processing_order,
        enable_progress_bar,
    )
    constant_buffer_size = self._process_constant_map(quantized_model)

    def _serialize_large_model(quantized_model: schema_py_generated.ModelT):
      return self._serialize_large_model(quantized_model, enable_progress_bar)
    # we leave 256MB for the model architecture.
    serialize_fun = (
        _serialize_large_model
        if constant_buffer_size > 2**31 - 2**28
        else self._serialize_small_model
    )
    serialized_quantized_model = serialize_fun(quantized_model)

    # Update signature defs if dequant is inserted before output.
    if self._has_transform_before_output(
        instructions, qtyping.QuantTransformation.ADD_DEQUANTIZE
    ):
      quantized_model = self._update_signature_defs(
          quantized_model, serialized_quantized_model, _DEQUANT_SUFFIX
      )
      serialized_quantized_model = serialize_fun(quantized_model)

    # Update signature defs if quant is inserted before output.
    if self._has_transform_before_output(
        instructions, qtyping.QuantTransformation.ADD_QUANTIZE
    ):
      quantized_model = self._update_signature_defs(
          quantized_model, serialized_quantized_model, _QUANT_SUFFIX
      )
      serialized_quantized_model = serialize_fun(quantized_model)

    return serialized_quantized_model

  def _update_signature_defs(
      self,
      model: tfl_flatbuffer_utils.ModelT,
      serialized_model: bytearray,
      suffix: str,
  ) -> tfl_flatbuffer_utils.ModelT:
    """Updates the signature definitions in the model.

    This function is called when a transformation (quantize or dequantize)
    is inserted before an output tensor. It updates the tensor index in the
    signature definitions to point to the newly inserted output tensor.

    Args:
      model: The TFlite ModelT object.
      serialized_model: The serialized bytearray of the TFlite model.
      suffix: The suffix to append to the tensor name.

    Returns:
      The updated TFlite ModelT object.
    """
    interpreter = tfl.Interpreter(model_content=bytes(serialized_model))

    for signature_def in model.signatureDefs:
      signature_key = signature_def.signatureKey.decode("utf-8")
      logging.info("Signature = %s", signature_key)
      subgraph_idx = tfl_interpreter_utils.get_signature_main_subgraph_index(
          interpreter, signature_key
      )
      output_details = interpreter.get_signature_runner(
          signature_key
      ).get_output_details()
      subgraph = model.subgraphs[subgraph_idx]
      graph_info = qtyping.GraphInfo(subgraph.tensors, model.buffers)

      for output in subgraph.outputs:
        tensor_name = tfl_flatbuffer_utils.get_tensor_name(
            graph_info.subgraph_tensors[output]
        )
        logging.info("\tOutput tensor = `%s`", tensor_name)

        for signature_name, tensor_details in output_details.items():
          if tensor_details["name"] + suffix == tensor_name:
            logging.info(
                "\t\tfound tensor mapping: `%s`->`%s` for signature name: `%s`",
                tensor_details["name"],
                tensor_name,
                signature_name,
            )
            for signature_item in signature_def.outputs:
              if signature_item.name.decode("utf-8") == signature_name:
                signature_item.tensorIndex = output
                logging.info(
                    "\t\t\tswapped tensor index: %s->%s",
                    tensor_details["index"],
                    output,
                )
                break
            break

    return model

  def _has_transform_before_output(
      self,
      instructions: dict[str, qtyping.TensorTransformationInsts],
      transformation: qtyping.QuantTransformation,
  ) -> bool:
    """Check if the model has transformation insert to output."""
    for tensor_name, tensor_trans_insts in instructions.items():
      for instr in tensor_trans_insts.instructions:
        if transformation == instr.transformation and instr.consumers == [-1]:
          logging.info(
              "Found %s insert to output for tensor: %s",
              transformation,
              tensor_name,
          )
          return True
    return False

  def _process_constant_map(
      self, quantized_model: tfl_flatbuffer_utils.ModelT
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
      bytearr.extend(b"\0" * padding_size)

  # TODO: b/333797307 - support > 2GB output model
  def _serialize_large_model(
      self, quantized_model: tfl_flatbuffer_utils.ModelT,
      enable_progress_bar: bool | None = None,
  ) -> bytearray:
    """serialize models > 2GB.

    Args:
      quantized_model: a quantized TFlite ModelT
      enable_progress_bar: Whether to enable the progress bar. If None, the
        progress bar will be disabled for models with less than 1000 buffers.

    Returns:
      a byte buffer that represents the serialized tflite model
    """
    # TODO: b/338244867 - we can have more efficient way to calculate the
    # buffer offsets.

    # remove all the constant from the model.

    # We go through the buffers 3 times.
    total_buffers = 3 * len(quantized_model.buffers)
    with progress_utils.ProgressBar(
        total_steps=total_buffers,
        description="Serializing Model:",
        enable=enable_progress_bar,
        # Progress bar will be skipped for smaller models.
    ) as progress_bar:
      for buffer in quantized_model.buffers:
        progress_bar.update_single_step()
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
        progress_bar.update_single_step()
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
        progress_bar.update_single_step()
        buffer_data = self._constant_map[buffer_idx]
        if buffer_data is None:
          continue
        model_bytearray += buffer_data
        self._pad_bytearray(model_bytearray)
    return model_bytearray

  def _serialize_small_model(
      self, quantized_model: tfl_flatbuffer_utils.ModelT
  ) -> bytearray:
    """serialize models < 2GB.

    Args:
      quantized_model: a quantized TFlite ModelT

    Returns:
      a byte buffer that represents the serialized tflite model
    """
    print("Serializing model.......")
    model_bytearray = flatbuffer_utils.convert_object_to_bytearray(
        quantized_model
    )
    return model_bytearray
