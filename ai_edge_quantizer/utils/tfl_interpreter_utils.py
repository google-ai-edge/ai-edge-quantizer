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

"""Util functions for TFL interpreter."""

from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from tensorflow.python.platform import gfile  # pylint: disable=g-direct-tensorflow-import


def create_tfl_interpreter(
    tflite_model: Union[str, bytearray],
    allocate_tensors: bool = True,
    use_reference_kernel: bool = False,
) -> tf.lite.Interpreter:
  """Creates a TFLite interpreter from a model file.

  Args:
    tflite_model: Model file path or bytearray.
    allocate_tensors: Whether to allocate tensors.
    use_reference_kernel: Whether to use the reference kernel for the
      interpreter.

  Returns:
    A TFLite interpreter.
  """
  if isinstance(tflite_model, str):
    with gfile.GFile(tflite_model, "rb") as f:
      tflite_model = f.read()
  else:
    tflite_model = bytes(tflite_model)
  if use_reference_kernel:
    op_resolver = tf.lite.experimental.OpResolverType.BUILTIN_REF
  else:
    op_resolver = (
        tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
  tflite_interpreter = tf.lite.Interpreter(
      model_content=tflite_model,
      experimental_op_resolver_type=op_resolver,
      experimental_preserve_all_tensors=True,
  )
  if allocate_tensors:
    tflite_interpreter.allocate_tensors()
  return tflite_interpreter


def is_tensor_quantized(tensor_detail: dict[str, Any]) -> bool:
  """Checks if a tensor is quantized.

  Args:
    tensor_detail: A dictionary of tensor details.

  Returns:
    True if the tensor is quantized.
  """
  quant_params = tensor_detail["quantization_parameters"]
  return bool(len(quant_params["scales"]))


def invoke_interpreter_signature(
    tflite_interpreter: tf.lite.Interpreter,
    signature_input_data: dict[str, Any],
    signature_key: Optional[str] = None,
    quantize_input: bool = True,
) -> dict[str, np.ndarray]:
  """Invokes the TFLite interpreter through signature runner.

  Args:
    tflite_interpreter: A TFLite interpreter.
    signature_input_data: The input data for the signature.
    signature_key: The signature key.
    quantize_input: Whether to quantize the input data.

  Returns:
    The output data of the signature.
  """
  # Make a copy to avoid in-place modification.
  signature_input = signature_input_data.copy()
  signature_runner = tflite_interpreter.get_signature_runner(signature_key)
  for input_name, input_detail in signature_runner.get_input_details().items():
    if is_tensor_quantized(input_detail) and quantize_input:
      input_data = signature_input[input_name]
      quant_params = qtyping.UniformQuantParams.from_tfl_tensor_details(
          input_detail
      )
      signature_input[input_name] = uniform_quantize_tensor.uniform_quantize(
          input_data, quant_params
      )
  return signature_runner(**signature_input)


def invoke_interpreter_once(
    tflite_interpreter: tf.lite.Interpreter,
    input_data_list: list[Any],
    quantize_input: bool = True,
):
  """Invokes the TFLite interpreter once.

  Args:
    tflite_interpreter: A TFLite interpreter.
    input_data_list: A list of input data.
    quantize_input: Whether to quantize the input data.
  """
  if len(input_data_list) != len(tflite_interpreter.get_input_details()):
    raise ValueError(
        "Input data must be a list with each element match the input sequence"
        " defined in .tflite. If the model has only one input, wrap it with a"
        " list (e.g., [input_data])"
    )
  for i, input_data in enumerate(input_data_list):
    input_details = tflite_interpreter.get_input_details()[i]
    if is_tensor_quantized(input_details) and quantize_input:
      quant_params = qtyping.UniformQuantParams.from_tfl_tensor_details(
          input_details
      )
      input_data = uniform_quantize_tensor.uniform_quantize(
          input_data, quant_params
      )
    tflite_interpreter.set_tensor(input_details["index"], input_data)
  tflite_interpreter.invoke()


def get_tensor_data(
    tflite_interpreter: Any,
    tensor_detail: dict[str, Any],
    dequantize: bool = True,
) -> np.ndarray:
  """Gets the tensor data from a TFLite interpreter.

  Args:
    tflite_interpreter: A TFLite interpreter.
    tensor_detail: A dictionary of tensor details.
    dequantize: Whether to dequantize the quantized tensor data.

  Returns:
    The tensor data.
  """
  tensor_data = tflite_interpreter.get_tensor(tensor_detail["index"])
  if is_tensor_quantized(tensor_detail) and dequantize:
    quant_params = qtyping.UniformQuantParams.from_tfl_tensor_details(
        tensor_detail
    )
    tensor_data = uniform_quantize_tensor.uniform_dequantize(
        tensor_data,
        quant_params,
    )
  return tensor_data


def get_tensor_name_to_content_map(
    tflite_interpreter: Any, dequantize: bool = False
) -> dict[str, Any]:
  """Gets internal tensors from a TFLite interpreter.

  Note the data will be copied to the returned dictionary, increasing the
  memory usage.

  Args:
    tflite_interpreter: A TFLite interpreter.
    dequantize: Whether to dequantize the tensor data.

  Returns:
    A dictionary of internal tensors.
  """
  tensors = {}
  for tensor_detail in tflite_interpreter.get_tensor_details():
    # Don't return temporary, unnamed tensors
    if not tensor_detail["name"]:
      continue
    tensors[tensor_detail["name"]] = get_tensor_data(
        tflite_interpreter, tensor_detail, dequantize
    )
  return tensors


def get_tensor_name_to_details_map(tflite_interpreter: Any) -> dict[str, Any]:
  """Gets internal tensors from a TFLite interpreter.

  Args:
    tflite_interpreter: A TFLite interpreter.

  Returns:
    A dictionary of internal tensors.
  """
  tensor_name_to_detail = {}
  for tensor_detail in tflite_interpreter.get_tensor_details():
    # Don't return temporary, unnamed tensors
    if not tensor_detail["name"]:
      continue
    tensor_name_to_detail[tensor_detail["name"]] = tensor_detail
  return tensor_name_to_detail


def get_input_tensor_names(tflite_model: Union[str, bytearray]) -> list[str]:
  """Gets input tensor names from a TFLite model.

  Args:
    tflite_model: Model file path or bytearray.

  Returns:
    A list of input tensor names.
  """

  tfl_interpreter = create_tfl_interpreter(tflite_model, allocate_tensors=False)
  input_tensor_names = []

  for input_detail in tfl_interpreter.get_input_details():
    input_tensor_names.append(input_detail["name"])
  return input_tensor_names


def get_output_tensor_names(tflite_model: Union[str, bytearray]) -> list[str]:
  """Gets output tensor names from a TFLite model.

  Args:
    tflite_model: Model file path or bytearray.

  Returns:
    A list of output tensor names.
  """
  tfl_interpreter = create_tfl_interpreter(tflite_model, allocate_tensors=False)
  output_tensor_names = []
  for output_detail in tfl_interpreter.get_output_details():
    output_tensor_names.append(output_detail["name"])
  return output_tensor_names


def get_constant_tensor_names(
    tflite_model: Union[str, bytearray], min_constant_size: int = 1
) -> list[str]:
  """Gets constant tensor names from a TFLite model.

  Args:
    tflite_model: Model file path or bytearray.
    min_constant_size: The minimum size of a constant tensor.

  Returns:
    A list of names for constant tensor that bigger than min_constant_size and a
    list of names for constant tensor that smaller than min_constant_size.
  """
  tfl_interpreter = create_tfl_interpreter(tflite_model, allocate_tensors=False)
  const_tensor_names = []
  for tensor_detail in tfl_interpreter.get_tensor_details():
    if tensor_detail["dtype"] == np.object_:
      continue
    try:
      tensor_data = get_tensor_data(tfl_interpreter, tensor_detail)
      if tensor_data.size >= min_constant_size:
        const_tensor_names.append(tensor_detail["name"])
    except ValueError:
      continue
  return const_tensor_names
