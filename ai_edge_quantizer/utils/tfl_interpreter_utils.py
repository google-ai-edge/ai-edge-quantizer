"""Util functions for TFL interpreter."""

from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from tensorflow.python.platform import gfile  # pylint: disable=g-direct-tensorflow-import


def create_tfl_interpreter(
    tflite_model: Union[str, bytearray],
) -> tf.lite.Interpreter:
  """Creates a TFLite interpreter from a model file.

  Args:
    tflite_model: Model file path or bytearray.

  Returns:
    A TFLite interpreter.
  """
  if isinstance(tflite_model, str):
    with gfile.GFile(tflite_model, "rb") as f:
      tflite_model = f.read()
  else:
    tflite_model = bytes(tflite_model)
  tflite_interpreter = tf.lite.Interpreter(
      model_content=tflite_model,
      experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
      experimental_preserve_all_tensors=True,
  )
  tflite_interpreter.allocate_tensors()
  return tflite_interpreter


def _is_tensor_quantized(tensor_detail: dict[str, Any]) -> bool:
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
  signature_runner = tflite_interpreter.get_signature_runner(signature_key)
  for input_name, input_detail in signature_runner.get_input_details().items():
    if _is_tensor_quantized(input_detail) and quantize_input:
      input_data = signature_input_data[input_name]
      quant_params = qtyping.UniformQuantParams.from_tfl_tensor_details(
          input_detail
      )
      signature_input_data[input_name] = (
          uniform_quantize_tensor.uniform_quantize(input_data, quant_params)
      )
  return signature_runner(**signature_input_data)


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
    if _is_tensor_quantized(input_details) and quantize_input:
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
  if _is_tensor_quantized(tensor_detail) and dequantize:
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
