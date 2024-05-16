"""quantize a given tensor."""

from typing import cast
import numpy as np
from quantization_toolkit import qtyping
from quantization_toolkit.transformations import transformation_utils
from tensorflow.lite.python import schema_py_generated  # pylint: disable=g-direct-tensorflow-import


# TODO(b/335014051): support distinguishing INT, FLOAT & UINT, BFLOAT
def quant_params_to_tflite_type(
    bitwidth: int,
) -> schema_py_generated.TensorType | None:
  """Given specifications from quant param return the corresponding tflite dtype.

  Args:
    bitwidth: bitwidth from UniformQuantParams

  Returns:
    the corresponding tflite tensortype
  """
  if bitwidth <= 8:
    return schema_py_generated.TensorType.INT8
  elif bitwidth == 16:
    return schema_py_generated.TensorType.INT16
  elif bitwidth == 32:
    return schema_py_generated.TensorType.INT32
  elif bitwidth == 64:
    return schema_py_generated.TensorType.INT64
  else:
    raise ValueError(f"Unsupported quant params: {bitwidth}")


def nonlinear_quant_params_to_tflite_type(
    bitwidth: int,
) -> schema_py_generated.TensorType | None:
  """Given specifications from quant param return the corresponding tflite dtype.

  Args:
    bitwidth: bitwidth from NonLinearQuantParams

  Returns:
    the corresponding tflite tensortype
  """
  if bitwidth <= 16:
    return schema_py_generated.TensorType.FLOAT16
  elif bitwidth == 32:
    return schema_py_generated.TensorType.FLOAT32
  else:
    raise ValueError(f"Unsupported nonlinear params: {bitwidth}")


# TODO(b/333797939): add INT4 packing
def quantize_tensor(
    transformation_input: transformation_utils.TransformationInput,
) -> qtyping.TransformationInfo:
  """Quantize the tensor at the tensor_id in the given subgraph.

  Args:
    transformation_input: input structure that contains all information needed
      for the transformation.

  Returns:
    TransformationInfo:
      op_id: the producer index for tensor
      num_ops_added: the total number of ops inserted by this operation, which
        is 0
  """
  tensor = transformation_input.subgraph.tensors[transformation_input.tensor_id]
  # TODO(b/336385820): suppport quantize buffer directly when quantized_data
  # is not provided
  if tensor.buffer:
    if transformation_input.quant_params.quantized_data is not None:
      transformation_input.buffers[tensor.buffer].data = np.frombuffer(
          cast(
              np.ndarray, transformation_input.quant_params.quantized_data
          ).tobytes(),
          dtype=np.uint8,
      ).flatten()
  if isinstance(transformation_input.quant_params, qtyping.UniformQuantParams):
    flatbuffer_quantization = schema_py_generated.QuantizationParametersT()
    flatbuffer_quantization.scale = list(
        transformation_input.quant_params.scale.flatten().astype(np.float32)
    )  # flatbuffer requires scale as list[float]
    flatbuffer_quantization.zeroPoint = list(
        transformation_input.quant_params.zero_point.flatten().astype(np.int64)
    )  # flatbuffer requires zeroPoint as list[int64]
    if transformation_input.quant_params.quantized_dimension is not None:
      flatbuffer_quantization.quantizedDimension = (
          transformation_input.quant_params.quantized_dimension
      )
    tensor.quantization = flatbuffer_quantization
    tensor.type = quant_params_to_tflite_type(
        transformation_input.quant_params.num_bits
    )

  if isinstance(
      transformation_input.quant_params, qtyping.NonLinearQuantParams
  ):
    tensor.type = nonlinear_quant_params_to_tflite_type(
        transformation_input.quant_params.num_bits
    )

  if isinstance(
      transformation_input.quant_params, qtyping.NonLinearQuantParams
  ):
    tensor.type = nonlinear_quant_params_to_tflite_type(
        transformation_input.quant_params.num_bits
    )

  return qtyping.TransformationInfo(
      0, num_ops_added=0, output_tensor_id=transformation_input.tensor_id
  )
