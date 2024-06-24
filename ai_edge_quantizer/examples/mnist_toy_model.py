"""Sample end-to-end run for the MNIST toy model.

This script quantizes the float mnist toy model and runs inference on a sample
mnist image.
"""

import os

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf

from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils

_OpExecutionMode = qtyping.OpExecutionMode
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig

_FLOAT_MODEL_PATH = flags.DEFINE_string(
    'float_model_path',
    test_utils.get_path_to_datafile('../tests/models/conv_fc_mnist.tflite'),
    'The trained float MNIST toy model path.',
)
_IMG_PATH = flags.DEFINE_string(
    'img_path',
    test_utils.get_path_to_datafile('mnist_samples/sample6.png'),
    'Path for the MNIST image to be predicted.',
)
_SAVE_PATH = flags.DEFINE_string(
    'save_path',
    '/tmp/',
    'Path to save the quantized model and recipe.',
)


def read_img(img_path: str):
  """Read MNIST image and normalize it.

  Args:
      img_path: Path to the image to be predicted.

  Returns:
      The normalized image.
  """
  image = Image.open(img_path)
  data = np.asarray(image, dtype=np.float32)
  if data.shape not in [(28, 28), (28, 28, 1)]:
    raise ValueError(
        'Invalid input image shape (MNIST image should have shape 28*28 or'
        ' 28*28*1)'
    )
  # Normalize the image if necessary.
  if data.max() > 1:
    data = data / 255.0
  return data.reshape((1, 28, 28, 1))


def quantize(
    float_model_path: str,
    execution_mode: _OpExecutionMode = _OpExecutionMode.WEIGHT_ONLY,
) -> quantizer.QuantizationResult:
  """Quantize the float model.

  Args:
      float_model_path: Path to the float model.
      execution_mode: Execution mode for the quantized model.

  Returns:
      QuantResult: quantization result
  """
  qt = quantizer.Quantizer(float_model_path)
  qt.update_quantization_recipe(
      regex='.*',
      operation_name=_OpName.FULLY_CONNECTED,
      op_config=_OpQuantConfig(
          weight_tensor_config=_TensorQuantConfig(
              num_bits=8,
              symmetric=False,
              channel_wise=True,
          ),
          execution_mode=execution_mode,
      ),
  )
  return qt.quantize()


def inference(quantized_tflite: bytes, image_path: str) -> np.ndarray:
  """Run inference on the quantized model.

  Args:
      quantized_tflite: Quantized model in bytes.
      image_path: Path to the image to be predicted.

  Returns:
      Predicted category probabilities for the image.
  """
  tflite_interpreter = tf.lite.Interpreter(model_content=quantized_tflite)
  tflite_interpreter.allocate_tensors()
  data = read_img(image_path)
  tfl_interpreter_utils.invoke_interpreter_once(tflite_interpreter, [data])
  tflite_output_details = tflite_interpreter.get_output_details()[0]
  result = tflite_interpreter.get_tensor(tflite_output_details['index'])
  return np.squeeze(result)


def main(_) -> None:
  if not os.path.exists(_FLOAT_MODEL_PATH.value):
    raise ValueError(
        'Model file does not exist. Please check the .tflite model path.'
    )
  if not os.path.exists(_IMG_PATH.value):
    raise ValueError('Image file does not exist. Please check the image path.')
  quant_result = quantize(_FLOAT_MODEL_PATH.value, _OpExecutionMode.WEIGHT_ONLY)
  category_probabilities = inference(
      quant_result.quantized_model, _IMG_PATH.value
  )
  predicted_category = np.argmax(category_probabilities)
  print(
      f'Model predicts the image as {predicted_category} with probability'
      f' {category_probabilities[predicted_category]}',
  )
  quant_result.save(_SAVE_PATH.value, model_name='mnist_toy_model')


if __name__ == '__main__':
  app.run(main)
