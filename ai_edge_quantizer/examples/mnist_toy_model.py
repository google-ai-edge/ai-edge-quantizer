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

"""Sample end-to-end run for the MNIST toy model.

This script quantizes the float mnist toy model and runs inference on a sample
mnist image.
"""

import os
import random

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
_QUANTIZATION_MODE = flags.DEFINE_enum(
    'quantization_mode',
    'af32w8float',
    ['af32w8float', 'af32w8int', 'a8w8', 'a16w8'],
    'How to quantize the model (e.g., af32w8float, af32w8int, a8w8, a16w8).',
)


def _get_calibration_data(
    num_samples: int = 256,
) -> list[dict[str, np.ndarray]]:
  (x_train, _), _ = tf.keras.datasets.mnist.load_data()
  x_train = x_train / 255.0  # Normalize pixel values to 0-1.
  x_train = x_train.astype(np.float32)
  x_train = x_train.reshape([-1, 28, 28, 1])
  calibration_data = []
  for _ in range(num_samples):
    sample = random.choice(x_train)
    calibration_data.append({'conv2d_input': sample.reshape([-1, 28, 28, 1])})
  return calibration_data


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
    quantization_mode: str,
) -> quantizer.QuantizationResult:
  """Quantize the float model.

  Args:
      float_model_path: Path to the float model.
      quantization_mode: How to quantize the model (e.g., af32w8float,
        af32w8int).

  Returns:
      QuantResult: quantization result
  """
  if quantization_mode == 'af32w8float':
    recipe_path = test_utils.get_path_to_datafile(
        '../recipes/default_af32w8float_recipe.json'
    )
  elif quantization_mode == 'af32w8int':
    recipe_path = test_utils.get_path_to_datafile(
        '../recipes/default_af32w8int_recipe.json'
    )
  elif quantization_mode == 'a8w8':
    recipe_path = test_utils.get_path_to_datafile(
        '../recipes/default_a8w8_recipe.json'
    )
  elif quantization_mode == 'a16w8':
    recipe_path = test_utils.get_path_to_datafile(
        '../recipes/default_a16w8_recipe.json'
    )
  else:
    raise ValueError(
        'Invalid quantization mode. Only af32w8float, af32w8int, a8w8, a16w8'
        ' are supported.'
    )

  qt = quantizer.Quantizer(float_model_path)
  qt.load_quantization_recipe(recipe_path)
  calibration_result = None
  if qt.need_calibration:
    calibration_result = qt.calibrate(_get_calibration_data())
  return qt.quantize(calibration_result)


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
  quant_result = quantize(_FLOAT_MODEL_PATH.value, _QUANTIZATION_MODE.value)
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
