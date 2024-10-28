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

"""Sample end-to-end example for the MNIST toy model.

This script quantizes an MNIST toy model and runs inference on a sample MNIST
image.
"""

import os
import random
from typing import Any, Optional

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf

from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_interpreter_utils
from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-direct-tensorflow-import

_OpExecutionMode = qtyping.OpExecutionMode
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig
_QuantGranularity = qtyping.QuantGranularity

_FLOAT_MODEL_PATH = flags.DEFINE_string(
    'float_model_path',
    test_utils.get_path_to_datafile('../../tests/models/conv_fc_mnist.tflite'),
    'The trained floating point MNIST toy model TFLite flatbuffer path.',
)
_RECIPE_PATH = flags.DEFINE_string(
    'recipe_path',
    test_utils.get_path_to_datafile(
        '../../recipes/default_af32w8float_recipe.json'
    ),
    'The quantization recipe path in JSON format.',
)
_IMG_PATH = flags.DEFINE_string(
    'img_path',
    test_utils.get_path_to_datafile('data/sample6.png'),
    'Path to the MNIST image to be predicted.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp/',
    'Directory to save the quantized model and recipe.',
)


def _check_user_inputs() -> None:
  """Check if the user input paths exist and create the save path if necessary."""
  if not os.path.exists(_FLOAT_MODEL_PATH.value):
    raise ValueError(
        'Model file does not exist. Please check the .tflite model path.'
    )
  if not os.path.exists(_RECIPE_PATH.value):
    raise ValueError(
        'Recipe file does not exist. Please check the recipe path.'
    )
  if not os.path.exists(_IMG_PATH.value):
    raise ValueError('Image file does not exist. Please check the image path.')

  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)


def quantize(
    float_model_path: str,
    recipe_json_path: str,
) -> Optional[bytearray]:
  """Quantize the float model.

  Args:
      float_model_path: Path to the float model.
      recipe_json_path: Path to the quantization recipe.

  Returns:
      Quantized model in bytes.
  """

  def _get_calibration_data(
      num_samples: int = 256,
  ) -> dict[str, list[dict[str, Any]]]:
    """Generate random dummy calibration data.

    The calibration data is a list of dictionaries, each of which contains an
    input data for a single calibration step. The key is the input tensor name
    and the value is the input tensor value.

    Args:
      num_samples: Number of samples to generate.

    Returns:
      A list of calibration data.
    """
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0  # Normalize pixel values to 0-1.
    x_train = x_train.astype(np.float32)
    x_train = x_train.reshape([-1, 28, 28, 1])
    calibration_samples = []
    for _ in range(num_samples):
      sample = random.choice(x_train)
      calibration_samples.append(
          {'conv2d_input': sample.reshape([-1, 28, 28, 1])}
      )
    calibration_data = {
        tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
    }
    return calibration_data

  # 1) Instantiate a new quantizer with the source float model path.
  qt = quantizer.Quantizer(float_model_path)

  # 2) Initialize the quantization recipe. The input is a json file that
  #    specifies the quantization configuration for the model.
  #    A collection of common recipes can be found in the top-level directory
  #    `recipes/`. See `recipes/sample_advanced_usage_recipe.json` for
  #    advanced usage.
  #    Alternatively, you can also use the `update_quantization_recipe()`
  #    method to specify the quantization recipe programmatically.
  qt.load_quantization_recipe(recipe_json_path)

  # 3) Calibrate the model if necessary. This is only necessary for quantization
  #    modes that involve integer computation. `qt.need_calibration` can be used
  #    to check if the quantization recipe needs calibration.
  calibration_result = (
      qt.calibrate(_get_calibration_data()) if qt.need_calibration else None
  )

  # 4) Quantize the model. The quantization result contains the actual
  #    recipe and the quantized model.
  quant_result = qt.quantize(calibration_result)

  # 5) Save the quantized model and the recipe used to the filesystem.
  quant_result.save(_OUTPUT_DIR.value, model_name='mnist_toy_model')

  return quant_result.quantized_model


def inference(quantized_tflite: bytes, image_path: str):
  """Run inference on the quantized model.

  Args:
      quantized_tflite: Quantized model in bytes.
      image_path: Path to the image to be predicted.
  """

  def _read_img(img_path: str) -> np.ndarray:
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

  tflite_interpreter = tfl_interpreter.Interpreter(
      model_content=quantized_tflite
  )
  tflite_interpreter.allocate_tensors()
  data = _read_img(image_path)
  tfl_interpreter_utils.invoke_interpreter_once(tflite_interpreter, [data])
  tflite_output_details = tflite_interpreter.get_output_details()[0]
  result = tflite_interpreter.get_tensor(tflite_output_details['index'])
  category_probabilities = np.squeeze(result)
  predicted_category = np.argmax(category_probabilities)
  print(
      f'Model predicts the image as {predicted_category} with probability'
      f' {category_probabilities[predicted_category]}',
  )


def main(_) -> None:
  _check_user_inputs()

  # Quantize the source floating point model into a quantized model based on the
  # given quantization recipe. This function demonstrates the end-to-end
  # quantization flow and the basic usage of AI Edge Quantizer's API.
  quantized_model = quantize(_FLOAT_MODEL_PATH.value, _RECIPE_PATH.value)

  # Run inference on the quantized model. This function demonstrates how to
  # use the quantized model for inference.
  inference(quantized_model, _IMG_PATH.value)


if __name__ == '__main__':
  app.run(main)
