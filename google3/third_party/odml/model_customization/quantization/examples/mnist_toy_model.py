"""sample e2e run for the mnist toy model."""

from absl import app
from google3.pyglib import gfile
from google3.third_party.odml.model_customization.quantization import model_modifier
from google3.third_party.odml.model_customization.quantization import model_validator
from google3.third_party.odml.model_customization.quantization import params_generator
from google3.third_party.odml.model_customization.quantization import recipe_manager
from google3.third_party.odml.model_customization.quantization import typing as qtyping
from google3.third_party.odml.model_customization.quantization.utils import test_utils
from google3.third_party.odml.model_customization.quantization.utils import tfl_interpreter_utils
from google3.third_party.odml.model_customization.quantization.utils import validation_utils


# TODO: b/336642947 - use the entry point API to generate the quantized .tflite.
def main(_) -> None:
  model_path = test_utils.get_path_to_datafile(
      '../test_models/conv_fc_mnist.tflite'
  )
  recipe_manager_instance = recipe_manager.RecipeManager()
  params_generator_instance = params_generator.ParamsGenerator(model_path)
  model_modifier_instance = model_modifier.ModelModifier(model_path)
  global_recipe = [
      {
          'regex': '.*',
          'operation': 'FULLY_CONNECTED',
          'algorithm_key': 'ptq',
          'op_config': {
              'weight_tensor_config': {
                  'dtype': qtyping.TensorDataType.INT,
                  'num_bits': 8,
                  'symmetric': False,
                  'channel_wise': True,
              },
              'execution_mode': qtyping.OpExecutionMode.WEIGHT_ONLY,
          },
          'override_algorithm': True,
      },
  ]
  recipe_manager_instance.load_quantization_recipe(global_recipe)
  tensor_quantization_params = (
      params_generator_instance.generate_quantization_parameters(
          recipe_manager_instance
      )
  )
  new_model_binary = model_modifier_instance.modify_model(
      tensor_quantization_params
  )
  save_model_path = test_utils.get_path_to_datafile('/tmp/test.tflite')
  with gfile.GFile(save_model_path, 'wb') as output_file_handle:
    output_file_handle.write(new_model_binary)

  reference_model_interpreter = tfl_interpreter_utils.create_tfl_interpreter(
      model_path
  )
  dataset = test_utils.create_random_normal_dataset(
      reference_model_interpreter.get_input_details(), 3, 666
  )
  result = model_validator.compare_model(
      model_path,
      save_model_path,
      dataset,
      False,
      validation_utils.mean_squared_difference,
  )

  print(result)


if __name__ == '__main__':
  app.run(main)
