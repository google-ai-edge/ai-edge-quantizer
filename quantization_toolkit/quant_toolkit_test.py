"""Tests for quant_toolkit."""

import json
import os
from tensorflow.python.platform import googletest
from quantization_toolkit import qtyping
from quantization_toolkit import quant_toolkit
from quantization_toolkit.utils import test_utils

_OpExecutionMode = qtyping.OpExecutionMode
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_TensorDataType = qtyping.TensorDataType

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('')


class QuantizationToolkitTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'test_models/conv_fc_mnist.tflite'
    )
    self._test_recipe_path = os.path.join(
        TEST_DATA_PREFIX_PATH,
        'test_models/recipes/conv_fc_mnist_weight_only_recipe.json',
    )
    with open(self._test_recipe_path) as json_file:
      self._test_recipe = json.load(json_file)
    self._quant_toolkit = quant_toolkit.QuantToolkit(
        self._test_model_path, self._test_recipe_path
    )

  def test_update_quantization_recipe_succeeds(self):
    self._quant_toolkit.load_quantization_recipe(self._test_recipe_path)
    scope_regex = '.*/Dense/.*'
    new_op_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=_TensorQuantConfig(num_bits=4, symmetric=True),
        execution_mode=_OpExecutionMode.DRQ,
    )
    self._quant_toolkit.update_quantization_recipe(
        regex=scope_regex,
        operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        algorithm_key='ptq',
        op_config=new_op_config,
        override_algorithm=True,
    )
    updated_recipe = self._quant_toolkit.get_quantization_recipe()
    self.assertLen(updated_recipe, 2)

    added_config = updated_recipe[-1]
    self.assertEqual(added_config['regex'], scope_regex)
    self.assertEqual(
        added_config['op_config']['execution_mode'],
        new_op_config.execution_mode,
    )

  def test_load_quantization_recipe_succeeds(self):
    qt = quant_toolkit.QuantToolkit(self._test_model_path, None)
    qt.load_quantization_recipe(self._test_recipe_path)
    self.assertEqual(qt.get_quantization_recipe(), self._test_recipe)

    # Load a different recipe.
    new_recipe_path = os.path.join(
        TEST_DATA_PREFIX_PATH,
        'test_models/recipes/conv_fc_mnist_drq_recipe.json',
    )
    with open(new_recipe_path) as json_file:
      new_recipe = json.load(json_file)
    qt.load_quantization_recipe(new_recipe_path)
    self.assertEqual(qt.get_quantization_recipe(), new_recipe)

  def test_quantize_succeeds(self):
    self._quant_toolkit.load_quantization_recipe(self._test_recipe_path)
    self.assertIsNone(self._quant_toolkit._result.quantized_model)
    quant_result = self._quant_toolkit.quantize()
    self.assertEqual(quant_result.recipe, self._test_recipe)
    self.assertIsNotNone(quant_result.quantized_model)

  def test_quantize_no_recipe_raise_error(self):
    qt = quant_toolkit.QuantToolkit(self._test_model_path, None)
    error_message = 'Can not quantize without a quantization recipe.'
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      qt.quantize()

  def test_save_succeeds(self):
    model_name = 'test_model'
    save_path = '/tmp/'
    self._quant_toolkit.load_quantization_recipe(self._test_recipe_path)
    result = self._quant_toolkit.quantize()
    result.save(save_path, model_name)
    saved_recipe_path = os.path.join(
        save_path, model_name, model_name + '_recipe.json'
    )
    with open(saved_recipe_path) as json_file:
      saved_recipe = json.load(json_file)
    self.assertEqual(saved_recipe, self._test_recipe)

  def test_save_no_quantize_raise_error(self):
    error_message = 'No quantized model to save.'
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      self._quant_toolkit._result.save('/tmp/', 'test_model')

  def test_compare_succeeds(self):
    self._quant_toolkit.quantize()
    comparison_result = self._quant_toolkit.compare()
    self.assertIsNotNone(comparison_result)
    self.assertIn('sequential/dense_1/MatMul', comparison_result)

  def test_save_compare_result_succeeds(self):
    self._quant_toolkit.quantize()
    test_json_path = '/tmp/test_compare_result.json'
    comparison_result = self._quant_toolkit.compare()
    self._quant_toolkit.save_comparison_result(
        comparison_result, test_json_path, [0, 1]
    )
    with open(test_json_path) as json_file:
      json_dict = json.load(json_file)
    self.assertIn('results', json_dict)
    results = json_dict['results']
    self.assertIn('sequential/dense_1/MatMul', results)


if __name__ == '__main__':
  googletest.main()
