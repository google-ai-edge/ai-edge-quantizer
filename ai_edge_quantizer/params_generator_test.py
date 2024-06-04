"""Tests for params_generator."""

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer import calibrator
from ai_edge_quantizer import params_generator
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe_manager
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils


_OpExecutionMode = qtyping.OpExecutionMode
_TensorDataType = qtyping.TensorDataType
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('')


def _single_fc_model_representative_dataset_gen(num_samples=5):
  for _ in range(num_samples):
    yield [np.random.rand(1, 8).astype(np.float32)]


class ParamsGeneratorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'test_models/conv_fc_mnist.tflite'
    )
    self._recipe_manager = recipe_manager.RecipeManager()
    self._params_generator = params_generator.ParamsGenerator(
        self._test_model_path
    )

  def test_update_model_quant_results(self):
    params_from_target = qtyping.TensorTransformationParams(
        tensor_name='test_tensor0',
        consumers=[
            qtyping.OpToTensorParams(
                subgraph_op_id=3,
                transformations=[
                    _QuantTransformation.ADD_QUANTIZE,
                    _QuantTransformation.ADD_DEQUANTIZE,
                ],
            )
        ],
    )
    # Test add new tensor from target
    self._params_generator._update_model_quant_results([params_from_target])
    self.assertIsNotNone(
        'test_tensor0' in self._params_generator.model_quant_results
    )
    tensor_params = self._params_generator.model_quant_results['test_tensor0']
    self.assertEqual(
        tensor_params,
        params_from_target,
    )
    # Test update new tensor from source
    params_from_source = qtyping.TensorTransformationParams(
        tensor_name='test_tensor0',
        producer=qtyping.OpToTensorParams(
            subgraph_op_id=3,
            transformations=[
                _QuantTransformation.ADD_DEQUANTIZE,
            ],
        ),
    )
    self._params_generator._update_model_quant_results([params_from_source])
    tensor_params = self._params_generator.model_quant_results['test_tensor0']
    self.assertEqual(
        tensor_params.producer,
        params_from_source.producer,
    )

    # We can have multiple target op params
    params_from_target2 = qtyping.TensorTransformationParams(
        tensor_name='test_tensor0',
        consumers=[
            qtyping.OpToTensorParams(
                subgraph_op_id=3,
                transformations=[
                    _QuantTransformation.NO_QUANTIZE,
                ],
            )
        ],
    )
    self._params_generator._update_model_quant_results([params_from_target2])
    tensor_params = self._params_generator.model_quant_results['test_tensor0']
    self.assertSequenceEqual(
        tensor_params.consumers,
        params_from_target.consumers + params_from_target2.consumers,
    )

    # but only a single source op params
    error_message = (
        'received multiple quantization parameters from the source op'
    )
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      self._params_generator._update_model_quant_results([params_from_source])

  def test_generate_config_global(self):
    # Quantize all fully_connected.
    global_recipe = [
        {
            'regex': '.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'ptq',
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 8,
                    'symmetric': False,
                    'channel_wise': True,
                },
                'execution_mode': _OpExecutionMode.WEIGHT_ONLY,
            },
            'override_algorithm': True,
        },
    ]
    self._recipe_manager.load_quantization_recipe(global_recipe)
    tensor_quantization_params = (
        self._params_generator.generate_quantization_parameters(
            self._recipe_manager
        )
    )
    # Every tensor in the model will have their params!
    flatbuffer_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    tensors = flatbuffer_model.subgraphs[0].tensors
    self.assertLen(tensor_quantization_params, len(tensors))

    # Input tensor
    tensor_name = 'serving_default_conv2d_input:0'
    self._test_tensor_transformation_params(
        0,
        tensor_quantization_params,
        tensor_name,
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )
    # Input tensor will have no source op
    transformation_config = tensor_quantization_params[tensor_name]
    self.assertIsNone(transformation_config.producer)

    # Intermediate tensor will have no_quantize at the both end
    tensor_name = 'sequential/average_pooling2d/AvgPool'
    self._test_tensor_transformation_params(
        1,
        tensor_quantization_params,
        tensor_name,
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )  # output from average pool
    self._test_tensor_transformation_params(
        2,
        tensor_quantization_params,
        tensor_name,
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )  # input to Reshape

    # First FC
    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'sequential/flatten/Reshape',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )  # input tensor

    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'arith.constant1',
        [
            _QuantTransformation.ADD_DEQUANTIZE,
        ],
        is_inbounding_tensor=True,
        num_bits=8,
        channel_wise=True,
        symmetric=False,
    )  # weight tensor
    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'arith.constant2',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )  # bias tensor
    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )  # output tensor

    # Second FC
    self._test_tensor_transformation_params(
        4,
        tensor_quantization_params,
        'sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )  # input tensor
    self._test_tensor_transformation_params(
        4,
        tensor_quantization_params,
        'arith.constant',
        [
            _QuantTransformation.ADD_DEQUANTIZE,
        ],
        is_inbounding_tensor=True,
        num_bits=8,
        channel_wise=True,
        symmetric=False,
    )  # weight tensor
    self._test_tensor_transformation_params(
        4,
        tensor_quantization_params,
        'sequential/dense_1/MatMul',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )  # output tensor

    # Model output tensor
    tensor_name = 'StatefulPartitionedCall:0'
    self._test_tensor_transformation_params(
        5,
        tensor_quantization_params,
        tensor_name,
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )
    # Output tensor will have no target op
    transformation_config = tensor_quantization_params[tensor_name]
    self.assertIsNone(transformation_config.consumers)

  # TODO(b/330770656): expand the test to cover mixed activation precision.
  def test_generate_config_selective(self):
    # Choose scope regex using Model Explorer
    selective_quantization_recipe = [
        {
            'regex': '.*/dense/.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'ptq',
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 8,
                    'symmetric': True,
                    'channel_wise': True,
                },
                'execution_mode': _OpExecutionMode.DRQ,
            },
            'override_algorithm': True,
        },
        {
            'regex': '.*/dense_1/.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'ptq',
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 4,
                    'symmetric': False,
                    'channel_wise': False,
                },
                'execution_mode': _OpExecutionMode.WEIGHT_ONLY,
            },
            'override_algorithm': True,
        },
    ]
    self._recipe_manager.load_quantization_recipe(selective_quantization_recipe)
    tensor_quantization_params = (
        self._params_generator.generate_quantization_parameters(
            self._recipe_manager
        )
    )
    # FC weights for scope "dense"
    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'arith.constant1',
        [_QuantTransformation.QUANTIZE_TENSOR],
        is_inbounding_tensor=True,
        num_bits=8,
        channel_wise=True,
        symmetric=True,
    )

    # FC weights for scope "dense1"
    self._test_tensor_transformation_params(
        4,
        tensor_quantization_params,
        'arith.constant',
        [
            _QuantTransformation.ADD_DEQUANTIZE,
        ],
        is_inbounding_tensor=True,
        num_bits=4,
        channel_wise=False,
        symmetric=False,
    )

  def test_generate_config_edge_cases(self):

    selective_quantization_recipe = [
        # Use the tensor name as scope directly.
        {
            'regex': 'sequential/dense_1/MatMul',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'ptq',
            'op_config': {
                'weight_tensor_config': {
                    'num_bits': 8,
                    'symmetric': True,
                    'channel_wise': True,
                },
                'execution_mode': _OpExecutionMode.DRQ,
            },
            'override_algorithm': True,
        },
        # Scope that does not exist in the model.
        {
            'regex': '.*/dense_3/.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'ptq',
            'op_config': {
                'weight_tensor_config': {
                    'num_bits': 4,
                    'symmetric': False,
                    'channel_wise': False,
                },
                'execution_mode': _OpExecutionMode.WEIGHT_ONLY,
            },
            'override_algorithm': True,
        },
    ]
    self._recipe_manager.load_quantization_recipe(selective_quantization_recipe)
    tensor_quantization_params = (
        self._params_generator.generate_quantization_parameters(
            self._recipe_manager
        )
    )
    # Only the second FC will be quantized
    self._test_tensor_transformation_params(
        3,
        tensor_quantization_params,
        'arith.constant1',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )

    self._test_tensor_transformation_params(
        4,
        tensor_quantization_params,
        'arith.constant',
        [_QuantTransformation.QUANTIZE_TENSOR],
        is_inbounding_tensor=True,
        num_bits=8,
        channel_wise=True,
        symmetric=True,
    )

  @parameterized.parameters(
      (True, True), (True, False), (False, True), (False, False)
  )
  def test_generate_config_int8xint8_single_fc(
      self, act_symmetric, channelwise_weight
  ):
    single_fc_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'test_models/single_fc.tflite'
    )
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=qtyping.TFLOperationName.ALL,
        algorithm_key='ptq',
        op_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=_TensorQuantConfig(
                num_bits=8, symmetric=act_symmetric
            ),
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8, symmetric=True, channel_wise=channelwise_weight
            ),
            execution_mode=_OpExecutionMode.SRQ,
        ),
    )

    params_generator_single_fc = params_generator.ParamsGenerator(
        single_fc_model_path
    )
    # Raise error when missing QSVs.
    error_message = 'Model quantization statistics values (QSVs) are required'
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      params_generator_single_fc.generate_quantization_parameters(
          self._recipe_manager
      )

    # Calibrate then quantize
    model_calibrator = calibrator.Calibrator(single_fc_model_path)
    model_calibrator.calibrate(
        _single_fc_model_representative_dataset_gen(), self._recipe_manager
    )
    model_qsvs = model_calibrator.get_model_qsvs()
    quant_params = params_generator_single_fc.generate_quantization_parameters(
        self._recipe_manager,
        model_qsvs,
    )
    self.assertLen(quant_params, 4)

    # Input tensor
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'serving_default_input_1:0',
        [_QuantTransformation.QUANTIZE_TENSOR],
        num_bits=8,
        channel_wise=False,
        symmetric=act_symmetric,
        is_inbounding_tensor=True,
    )

    # output tensor
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'StatefulPartitionedCall:0',
        [_QuantTransformation.QUANTIZE_TENSOR],
        num_bits=8,
        channel_wise=False,
        symmetric=act_symmetric,
        is_inbounding_tensor=False,
    )

    # weights
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'sequential/dense/MatMul',
        [_QuantTransformation.QUANTIZE_TENSOR],
        num_bits=8,
        channel_wise=channelwise_weight,
        symmetric=True,
        is_inbounding_tensor=True,
    )

    # bias
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'sequential/dense/BiasAdd/ReadVariableOp',
        [_QuantTransformation.QUANTIZE_TENSOR],
        num_bits=32,
        channel_wise=channelwise_weight,
        symmetric=True,
        is_inbounding_tensor=True,
    )

  def _test_tensor_transformation_params(
      self,
      subgraph_op_id,
      quant_params,
      tensor_name,
      transformations,
      is_inbounding_tensor,
      num_bits=8,
      channel_wise=False,
      symmetric=True,
      quantized_dimension=0,
  ):
    """Helper function to test tensor transformation parameters are correct."""
    self.assertIn(tensor_name, quant_params)
    transformation_config = quant_params[tensor_name]
    self.assertEqual(transformation_config.tensor_name, tensor_name)
    if is_inbounding_tensor:
      self.assertLen(transformation_config.consumers, 1)
      op_config = transformation_config.consumers[0]
    else:
      op_config = transformation_config.producer
    self.assertIsNotNone(op_config)
    self.assertEqual(op_config.subgraph_op_id, subgraph_op_id)
    self.assertSequenceEqual(op_config.transformations, transformations)
    if transformations == [_QuantTransformation.NO_QUANTIZE]:
      self.assertIsNone(op_config.parameters)
    else:
      quantization_params = op_config.parameters
      self.assertIsNotNone(quantization_params)
      if channel_wise:
        self.assertEqual(
            quantization_params.quantized_dimension, quantized_dimension
        )
      else:
        self.assertIsNone(quantization_params.quantized_dimension)
      self.assertEqual(quantization_params.num_bits, num_bits)
      if symmetric:
        self.assertEqual(np.sum(abs(quantization_params.zero_point)), 0)
      else:
        self.assertEqual(
            len(quantization_params.scale),
            len(quantization_params.zero_point),
        )


if __name__ == '__main__':
  googletest.main()