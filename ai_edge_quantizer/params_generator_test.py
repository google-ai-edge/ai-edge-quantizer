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


_ComputePrecision = qtyping.ComputePrecision
_TensorDataType = qtyping.TensorDataType
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation
_AlgorithmName = recipe_manager.AlgorithmName
_QuantGranularity = qtyping.QuantGranularity

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('')


def _single_fc_model_representative_dataset_gen(num_samples=5):
  for _ in range(num_samples):
    yield {'input_1': np.random.rand(1, 8).astype(np.float32)}


def _int_transpose_model_representative_dataset_gen(num_samples=5):
  data = []
  for _ in range(num_samples):
    data.append({'input_2': np.random.rand(1, 2, 3, 4).astype(np.int32)})
  return data


class ParamsGeneratorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/conv_fc_mnist.tflite'
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
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 8,
                    'symmetric': False,
                    'granularity': _QuantGranularity.CHANNELWISE,
                },
                # Equivalent to WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': True,
            },
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
    # Input tensor is produced from the virtual Input op.
    transformation_config = tensor_quantization_params[tensor_name]
    self.assertIsNotNone(transformation_config.producer)
    self.assertEqual(transformation_config.producer.subgraph_op_id, -1)

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
        granularity=_QuantGranularity.CHANNELWISE,
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
        granularity=_QuantGranularity.CHANNELWISE,
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
    # Output tensor is consumed by the virtual Output op.
    transformation_config = tensor_quantization_params[tensor_name]
    self.assertLen(transformation_config.consumers, 1)
    consumer = transformation_config.consumers[0]
    self.assertEqual(consumer.subgraph_op_id, -1)

  # TODO: b/330770656 - expand the test to cover mixed activation precision.
  def test_generate_config_selective(self):
    # Choose scope regex using Model Explorer
    selective_quantization_recipe = [
        {
            'regex': '.*/dense/.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': _QuantGranularity.CHANNELWISE,
                },
                # Equivalent to DRQ.
                'compute_precision': _ComputePrecision.INTEGER,
                'explicit_dequantize': False,
            },
        },
        {
            'regex': '.*/dense_1/.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'weight_tensor_config': {
                    'dtype': _TensorDataType.INT,
                    'num_bits': 4,
                    'symmetric': False,
                    'granularity': _QuantGranularity.TENSORWISE,
                },
                # Equivalent to WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': True,
            },
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
        granularity=_QuantGranularity.CHANNELWISE,
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
        granularity=_QuantGranularity.TENSORWISE,
        symmetric=False,
    )

  def test_generate_config_edge_cases(self):

    selective_quantization_recipe = [
        # Use the tensor name as scope directly.
        {
            'regex': 'sequential/dense_1/MatMul',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'weight_tensor_config': {
                    'num_bits': 8,
                    'symmetric': True,
                    'granularity': _QuantGranularity.CHANNELWISE,
                },
                # Equivalent to DRQ.
                'compute_precision': _ComputePrecision.INTEGER,
            },
        },
        # Scope that does not exist in the model.
        {
            'regex': '.*/dense_3/.*',
            'operation': 'FULLY_CONNECTED',
            'algorithm_key': 'min_max_uniform_quantize',
            'op_config': {
                'weight_tensor_config': {
                    'num_bits': 4,
                    'symmetric': False,
                    'granularity': _QuantGranularity.TENSORWISE,
                },
                # Equivalent to WEIGHT_ONLY.
                'compute_precision': _ComputePrecision.FLOAT,
                'explicit_dequantize': True,
            },
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
        granularity=_QuantGranularity.CHANNELWISE,
        symmetric=True,
    )

  @parameterized.parameters(
      (True, _QuantGranularity.CHANNELWISE),
      (True, _QuantGranularity.TENSORWISE),
      (False, _QuantGranularity.CHANNELWISE),
      (False, _QuantGranularity.TENSORWISE),
  )
  def test_generate_config_int8xint8_single_fc(
      self, act_symmetric, channelwise_weight
  ):
    single_fc_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/single_fc.tflite'
    )
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=_TensorQuantConfig(
                num_bits=8, symmetric=act_symmetric
            ),
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8, symmetric=True, granularity=channelwise_weight
            ),
            # Equivalent to SRQ.
            compute_precision=_ComputePrecision.INTEGER,
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

    # Input tensor producer (from the virtual input op).
    self._test_tensor_transformation_params(
        -1,  # virtual input op.
        quant_params,
        'serving_default_input_1:0',
        [_QuantTransformation.ADD_DEQUANTIZE],
        num_bits=8,
        granularity=_QuantGranularity.TENSORWISE,
        symmetric=act_symmetric,
        is_inbounding_tensor=False,
    )
    # Input tensor consumer.
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'serving_default_input_1:0',
        [_QuantTransformation.ADD_QUANTIZE],
        num_bits=8,
        granularity=_QuantGranularity.TENSORWISE,
        symmetric=act_symmetric,
        is_inbounding_tensor=True,
    )

    # output tensor producer.
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'StatefulPartitionedCall:0',
        [_QuantTransformation.ADD_DEQUANTIZE],
        num_bits=8,
        granularity=_QuantGranularity.TENSORWISE,
        symmetric=act_symmetric,
        is_inbounding_tensor=False,
    )
    # output tensor consumer (into the virtual output op).
    self._test_tensor_transformation_params(
        -1,  # virtual output op.
        quant_params,
        'StatefulPartitionedCall:0',
        [_QuantTransformation.ADD_QUANTIZE],
        num_bits=8,
        granularity=_QuantGranularity.TENSORWISE,
        symmetric=act_symmetric,
        is_inbounding_tensor=True,
    )

    # weights
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'sequential/dense/MatMul',
        [_QuantTransformation.QUANTIZE_TENSOR],
        num_bits=8,
        granularity=channelwise_weight,
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
        granularity=channelwise_weight,
        symmetric=True,
        is_inbounding_tensor=True,
    )

  @parameterized.parameters('weight_only', 'DRQ')
  def test_generate_params_buffer_sharing_graphs_succeeds(
      self, the_other_fc_difference
  ):
    model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/weight_sharing_fcs.tflite'
    )
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
            # Equivalent to WEIGHT_ONLY.
            compute_precision=_ComputePrecision.FLOAT,
            explicit_dequantize=True,
        ),
    )
    if the_other_fc_difference == 'DRQ':
      self._recipe_manager.add_quantization_config(
          regex='PartitionedCall_1:0',
          operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
          op_config=qtyping.OpQuantizationConfig(
              # Equivalent to DRQ.
              compute_precision=_ComputePrecision.INTEGER,
          ),
      )
    pg = params_generator.ParamsGenerator(model_path)
    quant_params = pg.generate_quantization_parameters(
        self._recipe_manager,
    )
    self.assertLen(quant_params, 6)

  @parameterized.parameters('no_quant', 'execution_mode', 'num_bits')
  def test_generate_params_buffer_sharing_graphs_fails(
      self, the_other_fc_difference
  ):
    model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/weight_sharing_fcs.tflite'
    )
    # Setup the quantization config for the first FC.
    self._recipe_manager.add_quantization_config(
        regex='PartitionedCall:0',
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(num_bits=8),
            compute_precision=_ComputePrecision.INTEGER,
        ),
    )
    # Setup the quantization config for the second FC (weight shared with the
    # first FC).
    if the_other_fc_difference == 'no_quant':
      pass
    elif the_other_fc_difference == 'num_bits':
      self._recipe_manager.add_quantization_config(
          regex='PartitionedCall_1:0',
          operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
          op_config=qtyping.OpQuantizationConfig(
              weight_tensor_config=_TensorQuantConfig(num_bits=4),
              compute_precision=_ComputePrecision.INTEGER,
          ),
      )
    pg = params_generator.ParamsGenerator(model_path)
    error_message = 'do not have the same quantization parameters'
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda err: error_message in str(err)
    ):
      pg.generate_quantization_parameters(
          self._recipe_manager,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='producer_incompatible',
          param1=qtyping.TensorTransformationParams(
              tensor_name='tfl.quantize',
              producer=qtyping.OpToTensorParams(
                  subgraph_op_id=0,
                  transformations=[qtyping.QuantTransformation.ADD_DEQUANTIZE],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              consumers=[
                  qtyping.OpToTensorParams(
                      subgraph_op_id=1,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=2,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=3,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=4,
                      transformations=[
                          qtyping.QuantTransformation.NO_QUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
              ],
          ),
          param2=qtyping.TensorTransformationParams(
              'tfl.other_quantize',
              qtyping.OpToTensorParams(
                  subgraph_op_id=0,
                  transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              [
                  qtyping.OpToTensorParams(
                      subgraph_op_id=1,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=2,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=3,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
              ],
          ),
          expected=False,
      ),
      dict(
          testcase_name='param2_consumer_incompatible',
          param1=qtyping.TensorTransformationParams(
              tensor_name='tfl.quantize',
              producer=qtyping.OpToTensorParams(
                  subgraph_op_id=0,
                  transformations=[qtyping.QuantTransformation.ADD_QUANTIZE],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              consumers=[
                  qtyping.OpToTensorParams(
                      subgraph_op_id=1,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=2,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=3,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
              ],
          ),
          param2=qtyping.TensorTransformationParams(
              'tfl.other_quantize',
              qtyping.OpToTensorParams(
                  subgraph_op_id=0,
                  transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
                  parameters=qtyping.UniformQuantParams(
                      8, None, np.array([1]), np.array([0])
                  ),
              ),
              [
                  qtyping.OpToTensorParams(
                      subgraph_op_id=1,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=2,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=3,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=4,
                      transformations=[
                          qtyping.QuantTransformation.QUANTIZE_TENSOR,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
              ],
          ),
          expected=False,
      ),
      dict(
          testcase_name='compatible',
          param1=qtyping.TensorTransformationParams(
              tensor_name='tfl.quantize',
              producer=None,
              consumers=[
                  qtyping.OpToTensorParams(
                      subgraph_op_id=2,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=3,
                      transformations=[
                          qtyping.QuantTransformation.NO_QUANTIZE,
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=4,
                      transformations=[
                          qtyping.QuantTransformation.NO_QUANTIZE,
                      ],
                  ),
              ],
          ),
          param2=qtyping.TensorTransformationParams(
              'tfl.other_quantize',
              None,
              [
                  qtyping.OpToTensorParams(
                      subgraph_op_id=1,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=2,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=3,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                          qtyping.QuantTransformation.ADD_DEQUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
                  qtyping.OpToTensorParams(
                      subgraph_op_id=4,
                      transformations=[
                          qtyping.QuantTransformation.ADD_QUANTIZE,
                      ],
                      parameters=qtyping.UniformQuantParams(
                          8, None, np.array([1]), np.array([0])
                      ),
                  ),
              ],
          ),
          expected=True,
      ),
  )
  def test_params_compatible(self, param1, param2, expected):
    # adding a test to make production coverage happy.
    self.assertEqual(
        params_generator._compatible_tensor_transformation_params(
            param1, param2
        ),
        expected,
    )

  def test_model_with_duplicated_tensor_names_fails(self):
    model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/duplicated_tensor_names.tflite'
    )
    error_message = 'Tensor name test_same_name is not unique in the model.'
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      params_generator.ParamsGenerator(model_path)

  def test_quantize_integer_input_output(self):
    model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/single_transpose_int32.tflite'
    )
    self._recipe_manager.add_quantization_config(
        regex='.*',
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=_TensorQuantConfig(
                num_bits=8, symmetric=False
            ),
            weight_tensor_config=_TensorQuantConfig(num_bits=8, symmetric=True),
            # Equivalent to SRQ.
            compute_precision=_ComputePrecision.INTEGER,
        ),
    )
    pg = params_generator.ParamsGenerator(model_path)

    # Calibrate then quantize.
    model_calibrator = calibrator.Calibrator(model_path)
    model_calibrator.calibrate(
        _int_transpose_model_representative_dataset_gen(), self._recipe_manager
    )
    model_qsvs = model_calibrator.get_model_qsvs()
    quant_params = pg.generate_quantization_parameters(
        self._recipe_manager,
        model_qsvs,
    )
    self.assertLen(quant_params, 3)

    self._test_tensor_transformation_params(
        -1,  # virtual input op.
        quant_params,
        'serving_default_input_2:0',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )
    # Input tensor consumer.
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'serving_default_input_2:0',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )

    # Output tensor producer.
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'PartitionedCall:0',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )
    # output tensor consumer (into the virtual output op).
    self._test_tensor_transformation_params(
        -1,  # virtual output op.
        quant_params,
        'PartitionedCall:0',
        [_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )

    # perm
    self._test_tensor_transformation_params(
        0,
        quant_params,
        'sequential_1/permute_1/transpose/perm',
        [_QuantTransformation.NO_QUANTIZE],
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
      granularity=_QuantGranularity.TENSORWISE,
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
      if granularity is _QuantGranularity.CHANNELWISE:
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


class ParamsGeneratorAlreadyQuantizedModelTest(googletest.TestCase):

  def test_check_is_float_model_succeeds_when_model_is_float(self):
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/conv_fc_mnist.tflite'
    )
    _ = params_generator.ParamsGenerator(test_model_path)

  def test_check_is_float_model_raises_error_when_model_is_quantized(self):
    test_model_path = os.path.join(
        TEST_DATA_PREFIX_PATH, 'tests/models/mnist_quantized.tflite'
    )
    with self.assertRaisesRegex(
        ValueError,
        'The input model for quantization parameters generation is not a float'
        ' model.',
    ):
      _ = params_generator.ParamsGenerator(test_model_path)


if __name__ == '__main__':
  googletest.main()
