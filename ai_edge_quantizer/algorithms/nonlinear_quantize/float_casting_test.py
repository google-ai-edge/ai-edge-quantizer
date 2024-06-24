import os
from absl.testing import parameterized
import numpy as np
from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.nonlinear_quantize import float_casting
from ai_edge_quantizer.utils import test_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../../tests/models")
_TFLOpName = qtyping.TFLOperationName
_OpExecutionMode = qtyping.OpExecutionMode
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation


class Fp16QuantizeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = os.path.join(
        _TEST_DATA_PREFIX_PATH, "conv_fc_mnist.tflite"
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._tensor_name_to_qsv = {}

  @parameterized.named_parameters(
      dict(
          testcase_name="fc",
          op_name=_TFLOpName.FULLY_CONNECTED,
      ),
      dict(
          testcase_name="conv2d",
          op_name=_TFLOpName.CONV_2D,
      ),
  )
  def test_check_op_quantization_config_succeeds(self, op_name):
    float_casting.check_op_quantization_config(
        op_name,
        qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT
            ),
            execution_mode=_OpExecutionMode.WEIGHT_ONLY,
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_fc",
          op_name=_TFLOpName.FULLY_CONNECTED,
      ),
      dict(
          testcase_name="invalid_conv2d",
          op_name=_TFLOpName.CONV_2D,
      ),
  )
  def test_check_op_quantization_config_invalid_activation_tensor_config_raises_exception(
      self, op_name
  ):
    # With activation tensor config.
    error_message = (
        "Activation tensor quantization is not supported for float casting"
        " quantization."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      float_casting.check_op_quantization_config(
          op_name,
          qtyping.OpQuantizationConfig(
              activation_tensor_config=_TensorQuantConfig(
                  num_bits=16, dtype=qtyping.TensorDataType.FLOAT
              ),
              weight_tensor_config=_TensorQuantConfig(
                  num_bits=16, dtype=qtyping.TensorDataType.FLOAT
              ),
              execution_mode=_OpExecutionMode.WEIGHT_ONLY,
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_fc",
          op_name=_TFLOpName.FULLY_CONNECTED,
      ),
      dict(
          testcase_name="invalid_conv2d",
          op_name=_TFLOpName.CONV_2D,
      ),
  )
  def test_check_op_quantization_config_invalid_bit_width_raises_exception(
      self, op_name
  ):
    error_message = (
        "float casting quantization requires number of bits to be set as 16,"
        " dtype as float"
    )
    # Wrong bit width.
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      float_casting.check_op_quantization_config(
          op_name,
          qtyping.OpQuantizationConfig(
              activation_tensor_config=None,
              weight_tensor_config=_TensorQuantConfig(
                  num_bits=8, dtype=qtyping.TensorDataType.FLOAT
              ),
              execution_mode=_OpExecutionMode.WEIGHT_ONLY,
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_fc",
          op_name=_TFLOpName.FULLY_CONNECTED,
      ),
      dict(
          testcase_name="invalid_conv2d",
          op_name=_TFLOpName.CONV_2D,
      ),
  )
  def test_check_op_quantization_config_invalid_dtype_raises_exception(
      self, op_name
  ):
    error_message = (
        "float casting quantization requires number of bits to be set as 16,"
        " dtype as float"
    )
    # Wrong dtype.
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      float_casting.check_op_quantization_config(
          op_name,
          qtyping.OpQuantizationConfig(
              activation_tensor_config=None,
              weight_tensor_config=_TensorQuantConfig(
                  num_bits=16, dtype=qtyping.TensorDataType.INT
              ),
              execution_mode=_OpExecutionMode.WEIGHT_ONLY,
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="averagepool2D",
          op_name=_TFLOpName.AVERAGE_POOL_2D,
      ),
      dict(
          testcase_name="reshape",
          op_name=_TFLOpName.RESHAPE,
      ),
  )
  def test_check_op_quantization_config_invalid_ops_raises_exception(
      self, op_name
  ):
    error_message = "Unsupported op"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      float_casting.check_op_quantization_config(
          op_name=op_name,
          op_quant_config=qtyping.OpQuantizationConfig(
              activation_tensor_config=None,
              weight_tensor_config=_TensorQuantConfig(
                  num_bits=16, dtype=qtyping.TensorDataType.FLOAT
              ),
              execution_mode=_OpExecutionMode.WEIGHT_ONLY,
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="fc_with_bias",
          subgraph_op_id=3,
          op_tensor_names={
              "weight": "arith.constant1",
              "bias": "arith.constant2",
              "input": "sequential/flatten/Reshape",
              "output": "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd",
          },
      ),
      dict(
          testcase_name="fc_with_no_bias",
          subgraph_op_id=4,
          op_tensor_names={
              "weight": "arith.constant",
              "input": "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd",
              "output": "sequential/dense_1/MatMul",
          },
      ),
  )
  def test_fully_connected_weight_only_succeeds(
      self, subgraph_op_id, op_tensor_names
  ):
    subgraph0 = self._test_model.subgraphs[0]
    fc_op = subgraph0.operators[subgraph_op_id]
    op_info = qtyping.OpInfo(
        op=fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT
            ),
            execution_mode=_OpExecutionMode.WEIGHT_ONLY,
        ),
    )

    self._test_fc_conv(
        op_info,
        self._graph_info,
        op_tensor_names,
        float_casting.materialize_fc_conv,
    )

  def test_conv2d_weight_only_succeeds(self):
    # Read from Model Explorer.
    subgraph0 = self._test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]

    op_info = qtyping.OpInfo(
        op=op,
        op_name=_TFLOpName.CONV_2D,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT
            ),
            execution_mode=_OpExecutionMode.WEIGHT_ONLY,
        ),
    )

    op_tensor_names = {}
    op_tensor_names["weight"] = "sequential/conv2d/Conv2D"
    op_tensor_names["bias"] = (
        "sequential/conv2d/Relu;sequential/conv2d/BiasAdd;sequential/conv2d/Conv2D;sequential/conv2d/BiasAdd/ReadVariableOp"
    )
    op_tensor_names["input"] = "serving_default_conv2d_input:0"
    op_tensor_names["output"] = (
        "sequential/conv2d/Relu;sequential/conv2d/BiasAdd;sequential/conv2d/Conv2D;sequential/conv2d/BiasAdd/ReadVariableOp1"
    )
    self._test_fc_conv(
        op_info,
        self._graph_info,
        op_tensor_names,
        float_casting.materialize_fc_conv,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_fc",
          op_name=_TFLOpName.FULLY_CONNECTED,
      ),
      dict(
          testcase_name="invalid_conv2d",
          op_name=_TFLOpName.CONV_2D,
      ),
  )
  def test_check_op_quantization_config_invalid_execution_mode_raises_exception(
      self, op_name
  ):
    # Use DRQ instead of WEIGHT-ONLY.
    error_message = (
        "Currently, only Weight-Only is supported for float casting"
        " quantization."
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      float_casting.check_op_quantization_config(
          op_name,
          qtyping.OpQuantizationConfig(
              activation_tensor_config=None,
              weight_tensor_config=_TensorQuantConfig(
                  num_bits=16, dtype=qtyping.TensorDataType.FLOAT
              ),
              execution_mode=_OpExecutionMode.DRQ,
          ),
      )

  def _test_fc_conv(
      self,
      op_info,
      graph_info,
      op_tensor_names,
      materialization_func,
  ):

    tensor_quant_params = materialization_func(
        op_info, graph_info, self._tensor_name_to_qsv
    )
    _, weight_tensor, bias_tensor, _ = (
        tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
            op_info.op, graph_info.subgraph_tensors
        )
    )

    num_configs = 4 if bias_tensor is not None else 3
    self.assertLen(tensor_quant_params, num_configs)

    # Test input tensor params.
    self._test_fp16_nonweight_tensor_transformation_params(
        op_tensor_names["input"],
        op_info.subgraph_op_index,
        transformation_params=tensor_quant_params[0],
        desired_transformations=[_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=True,
    )

    # Test weight tensor params.
    weight_tensor_data = tfl_flatbuffer_utils.get_tensor_data(
        weight_tensor,
        self._test_model.buffers,
    )

    self._test_fp16_weight_tensor_transformation_params(
        op_tensor_names["weight"],
        op_info.subgraph_op_index,
        tensor_quant_config=op_info.op_quant_config.weight_tensor_config,
        transformation_params=tensor_quant_params[1],
        desired_transformations=[_QuantTransformation.ADD_DEQUANTIZE],
        tensor_data=weight_tensor_data,
    )
    # Test output tensor params.
    self._test_fp16_nonweight_tensor_transformation_params(
        op_tensor_names["output"],
        op_info.subgraph_op_index,
        transformation_params=tensor_quant_params[2],
        desired_transformations=[_QuantTransformation.NO_QUANTIZE],
        is_inbounding_tensor=False,
    )

    # Test bias tensor params.
    if bias_tensor is not None:
      self._test_fp16_nonweight_tensor_transformation_params(
          op_tensor_names["bias"],
          op_info.subgraph_op_index,
          transformation_params=tensor_quant_params[3],
          desired_transformations=[_QuantTransformation.NO_QUANTIZE],
          is_inbounding_tensor=True,
      )

  def _test_fp16_weight_tensor_transformation_params(
      self,
      tensor_name,
      subgraph_op_id,
      tensor_quant_config,
      transformation_params,
      desired_transformations,
      tensor_data,
  ):
    self.assertEqual(transformation_params.tensor_name, tensor_name)
    # Weight-only means the transformation is added from the consumer.
    self.assertIsNone(transformation_params.producer)
    self.assertLen(transformation_params.consumers, 1)
    # Check op params.
    op_params = transformation_params.consumers[0]
    self.assertEqual(op_params.subgraph_op_id, subgraph_op_id)
    self.assertSequenceEqual(op_params.transformations, desired_transformations)
    # Check quantization params.
    quantization_params = op_params.parameters
    self.assertIsNotNone(quantization_params)
    self.assertEqual(quantization_params.num_bits, tensor_quant_config.num_bits)
    self.assertEqual(quantization_params.quantized_data.dtype, "float16")
    # fp16 quantization implies very small error.
    self.assertSequenceAlmostEqual(
        list(tensor_data.flatten()),  # pytype: disable=attribute-error
        list(quantization_params.quantized_data.flatten()),  # pytype: disable=attribute-error
        delta=5,
    )

  def _test_fp16_nonweight_tensor_transformation_params(
      self,
      tensor_name,
      subgraph_op_id,
      transformation_params,
      desired_transformations,
      is_inbounding_tensor,
  ):
    self.assertEqual(transformation_params.tensor_name, tensor_name)
    if is_inbounding_tensor:
      self.assertIsNone(transformation_params.producer)
      self.assertLen(transformation_params.consumers, 1)
      op_params = transformation_params.consumers[0]
    else:
      self.assertIsNone(transformation_params.consumers)
      op_params = transformation_params.producer
    self.assertIsNotNone(op_params)
    self.assertEqual(op_params.subgraph_op_id, subgraph_op_id)
    self.assertSequenceEqual(op_params.transformations, desired_transformations)
    self.assertIsNone(op_params.parameters)


if __name__ == "__main__":
  googletest.main()
