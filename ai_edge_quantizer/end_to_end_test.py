"""E2E tests for the quantizer."""

from absl.testing import parameterized
from tensorflow.python.platform import googletest
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.utils import test_utils

_OpExecutionMode = qtyping.OpExecutionMode
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig


class E2ETest(parameterized.TestCase):

  @parameterized.product(
      execution_mode=[_OpExecutionMode.WEIGHT_ONLY, _OpExecutionMode.DRQ],
      symmetric_weight=[True, False],
      channel_wise_weight=[True, False],
  )
  def test_mnist_toy_model_int8_ptq(
      self, execution_mode, symmetric_weight, channel_wise_weight
  ):

    # asym DRQ is not supported
    # TODO: b/335254997 - fail when trying to use unsupported recipe.
    if execution_mode == _OpExecutionMode.DRQ and not symmetric_weight:
      return

    float_model_path = test_utils.get_path_to_datafile(
        './test_models/conv_fc_mnist.tflite'
    )
    qt = quantizer.Quantizer(float_model_path)
    qt.update_quantization_recipe(
        regex='.*',
        operation_name=_OpName.FULLY_CONNECTED,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=symmetric_weight,
                channel_wise=channel_wise_weight,
            ),
            execution_mode=execution_mode,
        ),
    )
    _ = qt.quantize()
    comparion_result = qt.compare(error_metrics='mse')

    # Check weight tensors.
    tolerance = 1e-2 if channel_wise_weight else 1e-1
    conv_weight_mse = comparion_result['sequential/conv2d/Conv2D']
    self.assertLess(conv_weight_mse, tolerance)
    fc1_weight_mse = comparion_result['arith.constant1']
    self.assertLess(fc1_weight_mse, tolerance)
    fc2_weight_mse = comparion_result['arith.constant']
    self.assertLess(fc2_weight_mse, tolerance)
    # check logits.
    logits_mse = comparion_result['sequential/dense_1/MatMul']
    self.assertLess(logits_mse, tolerance)
    # check final output.
    output_mse = comparion_result['StatefulPartitionedCall:0']
    self.assertLess(output_mse, tolerance)


if __name__ == '__main__':
  googletest.main()
