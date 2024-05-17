import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from quantization_toolkit import qtyping
from quantization_toolkit.algorithms.uniform_quantize import naive_min_max_quantize
from quantization_toolkit.algorithms.uniform_quantize.naive_min_max_quantize_op_tests import test_utils as naive_min_max_test_utils
from quantization_toolkit.utils import test_utils
from quantization_toolkit.utils import tfl_flatbuffer_utils

_TFLOpName = qtyping.TFLOperationName
_OpExecutionMode = qtyping.OpExecutionMode
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation
_OpTestInfo = naive_min_max_test_utils.OpTestInfo

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../../../test_models")
_DEFAULT_ACTIVATION_QUANT_SETTING = (
    naive_min_max_test_utils.DEFAULT_ACTIVATION_QUANT_SETTING
)


class FullyConnectedTest(naive_min_max_test_utils.NaiveMinMaxQuantizeTest):

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = os.path.join(
        _TEST_DATA_PREFIX_PATH, "conv_fc_mnist.tflite"
    )
    self._op_test_info = _OpTestInfo(
        test_model=tfl_flatbuffer_utils.read_model(self._test_model_path),
        model_buffer=tfl_flatbuffer_utils.get_model_buffer(
            self._test_model_path
        ),
        op_tensor_names={},
        input_range=(np.array([[-10]]), np.array([[8]])),
        output_range=(np.array([[10]]), np.array([[88]])),
    )
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._op_test_info.test_model.subgraphs[0].tensors,
        buffers=self._op_test_info.test_model.buffers,
        whole_model_buffer=self._op_test_info.model_buffer,
    )

  # TODO(rewu): add int16 tests.
  @parameterized.product(
      num_bits_weight=(4, 8),
      symmetric_weight=(True, False),
      channel_wise_weight=(True, False),
      execution_mode=(
          _OpExecutionMode.WEIGHT_ONLY,
          _OpExecutionMode.DRQ,
          _OpExecutionMode.SRQ,
      ),
  )
  def test_materialize_fully_connected_succeeds(
      self,
      num_bits_weight,
      symmetric_weight,
      channel_wise_weight,
      execution_mode,
  ):
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 3
    fc_op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["weight"] = "arith.constant1"
    op_tensor_names["bias"] = "arith.constant2"
    op_tensor_names["input"] = "sequential/flatten/Reshape"
    op_tensor_names["output"] = (
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    )
    self._op_test_info.op_tensor_names = op_tensor_names

    activation_tensor_config = None
    if execution_mode == _OpExecutionMode.SRQ:
      activation_tensor_config = _DEFAULT_ACTIVATION_QUANT_SETTING
    op_info = qtyping.OpInfo(
        op=fc_op,
        op_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=num_bits_weight,
                symmetric=symmetric_weight,
                channel_wise=channel_wise_weight,
            ),
            execution_mode=execution_mode,
        ),
    )

    self._test_fc_bmm_conv(
        op_info,
        self._graph_info,
        self._op_test_info,
        naive_min_max_quantize.materialize_fc_conv,
    )


if __name__ == "__main__":
  googletest.main()
