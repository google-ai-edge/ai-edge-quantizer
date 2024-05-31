"""Tests for algorithm_manager_api."""

from absl.testing import parameterized
from tensorflow.python.platform import googletest
from ai_edge_quantizer import algorithm_manager_api
from ai_edge_quantizer import qtyping

_TFLOpName = qtyping.TFLOperationName


# Sample functions for test cases.
def _sample_init_qsvs(*_, **__):
  return 1.0, dict()


def _sample_calibration_func(*_, **__):
  return 2.0, dict()


def _sample_materialize_func(*_, **__):
  return 3.0, dict()


def _sample_check_op_config_func(_, op_config):
  if op_config.weight_tensor_config.num_bits == 17:
    raise ValueError("Unsupported number of bits.")


class AlgorithmManagerApiTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._alg_manager = algorithm_manager_api.AlgorithmManagerApi()

  def test_register_op_quant_config_validation_func_succeeds(self):
    self.assertEmpty(self._alg_manager._config_check_registry)
    test_algorithm_name = "test_algorithm"
    self._alg_manager.register_op_quant_config_validation_func(
        test_algorithm_name, _sample_check_op_config_func
    )
    self.assertIn(test_algorithm_name, self._alg_manager._config_check_registry)
    check_func = self._alg_manager._config_check_registry[test_algorithm_name]
    self.assertEqual(check_func, _sample_check_op_config_func)

  def test_register_quantized_op(self):
    self._alg_manager.register_quantized_op(
        algorithm_key="ptq",
        tfl_op_name=_TFLOpName.FULLY_CONNECTED,
        init_qsv_func=_sample_init_qsvs,
        calibration_func=_sample_calibration_func,
        materialize_func=_sample_materialize_func,
    )
    self._alg_manager.register_quantized_op(
        algorithm_key="gptq",
        tfl_op_name=_TFLOpName.CONV_2D,
        init_qsv_func=_sample_init_qsvs,
        calibration_func=_sample_calibration_func,
        materialize_func=_sample_materialize_func,
    )
    self.assertTrue(self._alg_manager.is_algorithm_registered("ptq"))
    self.assertTrue(self._alg_manager.is_algorithm_registered("gptq"))
    self.assertTrue(
        self._alg_manager.is_op_registered("ptq", _TFLOpName.FULLY_CONNECTED)
    )
    self.assertTrue(
        self._alg_manager.is_op_registered("gptq", _TFLOpName.CONV_2D)
    )
    self.assertFalse(
        self._alg_manager.is_op_registered("gptq", _TFLOpName.DEPTHWISE_CONV_2D)
    )

  def test_get_supported_ops(self):
    algorithm_key = "ptq"
    self._alg_manager.register_quantized_op(
        algorithm_key=algorithm_key,
        tfl_op_name=_TFLOpName.FULLY_CONNECTED,
        init_qsv_func=_sample_init_qsvs,
        calibration_func=_sample_calibration_func,
        materialize_func=_sample_materialize_func,
    )
    self._alg_manager.register_quantized_op(
        algorithm_key=algorithm_key,
        tfl_op_name=_TFLOpName.CONV_2D,
        init_qsv_func=_sample_init_qsvs,
        calibration_func=_sample_calibration_func,
        materialize_func=_sample_materialize_func,
    )
    supported_ops = self._alg_manager.get_supported_ops(algorithm_key)
    self.assertIn(_TFLOpName.CONV_2D, supported_ops)
    self.assertIn(_TFLOpName.FULLY_CONNECTED, supported_ops)
    self.assertNotIn(_TFLOpName.DEPTHWISE_CONV_2D, supported_ops)

  def test_get_quantization_func(self):
    algorithm_key = "ptq"
    tfl_op = _TFLOpName.FULLY_CONNECTED
    self._alg_manager.register_quantized_op(
        algorithm_key=algorithm_key,
        tfl_op_name=tfl_op,
        init_qsv_func=_sample_init_qsvs,
        calibration_func=_sample_calibration_func,
        materialize_func=_sample_materialize_func,
    )
    materialize_func = self._alg_manager.get_quantization_func(
        algorithm_key,
        tfl_op,
        qtyping.QuantizeMode.MATERIALIZE,
    )
    self.assertEqual(_sample_materialize_func()[0], materialize_func()[0])
    calibration_func = self._alg_manager.get_quantization_func(
        algorithm_key,
        tfl_op,
        qtyping.QuantizeMode.CALIBRATE,
    )
    self.assertEqual(_sample_calibration_func()[0], calibration_func()[0])

    # Query for unsupported operation.
    error_message = "Unsupported operation"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._alg_manager.get_quantization_func(
          algorithm_key,
          _TFLOpName.BATCH_MATMUL,
          qtyping.QuantizeMode.MATERIALIZE,
      )

    # Query for unregisted algorithm.
    error_message = "Unregistered algorithm"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._alg_manager.get_quantization_func(
          "gptq",
          tfl_op,
          qtyping.QuantizeMode.MATERIALIZE,
      )

  def test_get_init_qsv_func(self):
    algorithm_key = "ptq"
    tfl_op = _TFLOpName.FULLY_CONNECTED
    self._alg_manager.register_quantized_op(
        algorithm_key=algorithm_key,
        tfl_op_name=tfl_op,
        init_qsv_func=_sample_init_qsvs,
        calibration_func=_sample_calibration_func,
        materialize_func=_sample_materialize_func,
    )
    init_qsv_func = self._alg_manager.get_init_qsv_func(algorithm_key, tfl_op)
    self.assertEqual(_sample_init_qsvs()[0], init_qsv_func()[0])

    # Query for unsupported operation.
    error_message = "Unsupported operation"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._alg_manager.get_init_qsv_func(
          algorithm_key,
          _TFLOpName.BATCH_MATMUL,
      )

    # Query for unregisted algorithm.
    error_message = "Unregistered algorithm"
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: error_message in str(err)
    ):
      self._alg_manager.get_init_qsv_func(
          "gptq",
          tfl_op,
      )


if __name__ == "__main__":
  googletest.main()
