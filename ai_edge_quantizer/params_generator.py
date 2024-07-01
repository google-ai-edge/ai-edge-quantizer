"""Generate model tensor level quantization config."""

import copy
from typing import Any, Optional, Union

from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe_manager
from ai_edge_quantizer.utils import tfl_flatbuffer_utils


class ParamsGenerator:
  """Generate model tensor level quantization parameters."""

  def __init__(self, float_tflite: Union[str, bytearray]):
    self.flatbuffer_model = tfl_flatbuffer_utils.read_model(float_tflite)
    self.buffer_to_tensors: dict[int, list[Any]] = (
        tfl_flatbuffer_utils.buffer_to_tensors(self.flatbuffer_model)
    )
    self.model_quant_results: dict[str, qtyping.TensorTransformationParams] = {}

  def generate_quantization_parameters(
      self,
      model_recipe_manager: recipe_manager.RecipeManager,
      model_qsvs: Optional[dict[str, qtyping.QSV]] = None,
  ) -> dict[str, qtyping.TensorTransformationParams]:
    """Generate the quantization parameters for the model.

    Args:
      model_recipe_manager: the recipe manager for the model.
      model_qsvs: quantization statistics values (qsvs) for the model. This is
        obtained through calibration process.

    Returns:
      model_quant_results: the quantization parameters for tensors in the model.

    Raises:
      RuntimeError: if the calibration dataset is required but not provided.
    """
    if model_recipe_manager.need_calibration() and not model_qsvs:
      raise RuntimeError(
          'Model quantization statistics values (QSVs) are required for the'
          ' input recipe. This can be obtained by running calibration on sample'
          ' dataset.'
      )

    if model_qsvs is None:
      model_qsvs = {}

    op_codes = self.flatbuffer_model.operatorCodes
    for subgraph in self.flatbuffer_model.subgraphs:
      graph_info = qtyping.GraphInfo(
          subgraph.tensors, self.flatbuffer_model.buffers
      )
      for subgraph_op_id, op in enumerate(subgraph.operators):
        op_code = op_codes[op.opcodeIndex].builtinCode
        # Do not quantize unknown ops.
        if op_code not in tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME:
          op_quant_results = self._get_params_for_no_quant_op(
              subgraph_op_id, op, subgraph.tensors
          )
        else:
          op_key = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME[op_code]
          # Step1: query the quantization_recipe to get op config.
          op_scope = self._get_op_scope(op, subgraph.tensors)
          algorithm_name, op_quant_config = (
              model_recipe_manager.get_quantization_configs(op_key, op_scope)
          )
          if algorithm_name == algorithm_manager.AlgorithmName.NO_QUANTIZE:
            op_quant_results = self._get_params_for_no_quant_op(
                subgraph_op_id, op, subgraph.tensors
            )
          else:
            op_info = qtyping.OpInfo(
                op, op_key, subgraph_op_id, op_quant_config
            )
            # Step2: query algorithm_manager to get/call the related function.
            materialize_func = algorithm_manager.get_quantization_func(
                algorithm_name,
                op_key,
                qtyping.QuantizeMode.MATERIALIZE,
            )
            op_quant_results = materialize_func(
                op_info,
                graph_info,
                model_qsvs,
            )
        # Step3: update the results.
        self._update_model_quant_results(op_quant_results)

    self._post_process_results()
    return self.model_quant_results

  def _post_process_results(self) -> None:
    """Post process the quantization results.

    Raises:
      RuntimeError: if the tensors sharing the same buffer have different
      quantization settings.
    """
    self._check_buffer_sharing()
    # Modify quantization tensor for I/O tensors.
    for _, tensor_params in self.model_quant_results.items():
      self._modify_io_tensor_transformations(tensor_params)

  def _update_model_quant_results(
      self,
      op_tensor_results: list[qtyping.TensorTransformationParams],
  ) -> None:
    """Update the op quantization results to the final output.

    Args:
      op_tensor_results: list of tensor level quantization params for the op.

    Raises:
      RuntimeError: if the same tensor has multiple quantization configs.
    """

    for op_tensor_result in op_tensor_results:
      tensor_name = op_tensor_result.tensor_name
      if tensor_name not in self.model_quant_results:
        self.model_quant_results[tensor_name] = copy.deepcopy(op_tensor_result)
      else:
        tensor_params = self.model_quant_results[tensor_name]
        # Set source op.
        if op_tensor_result.producer is not None:
          # src params must be unique (a tensor can only be produced by one op).
          if tensor_params.producer is not None:
            raise RuntimeError(
                'Tensor %s received multiple quantization parameters from the'
                ' source op, which should not happen as every tensor should'
                ' have only one source op.' % tensor_name
            )
          tensor_params.producer = copy.deepcopy(op_tensor_result.producer)
        # Set target op, which can be multiple (a tensor can be consumed by
        # multiple ops).
        if op_tensor_result.consumers is not None:
          if tensor_params.consumers is None:
            tensor_params.consumers = copy.deepcopy(op_tensor_result.consumers)
          else:
            tensor_params.consumers += copy.deepcopy(op_tensor_result.consumers)
        self.model_quant_results[tensor_name] = tensor_params

  def _get_op_scope(self, op: Any, subgraph_tensors: list[Any]) -> str:
    """Get the op scope.

    Op scope is defined by the output tensor names (following the Model
    Explorer).

    Args:
      op: the op that need to be parsed.
      subgraph_tensors: list of tensors in the subgraph.

    Returns:
      scope: scope for the op.
    """
    scope = ''
    # Op scope is determined by output tensors.
    for output_tensor_idx in op.outputs:
      if output_tensor_idx != -1:
        scope += tfl_flatbuffer_utils.get_tensor_name(
            subgraph_tensors[output_tensor_idx]
        )
        scope += ';'  # split names
    return scope

  def _get_params_for_no_quant_op(
      self,
      subgraph_op_id: int,
      op: Any,
      subgraph_tensors: list[Any],
  ) -> list[qtyping.TensorTransformationParams]:
    """Get the quantization parameters for ops require no quantization.

    Args:
      subgraph_op_id: the op id in the subgraph.
      op: the op that need to be parsed.
      subgraph_tensors: list of tensors in the subgraph.

    Returns:
      tensor_params: list of tensor level quantization params for the op.
    """

    def no_quant_tensor_params():
      return qtyping.OpToTensorParams(
          subgraph_op_id=subgraph_op_id,
          transformations=[qtyping.QuantTransformation.NO_QUANTIZE],
      )

    tensor_params = []
    for input_tensor_idx in op.inputs:
      if input_tensor_idx != -1:
        tensor = subgraph_tensors[input_tensor_idx]
        input_tensor_params = qtyping.TensorTransformationParams(
            tensor_name=tfl_flatbuffer_utils.get_tensor_name(tensor),
            consumers=[no_quant_tensor_params()],
        )
        tensor_params.append(input_tensor_params)

    for output_tensor_idx in op.outputs:
      if output_tensor_idx != -1:
        tensor = subgraph_tensors[output_tensor_idx]
        output_tensor_params = qtyping.TensorTransformationParams(
            tensor_name=tfl_flatbuffer_utils.get_tensor_name(tensor),
            producer=no_quant_tensor_params(),
        )
        tensor_params.append(output_tensor_params)
    return tensor_params

  def _check_buffer_sharing(self) -> None:
    """Check if tensors sharing the same buffer have the same quantization.

    Raises:
      RuntimeError: if the tensors sharing the same buffer have different
      quantization settings.
    """
    for tensors in self.buffer_to_tensors.values():
      first_tensor = tensors[0]
      first_tensor_params = self.model_quant_results.get(
          tfl_flatbuffer_utils.get_tensor_name(first_tensor), None
      )
      for tensor in tensors[1:]:
        tensor_params = self.model_quant_results.get(
            tfl_flatbuffer_utils.get_tensor_name(tensor), None
        )
        error_msg = (
            f'The tensors {first_tensor.name} and {tensor.name} do not have the'
            ' same quantization parameters even though they share the same'
            ' buffer. Please modify your quantization recipe to make sure the'
            ' two tensors have the same quantization settings.'
        )
        if not _compatible_tensor_transformation_params(
            first_tensor_params, tensor_params
        ):
          raise RuntimeError(error_msg)

  def _modify_io_tensor_transformations(
      self,
      tensor_params: qtyping.TensorTransformationParams,
  ) -> None:
    """Modify quantization information for I/O tensors.

    This will not be trigged by weight-only/drq because they do not quantize
    activation tensors.
    Selective srq & emulated srq will be okay because only the I/O tensors will
    be left as quantized, if applicable. This is the intended behavior if user
    choose to SRQ ops contain I/O tensors.

    Args:
      tensor_params: tensor level quantization params for the tensor.
    """
    # Change ADD_QUANTIZE to QUANTIZE_TENSOR for unique input/constant tensors.
    if (
        tensor_params.producer is None
        and tensor_params.consumers is not None
        and len(tensor_params.consumers) == 1
        and tensor_params.consumers[0].transformations
        == [qtyping.QuantTransformation.ADD_QUANTIZE]
    ):
      tensor_params.consumers = [
          qtyping.OpToTensorParams(
              subgraph_op_id=consumer.subgraph_op_id,
              transformations=[qtyping.QuantTransformation.QUANTIZE_TENSOR],
              parameters=consumer.parameters,
          )
          for consumer in tensor_params.consumers
      ]
    # Change ADD_DEQUANTIZE to QUANTIZE_TENSOR for output tensors.
    elif (
        tensor_params.consumers is None
        and tensor_params.producer is not None
        and tensor_params.producer.transformations
        == [qtyping.QuantTransformation.ADD_DEQUANTIZE]
    ):
      tensor_params.producer = qtyping.OpToTensorParams(
          subgraph_op_id=tensor_params.producer.subgraph_op_id,
          transformations=[qtyping.QuantTransformation.QUANTIZE_TENSOR],
          parameters=tensor_params.producer.parameters,
      )


def _compatible_tensor_transformation_params(
    params1: qtyping.TensorTransformationParams,
    params2: qtyping.TensorTransformationParams,
) -> bool:
  """Check if two tensor transformation params are compatible."""
  if params1.producer is None or params2.producer is None:
    if params1.producer != params2.producer:
      return False
  elif not _compatible_tensor_params(params1.producer, params2.producer):
    return False
  if params1.consumers is None or params2.consumers is None:
    if params1.consumers != params2.consumers:
      return False
  else:
    # check all consumers within eah params are compatible.
    for params1_consumer in params1.consumers:
      if not _compatible_tensor_params(params1_consumer, params1.consumers[0]):
        return False
    for params2_consumer in params2.consumers:
      if not _compatible_tensor_params(params2_consumer, params2.consumers[0]):
        return False
    if not _compatible_tensor_params(
        params1.consumers[0], params2.consumers[0]
    ):
      return False
  return True


def _compatible_tensor_params(
    params1: qtyping.OpToTensorParams,
    params2: qtyping.OpToTensorParams,
) -> bool:
  """Check if two op to tensor params are compatible."""
  float_source_transformations = [
      qtyping.QuantTransformation.ADD_QUANTIZE,
      qtyping.QuantTransformation.NO_QUANTIZE,
  ]
  quantized_source_transformations = [
      qtyping.QuantTransformation.QUANTIZE_TENSOR,
      qtyping.QuantTransformation.ADD_DEQUANTIZE,
  ]
  if params1.parameters != params2.parameters:
    return False
  # we only need to check the first transformation because transformations are
  # applied in order, and as long as the one that's immediately after the tensor
  # is the same, it's compatible.
  if (
      params1.transformations[0] in float_source_transformations
      and params2.transformations[0] in float_source_transformations
  ):
    return True
  if (
      params1.transformations[0] in quantized_source_transformations
      and params2.transformations[0] in quantized_source_transformations
  ):
    return True
  return False
