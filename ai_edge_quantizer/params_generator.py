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

"""Generate model tensor level quantization config."""

from collections.abc import Sequence
import copy
from typing import Any, Optional, Union

from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import default_policy as policy
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe_manager
from ai_edge_quantizer.utils import tfl_flatbuffer_utils

_QuantTrans = qtyping.QuantTransformation
_OpName = qtyping.TFLOperationName


class ParamsGenerator:
  """Generate model tensor level quantization parameters."""

  def __init__(self, float_tflite: Union[str, bytes]):
    self.flatbuffer_model = tfl_flatbuffer_utils.read_model(float_tflite)

    if not tfl_flatbuffer_utils.is_float_model(self.flatbuffer_model):
      raise ValueError(
          'The input model for quantization parameters generation is not a'
          ' float model. Please check the model (e.g., if it is already'
          ' quantized).'
      )
    self._check_tensor_names_are_unique()
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
      model_recipe_manager: The recipe manager for the model.
      model_qsvs: Quantization statistics values (QSVs) for the model. This is
        obtained through calibration process.

    Returns:
      model_quant_results: The quantization parameters for tensors in the model.

    Raises:
      RuntimeError: If the calibration dataset is required but not provided.
    """
    if model_recipe_manager.need_calibration() and not model_qsvs:
      raise RuntimeError(
          'Model quantization statistics values (QSVs) are required for the'
          ' input recipe. This can be obtained by running calibration on sample'
          ' dataset.'
      )

    if model_qsvs is None:
      model_qsvs = {}

    skip_subgraphs = set()
    op_codes = self.flatbuffer_model.operatorCodes
    for sg_ind, subgraph in enumerate(self.flatbuffer_model.subgraphs):
      if sg_ind in skip_subgraphs:
        continue

      graph_info = qtyping.GraphInfo(
          subgraph.tensors, self.flatbuffer_model.buffers
      )
      # Add input/output operators to the subgraph.
      subgraph.operators += (
          tfl_flatbuffer_utils.get_subgraph_input_output_operators(subgraph)
      )
      for subgraph_op_id, op in enumerate(subgraph.operators):
        # Get the op key.
        if isinstance(op, qtyping.IOOperator):
          op_key = op.op_key
          subgraph_op_id = -1  # Virtual op, no real id.
        else:
          op_code = op_codes[op.opcodeIndex].builtinCode
          # Do not quantize unknown ops.
          if op_code not in tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME:
            op_quant_results = self._get_params_for_no_quant_op(
                subgraph_op_id, op, subgraph.tensors
            )
            self._update_model_quant_results(op_quant_results)
            continue
          op_key = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME[op_code]

        # Step1: query the quantization_recipe to get op config.
        op_scope = self._get_op_scope(op, subgraph.tensors)
        algorithm_name, op_quant_config = (
            model_recipe_manager.get_quantization_configs(op_key, op_scope)
        )
        if policy.is_conditionally_unquantized(op):
          algorithm_name = algorithm_manager.AlgorithmName.NO_QUANTIZE

        if algorithm_name == algorithm_manager.AlgorithmName.NO_QUANTIZE:
          side_effect_subgraphs = (
              tfl_flatbuffer_utils.get_op_side_effect_subgraphs(op)
          )
          skip_subgraphs.update(side_effect_subgraphs)

          op_quant_results = self._get_params_for_no_quant_op(
              subgraph_op_id, op, subgraph.tensors
          )

        else:
          op_info = qtyping.OpInfo(op, op_key, subgraph_op_id, op_quant_config)
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

  def _check_tensor_names_are_unique(self):
    """Checks if the tensor names are unique in the model."""
    global_tensor_names = set()
    for subgraph in self.flatbuffer_model.subgraphs:
      for tensor in subgraph.tensors:
        tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
        if tensor_name in global_tensor_names:
          raise ValueError(
              'Tensor name %s is not unique in the model. Please check your'
              ' model and rename the tensor as ParamsGenerator assumes tensor'
              ' names are unique.' % tensor_name
          )
        global_tensor_names.add(tensor_name)

  def _post_process_results(self) -> None:
    """Post process the quantization results.

    Raises:
      RuntimeError: If the tensors sharing the same buffer have different
      quantization settings.
    """
    self._check_buffer_sharing()
    self._mark_tensors_requiring_buffer_duplication()
    self._mark_tensors_requiring_tensor_duplication()

  def _update_model_quant_results(
      self,
      op_tensor_results: list[qtyping.TensorTransformationParams],
  ) -> None:
    """Update the op quantization results to the final output.

    Args:
      op_tensor_results: Tensor level quantization params for the op.

    Raises:
      RuntimeError: If the same tensor has multiple quantization configs.
    """

    for op_tensor_result in op_tensor_results:
      tensor_name = op_tensor_result.tensor_name
      if tensor_name not in self.model_quant_results:
        self.model_quant_results[tensor_name] = copy.deepcopy(op_tensor_result)
      else:
        tensor_params = self.model_quant_results[tensor_name]
        # Set source op.
        if op_tensor_result.producer is not None:
          # Src params must be unique (a tensor can only be produced by one op).
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
      op: The op that needs to be parsed.
      subgraph_tensors: Tensors in the subgraph.

    Returns:
      Scope for the op.
    """
    scope = ''
    # Op scope is determined by output tensors.
    for output_tensor_idx in op.outputs:
      if output_tensor_idx != -1:
        scope += tfl_flatbuffer_utils.get_tensor_name(
            subgraph_tensors[output_tensor_idx]
        )
        scope += ';'  # Split names.
    return scope

  def _get_params_for_no_quant_op(
      self,
      subgraph_op_id: int,
      op: Any,
      subgraph_tensors: list[Any],
  ) -> list[qtyping.TensorTransformationParams]:
    """Get the quantization parameters for ops require no quantization.

    Args:
      subgraph_op_id: The op id in the subgraph.
      op: The op that needs to be parsed.
      subgraph_tensors: Tensors in the subgraph.

    Returns:
      Tensor level quantization params for the op.
    """

    def no_quant_tensor_params():
      return qtyping.OpToTensorParams(
          subgraph_op_id=subgraph_op_id,
          transformations=[_QuantTrans.NO_QUANTIZE],
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

  def _mark_tensors_requiring_buffer_duplication(self) -> None:
    """Mark tensors that require buffer duplication.

    Marking a tensor means adding a DUPLICATE_BUFFER transformation as the first
    transformation to be applied for each consumer of the tensor. Need to do
    that for each consumer to preserve a zero layer and not affect the
    horizontal optimization later in the transformation instructions generator.

    Marks all tensors within each of the provided buffers as requiring buffer
    duplication, except for the last tensor. The order of tensors is assumed to
    be the same during both the marking and transformation performer steps, as
    determined by `self.buffer_to_tensors`. This allows the final tensor to
    reuse the original buffer, as it is not marked for duplication.
    """
    for buffer_idx in self._buffers_to_duplicate:
      for tensor in self.buffer_to_tensors[buffer_idx][:-1]:
        tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
        for consumer_params in self.model_quant_results[tensor_name].consumers:
          consumer_params.transformations.insert(
              0, _QuantTrans.DUPLICATE_BUFFER
          )

  def _mark_tensors_requiring_tensor_duplication(self) -> None:
    """Mark tensors that require tensor duplication.

    Marking a tensor means adding a DUPLICATE_TENSOR transformation as the first
    transformation to be applied for each consumer of the tensor. Need to do
    that for each consumer to preserve a zero layer and not affect the
    horizontal optimization later in the transformation instructions generator.
    """
    for tensor_name in self._tensor_names_to_duplicate:
      for consumer_params in self.model_quant_results[tensor_name].consumers:
        consumer_params.transformations.insert(0, _QuantTrans.DUPLICATE_TENSOR)

  def _check_buffer_sharing_for_tensor_pair(self, tensor1, tensor2) -> bool:
    """Check buffer sharing for the first tensor against the second tensor.

    Args:
      tensor1: The first tensor to check.
      tensor2: The second tensor to check.

    Returns:
      Whether the tensors have compatible quantization parameters.

    Raises:
      RuntimeError: If the tensors have incompatible quantization parameters
      and the buffer is not constant.
    """
    tensor1_params = self.model_quant_results.get(
        tfl_flatbuffer_utils.get_tensor_name(tensor1), None
    )
    tensor2_params = self.model_quant_results.get(
        tfl_flatbuffer_utils.get_tensor_name(tensor2), None
    )
    if tensor1_params is None or tensor2_params is None:
      return True

    if _compatible_tensor_transformation_params(tensor1_params, tensor2_params):
      return True
    elif _is_constant_tensor(tensor1, self.flatbuffer_model.buffers):
      return False
    else:
      error_msg = (
          f'The tensors {tensor1.name} and {tensor2.name} do not have'
          ' the same quantization parameters even though they share the'
          ' same buffer. Please modify your quantization recipe to make'
          ' sure the two tensors have the same quantization settings.'
      )
      raise RuntimeError(error_msg)

  def _check_buffer_sharing(self) -> None:
    """Check if tensors sharing the same buffer have the same quantization.

    Function adds two lists to the self object: a list of buffers to duplicate
    and a list of tensor names to duplicate.

    Raises:
      RuntimeError: If the tensors sharing the same buffer have different
        quantization settings and it can't be resolved by duplicating the
        buffer/tensor (meaning the buffer is not constant).
    """
    self._buffers_to_duplicate = []
    self._tensor_names_to_duplicate = []
    for tensors in self.buffer_to_tensors.values():
      if not tensors:
        continue
      # Check if the buffer needs to be duplicated.
      tensor1 = tensors[0]
      for tensor2 in tensors[1:]:
        if not self._check_buffer_sharing_for_tensor_pair(tensor1, tensor2):
          self._buffers_to_duplicate.append(tensor1.buffer)
      # Check if any of the tensors needs to be duplicated.
      for tensor in tensors:
        if not self._check_buffer_sharing_for_tensor_pair(tensor, tensor):
          self._tensor_names_to_duplicate.append(
              tfl_flatbuffer_utils.get_tensor_name(tensor)
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
    # Check all consumers within each params are compatible.
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


def _same_tensor_params_except_id(
    params1: qtyping.OpToTensorParams,
    params2: qtyping.OpToTensorParams,
) -> bool:
  """Check if two op to tensor params are the same except for subgraph_op_id."""
  return params1.transformations == params2.transformations and (
      params1.parameters == params2.parameters
      or params1.parameters is None
      and params2.parameters is None
  )


def _compatible_tensor_params(
    params1: qtyping.OpToTensorParams,
    params2: qtyping.OpToTensorParams,
) -> bool:
  """Check if two op to tensor params are compatible."""
  float_source_transformations = [
      _QuantTrans.ADD_QUANTIZE,
      _QuantTrans.NO_QUANTIZE,
  ]
  quantized_source_transformations = [
      _QuantTrans.QUANTIZE_TENSOR,
      _QuantTrans.ADD_DEQUANTIZE,
  ]
  if _same_tensor_params_except_id(params1, params2):
    return True
  # We only need to check the first transformation because transformations are
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
      and params1.parameters == params2.parameters
  ):
    return True
  return False


def _is_constant_tensor(tensor: Any, buffers: Sequence[Any]) -> bool:
  """Check if the tensor is a constant tensor."""
  return buffers[tensor.buffer].data is not None
