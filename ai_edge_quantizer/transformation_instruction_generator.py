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

"""Create transformation instructions for transformation_performer.

Given quantization parameters, create a list of transformation instructions that
can then be used by transformation_performer. Includes necessary optimizations
"""

from collections.abc import Iterator
import dataclasses
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from tensorflow.lite.python import schema_py_generated  # pylint: disable=g-direct-tensorflow-import


# When a tensor has no producer, we'll assign -1 to the producer field
# When a tensor is a graph output, we'll also include a -1 in the consumer list
def check_horizontal_optimization(
    param1: qtyping.OpToTensorParams,
    param2: qtyping.OpToTensorParams,
    index: int,
) -> bool:
  """check if horizontal optimization can be applied.

  check if two transformations at the same index (which belongs to two
  different
  OpToTensorParams) can be merged together.

  Args:
    param1: first parameters to be compared
    param2: second parameters to be compared
    index: the index for which the transformation will be compared

  Returns:
    True if the two transformations can be merged, False otherwise
  """
  return (
      param1.parameters == param2.parameters
      and len(param1.transformations) > index
      and len(param2.transformations) > index
      and param1.transformations[index] == param2.transformations[index]
  )


def check_dq_q_elimination(
    producer_inst: qtyping.TransformationInst,
    consumer_inst: qtyping.TransformationInst,
) -> bool:
  """Check if a pair of dequantize & quantize transformation can be eliminated.

  This can only happen when the dequantize & quantize have the same quant
  parameters and dequantize belongs to producer and quantize belongs to a
  consumer.

  Args:
    producer_inst: TransformationInst from producer.
    consumer_inst: TransformationInst from consumer.

  Returns:
    True if dequantize & quantize can be eliminated, False otherwise.
  """
  is_dequantize_in_producer = (
      producer_inst.transformation == qtyping.QuantTransformation.ADD_DEQUANTIZE
  )
  is_quantize_in_consumer = (
      consumer_inst.transformation == qtyping.QuantTransformation.ADD_QUANTIZE
  )
  is_same_parameters = producer_inst.parameters == consumer_inst.parameters
  return (
      is_dequantize_in_producer
      and is_quantize_in_consumer
      and is_same_parameters
  )


def check_replace_dq_q_with_rq(
    producer_inst: qtyping.TransformationInst,
    consumer_inst: qtyping.TransformationInst,
) -> bool:
  """Check if a pair of dequantize & quantize can be replaced by a requantize.

  This can only happen when the dequantize belongs to producer and quantize
  belongs to a consumer.

  Args:
    producer_inst: TransformationInst from producer.
    consumer_inst: TransformationInst from consumer.

  Returns:
    True if dequantize & quantize can be replaced, False otherwise.
    Note that we consider the case where DQ & Q can be eliminated as a false
    case.
  """
  is_dequantize_in_producer = (
      producer_inst.transformation == qtyping.QuantTransformation.ADD_DEQUANTIZE
  )
  is_quantize_in_consumer = (
      consumer_inst.transformation == qtyping.QuantTransformation.ADD_QUANTIZE
  )
  is_same_parameters = producer_inst.parameters == consumer_inst.parameters

  return (
      is_dequantize_in_producer
      and is_quantize_in_consumer
      and not is_same_parameters
  )


def check_dq_no_quant_elimination(
    producer_inst: qtyping.TransformationInst,
    consumer_inst: qtyping.TransformationInst,
) -> bool:
  """Check if a pair of dequantize & no quantize transformation can be eliminated.

  This can only happen when the dequantize belongs to producer and no quantize
  belongs to a consumer.

  Args:
    producer_inst: TransformationInst from producer.
    consumer_inst: TransformationInst from consumer.

  Returns:
    True if dequantize & no quantize can be eliminated, False otherwise.
  """
  is_dequantize_in_producer = (
      producer_inst.transformation == qtyping.QuantTransformation.ADD_DEQUANTIZE
  )
  is_no_quant_in_consumer = (
      consumer_inst.transformation == qtyping.QuantTransformation.NO_QUANTIZE
  )
  return is_dequantize_in_producer and is_no_quant_in_consumer


class TransformationInstructionsGenerator:
  """Generates transformation instructions from tensor quant params."""

  def __init__(self, float_tflite):
    """Constructor.

    Args:
      float_tflite: the original TFlite model in bytearray or file path.
    """
    self.flatbuffer_model = tfl_flatbuffer_utils.read_model(float_tflite)
    self._create_tensor_name_to_graph_info_map()

  @dataclasses.dataclass(frozen=True)
  class TensorGraphInfo:
    tensor_id: int
    subgraph_id: int
    producer: int
    consumers: list[int]

  def _tensor_info_generator(
      self, subgraph_id: int, subgraph: schema_py_generated.SubGraphT
  ) -> Iterator[tuple[str, TensorGraphInfo]]:
    """Generator function for tensor info.

    Args:
      subgraph_id: Index for the given subgraph,
      subgraph: Subgraph struct to generate tensor info on.

    Yields:
      A tuple of tensor_name and TensorGraphInfo.
    """
    for tensor_id, tensor in enumerate(subgraph.tensors):
      consumers = [
          op_id
          for (op_id, op) in enumerate(subgraph.operators)
          if tensor_id in op.inputs
      ]
      producer = -1
      for op_id, op in enumerate(subgraph.operators):
        if tensor_id in op.outputs:
          producer = op_id
          break
      if tensor_id in subgraph.outputs:
        consumers.insert(0, -1)
      tensor_info = self.TensorGraphInfo(
          tensor_id, subgraph_id, producer, consumers
      )
      tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
      yield tensor_name, tensor_info

  def _create_tensor_name_to_graph_info_map(self):
    """Create a mapping between tensor name and tensor info."""
    self._tensor_name_to_graph_info = {}
    # TODO: b/333607428 - support graph input & output
    for subgraph_id, subgraph in enumerate(self.flatbuffer_model.subgraphs):
      for tensor_name, tensor_info in self._tensor_info_generator(
          subgraph_id, subgraph
      ):
        self._tensor_name_to_graph_info[tensor_name] = tensor_info

  def _group_consumer_transformations(
      self, param: qtyping.TensorTransformationParams
  ) -> list[list[set[int]]]:
    """Group transformations between consumers into common groups.

    Args:
      param: TensorTransformationParams for a tensor

    Returns:
      A list of list of sets where the set represents indices of transformations
      that can be merged horizontally
      E.g:
        For the following consumer:
         [(1, [ADD_QUANTIZE, ADD_DEQUANTIZE], param1),
          (2, [ADD_QUANTIZE], param2),
          (3, [ADD_QUANTIZE], param1)]
        this function returns:
        [[{1, 2, 3}],
         [{1, 3}, {2}],
         [{1}]]

        Where the 0 depth list is the initial state, since all consumer comes
        from the same producer.
        In depth 1, the ADD_QUANTIZE in 1 & 3 can be merged, so they are in the
        same group
        In depth 2, there is only one transformation from 1, so there is only
        one group with 1 in there
    """
    if not param or not param.consumers:
      return []

    # consumer group contains indices of operations that can be horizontally
    # optimized together. The outermost list is the depth of the transformation
    # and the second list contains sets that represents the consumer indices
    # that can be grouped together at the given depth
    consumer_groups = [[set()]]
    # the max number of transformations applied before a particular consumer
    longest_trans_chain = 0
    for i, consumer_param in enumerate(param.consumers):
      consumer_groups[0][0].add(i)
      longest_trans_chain = max(
          longest_trans_chain, len(consumer_param.transformations)
      )

    #  looping over transformations of the same depth
    for transformation_depth in range(longest_trans_chain):
      next_depth_groups = []
      for consumer_param_index, consumer_param in enumerate(param.consumers):
        if len(consumer_param.transformations) > transformation_depth:
          for current_depth_groups in consumer_groups[transformation_depth]:
            if consumer_param_index in current_depth_groups:
              # if the transformation of the particular edge has been processed
              trans_assigned = False
              for new_group in next_depth_groups:
                # get an index in the existing group, any of them work since
                # they have the same quantization
                index = next(iter(new_group))
                if (
                    index in current_depth_groups
                    and check_horizontal_optimization(
                        param.consumers[index],
                        consumer_param,
                        transformation_depth,
                    )
                ):
                  new_group.add(consumer_param_index)
                  trans_assigned = True
                  break
              if not trans_assigned:
                next_depth_groups.append(set([consumer_param_index]))
      consumer_groups.append(next_depth_groups)
    return consumer_groups

  def _produce_transformation_for_vertical_opt(
      self,
      consumer_group: list[list[set[int]]],
      param: qtyping.TensorTransformationParams,
  ) -> list[qtyping.TransformationInst]:
    """Create a list of transformation rules available for vertical optimization.

    A consumer transformation is available to vertical transformation IFF it's
    the first transformation for a given consumer.

    This function relies on the consumer_group argument already being optimized
    for horizontal transformations.

    Args:
      consumer_group: a list of grouped indices for consumer transformationns
      param: a TensorTransformationParams for the tensor

    Returns:
      A list of transformation rules available for vertical optimization
    """
    tensor_info = self._tensor_name_to_graph_info[param.tensor_name]
    transformations_available_for_vertical_optimization = []
    # we start at 1 because consumer groups in index 0 is the inital state
    # and does not contain actual information
    if len(consumer_group) > 1:
      for group in consumer_group[1]:
        op_list = list(group)
        op_idx_list = []
        for index in op_list:
          op_idx_list.append(param.consumers[index].subgraph_op_id)
        transformations_available_for_vertical_optimization.append(
            qtyping.TransformationInst(
                param.consumers[op_list[0]].transformations[0],
                tensor_info.tensor_id,
                tensor_info.producer,
                op_idx_list,
                param.consumers[op_list[0]].parameters,
            )
        )
    return transformations_available_for_vertical_optimization

  def _produce_consumer_transformations_unavailable_for_vertical_opt(
      self,
      consumer_group: list[list[set[int]]],
      param: qtyping.TensorTransformationParams,
  ) -> list[qtyping.TransformationInst]:
    """Produce a list of consumer transformation that can't be used for vertical optimization.

    A consumer transformation is available to vertical optimization if and only
    if it's the first transformation for a given consumer.

    This function relies on the consumer_group argument already being optimized
    for horizontal transformations

    Args:
      consumer_group: a list of grouped indices for consumer transformationns
      param: a TensorTransformationParams for the tensor

    Returns:
      A list of transformation rules unavailable for vertical optimization
    """
    tensor_info = self._tensor_name_to_graph_info[param.tensor_name]
    other_consumer_transformations = []
    for transformation_idx in range(2, len(consumer_group)):
      for group in consumer_group[transformation_idx]:
        op_list = list(group)
        op_idx_list = []
        if (
            len(param.consumers[op_list[0]].transformations)
            <= transformation_idx - 1
        ):
          continue
        for index in op_list:
          op_idx_list.append(param.consumers[index].subgraph_op_id)
        other_consumer_transformations.append(
            qtyping.TransformationInst(
                param.consumers[op_list[0]].transformations[
                    transformation_idx - 1
                ],
                tensor_info.tensor_id,
                tensor_info.producer,
                op_idx_list,
                param.consumers[op_list[0]].parameters,
            )
        )
    return other_consumer_transformations

  def _apply_vertical_optimization(
      self,
      producer_trans_rule: qtyping.TransformationInst,
      consumer_trans_rules: list[qtyping.TransformationInst],
  ) -> list[qtyping.TransformationInst]:
    """Apply vertical optimization.

    There are two types of transformations we consider:
      1. when DQ & Q has the same parameter eliminate the operators and quantize
      the tensor only
      2. when DQ & Q has different parameters, then replace the DQ & Q with an
      RQ op

    vertical optimization can only happen with the last producer rules and the
    first consumer rules that are on the first.

    Args:
      producer_trans_rule: the last producer transformation rules.
      consumer_trans_rules: a list of consumer transformation rules that are
        avilable for vertical transformations.

    Returns:
      A list of transformations after vertical optimization has been applied,
      note producer transformation is included.
    """
    transformations = []
    for trans_rule in consumer_trans_rules:
      if check_dq_q_elimination(producer_trans_rule, trans_rule):
        for consumer_id in trans_rule.consumers:
          if consumer_id in producer_trans_rule.consumers:
            producer_trans_rule.consumers.remove(consumer_id)
        transformations.append(
            qtyping.TransformationInst(
                qtyping.QuantTransformation.QUANTIZE_TENSOR,
                trans_rule.tensor_id,
                trans_rule.producer,
                trans_rule.consumers,
                trans_rule.parameters,
            )
        )
        continue
      elif check_replace_dq_q_with_rq(producer_trans_rule, trans_rule):
        for consumer_id in trans_rule.consumers:
          producer_trans_rule.consumers.remove(consumer_id)
        transformations.append(
            qtyping.TransformationInst(
                qtyping.QuantTransformation.QUANTIZE_TENSOR,
                trans_rule.tensor_id,
                trans_rule.producer,
                trans_rule.consumers,
                producer_trans_rule.parameters,
            )
        )
        transformations.append(
            qtyping.TransformationInst(
                qtyping.QuantTransformation.ADD_QUANTIZE,
                trans_rule.tensor_id,
                trans_rule.producer,
                trans_rule.consumers,
                trans_rule.parameters,
            )
        )
        continue
      elif check_dq_no_quant_elimination(producer_trans_rule, trans_rule):
        for consumer_id in trans_rule.consumers:
          if consumer_id in producer_trans_rule.consumers:
            producer_trans_rule.consumers.remove(consumer_id)
        transformations.append(
            qtyping.TransformationInst(
                qtyping.QuantTransformation.ADD_DEQUANTIZE,
                trans_rule.tensor_id,
                trans_rule.producer,
                trans_rule.consumers,
                producer_trans_rule.parameters,
            )
        )
        continue
      else:
        transformations.append(trans_rule)
    if producer_trans_rule.consumers:
      transformations.insert(0, producer_trans_rule)
    return transformations

  def _quant_params_to_transformation_insts(
      self,
      param: qtyping.TensorTransformationParams,
  ) -> qtyping.TensorTransformationInsts:
    """Converts a single quantization params to transformation instructions.

    Args:
      param: quantization parameter of a tensor in the graph

    Returns:
      a list of transformations to be applied to the same tensor
    """
    # setup the structure
    tensor_info = self._tensor_name_to_graph_info[param.tensor_name]
    tensor_trans_insts = qtyping.TensorTransformationInsts(
        param.tensor_name, tensor_info.subgraph_id, []
    )

    # horizontal optimization
    consumer_group = self._group_consumer_transformations(param)
    # at this point, starting from index 1 of consumer_group, we're having sets
    # that represents transformations that can be grouped together
    transformations_available_for_vertical_optimization = (
        self._produce_transformation_for_vertical_opt(consumer_group, param)
    )
    other_consumer_transformations = (
        self._produce_consumer_transformations_unavailable_for_vertical_opt(
            consumer_group, param
        )
    )

    transformations = []
    # adding all producer rules
    producer_params = param.producer
    if producer_params:
      for transformation in producer_params.transformations:
        transformations.append(
            qtyping.TransformationInst(
                transformation,
                tensor_info.tensor_id,
                tensor_info.producer,
                tensor_info.consumers,
                producer_params.parameters,
            )
        )

    # apply vertical optimization
    last_producer_rule_idx = len(transformations) - 1
    if last_producer_rule_idx >= 0:
      transformations += self._apply_vertical_optimization(
          transformations.pop(),
          transformations_available_for_vertical_optimization,
      )
    else:
      transformations += transformations_available_for_vertical_optimization
    # Adding other consumers rules
    transformations += other_consumer_transformations
    tensor_trans_insts.instructions = transformations
    # Check the generated transformation instructions are valid, the function
    # will raise an error if the instructions are not valid
    self._check_tensor_transformation_instructions_valid(tensor_trans_insts)

    return tensor_trans_insts

  def _check_tensor_transformation_instructions_valid(
      self, instructions: qtyping.TensorTransformationInsts
  ):
    """Check if the tensor transformation instructions are valid.

    Args:
      instructions: Transformation instructions for a tensor.

    Raises:
      ValueError: If the instructions are not valid.
    """
    is_tensor_unquantized = False
    is_tensor_quantized = False
    is_operator_emulated = False
    for instruction in instructions.instructions:
      transform_type = instruction.transformation
      if transform_type == qtyping.QuantTransformation.NO_QUANTIZE:
        is_tensor_unquantized = True
      elif (
          transform_type == qtyping.QuantTransformation.QUANTIZE_TENSOR
          or transform_type == qtyping.QuantTransformation.ADD_DEQUANTIZE
      ):
        is_tensor_quantized = True
      elif transform_type == qtyping.QuantTransformation.EMULATED_SUBCHANNEL:
        is_operator_emulated = True
    if is_tensor_unquantized and is_tensor_quantized:
      raise ValueError(
          "Tensor %s can not be both quantized and unquantized"
          % instructions.tensor_name
      )
    if is_operator_emulated and len(instructions.instructions) > 1:
      raise ValueError(
          "Tensor %s : op replacement transformation can not be combined with"
          " other transformations."
          % instructions.tensor_name
      )

  def quant_params_to_transformation_insts(
      self,
      params: dict[str, qtyping.TensorTransformationParams],
  ) -> dict[str, qtyping.TensorTransformationInsts]:
    """Converts quantization params to transformation instructions.

    Args:
      params: quantization parameters generated by params_generator. The data
        type is designed to be the same as the output of
        generate_quantization_parameters.

    Returns:
      a dictionary with tensor name as key and transformation instructions as
      value
    """
    insts = {}
    for tensor_name in params:
      insts[tensor_name] = self._quant_params_to_transformation_insts(
          params[tensor_name]
      )
    return insts
