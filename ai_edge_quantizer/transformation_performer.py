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

"""Python manager for transformations to be applied to TFlite models."""

import numpy as np
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.transformations import dequant_insert
from ai_edge_quantizer.transformations import emulated_subchannel
from ai_edge_quantizer.transformations import quant_insert
from ai_edge_quantizer.transformations import quantize_tensor
from ai_edge_quantizer.transformations import transformation_utils
from tensorflow.lite.python import schema_py_generated  # pylint: disable=g-direct-tensorflow-import


class TransformationPerformer:
  """Wrapper class for transformations.

  all transformations supported by the AI Edge Quantizer should be registered in
  the Transformation Performer.

  transformation to be appied to a tensor in the Instruction Generator is
  specified
  as a key. Given this key, the transformation performer will return a function
  that will apply the corresponding transformation on the graph

  A transformation is defined as a Callable that takes the following parameters:
  tensor_id: A tensor id that represents the tensor to be applied
  operatorCodes: list of OperatorCodesT from the source TFlite ModelT
  buffers: list of BufferT from the source TFLite ModelT
  subgraph: the specific subgraph where the transformation should be applied
  producer: the op index for the producer of the tensor
  consumers: a list of op index representing consumers to apply the change on
  quant_param: the quantization parameters in qtyping.UniformQuantParams
  And returns a qtyping.TransformationInfo which contains the index where the
  ops are added and how many ops are added

  Additionally, op additions must be consecutive

  this class is expected to be created by the Model Modifier and nothing else

  Model modifier would pass in a dict of transformations to be applied, this
  class will apply the transformations in a pre-determined static order
  """

  def __init__(self):
    """Initializes the TransformationPerformer."""
    self._transformation_registration = {
        qtyping.QuantTransformation.ADD_DEQUANTIZE: (
            dequant_insert.insert_dequant
        ),
        qtyping.QuantTransformation.QUANTIZE_TENSOR: (
            quantize_tensor.quantize_tensor
        ),
        qtyping.QuantTransformation.EMULATED_SUBCHANNEL: (
            emulated_subchannel.emulated_subchannel
        ),
        qtyping.QuantTransformation.ADD_QUANTIZE: quant_insert.insert_quant,
    }
    # transformations are seprated in two categories:
    # op_insertion_transformations are transformations that only insert ops
    # into the graph, whereas op_replacement_transformations will replace one op
    # with a pattern
    self._op_insertion_transformations = set([
        qtyping.QuantTransformation.ADD_DEQUANTIZE,
        qtyping.QuantTransformation.QUANTIZE_TENSOR,
        qtyping.QuantTransformation.ADD_QUANTIZE,
    ])
    self._op_replacement_transformations = set(
        [qtyping.QuantTransformation.EMULATED_SUBCHANNEL]
    )
    self._original_op_id_map = []
    self._added_op_id_map = []

  def _create_op_id_map(self, tflite_model: schema_py_generated.ModelT):
    """init the original op_id to modified op_id map.

    At the beginning the graph has not been updated, so op_id maps to it's
    current id.

    Args:
      tflite_model: the model we're create op_id mapping

    Returns:
      None, modifies self._original_op_id_map inplace
    """
    for subgraph in tflite_model.subgraphs:
      self._original_op_id_map.append(list(range(len(subgraph.operators))))
      self._added_op_id_map.append([])

  def _update_op_id_map(
      self, subgraph_id: int, original_op_id: int, num_ops_added: int
  ):
    """Update the mapping between the original op id and modified op ids.

    Args:
      subgraph_id: the index of subgraph that we're interested in
      original_op_id: the original id for which the first op is added
      num_ops_added: the number of ops added starting from the op id

    Returns:
      None, modify self._original_op_id_map
    """
    np_op_id_map = np.array(self._original_op_id_map[subgraph_id])
    np_op_id_map[original_op_id:] += num_ops_added
    self._original_op_id_map[subgraph_id] = np_op_id_map.tolist()

  def _update_instructions(
      self,
      prev_transformation_index: int,
      transformations: list[qtyping.TransformationInst],
      subgraph_id: int,
      trans_info: qtyping.TransformationInfo,
  ):
    """Update the instructions after the graph is modified.

    After an op is inserted, the topology is changed and this may impact the
    following transformation to be applied. So we need to update instructions
    that have yet to be applied.

    Args:
      prev_transformation_index: the index of the last applied transformation
      transformations: the list of transformations we're applying
      subgraph_id: the subgraph where the provided instrucitons belongs to
      trans_info: transformation info returned by a transformation

    Returns:
      None, modifies the transformation in place
    """
    # if no ops were added, then no need for update
    if trans_info.num_ops_added == 0:
      return
    prev_transformation = transformations[prev_transformation_index]
    self._added_op_id_map[subgraph_id].append(
        trans_info.op_id + trans_info.num_ops_added - 1
    )
    for transformations_index in range(
        prev_transformation_index + 1, len(transformations)
    ):
      transformation = transformations[transformations_index]
      for consumer_index in transformation.consumers:
        # if the consumer need to use newly added ops, then the new added op
        # index needs to be outside of the range of the orignal op ids.
        if consumer_index in prev_transformation.consumers:
          transformation.producer = (
              len(self._original_op_id_map[subgraph_id])
              + len(self._added_op_id_map[subgraph_id])
              - 1
          )
          transformation.tensor_id = trans_info.output_tensor_id

  def _apply_single_transformation(
      self,
      transformation_inst: qtyping.TensorTransformationInsts,
      transformation_index: int,
      tflite_model: schema_py_generated.ModelT,
  ):
    """Apply a single transformation.

    Args:
      transformation_inst: a TensorTransformationInsts type that contains all
        transformations on a tensor
      transformation_index: the index of the transformation to be applied
      tflite_model: source tflite model to be updated

    Returns:
      None, update the transformation_inst & tflite_model in place
    """
    instruction = transformation_inst.instructions[transformation_index]
    if not instruction.producer or instruction.producer < 0:
      producer = -1
    elif instruction.producer < len(
        self._original_op_id_map[transformation_inst.subgraph_id]
    ):
      producer = self._original_op_id_map[transformation_inst.subgraph_id][
          instruction.producer
      ]
    else:
      # if the producer id is not in the original op map, it's an added op,
      # go the corresponding new maps
      producer = self._added_op_id_map[transformation_inst.subgraph_id][
          instruction.producer
          - len(self._original_op_id_map[transformation_inst.subgraph_id])
      ]
    consumers = []
    for original_op_id in instruction.consumers:
      consumers.append(
          self._original_op_id_map[transformation_inst.subgraph_id][
              original_op_id
          ]
      )
    trans_info = self._transformation_registration[instruction.transformation](
        transformation_utils.TransformationInput(
            instruction.tensor_id,
            tflite_model.operatorCodes,
            tflite_model.buffers,
            tflite_model.subgraphs[transformation_inst.subgraph_id],
            producer,
            consumers,
            instruction.parameters,
        )
    )
    self._update_instructions(
        transformation_index,
        transformation_inst.instructions,
        transformation_inst.subgraph_id,
        trans_info,
    )
    self._update_op_id_map(
        transformation_inst.subgraph_id,
        min(instruction.consumers),
        trans_info.num_ops_added,
    )

  def _apply_transformations(
      self,
      transformation_inst: qtyping.TensorTransformationInsts,
      tflite_model: schema_py_generated.ModelT,
  ):
    """Apply all transformations for a tensor.

    transformations are separated in two types and applied separately in two
    different passes

    Args:
      transformation_inst: a TensorTransformationInsts type that contains all
        transformation on a tensor
      tflite_model: source tflite model to be updated

    Returns:
      None, update the transformation_inst & tflite_model in place
    """
    # pass 1: apply all the op insertion transformation, because op replacement
    # may remove consumer or producer of some tensors
    for index, instruction in enumerate(transformation_inst.instructions):
      if instruction.transformation in self._op_insertion_transformations:
        self._apply_single_transformation(
            transformation_inst, index, tflite_model
        )
    # pass 2: apply all the op replacement transformation
    for index, instruction in enumerate(transformation_inst.instructions):
      if instruction.transformation in self._op_replacement_transformations:
        self._apply_single_transformation(
            transformation_inst, index, tflite_model
        )

  def transform_graph(
      self,
      transformation_instructions: dict[str, qtyping.TensorTransformationInsts],
      tflite_model: schema_py_generated.ModelT,
  ):
    """Apply all transformations to the given tflite_model.

    Args:
      transformation_instructions: a dict of transformation instructions grouped
        by tensors, produced by transformation_instruction_generator
      tflite_model: the tflite model to apply quantization on

    Returns:
      None, modifies the input tflite_model in place
    """
    self._original_op_id_map = []
    self._added_op_id_map = []
    self._create_op_id_map(tflite_model)
    for transformation_inst in transformation_instructions.values():
      self._apply_transformations(transformation_inst, tflite_model)
