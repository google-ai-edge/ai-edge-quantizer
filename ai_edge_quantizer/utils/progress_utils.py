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

"""Utility functions to display a progress bar and progress report."""

from collections.abc import Sequence
import os
import time
import tracemalloc
from typing import Any

import tqdm

import os
import io
from ai_edge_litert.tools import flatbuffer_utils
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from ai_edge_litert import schema_py_generated


class ProgressBar:
  """A Progress Bar that can be used to track the progress of a process."""

  def __init__(
      self,
      total_steps: int,
      description: str = '',
      disappear_on_finish: bool = False,
      enable: bool | None = None,
  ):

    if enable is None:
      # Progress bar will be skipped for smaller models.
      disable = total_steps < 1000
    else:
      disable = not enable
    self._progress_bar = tqdm.tqdm(
        total=total_steps,
        desc=description,
        leave=not disappear_on_finish,
        disable=disable,
    )

  def __enter__(self):
    """Enters the context manager."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the context manager and closes the progress bar."""
    self.close()

  def update_single_step(self):
    """Updates the progress bar by a single step."""
    self._progress_bar.update(1)

  def close(self):
    """Closes the progress bar."""
    self._progress_bar.close()


class ProgressReport:
  """A class to generate a progress report for the quantization process."""

  def __init__(self, description: str = ''):
    self._description = description
    self._start_time = None

  def capture_progess_start(self):
    self._start_time = time.time()
    tracemalloc.start()

  def generate_per_op_report(self, quantized_model):
    op_codes = quantized_model.operatorCodes
    short_summary_op_types = [
        'FULLY_CONNECTED',
        'CONV_2D',
        'BATCH_MATMUL',
        'EMBEDDING_LOOKUP',
        'DEPTHWISE_CONV_2D',
    ]
    report_path = '/tmp/aeq/per_op_report.txt'
    os.makedirs(os.path.dirname(report_path))
    with open(report_path, 'w') as f:
      for subgraph in quantized_model.subgraphs:
        for index, op in enumerate(subgraph.operators):
          op_code_name = self.builtin_code_to_name(
              op_codes[op.opcodeIndex].builtinCode
          )
          for input in op.inputs:
            if _is_constant_tensor(
                subgraph.tensors[input], quantized_model.buffers
            ):
              op_report = self._print_op_report(
                  index,
                  op_code_name,
                  self.type_to_name(subgraph.tensors[input].type),
                  input,
              )
              f.write(op_report + '\n')
              if op_code_name not in short_summary_op_types:
                continue
              print(op_report)
    print(f'Detailed Per-op report saved to: {os.path.abspath(report_path)}')

  def _print_op_report(
      self, op_index, op_code, tensor_type, tensor_index
  ) -> str:
    return (
        f'op {op_index}: op_code: {op_code}: type: {tensor_type},'
        f' tensor_index: {tensor_index}'
    )

  def type_to_name(self, tensor_type):
    """Converts a numerical enum to a readable tensor type."""
    for name, value in schema_py_generated.TensorType.__dict__.items():
      if value == tensor_type:
        return name
    return None

  def builtin_code_to_name(self, builtin_code):
    """Converts a numerical enum to a readable builtin operator name."""
    for name, value in schema_py_generated.BuiltinOperator.__dict__.items():
      if value == builtin_code:
        return name
    return None

  def render_report(
      self,
      original_size: int,
      quantized_size: int,
      quantization_ratio: float,
      memory_peak: float,
      total_time: float,
  ):
    """Prints out the progress report."""
    print(f'Original model size: {original_size/1024:.2f} KB')
    print(f'Quantized model size: {quantized_size/1024:.2f} KB')
    print(f'Quantization Ratio: {quantization_ratio:.2f}')
    print(f'Total time: {total_time:.2f} seconds')
    print(f'Memory peak: {memory_peak:.2f} MB')

  def generate_progress_report(
      self, original_model: tfl_flatbuffer_utils.ModelT, quantized_model
  ):
    """Generates and prints a comprehensive progress report.

    This report includes model sizes, quantization ratio, total execution time,
    memory peak usage, and a per-operator analysis.

    Args:
      original_model: The original TFLite model in flatbuffer object format.
      quantized_model: The quantized TFLite model as a byte array.
    """
    original_size = len(
        flatbuffer_utils.convert_object_to_bytearray(original_model)
    )
    quantized_size = len(quantized_model)
    quantization_ratio = quantized_size / original_size
    total_time = time.time() - self._start_time
    _, mem_peak_bytes = tracemalloc.get_traced_memory()
    mem_peak_mb = mem_peak_bytes / 1024 / 1024
    self.render_report(
        original_size,
        quantized_size,
        quantization_ratio,
        mem_peak_mb,
        total_time,
    )
    print('##### Per-op report #####')
    self.generate_per_op_report(
        tfl_flatbuffer_utils.read_model(quantized_model)
    )


def _is_constant_tensor(tensor: Any, buffers: Sequence[Any]) -> bool:
  """Check if the tensor is a constant tensor."""
  return buffers[tensor.buffer].data is not None
