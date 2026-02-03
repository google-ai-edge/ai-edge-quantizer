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

import time
import tracemalloc
import tqdm


class ProgressBar:
  """A Progress Bar that can be used to track the progress of a process."""

  def __init__(
      self,
      total_steps: int,
      description: str = '',
      disappear_on_finish: bool = False,
      disable: bool = False,
  ):
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

  def generate_progress_report(self, original_model, quantized_model):
    original_size = len(original_model)
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
