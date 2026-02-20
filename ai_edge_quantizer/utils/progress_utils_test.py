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

import io
import sys
import time
import tracemalloc
from unittest import mock

import absl.testing.absltest as absltest
from ai_edge_quantizer.utils import progress_utils


class ProgressBarTest(absltest.TestCase):

  @mock.patch('tqdm.tqdm')
  def test_progress_bar_update(self, mock_tqdm):
    mock_progress_bar_instance = mock_tqdm.return_value
    with progress_utils.ProgressBar(total_steps=10) as pb:
      pb.update_single_step()
      pb.update_single_step()

    mock_tqdm.assert_called_once_with(
        total=10, desc='', leave=True, disable=False
    )
    self.assertEqual(mock_progress_bar_instance.update.call_count, 2)
    mock_progress_bar_instance.update.assert_called_with(1)
    mock_progress_bar_instance.close.assert_called_once()

  @mock.patch('tqdm.tqdm')
  def test_progress_bar_disable(self, mock_tqdm):
    mock_progress_bar_instance = mock_tqdm.return_value
    with progress_utils.ProgressBar(total_steps=10, disable=True):
      pass
    mock_tqdm.assert_called_once_with(
        total=10, desc='', leave=True, disable=True
    )
    mock_progress_bar_instance.close.assert_called_once()

  @mock.patch('tqdm.tqdm')
  def test_progress_bar_disappear_on_finish(self, mock_tqdm):
    mock_progress_bar_instance = mock_tqdm.return_value
    with progress_utils.ProgressBar(total_steps=10, disappear_on_finish=True):
      pass
    mock_tqdm.assert_called_once_with(
        total=10, desc='', leave=False, disable=False
    )
    mock_progress_bar_instance.close.assert_called_once()


class ProgressReportTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_time = self.enter_context(
        mock.patch.object(time, 'time', autospec=True)
    )
    self.mock_tracemalloc_start = self.enter_context(
        mock.patch.object(tracemalloc, 'start', autospec=True)
    )
    self.mock_tracemalloc_get_traced_memory = self.enter_context(
        mock.patch.object(tracemalloc, 'get_traced_memory', autospec=True)
    )

  def test_generate_progress_report(self):
    self.mock_time.side_effect = [100.0, 105.5]  # Start time, end time.
    # Mock memory: current=1MB, peak=2MB.
    self.mock_tracemalloc_get_traced_memory.return_value = (
        1 * 1024 * 1024,
        2 * 1024 * 1024,
    )

    progress_report = progress_utils.ProgressReport()
    progress_report.capture_progess_start()

    original_model = b'\x01' * 2048  # 2KB.
    quantized_model = b'\x02' * 1024  # 1KB

    mock_stdout = io.StringIO()
    with mock.patch.object(sys, 'stdout', mock_stdout):
      progress_report.generate_progress_report(original_model, quantized_model)

    self.mock_tracemalloc_start.assert_called_once()
    output = mock_stdout.getvalue()
    self.assertIn('Original model size: 2.00 KB', output)
    self.assertIn('Quantized model size: 1.00 KB', output)
    self.assertIn('Quantization Ratio: 0.50', output)
    self.assertIn('Total time: 5.50 seconds', output)
    self.assertIn('Memory peak: 2.00 MB', output)


if __name__ == '__main__':
  absltest.main()
