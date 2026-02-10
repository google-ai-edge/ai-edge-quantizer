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

"""Tests for the Hadamard matrix construction functions."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.platform import googletest
from ai_edge_quantizer.algorithms.uniform_quantize import hadamard_matrices


class HadamardMatricesTest(parameterized.TestCase):

  def _check_hadamard_matrix(self, h: np.ndarray, normalized: bool):
    # Make sure the matrix is square.
    self.assertLen(h.shape, 2)
    size = h.shape[0]
    self.assertEqual(h.shape[0], h.shape[1])

    if normalized:
      # Check that the matrix is orthonormal.
      self.assertLessEqual(
          np.max(np.abs(np.matmul(h, h.T) - np.eye(size))), size * 1e-6
      )
    else:
      # Check that the matrix only contains +/-1.
      self.assertContainsSubset(h.flatten().tolist(), {-1, 1})
      # Check that the matrix is orthonormal (since its values are +/-1, we can
      # assume that the matrix multiplication is exact.
      self.assertEqual(
          np.max(np.abs(np.matmul(h, h.T) - np.eye(size) * size)),
          0,
      )

  @parameterized.named_parameters(('integer', False), ('normalized', True))
  def test_powers_of_two(self, normalize: bool):
    size = 1
    while size <= 1024:
      self._check_hadamard_matrix(
          hadamard_matrices.make_hadamard_matrix(size, normalize=normalize),
          normalized=normalize,
      )
      size *= 2

  @parameterized.named_parameters(('integer', False), ('normalized', True))
  def test_different_sizes(self, normalize: bool):
    for size in range(4, 101, 4):
      h = hadamard_matrices.make_hadamard_matrix(size, normalize=normalize)
      self.assertEqual(h.shape[0], size)
      self._check_hadamard_matrix(h, normalized=normalize)

  @parameterized.named_parameters(('integer', False), ('normalized', True))
  def test_max_size(self, normalize: bool):
    max_size = 32
    for size in range(4, 101, 4):
      h, rem = hadamard_matrices.make_diag_hadamard_matrix(
          size, max_size=max_size, normalize=normalize
      )
      self.assertLessEqual(h.shape[0], max_size)
      self.assertEqual(h.shape[0] * rem, size)
      self._check_hadamard_matrix(h, normalized=normalize)


if __name__ == '__main__':
  googletest.main()
