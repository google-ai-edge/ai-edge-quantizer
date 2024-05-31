"""Test for validation_utils."""
import numpy as np
from tensorflow.python.platform import googletest
from ai_edge_quantizer.utils import validation_utils


class ValidationUtilTest(googletest.TestCase):

  def test_preprocess_same_size_arrays_nan(self):
    data1 = [1, 2, np.nan]
    data2 = [1, 2, 3]
    processed_data1, processed_data2 = (
        validation_utils._preprocess_same_size_arrays(data1, data2)
    )
    self.assertAlmostEqual(processed_data1[2], 1e-9)
    self.assertListEqual(list(processed_data2), data2)

  def test_preprocess_same_size_arrays_inf(self):
    data1 = [1, 2, -np.inf]
    data2 = [1, 2, np.inf]
    processed_data1, processed_data2 = (
        validation_utils._preprocess_same_size_arrays(data1, data2)
    )
    self.assertAlmostEqual(processed_data1[2], -1e9)
    self.assertAlmostEqual(processed_data2[2], 1e9)

  def test_preprocess_same_size_arrays_fails_with_incorrect_shape(self):
    data1 = [1, 2, 3]
    data2 = [1, 2]
    self.assertRaises(
        ValueError, validation_utils._preprocess_same_size_arrays, data1, data2
    )

  def test_mean_squared_difference(self):
    data1 = [1, 2, 3]
    data2 = [1, 2, 3]
    result = validation_utils.mean_squared_difference(data1, data2)
    self.assertEqual(result, 0)

  def test_mean_squared_difference_multidim(self):
    data1 = [[1, 2], [4, 5]]
    data2 = [[1, 3], [2, 2]]
    result = validation_utils.mean_squared_difference(data1, data2)
    self.assertAlmostEqual(result, 3.5)

  def test_mean_squared_difference_0d(self):
    data1 = []
    data2 = []
    result = validation_utils.mean_squared_difference(data1, data2)
    self.assertEqual(result, 0)

  def test_median_diff_ratio(self):
    data1 = [1, 2, 3]
    data2 = [1, 2, 3]
    result = validation_utils.median_diff_ratio(data1, data2)
    self.assertEqual(result, 0)

  def test_median_diff_ratio_multidim(self):
    data1 = [[1, 2], [4, 5]]
    data2 = [[1, 3], [2, 2]]
    result = validation_utils.median_diff_ratio(data1, data2)
    self.assertAlmostEqual(result, 0.6666664)

  def test_median_diff_ratio_0d(self):
    data1 = []
    data2 = []
    result = validation_utils.median_diff_ratio(data1, data2)
    self.assertEqual(result, 0)


if __name__ == "__main__":
  googletest.main()
