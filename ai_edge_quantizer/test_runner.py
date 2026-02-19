import sys
import unittest

def main():
  """Runs unit tests for the project."""
  # Discover and run tests matching the pattern '*_test.py'
  # mimicking 'python -m unittest discover --pattern "*_test.py"'
  loader = unittest.TestLoader()
  suite = loader.discover('.', pattern='*_test.py')
  runner = unittest.TextTestRunner(verbosity=2)
  result = runner.run(suite)
  if not result.wasSuccessful():
    sys.exit(1)

if __name__ == '__main__':
  main()
