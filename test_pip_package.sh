#!/bin/bash
# Test AI Edge Quantizer's pip package and unit tests.

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "Testing on Python version ${PYTHON_VERSION}"

function create_venv {
  PYENV_ROOT="$(pwd)/pyenv"
  if ! git clone https://github.com/pyenv/pyenv.git 2>/dev/null && [ -d "${PYENV_ROOT}" ] ; then
      echo "${PYENV_ROOT} exists"
  fi

  export PATH="$PYENV_ROOT/bin:$PATH"

  eval "$(pyenv init -)"
  pyenv install -s "${PYTHON_VERSION}"
  pyenv global "${PYTHON_VERSION}"

  PYTHON_BIN=$(which python)
  export PYTHON_BIN

  ${PYTHON_BIN} -m pip  install virtualenv
  ${PYTHON_BIN} -m virtualenv ai_edge_quantizer_env
  source ai_edge_quantizer_env/bin/activate
}

function build_pip_and_install {
  # Build and install pip package.
  if [[ "${PYTHON_BIN}" == "" ]]; then
    echo "python is not available."
    exit 1
  fi

  ${PYTHON_BIN} -m pip install --upgrade pip
  ${PYTHON_BIN} -m pip install build wheel

  echo "------ build pip and install -----"
  pushd "${SCRIPT_DIR}" > /dev/null

  rm -r -f dist   # Clean up distributions.

  ${PYTHON_BIN} setup.py bdist_wheel

  local dist_pkg="$(ls dist/${pkg}*.whl)"
  ${PYTHON_BIN} -m pip install ${dist_pkg?} --ignore-installed

  popd > /dev/null
  echo
}

function uninstall_pip {
  # Uninstall pip package.
  echo "------ uninstall pip -----"

  local pip_pkg="ai-edge-quantizer"

  yes | ${PYTHON_BIN} -m pip uninstall ${pip_pkg}
  echo
}

function test_import {
  # Test whether import is successful
  echo "------ Test import -----"
  ${PYTHON_BIN} -c "import ai_edge_quantizer"
  echo
}

function test_unittest {
  TEST_DIR="${WORKSPACE_DIR}/tensorflow_examples/lite/model_maker"

  echo "=== BEGIN UNIT TESTS FOR: ${TEST_DIR} ==="

  pushd "${SCRIPT_DIR}" > /dev/null

  ${PYTHON_BIN} -m pip install -r requirements.txt
  ${PYTHON_BIN} -m unittest discover --pattern *_test.py

  popd > /dev/null
  echo "=== END UNIT TESTS: ${TEST_DIR} ==="
  echo
  echo
}

function test_ai_edge_quantizer {
  echo "===== Test AI Edge Quantizer ====="

  create_venv
  build_pip_and_install
  test_import
  test_unittest
  uninstall_pip
  echo
}

test_ai_edge_quantizer
deactivate  # deactivate virtualenv.
