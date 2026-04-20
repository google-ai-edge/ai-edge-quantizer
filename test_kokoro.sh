#!/bin/bash
# Note: this script is primarily used within the Kokoro CI system.

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

function ensure_uv {
  if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "${HOME}/.local/bin/env"
  else
    echo "uv is already installed."
  fi
}

GET_REQUIRED_ENVIRONMENTS="import sys
import platform
print('\n[tool.uv]')
print('required-environments = [')
print(f'    \"sys_platform == \'{sys.platform}\' and platform_machine == \'{platform.machine()}\' and python_version == \'PYTHON_VERSION_PLACEHOLDER\'\",')
print(']')
"
python -c "${GET_REQUIRED_ENVIRONMENTS}" >> pyproject.toml

# Create a copy of the templated pyproject.toml file.
cp -f pyproject.toml pyproject.toml.bak

ensure_uv

PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13")

for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do
  echo "----------------------------------------------------------------"
  echo "Testing on Python version ${PYTHON_VERSION}"
  echo "----------------------------------------------------------------"

  # Set up required environment to match the target architecture.
  sed "s/PYTHON_VERSION_PLACEHOLDER/${PYTHON_VERSION}/g" pyproject.toml.bak > pyproject.toml
  uv sync \
    --python "${PYTHON_VERSION}" \
    --all-packages


  # Install Python version
  uv python install "${PYTHON_VERSION}"
  uv run --python "${PYTHON_VERSION}" pytest
done

echo "All tests passed!"
