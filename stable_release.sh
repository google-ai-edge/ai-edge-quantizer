#!/bin/bash
# Note: this script is used to build a custom release of ai-edge-quantizer.
# It pins the ai-edge-litert dependency to a specific version.
# The quantizer version is read from ../VERSION.
#
# Environment Variables:
#   LITERT_VERSION:  The version of ai-edge-litert to pin as a dependency.
#   UV_PUBLISH_TOKEN: (Optional) If set, the script will attempt to publish the
#                    built package to PyPI using this token.

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${SCRIPT_DIR}"

if [ ! -f VERSION ] || [ -z "${LITERT_VERSION}" ]; then
    echo "Error: VERSION file not found, or LITERT_VERSION env var is not set."
    exit 1
fi

function ensure_uv {
  # Add ~/.local/bin to PATH if it's not already there
  if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi

  # Check for uv binary specifically in ~/.local/bin or in PATH (avoiding functions)
  if [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
  elif command -v uv &> /dev/null && ! type uv | grep -q "function"; then
    UV_BIN=$(command -v uv)
  else
    echo "uv binary not found or is a function. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_BIN="$HOME/.local/bin/uv"
    # Ensure it's in the current shell's path after installation
    export PATH="$HOME/.local/bin:$PATH"
  fi
  echo "Using uv binary at: $UV_BIN"
}

ensure_uv

# Update pyproject.toml to pin ai-edge-litert version
# Matches "ai-edge-litert-nightly" or "ai-edge-litert == <version>"
sed -i "s/\"ai-edge-litert[^\"]*\"/\"ai-edge-litert==${LITERT_VERSION}\"/" pyproject.toml

echo "Building ai-edge-quantizer version $(cat VERSION) with ai-edge-litert == ${LITERT_VERSION}"

# Build the package using the discovered binary and public PyPI.
# We must ignore the global /etc/uv/uv.toml which points to a restricted index.
export UV_NO_CONFIG=1
export UV_INDEX_URL=https://pypi.org/simple
"$UV_BIN" build

echo "Build complete!"

if [ -n "${UV_PUBLISH_TOKEN}" ]; then
    echo "Publishing to PyPI..."
    "$UV_BIN" publish
fi
