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

"""A quantizer for advanced developers to quantize converted AI Edge models.

It aims to facilitate advanced users to strive for optimal performance on
resource demanding models (e.g., GenAI models).
"""

import os
import pathlib
import setuptools

here = pathlib.Path(__file__).parent.resolve()

DOCLINES = __doc__.split("\n")

name = "ai-edge-quantizer"
# The next version of ai-edge-quantizer.
# The minor version code should be bumped after every release.
version = "0.5.0"
if nightly_release_date := os.environ.get("NIGHTLY_RELEASE_DATE"):
  name += "-nightly"
  version += ".dev" + nightly_release_date


setuptools.setup(
    name=name,
    version=version,
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    long_description_content_type="text/markdown",
    url="https://github.com/google-ai-edge/ai-edge-quantizer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="On-Device ML, AI, Google, TFLite, Quantization, LLMs, GenAI",
    packages=setuptools.find_packages(
        include=["ai_edge_quantizer*"],
    ),
    python_requires=">=3.10",
    install_requires=[
        "absl-py",
        "immutabledict",
        "numpy",
        "ml_dtypes",
        "ai-edge-litert-nightly",
    ],
)
