"""A quantizer for advanced developers to quantize converted ODML models.

It aims to facilitate advanced users to strive for optimal performance on
resource demanding models (e.g., GenAI models).
"""

import os
import pathlib
import setuptools

here = pathlib.Path(__file__).parent.resolve()

DOCLINES = __doc__.split("\n")

name = "ai-edge-quantizer"
version = "0.0.1"
if nightly_release_datetime := os.environ.get("NIGHTLY_RELEASE_DATETIME"):
  name += "-nightly"
  version += ".dev" + nightly_release_datetime


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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
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
    python_requires=">=3.9, <3.12",
    install_requires=[
        "immutabledict",
        "numpy",
        "tf-nightly>=2.17.0.dev20240509",
    ],
)
