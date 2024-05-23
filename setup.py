"""A toolkit for advanced developers to quantize converted ODML models.

It aims to facilitate advanced users to strive for optimal performance on
resource demanding models (e.g., GenAI models).
"""

import pathlib
import setuptools

here = pathlib.Path(__file__).parent.resolve()

DOCLINES = __doc__.split("\n")

setuptools.setup(
    name="quantization-toolkit",
    version="0.0.1",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    long_description_content_type="text/markdown",
    url="https://github.com/google-ai-edge/quantization-toolkit",
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
        include=["quantization_toolkit*"],
    ),
    python_requires=">=3.9, <3.12",
    install_requires=[
        "immutabledict",
        "numpy",
        "tf-nightly==2.17.0.dev20240509",
    ],
)
