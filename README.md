# ODML Quantizer

A quantizer for advanced developers to quantize converted ODML models. It aims to
facilitate advanced users to strive for optimal performance on resource
demanding models (e.g., GenAI models).

## Build Status

Build Type         |    Status     |
-----------        | --------------|
Unit Tests (Linux) | [![](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_unittests.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_unittests.yml) |
PyPi Package       | [![](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/build_release.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/build_release.yml) |

## Development Manual

### Requirements and Dependencies

 * Python versions:  3.9, 3.10, 3.11
 * Operating system: Linux, MacOS
 * TensorFlow: [![tf-nightly](https://img.shields.io/badge/tf--nightly-2.17.0.dev20240509-blue)](https://pypi.org/project/tf-nightly/)

### Python Virtual Env

Set up a Python virtualenv:

```bash
python -m venv --prompt ai-edge-quantizer venv
source venv/bin/activate
```

### Build PyPi Package at Local

```bash
pip install wheel
python setup.py bdist_wheel
```

It will build a PyPi package `ai_edge_quantizer-0.0.1-py3-none-any.whl` unde
the `dist` folder, which you could install at local using command:

```bash
pip install dist/ai_edge_quantizer-0.0.1-py3-none-any.whl
```

### Run Unit Tests

```bash
pip install -r requirements.txt
python -m unittest discover --pattern *_test.py
```
