# AI Edge Quantizer

A quantizer for advanced developers to quantize converted LiteRT models. It aims to
facilitate advanced users to strive for optimal performance on resource
demanding models (e.g., GenAI models).

## Build Status

Build Type         |    Status     |
-----------        | --------------|
Unit Tests (Linux) | [![](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_unittests.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_unittests.yml) |
Nightly Release    | [![](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_release.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_release.yml) |
Nightly Colab      | [![](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_colabs.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_colabs.yml) |

## Installation

### Requirements and Dependencies

 * Python versions: 3.9, 3.10, 3.11, 3.12
 * Operating system: Linux, MacOS
 * TensorFlow: [![tf-nightly](https://img.shields.io/badge/tf--nightly-latest-blue)](https://pypi.org/project/tf-nightly/)

### Install

Nightly PyPi package:

```bash
pip install ai-edge-quantizer-nightly
```

## API Usage

The quantizer requires two inputs:

1. An unquantized source LiteRT model (with FP32 data type in the FlatBuffer format with `.tflite` extension)
2. A quantization recipe (details below)

and outputs a quantized LiteRT model that's ready for deployment on edge devices.

### Basic Usage

In a nutshell, the quantizer works according to the following steps:

1. Instantiate a `Quantizer` class. This is the entry point to the quantizer's functionalities that the user accesses.
2. Load a desired quantization recipe (details in subsection).
3. Quantize (and save) the model. This is where most of the quantizer's internal logic works.

```python
qt = quantizer.Quantizer("path/to/input/tflite")
qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
qt.quantize().export_model("/path/to/output/tflite")
```

Please see the [getting started colab](colabs/getting_started.ipynb) for the simplest quick start guide on those 3 steps, and the [selective quantization colab](colabs/selective_quantization_isnet.ipynb) with more details on advanced features.

#### LiteRT Model

Please refer to the [LiteRT documentation](https://ai.google.dev/edge/litert) for ways to generate LiteRT models from Jax, PyTorch and TensorFlow. The input source model should be an FP32 (unquantized) model in the FlatBuffer format with `.tflite` extension.

#### Quantization Recipe

The user needs to specify a quantization recipe using AI Edge Quantizer's API to
apply to the source model. The quantization recipe encodes all information on how
a model is to be quantized, such as number of bits, data type, symmetry, scope
name, etc.

Essentially, a quantization recipe is defined as a collection of commands of the
following type:

_“Apply **Quantization Algorithm X** on **Operator Y** under **Scope Z** with **ConfigN**”._

For example:

_\"**Uniformly quantize** the **FullyConnected op** under scope **'dense1/'** with **INT8 symmetric with Dynamic Quantization**"._

All the unspecified ops will be kept as FP32 (unquantized). The scope of an operator in TFLite is defined as the output tensor name of the op, which preserves the hierarchical model information from the source model (e.g., scope in TF). The best way to obtain scope name is by visualizing the model with [Model Explorer](https://ai.google.dev/edge/model-explorer).

Currently, there are three ways to quantize an operator:

* **dynamic quantization (recommended)**: weights are quantized while activations
  remain in a float format and are not processed by AI Edge Quantizer (AEQ). The
  runtime kernel handles the on-the-fly quantization of these activations, as
  identified by `compute_precision=integer` and `explicit_dequantize=False`.
  * Pros: reduced model size and memory usage. Latency improvement due to integer computation. No sample data requirement (calibration).
  * Cons: on-the-fly quantization of activation tensors can affect model quality. Not supported in all hardware (e.g., some GPU and NPU).

* **weight only quantization**: only model weights are quantized, not activations. The
  actual operation (op) computation remains in float. The quantized weight is
  explicitly dequantized before being fed into the op, by inserting a dequantize
  op between the quantized weight and the consuming op. To enable this,
  `compute_precision` will be set to `float` and `explicit_dequantize` to `True`.
  * Pros: reduced model size and memory usage. No sample data requirement (calibration). Usually has the best model quality.
  * Cons: no latency benefit (may be worse) due to float computation with explicit dequantization.

* **static quantization**: both weights and activations are quantized. This requires a calibration phase to estimate quantization parameters of runtime tensors (activations).
  * Pros: reduced model size, memory usage, and latency.
  * Cons: requires sample data for calibration. Imposing static quantization parameters (derived from calibration) on runtime tensors can compromise quality.

Generally, we recommend dynamic quantization for CPU/GPU deployment and static
quantization for NPU deployment.

We include commonly used recipes in [recipe.py](ai_edge_quantizer/recipe.py). This is demonstrated in the [getting started colab](colabs/getting_started.ipynb) example. Advanced users can build their own recipe through the quantizer API.

#### Deployment
Please refer to the [LiteRT deployment documentation](https://ai.google.dev/edge/litert/inference) for ways to deploy a quantized LiteRT model.

### Advanced Recipes

There are many ways the user can configure and customize the quantization recipe beyond using a template in [recipe.py](ai_edge_quantizer/recipe.py). For example, the user can configure the recipe to achieve these features:

* Selective quantization (exclude selected ops from being quantized)
* Flexible mixed scheme quantization (mixture of different precision, compute precision, scope, op, config, etc)
* 4-bit weight quantization

The [selective quantization colab](colabs/selective_quantization_isnet.ipynb) shows some of these more advanced features.

For specifics of the recipe schema, please refer to the `OpQuantizationRecipe` in [recipe_manager.py].

For advanced usage involving mixed quantization, the following API may be useful:

* Use `Quantizer:load_quantization_recipe()` in [quantizer.py](ai_edge_quantizer/quantizer.py) to load a custom recipe.
* Use `Quantizer:update_quantization_recipe()` in [quantizer.py](ai_edge_quantizer/quantizer.py) to extend or override specific parts of the recipe.

### Operator coverage

The table below outlines the allowed configurations for available recipes.

|     |     |     |     |     |     |    |    |    |    |
| --- | --- | --- | --- | --- | --- |--- |--- |--- |--- |
| **Config** | | DYNAMIC_WI8_AFP32 | DYNAMIC_WI4_AFP32 | STATIC_WI8_AI16 | STATIC_WI4_AI16 | STATIC_WI8_AI8 | STATIC_WI4_AI8 | WEIGHTONLY_WI8_AFP32 | WEIGHTONLY_WI4_AFP32 |
|activation| num\_bits | None | None | 16 | 16 | 8 | 8 | None | None |
| | symmetric |None | None | TRUE | TRUE | [TRUE, FALSE] | [TRUE, FALSE] | None | None |
| | granularity |None | None | TENSORWISE | TENSORWISE | TENSORWISE | TENSORWISE | None | None |
| | dtype| None | None |INT | INT | INT | INT | None | None |
| weight | num\_bits | 8 | 4 | 8 | 4 | 8 | 4 | 8 | 4 |
| | symmetric | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | [TRUE, FALSE] | [TRUE, FALSE] |
| | granularity | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] |
| | dtype | INT | INT | INT | INT | INT | INT | INT | INT |
| explicit\_dequantize | | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | TRUE | TRUE |
| compute\_precision || INTEGER | INTEGER | INTEGER | INTEGER | INTEGER | INTEGER | FLOAT | FLOAT |


**Operators Supporting Quantization**

|     |     |     |     |     |     |    |    |    |
| --- | --- | --- | --- | --- | --- |--- |--- |--- |
| **Config** | DYNAMIC_WI8_AFP32 | DYNAMIC_WI4_AFP32 | STATIC_WI8_AI16 | STATIC_WI4_AI16 | STATIC_WI8_AI8 | STATIC_WI4_AI8 | WEIGHTONLY_WI8_AFP32 | WEIGHTONLY_WI4_AFP32 |
|FULLY_CONNECTED  |<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|CONV_2D          |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|    |
|BATCH_MATMUL     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |<div align="center"> &check; </div>|    |
|EMBEDDING_LOOKUP |<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|    |
|DEPTHWISE_CONV_2D|<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |<div align="center"> &check; </div>|    |
|AVERAGE_POOL_2D  |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|RESHAPE          |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|SOFTMAX          |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|TANH             |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|TRANSPOSE        |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|GELU             |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|ADD              |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|CONV_2D_TRANSPOSE|<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|SUB              |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|MUL              |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|MEAN             |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|RSQRT            |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|CONCATENATION    |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|STRIDED_SLICE    |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|SPLIT            |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|LOGISTIC         |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|SLICE            |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|SELECT           |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|SELECT_V2        |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|SUM              |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|PAD              |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|PADV2            |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|MIRROR_PAD       |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|SQUARED_DIFFERENCE |     |     |     |     |<div align="center"> &check; </div>|    |    |    |
|MAX_POOL_2D      |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|RESIZE_BILINEAR  |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|GATHER_ND        |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|PACK             |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|UNPACK           |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|DIV              |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|SQRT             |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|GATHER           |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|HARD_SWISH       |     |     |    |     |<div align="center"> &check; </div>|    |    |    |
|MAXIMUM          |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|REDUCE_MIN       |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|EQUAL            |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|NOT_EQUAL        |     |     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |    |    |
|SPACE_TO_DEPTH   |     |     |     |     |<div align="center"> &check; </div>|    |    |    |
