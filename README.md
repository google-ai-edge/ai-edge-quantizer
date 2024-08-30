# AI Edge Quantizer

A quantizer for advanced developers to quantize converted LiteRT models. It aims to
facilitate advanced users to strive for optimal performance on resource
demanding models (e.g., GenAI models).

## Build Status

Build Type         |    Status     |
-----------        | --------------|
Unit Tests (Linux) | [![](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_unittests.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_unittests.yml) |
Nightly Release    | [![](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_release.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_release.yml) |

## Development Manual

### Requirements and Dependencies

 * Python versions:  3.9, 3.10, 3.11
 * Operating system: Linux, MacOS
 * TensorFlow: [![tf-nightly](https://img.shields.io/badge/tf--nightly-latest-blue)](https://pypi.org/project/tf-nightly/)

### Install

Stable version:
```bash
pip install ai-edge-quantizer
```

Nightly version:
```bash
pip install ai-edge-quantizer-nightly
```

## API Usage

The quantizer requires two inputs:

1. An unquantized source LiteRT model (with FP32 data type in the FlatBuffers format with `.tflite` extension)
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
qt.quantize().save("/path/to/output/tflite")
```

TODO(b/354275253, b/362387762): Update colab path
Please visit the [MNIST colab]() for the simplest quick start guide on those 3 steps, and the [ISNET colab]() with more details on advanced features.

#### LiteRT Model

Please refer to the [LiteRT model documentation](https://ai.google.dev/edge/lite) for ways to generate LiteRT models. The input source model should be an unquantized FP32 model in the FlatBuffers format with `.tflite` extension.

#### Quantization Recipe

The user needs to specify a quantization recipe using AI Edge Quantizer's API to apply to the source model. The quantization recipe encodes all information on how a model is to be quantized, such as number of bits, data type, symmetry, scope name, etc.

Essentially, a quantization recipe is defined as a collection of the following command:

_“Apply **Quantization Algorithm X** on **Operator Y** under **Scope Z** with **ConfigN**”._

For example:

_"**Uniformly quantize** the **FullyConnected op** under scope **'dense1/'** with **INT8 symmetric with Dynamic Quantization**"._

All the unspecified ops will be kept as FP32 (unquantized). The scope of an operator in TFLite is defined as the output tensor name of the op, which preserves the hierarchical model information from the source model (e.g., scope in TF). The best way to obtain scope name is by visualizing the model with [Model Explorer](https://github.com/google-ai-edge/model-explorer).

TODO(b/362387762): Update colab path
The simplest recipe to get started with is provided in [recipe.py](ai_edge_quantizer/recipe.py):`dynamic_wi8_afp32()`. This is demonstrated in the [MNIST colab]() example.

#### Deployment
Please refer to the [LiteRT deployment documentation](https://ai.google.dev/edge/lite/inference) for ways to deploy a quantized LiteRT model.

### Advanced Recipes

There are many ways the user can configure and customize the quantization recipe beyond using a template in [recipe.py](ai_edge_quantizer/recipe.py). For example, the user can configure the recipe to achieve these features:

* Selective quantization (exclude selected ops from being quantized)
* Flexible mixed scheme quantization (mixture of different precision, compute precision, scope, op, config, etc)
* 4-bit weight quantization

TODO(b/354275253): Update colab path
These recipes are explored and constructed in the [ISNET colab]() example.

For specifics of the recipe schema, please refer to [recipe_manager.py](ai_edge_quantizer/recipe_manager.py):`OpQuantizationRecipe` which is the source of truth.

For advanced usage involving mixed quantization, the following API may be useful:

* Use [quantizer.py](ai_edge_quantizer/quantizer.py):`Quantizer:load_quantization_recipe()` to load a custom recipe.
* Use [quantizer.py](ai_edge_quantizer/quantizer.py):`Quantizer:update_quantization_recipe()` to add or override (if there's a conflict in the same scope) specific parts of the recipe.

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
|     |     |     |     |     |     |    |    |    |    |
| --- | --- | --- | --- | --- | --- |--- |--- |--- |--- |


**Operators Supporting Quantization**

|     |     |     |     |     |     |    |    |    |
| --- | --- | --- | --- | --- | --- |--- |--- |--- |
| **Config** | DYNAMIC_WI8_AFP32 | DYNAMIC_WI4_AFP32 | STATIC_WI8_AI16 | STATIC_WI4_AI16 | STATIC_WI8_AI8 | STATIC_WI4_AI8 | WEIGHTONLY_WI8_AFP32 | WEIGHTONLY_WI4_AFP32 |
|FULLY_CONNECTED  |<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|CONV_2D          |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|    |
|BATCH_MATMUL     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|     |<div align="center"> &check; </div>|    |<div align="center"> &check; </div>|    |
|EMBEDDING_LOOKUP |<div align="center"> &check; </div>|<div align="center"> &check; </div>|     |     |     |<div align="center"> &check; </div>|<div align="center"> &check; </div>|    |
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
| --- | --- | --- | --- | --- | --- |--- |--- |--- |
