*AI Edge Quantizer*

AI Edge Quantizer is a **post training quantization (PTQ) tool** for LiteRT models (formerly known as TFLite), with the following key features:

* Selective Quantization:
    * Enables users to quantize specific operations at layer level (e.g., only quantizing FullyConnected Ops in FeedForward layers and leave all other Ops as float).
* Mixed Precision Quantization:
    * Allows users to specify different precision levels (activation and weight) for operators at layer level (INT4 weights FullyConnected in FeedForward layers but INT8 weights in Attention layers).
* Advanced Quantization Functions:
    * Provides functionalities like block-based/sub-channel quantization, along with weight-only and dynamic range quantization, ensuring compatibility with the DarwiNN toolchain.
* Full Integer Quantization:
    * Offers full integer quantization, including INT16/INT8 activations with INT8/INT4 weights, a requirement for many Android OEM toolchains and hardware like Qualcomm and Samsung NPUs.

The AI Edge Quantizer is one of many different tools for quantization across various teams and use-cases. Other quantization tooling with a path to on-device deployment include [TensorFlow Quantizer](go/tf-quantizer) and [TfLite quantizer](https://www.tensorflow.org/lite/performance/post_training_quantization). Feature parity of AI Edge Quantizer to TFLite Quantizer is WIP.

See go/ai-edge-quantizer for more details.
