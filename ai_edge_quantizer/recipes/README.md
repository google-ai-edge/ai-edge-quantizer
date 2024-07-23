# Collection of Quantization Recipes

This directory contains a collection of quantization recipes that are commonly
used and supported on the TFLite runtime.

## Usage

These quantization recipes are provided in the JSON format. They can be used by
directly loading into a `Quantizer` instance as such:

```
qt = quantizer.Quantizer(f32_model_path)
qt.load_quantization_recipe(default_a8w8_recipe.json)
```

## Nomenclature

The filenames of recipes in this directory can be interpreted as follows:

`default` specifies the same scheme for all scopes and ops in the model.

`a<type>` means that the activation tensors are to be quantized to `type`.
If `type` is not prefixed by some letter, it indicates an integer type. E.g.
`a8` = int8 activation and `af32` = float32 activation.

`w<type>` means that the weight tensors are to be quantized to `type`.
If `type` is not prefixed by some letter, it indicates an integer type. E.g.
`a8` = int8 weight.

`[float|int]` specifies whether computation should be performed in floating
point or integer. If unspecified that means there's no ambiguity and that only
one computation type is relevant.