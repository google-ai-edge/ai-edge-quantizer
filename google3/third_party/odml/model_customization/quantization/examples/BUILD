load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_binary(
    name = "mnist_toy_model",
    srcs = ["mnist_toy_model.py"],
    data = ["//third_party/odml/model_customization/quantization/test_models:test_tfl_models"],
    deps = [
        "//pyglib:gfile",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:model_modifier",
        "//third_party/odml/model_customization/quantization:model_validator",
        "//third_party/odml/model_customization/quantization:params_generator",
        "//third_party/odml/model_customization/quantization:recipe_manager",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_interpreter_utils",
        "//third_party/odml/model_customization/quantization/utils:validation_utils",
        "//third_party/py/absl:app",
    ],
)
