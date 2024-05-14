load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "mnist_samples",
    srcs = glob(["mnist_samples/*.png"]),
    visibility = ["//third_party/odml:__subpackages__"],
)

pytype_strict_binary(
    name = "mnist_toy_model",
    srcs = ["mnist_toy_model.py"],
    data = [
        ":mnist_samples",
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:quantization_toolkit",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_interpreter_utils",
        "//third_party/py/PIL:pil",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)
