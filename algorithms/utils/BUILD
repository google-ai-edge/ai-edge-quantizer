load("//devtools/python/blaze:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//third_party/odml/model_customization/quantization:__subpackages__"],
)

pytype_strict_library(
    name = "min_max_quantize_utils",
    srcs = ["min_max_quantize_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:uniform_quantize_tensor",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "min_max_quantize_utils_test",
    srcs = ["min_max_quantize_utils_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        ":min_max_quantize_utils",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:uniform_quantize_tensor",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)
