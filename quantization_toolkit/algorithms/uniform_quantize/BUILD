load("//devtools/python/blaze:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "uniform_quantize_tensor",
    srcs = ["uniform_quantize_tensor.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/odml/model_customization/quantization:qtyping",
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "uniform_quantize_tensor_test",
    srcs = ["uniform_quantize_tensor_test.py"],
    deps = [
        ":uniform_quantize_tensor",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:qtyping",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "naive_min_max_quantize",
    srcs = ["naive_min_max_quantize.py"],
    srcs_version = "PY3",
    deps = [
        ":uniform_quantize_tensor",
        "//third_party/odml/model_customization/quantization:qtyping",
        "//third_party/odml/model_customization/quantization/algorithms/utils:min_max_quantize_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "naive_min_max_quantize_test",
    srcs = ["naive_min_max_quantize_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        ":naive_min_max_quantize",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:qtyping",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)
