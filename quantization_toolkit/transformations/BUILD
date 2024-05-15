# libraries of a transfomrmations used to support quantization emulation

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "dequant_insert",
    srcs = ["dequant_insert.py"],
    srcs_version = "PY3",
    deps = [
        ":quantize_tensor",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/tensorflow/lite/python:schema_py",
    ],
)

pytype_strict_contrib_test(
    name = "dequant_insert_tests",
    srcs = ["dequant_insert_tests.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    srcs_version = "PY3",
    deps = [
        ":dequant_insert",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/numpy",
        "//third_party/tensorflow/lite/python:schema_py",
    ],
)

pytype_strict_library(
    name = "quantize_tensor",
    srcs = ["quantize_tensor.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/py/numpy",
        "//third_party/tensorflow/lite/python:schema_py",
    ],
)

pytype_strict_contrib_test(
    name = "quantize_tensor_tests",
    srcs = ["quantize_tensor_tests.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    srcs_version = "PY3",
    deps = [
        ":quantize_tensor",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/numpy",
    ],
)
