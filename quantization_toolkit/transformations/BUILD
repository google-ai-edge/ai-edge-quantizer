# libraries of a transfomrmations used to support quantization emulation

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "transformation_utils",
    srcs = ["transformation_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/py/numpy",
        "//third_party/tensorflow/lite/python:schema_py",
    ],
)

pytype_strict_contrib_test(
    name = "transformation_utils_test",
    srcs = ["transformation_utils_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    srcs_version = "PY3",
    deps = [
        ":transformation_utils",
        "//testing/pybase",
        "//testing/pybase:parameterized",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/numpy",
        "//third_party/tensorflow/lite/python:schema_py",
    ],
)

pytype_strict_library(
    name = "emulated_subchannel",
    srcs = ["emulated_subchannel.py"],
    srcs_version = "PY3",
    deps = [
        ":transformation_utils",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/py/numpy",
        "//third_party/tensorflow/lite/python:schema_py",
    ],
)

pytype_strict_contrib_test(
    name = "emulated_subchannel_test",
    srcs = ["emulated_subchannel_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    srcs_version = "PY3",
    deps = [
        ":emulated_subchannel",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/numpy",
        "//third_party/tensorflow/lite/python:schema_py",
    ],
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
    name = "dequant_insert_test",
    srcs = ["dequant_insert_test.py"],
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
    name = "quantize_tensor_test",
    srcs = ["quantize_tensor_test.py"],
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
