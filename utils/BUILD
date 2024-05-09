load("//devtools/python/blaze:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//third_party/odml/model_customization/quantization:__subpackages__"],
)

pytype_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":tfl_interpreter_utils",
        "//pyglib:resources",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "tfl_flatbuffer_utils",
    srcs = ["tfl_flatbuffer_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//pyglib:gfile",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/py/immutabledict",
        "//third_party/py/numpy",
        "//third_party/tensorflow/lite/python:schema_py",
        "//third_party/tensorflow/lite/tools:flatbuffer_utils",
    ],
)

pytype_strict_contrib_test(
    name = "tfl_flatbuffer_utils_test",
    srcs = ["tfl_flatbuffer_utils_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        ":test_utils",
        ":tfl_flatbuffer_utils",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "tfl_interpreter_utils",
    srcs = ["tfl_interpreter_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//pyglib:gfile",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:uniform_quantize_tensor",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_contrib_test(
    name = "tfl_interpreter_utils_test",
    srcs = ["tfl_interpreter_utils_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        ":test_utils",
        ":tfl_interpreter_utils",
        "//testing/pybase",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "validation_utils",
    srcs = ["validation_utils.py"],
    deps = [
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "validation_utils_test",
    srcs = ["validation_utils_test.py"],
    deps = [
        ":validation_utils",
        "//testing/pybase",
        "//third_party/py/numpy",
    ],
)
