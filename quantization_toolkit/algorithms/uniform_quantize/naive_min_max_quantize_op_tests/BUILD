load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//third_party/odml/model_customization/quantization:__subpackages__"],
)

py_strict_test(
    name = "conv2d_test",
    srcs = ["conv2d_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        ":test_utils",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:naive_min_max_quantize",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:uniform_quantize_tensor",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "fully_connected_test",
    srcs = ["fully_connected_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        ":test_utils",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:naive_min_max_quantize",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:uniform_quantize_tensor",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "reshape_test",
    srcs = ["reshape_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        ":test_utils",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:naive_min_max_quantize",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:uniform_quantize_tensor",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "average_pool_2d_test",
    srcs = ["average_pool_2d_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        ":test_utils",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:naive_min_max_quantize",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:uniform_quantize_tensor",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "batch_matmul_test",
    srcs = ["batch_matmul_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        ":test_utils",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:naive_min_max_quantize",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:uniform_quantize_tensor",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "embedding_lookup_test",
    srcs = ["embedding_lookup_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        ":test_utils",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:naive_min_max_quantize",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:uniform_quantize_tensor",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:naive_min_max_quantize",
        "//third_party/odml/model_customization/quantization/algorithms/uniform_quantize:uniform_quantize_tensor",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)
