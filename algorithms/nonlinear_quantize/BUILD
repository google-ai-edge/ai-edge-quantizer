load("//devtools/python/blaze:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "fp16_quantize",
    srcs = ["fp16_quantize.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "fp16_quantize_test",
    srcs = ["fp16_quantize_test.py"],
    data = [
        "//third_party/odml/model_customization/quantization/test_models:test_tfl_models",
    ],
    deps = [
        ":fp16_quantize",
        "//testing/pybase",
        "//third_party/odml/model_customization/quantization:typing",
        "//third_party/odml/model_customization/quantization/utils:test_utils",
        "//third_party/odml/model_customization/quantization/utils:tfl_flatbuffer_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
    ],
)
