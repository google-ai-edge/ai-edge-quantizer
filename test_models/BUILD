package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//third_party/odml:internal"],
)

filegroup(
    name = "test_tfl_models",
    srcs = glob(["*.tflite"]),
    visibility = ["//third_party/odml:__subpackages__"],
)

filegroup(
    name = "test_recipes",
    srcs = [
        "recipes/conv_fc_mnist_drq_recipe.json",
        "recipes/conv_fc_mnist_weight_only_recipe.json",
    ],
    visibility = ["//third_party/odml:__subpackages__"],
)
