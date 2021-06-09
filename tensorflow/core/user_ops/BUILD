load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "cuda_op_kernel.so",
    srcs = ["cuda_op_kernel.cc"],
    gpu_srcs = ["cuda_op_kernel.cu.cc"],
)

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)