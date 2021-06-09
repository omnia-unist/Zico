/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// #define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <chrono>
#include <cmath>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

typedef std::chrono::time_point<
    std::chrono::_V2::steady_clock,
    std::chrono::duration<long int, std::ratio<1, 1000000000> > >
    time_point;

REGISTER_OP("K2")
    .Input("input: float")
    .Output("output: float")
    .Doc(R"doc(K2 Kernel)doc");

// void K2_OP_Launcher(const float* in, const int N, float* out, float* temp, time_point basetime);
void K2_OP_Launcher(const float* in, const int N, float* temp, time_point basetime);

class K2_OP : public OpKernel {
 public:
  time_point basetime = std::chrono::steady_clock::now();

  explicit K2_OP(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // auto time_span = std::chrono::steady_clock::now() - basetime;
    // double nsec = double(time_span.count()) *
    //               std::chrono::steady_clock::period::num /
    //               std::chrono::steady_clock::period::den;

    // Create an temporary tensor
    Tensor temp_tensor;
    int byte_size = pow(2,26);
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                   TensorShape({byte_size}),
                                                   &temp_tensor));
    printf(
        " cccccccccccccc K2 OP: allocate_temp (underlying data addr: %p), "
        " cccccccccccccc (gm)\n",
        temp_tensor.public_base<void*>());

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    printf(
        " cccccccccccccc K2 OP: allocate_output (underlying data addr: %p), "
        " cccccccccccccc (gm)\n",
        output_tensor->public_base<void*>());

    auto temp = temp_tensor.flat<float>();
    // auto output = output_tensor->template flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    // Call the cuda kernel launcher
    // K2_OP_Launcher(input.data(), N, output.data(), temp.data(), basetime);
    printf(
        " cccccccccccccc Launching K2_OP "
        " cccccccccccccc (gm)\n");
    K2_OP_Launcher(input.data(), N, temp.data(), basetime);
    printf(
        " cccccccccccccc Returned from K2_OP "
        " cccccccccccccc (gm)\n\n");
  }
};

REGISTER_KERNEL_BUILDER(Name("K2").Device(DEVICE_GPU), K2_OP);
