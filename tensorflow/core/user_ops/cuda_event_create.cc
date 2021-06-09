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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <chrono>

using namespace tensorflow;  // NOLINT(build/namespaces)

typedef std::chrono::time_point< std::chrono::_V2::steady_clock, std::chrono::duration<long int, std::ratio<1, 1000000000> > > time_point;

REGISTER_OP("CudaEventCreate")
    .Input("input: float")
    .Output("output: float")
    .Doc(R"doc(K2 Kernel)doc");

void CudaEventCreate_Launcher(const float* in, const int N, float* out, float* temp, time_point basetime);

class CudaEventCreate : public OpKernel {
 public:
  time_point basetime = std::chrono::steady_clock::now();

  explicit K2_OP(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    auto time_span = std::chrono::steady_clock::now() - basetime;
    double nsec = double(time_span.count()) *
                  std::chrono::steady_clock::period::num /
                  std::chrono::steady_clock::period::den;
    printf("(gangmuk) K2 OP: allocate_output, time_,%lf\n", nsec);
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    printf("(gangmuk) K2 OP: allocate_temp, time_,%lf\n", nsec);
    Tensor temp_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<float>::value,
                                          input_tensor.shape(), &temp_tensor));

    auto temp = temp_tensor.flat<float>();
    auto output = output_tensor->template flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    // Call the cuda kernel launcher
    K2_OP_Launcher(input.data(), N, output.data(), temp.data(), basetime);
  }
};

REGISTER_KERNEL_BUILDER(Name("K2").Device(DEVICE_GPU), K2_OP);
