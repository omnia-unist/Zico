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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <chrono>

typedef std::chrono::time_point< std::chrono::_V2::steady_clock, std::chrono::duration<long int, std::ratio<1, 1000000000> > > time_point;

// __global__ void K1_Read(const float* in, const int N, float* out,
// float* temp, float* dptr) {
__global__ void K1_Read(const float* in, const int N, float* temp,
                        float* dptr) {
  clock_t clock_count = 300000000;
  clock_t start_clock = clock64();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count) {
    clock_offset = clock64() - start_clock;
  }
  *dptr = temp[0]; // not necessarily (K2 will check the corruption)
  printf("================== K1_Read: temp[0] is %f. ================== (gm)\n",
         temp[0]);
}

// __global__ void K1_Write(const float* in, const int N, float* out, float* temp, float* dptr) {
__global__ void K1_Write(const float* in, const int N, float* temp,
                               float* dptr) {
  clock_t clock_count = 300000000;
  clock_t start_clock = clock64();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count) {
    clock_offset = clock64() - start_clock;
  }
  temp[0] = -100;
  printf(
      "================== K1_Write: Now, temp[0] is %f. "
      "================== (gm)\n",
      temp[0]);
}

void K1_OP_Launcher(const float* in, const int N, float* out, float* temp,
                    time_point basetime) {
// void K1_OP_Launcher(const float* in, const int N, float* temp, time_point
// basetime) {
  float* dptr;
  cudaMallocManaged((void**)&dptr, 4);
  cudaStream_t stream_;
  cudaStreamCreate(&stream_);
  K1_Read<<<1, 1, 0, stream_>>>(in, N, temp, dptr);
  K1_Write<<<1, 1, 0, stream_>>>(in, N, temp, dptr);
  // K1_Read<<<1, 1>>>(in, N, temp, dptr);
  // K1_Write<<<1, 1>>>(in, N, temp, dptr);
}

/**********************************************************************************************************/
/**********************************************************************************************************/
/**********************************************************************************************************/

// __global__ void K1_Kernel(const float* in, const int N, float* out, float* temp, float val, int* dptr) {
//   dptr[0] = 1;
//   float cur = out[0];
//   printf("(gangmuk) K1_Kernel Start: Existing out[0] is %f\n", out[0]);
//   // int busy_waiting = 10000; 
//   // for (int i = 0; i < busy_waiting; i++) {
//   //   for (int j = 0; j < busy_waiting; j++) {
//   //     dptr[0] = i * j * dptr[0];
//   //   }
//   // }
//   if (cur == out[0]) {
//     out[0]++;
//   }
//   else {
//     printf("(gangmuk) [FATAL ERROR]: Something touches my(K1) memory space!\n");
//     printf("(gangmuk) out[0] must be %f, but it is %f.\n", cur, out[0]);
//     assert(1);
//   }
//   printf("(gangmuk) K1_Kernel Done: Now Set out[0] to %f\n", out[0]);
// }


// void K1_OP_Launcher_2(const float* in, const int N, float* out, float* temp) {
//   int* dptr;
//   cudaMallocManaged((void**)&dptr, 4);
//   float val = 1;
//   printf("(gangmuk) Launching K1_OP(%f)\n", val);
//   K1_Kernel<<<1, 1>>>(in, N, out, temp, val, dptr);
//   printf("(gangmuk) Returned from K1_OP(%f)\n", val);
// }

#endif
