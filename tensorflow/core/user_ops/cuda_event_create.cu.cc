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

__global__ void K2_Kernel(const float* in, const int N, float* out, float val, float* dptr, clock_t* ct) {
  printf("(gangmuk) K2_Kernel Start: Existing out[0] is %f, time_,%lf\n", out[0], 0);
  dptr[0] = 1;
  float cur = out[0];

  //////////////////////////////////////////////
  // clock_t clock_count = 100000000000;
  // // clock_t d_o;
  // clock_t start_clock = clock64();
  // clock_t clock_offset = 0;
  // printf("clocking...\n");
  // while (clock_offset < clock_count) {
  //   clock_offset = clock64() - start_clock;
  // }
  // ct[0] = clock_offset;
  // printf("clocking finishes!\n");
  //////////////////////////////////////////////

  //////////////////////////////////////////////
  // int busy_waiting = 51474;  // around 10s
  // int busy_waiting = 31474;  // around 3s
  // int busy_waiting = 51374;     // less than 1s
  // int busy_waiting = 31074;  // less than 1s
  // int busy_waiting = 30474;  // less than 1s
  // int busy_waiting = 25474;  // less than 1s
  // for (int i = 0; i < busy_waiting; i++) {
  //   for (int j = 0; j < busy_waiting; j++) {
  //     dptr[0] = i * j * dptr[0];
  //   }
  // }
  //////////////////////////////////////////////

  if (cur == out[0]) {
    out[0]++;
  }
  else {
    printf("(gangmuk) [FATAL ERROR]: Something touches my(K2) memory space!\n");
    printf("(gangmuk) out[0] must be %f, but it is %f.\n", cur, out[0]);
    assert(2);
  }
  printf("(gangmuk) K2_Kernel Done: Now Set out[0] to %f, time_,%lf\n", out[0], 0);
}

__global__ void K2_Read(const float* in, const int N, float* out, float* temp, float* dptr) {
  *dptr = temp[0];
  printf("(gangmuk) K2_Read: Existing out[0] is %f and temp[0] is %f.\n", out[0], temp[0]);
}

__global__ void K2_Write(const float* in, const int N, float* out, float* temp, float* dptr) {
  
  ////////////////////////////////////////////
  clock_t clock_count = 1000000000;
  // clock_t d_o;
  clock_t start_clock = clock64();
  clock_t clock_offset = 0;
  printf("(gangmuk) clocking...\n");
  while (clock_offset < clock_count) {
    clock_offset = clock64() - start_clock;
  }
  // ct[0] = clock_offset;
  printf("(gangmuk) clocking finishes!\n");
  ////////////////////////////////////////////
  if (dptr[0] != temp[0]) {
    printf("(gangmuk) [ !!! FATAL ERROR !!! ]: previous temp[0] (%f) != current temp[0] (%f)\n", dptr[0], temp[0]);
  }
  out[0]++;
  temp[0]+=2;
  printf("(gangmuk) K2_Write: Now, out[0] it is %f and temp[0] is %f.\n", out[0], temp[0]);
  // ////////////////////////////////////////////
  // clock_t clock_count2 = 3000000000;
  // // clock_t d_o;
  // clock_t start_clock2 = clock64();
  // clock_t clock_offset2 = 0;
  // printf("clocking...\n");
  // while (clock_offset2 < clock_count2) {
  //   clock_offset2 = clock64() - start_clock2;
  // }
  // // ct[0] = clock_offset;
  // printf("clocking finishes!\n");
  // ////////////////////////////////////////////
}

void K2_OP_Launcher_2(const float* in, const int N, float* out, float* temp, time_point basetime) {
  float* dptr;
  cudaMallocManaged((void**)&dptr, 4);

  float val = 2;
  auto time_span = std::chrono::steady_clock::now() - basetime;
  double nsec = double(time_span.count()) *
                std::chrono::steady_clock::period::num /
                std::chrono::steady_clock::period::den;
  printf("(gangmuk) ======================= Launching K2_OP, time_,%lf =======================\n", nsec);

  clock_t* ct;
  cudaMalloc((void**)&ct, 8);

  K2_Kernel<<<1, 1>>>(in, N, out, val, dptr, ct);

  // cudaFree(dptr);
  auto time_span2 = std::chrono::steady_clock::now() - basetime;
  double nsec2 = double(time_span2.count()) *
                 std::chrono::steady_clock::period::num /
                 std::chrono::steady_clock::period::den;
  printf("(gangmuk) ======================= Returned from K2_OP, time_,%lf =======================\n", nsec2);
}

void K2_OP_Launcher(const float* in, const int N, float* out, float* temp, time_point basetime) {
  float* dptr;
  cudaMallocManaged((void**)&dptr, 4);

  auto time_span = std::chrono::steady_clock::now() - basetime;
  double nsec = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
  printf("(gangmuk) ======================= Launching K2_OP, time_,%lf =======================\n", nsec);

  K2_Read<<<1, 1>>>(in, N, out, temp, dptr);
  printf("(gangmuk) CPU sleeps...\n");
  // sleep(5);
  printf("(gangmuk) CPU wakes up!\n");
  K2_Write<<<1, 1>>>(in, N, out, temp, dptr);

  auto time_span2 = std::chrono::steady_clock::now() - basetime;
  double nsec2 = double(time_span2.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
  printf("(gangmuk) ======================= Returned from K2_OP, time_,%lf =======================\n", nsec2);
}

#endif
