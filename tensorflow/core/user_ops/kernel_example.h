// kernel_example.h
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct ExampleFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif KERNEL_EXAMPLE_H_