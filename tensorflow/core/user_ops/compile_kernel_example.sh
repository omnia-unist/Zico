#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# CPU kernel compilation
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
# reference: https://www.tensorflow.org/guide/create_op#gpu_kernels

# nvcc -std=c++11 -c -o kernel_example.cu.o kernel_example.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 -shared -o kernel_example.so kernel_example.cc kernel_example.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
# reference: https://stackoverflow.com/questions/44403127/adding-a-gpu-op-in-tensorflow--

