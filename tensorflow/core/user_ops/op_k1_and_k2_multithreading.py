import tensorflow as tf
import numpy as np
import time
from tensorflow.core.protobuf import config_pb2
from threading import Thread

k1_kernel = tf.load_op_library('./op_k1.so')
k2_kernel = tf.load_op_library('./op_k2.so')

# Set up tf.ConfigProto
vGPU1_memory_limit = 32000
vGPU2_memory_limit = 32000
virtual_device_gpu_options = config_pb2.GPUOptions(
    visible_device_list='0',
    experimental=config_pb2.GPUOptions.Experimental(virtual_devices=[
        config_pb2.GPUOptions.Experimental.VirtualDevices(
            memory_limit_mb=[vGPU1_memory_limit, vGPU2_memory_limit])]))

config = config_pb2.ConfigProto(gpu_options=virtual_device_gpu_options)

config.gpu_options.allow_growth = True

config.gpu_options.gangmuk.recording_tensors = True
config.gpu_options.gangmuk.tensor_recording_original_version = True
config.gpu_options.gangmuk.record_output_tensor_only = True

config.gpu_options.gangmuk.print_raw_internal_csv = True
config.gpu_options.gangmuk.adaptive_alloc = False
config.gpu_options.gangmuk.activate_global_pool = True
config.gpu_options.gangmuk.global_pool_release_ar = False
config.gpu_options.gangmuk.global_pool_release_all = True
config.gpu_options.gangmuk.gpu_release_ar = False
config.gpu_options.gangmuk.gpu_release_all = False
config.gpu_options.gangmuk.tighten_ar_size = True

config.gpu_options.gangmuk.skip_all_test = False
config.gpu_options.gangmuk.skip_ref_count_is_one = False

config.gpu_options.gangmuk.throttle_kernel_launch = False
config.gpu_options.gangmuk.max_launched_kernels = False
config.gpu_options.gangmuk.forward_input_profiling_mode = False

# Declare Session globally
# sess = tf.Session(config=config)

# sync = "init"

def run_k2_kernel():
  with tf.Session(config=config) as sess:
    with tf.device("/gpu:0"):
      array_size = 2**22
      tf_var = tf.Variable(tf.random_uniform([array_size], -1., 1.))

      K1 = k1_kernel.k1
      K1_L1 = K1(tf_var)

      K2 = k2_kernel.k2
      K2_L1 = K2(tf_var)

      init = tf.global_variables_initializer()
      sess.run(init)
      iters = 1
      last_iter = iters - 1
      for i in range(iters):
        if i is last_iter:
          while True:
            print("Thread 2 (K2): pause! Press \"2\" to continue (gm)")
            input_2 = input()
            if input_2 is "2":
              print("******************* Thread 2: start session.run(K2_L1) step: {} ******************* (gm)".format(i))
              sess.run(K2_L1)
              print("******************* Thread 2: return from session.run(K2_L1) step: {} ******************* (gm)".format(i))
              break
            else:
              print("Thread 2: no no no (gm)")
        # else:
        #   print("******************* Thread 2: start session.run(K2_L1) step: {} ******************* (gm)".format(i))
        #   sess.run(K2_L1)
        #   print("******************* Thread 2: return from session.run(K2_L1) step: {} ******************* (gm)".format(i))

def run_k1_kernel():
  with tf.Session(config=config) as sess:
    with tf.device("/gpu:1"):
      array_size = 2**22
      tf_var = tf.Variable(tf.random_uniform([array_size], -1., 1.))

      K1 = k1_kernel.k1
      K1_L1 = K1(tf_var)

      init = tf.global_variables_initializer()
      sess.run(init)
      iters = 1
      last_iter = iters - 1
      for i in range(iters):
        if i is last_iter:
          while True:
            print("Thread 1 (K1): pause! Press \"1\" to continue (gm)")
            input_1 = input()
            if input_1 is "1":
              print("****************** Thread 1: start session.run(K1_L1) step: {} ******************* (gm)".format(i))
              sess.run(K1_L1)
              print("******************* Thread 1: return from session.run(K1_L1) step: {} ******************* (gm)".format(i))
              break
            else:
              print("Thread 1: no no no (gm)")
        # else:
        #   print("****************** Thread 1: start session.run(K1_L1) step: {} ******************* (gm)".format(i))
        #   sess.run(K1_L1)
        #   print("******************* Thread 1: return from session.run(K1_L1) step: {} ******************* (gm)".format(i))


# Threading
th1 = Thread(target=run_k2_kernel, args=())
th2 = Thread(target=run_k1_kernel, args=())

th1.start()
# time.sleep(5)  # time sleep between (Kernel 1 launch) and (Kernel 2 launch)
th2.start()

th1.join()
th2.join()
