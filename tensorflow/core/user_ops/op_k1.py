import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from threading import Thread

k1_kernel = tf.load_op_library('./op_k1.so')
# k2_kernel = tf.load_op_library('./op_k2.so')

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
config.gpu_options.gangmuk.record_output_tensor_only = True
config.gpu_options.gangmuk.throttle_kernel_launch = False
config.gpu_options.gangmuk.max_launched_kernels = False
config.gpu_options.gangmuk.forward_input_profiling_mode = False


def run_k1_kernel():
  with tf.Session(config=config) as sess:
    with tf.device("/gpu:1"):
      array_size = 2**22
      tf_var = tf.Variable(tf.random_uniform([array_size], -1., 1.))

      k1 = k1_kernel.k1
      L1 = k1(tf_var)
      L2 = k1(L1)
      L3 = k1(L2)

      init = tf.global_variables_initializer()
      sess.run(init)
      iters = 200
      last_iter = iters - 1
      for i in range(iters):
        if i is last_iter:
          print("pause!")
          dummy = input()
        print(
            "******************* start session.run(K2) step: {} *******************".format(i))
        sess.run(L1)
        print("******************* return from session.run(K2) step: {} *******************\n".format(i))



run_k1_kernel()

# Threading
# th1 = Thread(target=run_k1_kernel, args=())
# th2 = Thread(target=run_k1_kernel, args=())

# th1.start()
# # time.sleep(5)  # time sleep between (Kernel 1 launch) and (Kernel 2 launch)
# th2.start()

# th1.join()
# th2.join()
