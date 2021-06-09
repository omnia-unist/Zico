import tensorflow as tf

add_one_kernel = tf.load_op_library('./cuda_op_kernel.so')
with tf.Session() as sess:
  result = add_one_kernel.add_one([[1, 2, 3, 4]])
  print("==================================",sess.run(result))
  
# Prints
# [2 3 4 5]
