import tensorflow as tf
from tensorflow.core.protobuf import config_pb2

config = config_pb2.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
  array_size = 2**10
  tf_var = tf.Variable(tf.random_uniform([array_size, array_size], -1., 1.))
  L1 = tf.matmul(tf_var,tf_var)
#  L2 = L1 + tf_var
#  L3 = L2 + tf_var
#  L3 = 
  sess.run(tf.global_variables_initializer())
  sess.run(L1)
