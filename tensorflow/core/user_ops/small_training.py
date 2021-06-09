import tensorflow as tf
import numpy as np
import pdb
import random

dim = 1024
num_class = 100
hidden_layer1_dim = 4096

# x_data = np.array(tf.random_uniform([hidden_layer1_dim, dim], -1., 1.))
x_data = [[random.randint(-100, 100) for r in range(dim)]] * 20

# y_data = np.array(tf.random_uniform([hidden_layer1_dim, num_class], -1., 1.))
y_data = [[random.randint(0, 1) for r in range(num_class)]] * 20

with tf.device("/GPU:0"):
  X = tf.placeholder(tf.float32)
  Y = tf.placeholder(tf.float32)

  W1 = tf.Variable(tf.zeros([dim, hidden_layer1_dim]), name = "Weight1")
  W2 = tf.Variable(tf.zeros([hidden_layer1_dim, num_class]), name="Weight2")

  b1 = tf.Variable(tf.zeros([hidden_layer1_dim]), name = "Bias1")
  b2 = tf.Variable(tf.zeros([num_class]), name="Bias2")

  L1 = tf.add(tf.matmul(X, W1), b1)
  L1 = tf.nn.relu(L1)

  model = tf.add(tf.matmul(L1, W2), b2)

  cost = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

#optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train_op = optimizer.minimize(cost)

  init = tf.global_variables_initializer()
  print("@@@@@@@@@@@@@@@@@@@@ sess = tf.Session() @@@@@@@@@@@@@@@@@@@@@@")
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  print("@@@@@@@@@@@@@@@@@@@@ sess.run(init) @@@@@@@@@@@@@@@@@@@@@@@@@@@")
  sess.run(init)

  sess.run(train_op, feed_dict={X: x_data, Y: y_data})
