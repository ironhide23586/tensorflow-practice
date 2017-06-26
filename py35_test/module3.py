import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

batch_size = 128

def weight_var(shape):
  init = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(init)

def bias_var(shape):
  init = tf.constant(0.1, shape=shape)
  return tf.Variable(init)

def conv2d(x_inp, filt, h_stride=1, w_stride=1):
   return tf.nn.conv2d(x_inp, filt, [1, h_stride, w_stride, 1], padding='SAME')

def max_pool_2x2(x_inp):
  return tf.nn.max_pool(x_inp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_in = tf.placeholder(tf.float32, [None, 784])
y_in = tf.placeholder(tf.float32, [None, 10])

x_in_reshaped = tf.reshape(x_in, [-1, 28, 28, 1])

conv1_w = weight_var([5, 5, 1, 32])
conv1_b = bias_var([32])

conv1_out = tf.nn.relu(conv2d(x_in_reshaped, conv1_w) + conv1_b)
conv1_pool = max_pool_2x2(conv1_out)

conv2_w = weight_var([5, 5, 32, 64])
conv2_b = bias_var([64])

conv2_out = tf.nn.relu(conv2d(conv1_out, conv2_w) + conv2_b)
conv2_pool = max_pool_2x2(conv2_out)


conv2_output_neuron_dims = np.array(conv2_pool.shape.as_list())[-3:]
conv2_output_neurons = 1
for d in conv2_output_neuron_dims:
  conv2_output_neurons *= d

conv2_pool_reshaped = tf.reshape(conv2_pool, [-1, int(conv2_output_neurons)])

fc1_w = weight_var([conv2_output_neurons, 1024])
fc1_b = bias_var([1024])

fc1_out = tf.nn.relu(tf.matmul(conv2_pool_reshaped, fc1_w) + fc1_b)

keep_prob = tf.placeholder(tf.float32)
fc1_out_dropout = tf.nn.dropout(fc1_out, keep_prob)

fc2_w = weight_var([1024, 10])
fc2_b = bias_var([10])

fc2_out = tf.matmul(fc1_out_dropout, fc2_w) + fc2_b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_in, logits=fc2_out))

y_out = tf.nn.softmax(fc2_out)
matches = tf.equal(tf.arg_max(y_in, 1), tf.arg_max(y_out, 1))
acc = tf.reduce_mean(tf.cast(matches, tf.float32))

opt = tf.train.GradientDescentOptimizer(0.05)
train_step = opt.minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  x_train, y_train = mnist.train.next_batch(batch_size)
  loss, _ = sess.run([cross_entropy, train_step], {x_in: x_train, y_in: y_train, keep_prob: 0.5})
  accuracy = sess.run(acc, {x_in: mnist.test.images, y_in: mnist.test.labels, keep_prob: 1.0})
  print('Iter ', i, ', loss =', loss, ', accuracy =', accuracy)