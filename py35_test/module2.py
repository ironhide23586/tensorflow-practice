import tensorflow as tf
import numpy as np
import idx2numpy
from tensorflow.examples.tutorials.mnist import input_data

def get_mnist_worker(x_fname, y_fname):
  x_all = idx2numpy.convert_from_file(x_fname)
  x_all = np.array([x.flatten() for x in x_all])
  y_all = idx2numpy.convert_from_file(y_fname)
  return x_all, y_all

def get_mnist_testset():
    return get_mnist_worker('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

def get_mnist_trainset():
    return get_mnist_worker('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')


#x_train_all, y_train_all = get_mnist_trainset()
#x_test_all, y_test_all = get_mnist_testset()

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

batch_size = 128
epochs = 10

neuron_counts = np.array([784, 12, 24, 10])

x_in = tf.placeholder(tf.float32, [None, 784])
#y_in = tf.placeholder(tf.int64)

#y_in_onehot = tf.one_hot(y_in, 10, axis=-1)
y_in_onehot = tf.placeholder(tf.float32, [None, 10])

weights = [tf.Variable(np.random.randn(neuron_counts[i], neuron_counts[i + 1]), dtype=tf.float32) for i in range(neuron_counts.shape[0] - 1)]
#weights = [tf.Variable(tf.zeros([neuron_counts[i], neuron_counts[i + 1]]), dtype=tf.float32) for i in range(neuron_counts.shape[0] - 1)]
biases = [tf.Variable(tf.zeros(neuron_counts[i + 1]), dtype=tf.float32) for i in range(neuron_counts.shape[0] - 1)]
layer_outs = [tf.Variable(tf.zeros([batch_size, neuron_counts[i + 1]]), dtype=tf.float32) for i in range(neuron_counts.shape[0] - 1)]

if neuron_counts.shape[0] > 2:
  layer_outs[0] = tf.matmul(x_in, weights[0]) + biases[0]
  for i in range(1, neuron_counts.shape[0] - 1):
    layer_outs[i] = tf.nn.relu(tf.matmul(layer_outs[i - 1], weights[i]) + biases[i])
  #layer_outs[-1] = tf.nn.softmax(tf.matmul(layer_outs[-2], weights[-1]) + biases[-1])
  layer_outs[-1] = tf.matmul(layer_outs[-2], weights[-1]) + biases[-1]
else:
  #layer_outs[0] = tf.nn.softmax(tf.matmul(x_in, weights[0]) + biases[0])
  layer_outs[0] = tf.matmul(x_in, weights[0]) + biases[0]

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_in_onehot * tf.log(layer_outs[-1]), reduction_indices=[1]))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_in_onehot, logits=layer_outs[-1]))

correct_preds = tf.equal(tf.arg_max(tf.nn.softmax(layer_outs[-1]), 1), tf.arg_max(y_in_onehot, 1))
acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

opt = tf.train.GradientDescentOptimizer(0.05)
trainer = opt.minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

start_idx = 0
epoch = 1
iter = 1

#train_limit = (x_train_all.shape[0] / batch_size) * batch_size
train_limit = int(mnist.train.images.shape[0] / batch_size) * batch_size

x_test = mnist.test.images
y_test_onehot = mnist.test.labels

while epoch <= epochs:
  if (start_idx + 1) * batch_size > train_limit:
    start_idx = 0
    epoch += 1
    iter = 1
  #x_train = x_train_all[start_idx * batch_size : batch_size * (start_idx + 1)]
  #y_train = y_train_all[start_idx * batch_size : batch_size * (start_idx + 1)]
  x_train, y_train = mnist.train.next_batch(batch_size)
  start_idx += 1
  
  loss, _ = sess.run([cross_entropy, trainer], {x_in: x_train, y_in_onehot: y_train})
  accuracy = sess.run(acc, {x_in: x_test, y_in_onehot: y_test_onehot})

  print('Batch', iter, 'Epoch =', epoch, 'Loss =', loss, 'Accuracy =', accuracy, 'TENSORFLOW_MNIST_GPU')
  iter += 1
  k = 0

k = 0