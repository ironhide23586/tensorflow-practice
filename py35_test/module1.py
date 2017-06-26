import tensorflow as tf
import numpy as np

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

sess.run(linear_model, {x: np.random.rand(1, 10)})

y = tf.placeholder(tf.float32)
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

print(sess.run(loss, {x: [1, 2, 3, 4], y:[0, -1, -2, -3]}))

#fixW = tf.assign(W, [-1.])
#fixB = tf.assign(b, [1.])

#sess.run([fixW, fixB])

#l1 = sess.run(loss, {x: [1, 2, 3, 4], y:[0, -1, -2, -3]})
#k=0

opt = tf.train.GradientDescentOptimizer(.01)
train = opt.minimize(loss)

for i in range(1000):
  a = sess.run([loss, train], {x: [1, 2, 3, 4], y:[0, -1, -2, -3]})
  k = 0

k = 0