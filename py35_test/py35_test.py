import numpy as np
import tensorflow as tf

n1 = tf.constant(2.5, dtype=tf.float32)
n2 = tf.constant(2.0)
sess = tf.Session()
print(sess.run([n1, n2]))

sess.close()
sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = a + b
d = c * 3.1
e = d + a

h = 1000
w = 1200

print(sess.run(e, {a: np.random.rand(h, w), b: np.random.rand(h, w)}))


k=0