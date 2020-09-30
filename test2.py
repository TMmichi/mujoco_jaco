import os
import tensorflow as tf
import numpy as np
import time

shapes = 3
with tf.variable_scope("hi"):
    a = tf.placeholder(shape=(None, shapes), dtype=tf.float32)
    a_in = tf.layers.flatten(a)
    #a_in = tf.layers.dense(a_in, len(a_in.shape)-1, activation=tf.nn.softplus)
    b = tf.placeholder(shape=(None, shapes), dtype=tf.float32)
    b_in = tf.layers.flatten(b)
    #b_in = tf.layers.dense(b_in, len(b_in.shape)-1, activation=tf.nn.softplus)

    dist = tf.distributions.Beta(concentration1=a_in, concentration0=b_in, validate_args=True, allow_nan_stats=False)

    gaus = a_in + b_in * tf.random_normal(tf.shape(a_in), dtype=a_in.dtype)
    # seive = np.zeros((shapes, 3), dtype=np.float32)
    # seive[0][0] = 1
    # seive[2][1] = 1
    # seive[3][2] = 1

    # x1 = tf.matmul(x_in,seive)
    # x2 = tf.layers.dense(x1, 16)
    # x_out = tf.layers.dense(x2, 4, activation='softmax')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
#inp = np.array([[2,5,8,0,11],[0,1,2,3,4]])
ainp = np.array([[1,2,3],[1,2,3],[1,2,3]])
binp = np.array([[1,2,3],[1,2,3],[1,2,3]])

then = time.time()
for i in range(200):
    c = sess.run(dist.mean(), feed_dict={a:ainp, b:binp})
print(time.time()-then)
then = time.time()
for i in range(200):
    d = sess.run(gaus, feed_dict={a:ainp, b:binp})
print(time.time()-then)
    