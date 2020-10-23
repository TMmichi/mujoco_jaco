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

    c = tf.zeros_like(a)
    d = tf.zeros((3,5))
    e = tf.matmul(c,d)
    #print(c.shape)
    print(e.shape)

    dist = tf.distributions.Beta(a_in, b_in, True, False)

    # gaus = a_in + b_in * tf.random_normal(tf.shape(a_in), dtype=a_in.dtype)
    # seive = np.zeros((shapes, 3), dtype=np.float32)
    # seive[0][0] = 1
    # seive[2][1] = 1
    # seive[3][2] = 1

    # x1 = tf.matmul(x_in,seive)
    # x2 = tf.layers.dense(x1, 16)
    # x_out = tf.layers.dense(x2, 4, activation='softmax')
    samples = dist.sample(30)
    log_probs = dist.log_prob(samples)
    mean_probs = tf.reduce_mean(log_probs,axis=0)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
#inp = np.array([[2,5,8,0,11],[0,1,2,3,4]])
ainp = np.array([[23,24,13],[7,14,12.5]])
binp = np.array([[23,24,3],[14,7,3.5]])

c,d = sess.run([log_probs, mean_probs], feed_dict={a:ainp, b:binp})
print(c)
print(d)