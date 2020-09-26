import os
import tensorflow as tf
import numpy as np


shapes = 5
act_shape = 2
with tf.variable_scope("hi"):
    x = tf.placeholder(shape=(None, shapes), dtype=tf.float32)
    action = tf.placeholder(shape=(None, act_shape), dtype=tf.float32)
    x_in = tf.layers.flatten(x)

    sieve = np.eye(shapes, dtype=np.float32)
    sieve[0][0] = 0
    sieve[3][3] = 0

    act_sieve = np.zeros([act_shape, shapes], dtype=np.float32)
    act_sieve[0][0] = 1
    act_sieve[1][3] = 1

    x1 = tf.matmul(x_in, sieve)
    a_shaped = tf.matmul(action, act_sieve)
    x11 = x1 + a_shaped
    x2 = tf.layers.dense(x1, 16)
    x_out = tf.layers.dense(x2, 4, activation='softmax')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    inp = np.array(
            [[2,5,8,1,9],
            [0,1,2,3,4]]
        )
    act = np.array(
        [[10,20],
        [30,40]]
        )

    #c = sess.run(x_out, feed_dict={x:inp})
    
    d = sess.run(x1, feed_dict={x:inp, action:act})
    e = sess.run(x11, feed_dict={x:inp, action:act})
    print(d)
    print(e)

'''
weight = []
for i in [1,10,100]:
    sub_ = []
    for j in range(1,4):
        sub_.append(j*i)
    weight.append(sub_)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# a -> batch: 3 // primitives: 4
a = tf.constant(weight, tf.float32)
print(sess.run(a))
b = tf.reshape(a[:,2],[-1,1])
c = tf.tile(b, tf.constant([1,20])) # total action: 20
print(sess.run(c))

# action dimension differ by primitive
act_index = [[0,1,2,3,4], [5,6,7,8], [0,1,2,17,18,19], [9,10]]

#for primitive 2
mask = np.zeros([c.shape[0],c.shape[1]])
for index in act_index[2]:
    mask[:, index] = 1

d = c * tf.constant(mask, dtype=tf.float32)
print(sess.run(d))

e = f = c * 0
f += 1
print(sess.run(e))
print(sess.run(f))

g = tf.concat([0,f],1)
print(sess.run(g))

tf.zero'''