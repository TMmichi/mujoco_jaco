import os
import tensorflow as tf
import numpy as np

shapes = 5

with tf.variable_scope("hi"):
    x = tf.placeholder(shape=(None, shapes), dtype=tf.float32)
    x_in = tf.layers.flatten(x)

    seive = np.zeros((shapes, 3), dtype=np.float32)
    seive[0][0] = 1
    seive[2][1] = 1
    seive[3][2] = 1

    x1 = tf.matmul(x_in,seive)
    x2 = tf.layers.dense(x1, 16)
    x_out = tf.layers.dense(x2, 1, activation='softmax')
    x_out = tf.reshape(x_out, [-1])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    inp = np.array([[2,5,8,0,11],[0,1,2,3,4]])

    c = sess.run(x_out, feed_dict={x:inp})
    #d = sess.run(x1, feed_dict={x:inp})
    print(c)



'''print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hi'))

a = tf.constant([[1,2,3],[4,5,6],[7,5,6]], tf.float32)
print(sess.run(tf.reshape(a[:,0],[-1,1])))
b = tf.tile(tf.reshape(a[:,0],[-1,1]),tf.constant([1,2])) + 1
print(sess.run(b))
c = tf.exp(b)
print(sess.run(c))
d = tf.log(c)
print(sess.run(d))
rand = tf.random_normal(tf.shape(d))
e = rand * c
print((e))
e1 = tf.math.multiply(rand,c)
print((e1))'''