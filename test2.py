import os
import tensorflow as tf
import numpy as np

'''
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
    x_out = tf.layers.dense(x2, 4, activation='softmax')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    inp = np.array([[2,5,8,0,11],[0,1,2,3,4]])

    c = sess.run(x_out, feed_dict={x:inp})
    #d = sess.run(x1, feed_dict={x:inp})
    print(x1.name)
    print(c)
    print(np.sum(c,axis=1,))
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hi'))
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

tf.zero