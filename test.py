import tensorflow as tf
import numpy as np

shapes = 5

x = tf.placeholder(shape=(None, shapes), dtype=tf.float32)
x_in = tf.layers.flatten(x)

seive = np.zeros((shapes, 2), dtype=np.float32)
seive[0][0] = 1
seive[2][1] = 1

x1 = tf.matmul(x_in,seive)
x2 = tf.layers.dense(x1, 16)
x_out = tf.layers.dense(x2, 4)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

inp = np.array([[2,5,8,0,11],[0,1,2,3,4]])

c = sess.run(x_out, feed_dict={x:inp})
d = sess.run(x1, feed_dict={x:inp})
print(c)
print(d)
print(x1.shape[:])