import tensorflow as tf
import numpy as np
import os
import vae_util
from matplotlib import pyplot as plt

plt.figure()

# train
n_epochs = 1000
batch_size = 16
learn_rate = 5e-6

""" Image data """
data = np.load("/home/ljh/Project/vrep_jaco/src/vrep_jaco_data/data/dummy_data.npy",allow_pickle=True)
data = data[0][0][0]/5000
data = np.reshape(data,[1,data.shape[0],data.shape[1],1])

#train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
#n_samples = train_size

""" build graph """
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 480,640,1], name='input_img')
keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
x_hat, z, loss, neg_marginal_likelihood, KL_divergence = vae_util.autoencoder(x, dim_z=64 ,trainable=True)
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

""" training """
# train
#total_batch = int(n_samples / batch_size)
min_tot_loss = 1e99

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
outputlist = []

with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer(), feed_dict={keep_prob : 0.9})

    for epoch in range(n_epochs):
        # Random shuffling
        #np.random.shuffle(train_total_data)
        #train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]

        # Loop over all batches
        #for i in range(total_batch):
        for i in range(2):
            # Compute the offset of the current minibatch in the data.
            #offset = (i * batch_size) % (n_samples)
            #batch_xs_input = train_data_[offset:(offset + batch_size), :]

            _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                (train_op, loss, neg_marginal_likelihood, KL_divergence),
                feed_dict={x: data, keep_prob : 0.8})
            #    feed_dict={x: batch_xs_target, keep_prob : 0.8})

        # print cost every epoch
        print(f"epoch {epoch}: L_tot {tot_loss} L_likelihood {loss_likelihood} L_divergence {loss_divergence}")
        #output[0][0] = -10
        if epoch % 10 == 0:
            output = sess.run((x_hat),feed_dict={x:data,keep_prob:1})
            output = np.reshape(output,[480,640])*255
            print(np.count_nonzero(np.isnan(output)))
            outputlist.append(output)

data = np.reshape(data,[480,640])

np.save("vae_result_5e6_1000",outputlist)

for i in range(int(len(outputlist)/3)):
    plt.imshow(outputlist[i])
    plt.show()