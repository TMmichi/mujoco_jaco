import tensorflow as tf
import numpy as np
import random
from datetime import datetime as dt

#TODO: Implement FiLM Architecture
def FiLM(x,gamma_tensor,beta_tensor):
    return tf.add(tf.multiply(x,gamma_tensor),beta_tensor)

def CNN_Encoder(x, z_dim, drop_rate=0.2, trainable=True, d_switch=False):
    d_switch = d_switch or trainable
    random.seed(dt.now())
    seed = random.randint(0,12345)
    with tf.compat.v1.variable_scope("CNN_Encoder"):
        print(x.shape)
        batch1 = tf.layers.batch_normalization(
            inputs=x, axis=-1, scale=True, training=trainable, name="BN1")
        layer1 = tf.layers.conv2d(
            inputs=batch1, filters=16, kernel_size=8, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv1")
        pool1 = tf.layers.max_pooling2d(
            inputs=layer1,pool_size=[2,2],strides=2)
        dropout1 = tf.layers.dropout(
            inputs=pool1, rate=drop_rate, seed=seed, training=d_switch)
        print(dropout1.shape)

        batch2 = tf.layers.batch_normalization(
            inputs=dropout1, axis=-1, scale=True, training=trainable, name="BN2")
        layer2 = tf.layers.conv2d(
            inputs=batch2, filters=32, kernel_size=4, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv2")
        pool2 = tf.layers.max_pooling2d(
            inputs=layer2,pool_size=[2,2],strides=2)
        dropout2 = tf.layers.dropout(
            inputs=pool2, rate=drop_rate, seed=seed, training=d_switch)
        print(dropout2.shape)

        batch3 = tf.layers.batch_normalization(
            inputs=dropout2, axis=-1, scale=True, training=trainable, name="BN3")
        layer3 = tf.layers.conv2d(
            inputs=batch3, filters=64, kernel_size=4, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv3")
        pool3 = tf.layers.max_pooling2d(
            inputs=layer3,pool_size=[2,2],strides=2)
        dropout3 = tf.layers.dropout(
            inputs=pool3, rate=drop_rate, seed=seed, training=d_switch)
        print(dropout3.shape)

        batch4 = tf.layers.batch_normalization(
            inputs=dropout3, axis=-1, scale=True, training=trainable, name="BN4")
        layer4 = tf.layers.conv2d(
            inputs=batch4, filters=128, kernel_size=4, strides=(1,1), padding='same',
            activation=tf.nn.relu, trainable=trainable, name="Conv4")
        pool4 = tf.layers.max_pooling2d(
            inputs=layer4,pool_size=[2,2],strides=2)
        dropout4 = tf.layers.dropout(
            inputs=pool4, rate=drop_rate, seed=seed, training=d_switch)
        print(dropout4.shape)

        flat = tf.layers.flatten(dropout4)
        print(flat.shape)
        fc1 = tf.layers.dense(
            inputs=flat, units=512, name="FC1")
        dropout_fc1 = tf.layers.dropout(
            inputs=fc1, rate=drop_rate, seed=seed, training=d_switch)
        print(dropout_fc1.shape)
        
        fc2 = tf.layers.dense(
            inputs=dropout_fc1, units=2*z_dim, name="FC2")
        print(fc2.shape)
        
        mean = fc2[:, :z_dim]
        stddev = 1e-6 + tf.nn.softplus(fc2[:, z_dim:])

    return mean, stddev


def CNN_Decoder(z, drop_rate=0.2, reuse=False):
    with tf.variable_scope("CNN_Decoder", reuse=reuse):
        de_fc1 = tf.layers.dense(
            inputs=z, units=512, name="de_FC1")
        dropout_de_fc1 = tf.layers.dropout(
            inputs=de_fc1, rate=drop_rate)

        de_fc2 = tf.layers.dense(
            inputs=dropout_de_fc1, units=153600, name="de_FC2")
        dropout_de_fc2 = tf.layers.dropout(
            inputs=de_fc2, rate=drop_rate)

        unflat = tf.reshape(
            tensor=dropout_de_fc2, shape=[-1,30,40,128])

        de_layer1 = tf.layers.conv2d_transpose(
            inputs=unflat, filters=64, kernel_size=5, strides=(2,2), padding='same',
            activation=tf.nn.relu, name="DeConv1")
        de_dropout1 = tf.layers.dropout(
            inputs=de_layer1, rate=drop_rate)

        de_layer2 = tf.layers.conv2d_transpose(
            inputs=de_dropout1, filters=32, kernel_size=5, strides=(2,2), padding='same',
            activation=tf.nn.relu, name="DeConv2")
        de_dropout2 = tf.layers.dropout(
            inputs=de_layer2, rate=drop_rate) 

        de_layer3 = tf.layers.conv2d_transpose(
            inputs=de_dropout2, filters=16, kernel_size=6, strides=(2,2), padding='same',
            activation=tf.nn.relu, name="DeConv3")
        de_dropout3 = tf.layers.dropout(
            inputs=de_layer3, rate=drop_rate)
        
        de_layer4 = tf.layers.conv2d_transpose(
            inputs=de_dropout3, filters=1, kernel_size=8, strides=(2,2), padding='same', name="DeConv4")
        
        x_hat = tf.sigmoid(de_layer4)

    return x_hat


# Gateway
def autoencoder(x, dim_z, drop_rate=0.2, trainable=True):

    # encoding
    mu, sigma = CNN_Encoder(x, dim_z, drop_rate, trainable)

    # sampling by re-parameterization technique
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    # decoding
    x_hat = CNN_Decoder(z, drop_rate)
    x_hat = tf.clip_by_value(x_hat, 1e-8, 1 - 1e-8)

    # loss
    marginal_likelihood = tf.reduce_sum(x * tf.log(1e-8 +x_hat) + (1 - x) * tf.log(1e-8 + 1 - x_hat), 1)
    #marginal_likelihood = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x,logits=x_hat))
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return x_hat, z, loss, -marginal_likelihood, KL_divergence


def feature_fushion_MLP(mean_feature, input_placeholder):

    return []

#TODO: build data_fusion graph
def data_fusion_graph(input_placeholder):
    mean_feature = []
    # depth_arm
    mean_feature.append(CNN_Encoder(input_placeholder[0], z_dim=64, trainable=False)[0])
    # depth_bed
    mean_feature.append(CNN_Encoder(input_placeholder[1], z_dim=64, trainable=False)[0])
    # image_arm
    mean_feature.append(CNN_Encoder(input_placeholder[2], z_dim=64, trainable=False)[0])
    # image_bed
    mean_feature.append(CNN_Encoder(input_placeholder[3], z_dim=64, trainable=False)[0])

    state = feature_fushion_MLP(mean_feature, input_placeholder)

    return state