import tensorflow as tf
import numpy as np
import utils


def leaky_relu(input, name, leak=0.2):
    return tf.maximum(input, leak * input, name=name)


def get_discriminator(input, is_train, output_dim=1, reuse=False):
    # name used later to pull all variables used in discriminator
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()
        # Possible bug: first layer does batch_norm but doesn't use it
        activated_conv1 = _conv_bn_relu(is_train, "dis_conv1", input, 64)
        #utils.print_shape(activated_conv1)

        activated_conv2 = _conv_bn_relu(is_train, "dis_conv2", activated_conv1, 128)
        #utils.print_shape(activated_conv2)

        activated_conv3 = _conv_bn_relu(is_train, "dis_conv3", activated_conv2, 256)
        #utils.print_shape(activated_conv3)

        activated_conv4 = _conv_bn_relu(is_train, "dis_conv4", activated_conv3, 512)
        #utils.print_shape(activated_conv4)

        # go into a fully connected layer
        conv4_dim = int(np.prod(activated_conv4.get_shape()[1:]))
        fc1 = tf.reshape(activated_conv4, shape=[-1, conv4_dim], name='fc1')
        #utils.print_shape(fc1)

        # down to the output_dimension of 1
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], output_dim], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[output_dim], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        #utils.print_shape(logits)
        return logits


def _conv_bn_relu(is_train, name, last_layer, num_filters, filter_size=[5, 5], strides=[2, 2]):
    conv = tf.layers.conv2d(last_layer, num_filters, kernel_size=filter_size, strides=strides,
                            padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            name=name + "_conv")
    bn = tf.contrib.layers.batch_norm(conv, is_training=is_train, epsilon=1e-5, decay=0.9,
                                      updates_collections=None, scope=name + "_bn")
    return leaky_relu(bn, name=name + "_act")
