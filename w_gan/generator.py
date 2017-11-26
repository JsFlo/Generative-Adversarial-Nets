import tensorflow as tf

import utils


def get_generator(input, input_dim, output_dim, is_train, reuse=False):
    # name used later to pull all variables belonging to the generator
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        # input is 100 random numbers
        flat_conv1 = _block_1_get_fc(input, input_dim, 4 * 4 * 512)
        # utils.print_shape(flat_conv1)  # (?, 8192)
        activated_conv1 = _block_2_conv1(is_train, flat_conv1, [-1, 4, 4, 512])
        # utils.print_shape(activated_conv1)  # (?, 4, 4, 512)
        activated_conv2 = _conv_bn_relu(is_train, "conv2", activated_conv1, 256)
        # utils.print_shape(activated_conv2)  # (?, 8, 8, 256)
        activated_conv3 = _conv_bn_relu(is_train, "conv3", activated_conv2, 128)
        # utils.print_shape(activated_conv3)  # (?, 16, 16, 128)
        activated_conv4 = _conv_bn_relu(is_train, "conv4", activated_conv3, 64)
        # utils.print_shape(activated_conv4)  # (?, 32, 32, 64)
        activated_conv5 = _conv_bn_relu(is_train, "conv5", activated_conv4, 32)
        # utils.print_shape(activated_conv5)  # (?, 64, 64, 32)
        conv6 = tf.layers.conv2d_transpose(activated_conv5, output_dim, kernel_size=[5, 5], strides=[2, 2],
                                           padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        # utils.print_shape(conv6)  # (? 128, 128, 3)
        return tf.nn.tanh(conv6, name='act6')


def _block_1_get_fc(input, input_dim, flat_size):
    w1 = tf.get_variable('w1', shape=[input_dim, flat_size], dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.02))

    b1 = tf.get_variable('b1', shape=[flat_size], dtype=tf.float32,
                         initializer=tf.constant_initializer(0.0))

    return tf.add(tf.matmul(input, w1), b1, name='flat_conv1')


def _block_2_conv1(is_train, flat_conv, reshape_shape):
    conv1 = tf.reshape(flat_conv, shape=reshape_shape, name='conv1')
    bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay=0.9,
                                       updates_collections=None, scope='bn1')
    return tf.nn.relu(bn1, name='act1')


def _conv_bn_relu(is_train, name, last_layer, num_filters, filter_size=[5, 5], strides=[2, 2]):
    conv = tf.layers.conv2d_transpose(last_layer, num_filters, kernel_size=filter_size, strides=strides,
                                      padding="SAME",
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                      name=name + "_conv")
    bn = tf.contrib.layers.batch_norm(conv, is_training=is_train, epsilon=1e-5, decay=0.9,
                                      updates_collections=None, scope=name + "_bn")
    return tf.nn.relu(bn, name=name + "_act")
