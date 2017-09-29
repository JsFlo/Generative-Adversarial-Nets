import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# takes in a 28 x 28 x 1 and outputs a 1(real) or a 0(fake)
disc_weights = {  # 28 x 28 x 1
    'wc1': weight_variable([2, 2, 1, 32]),  # 14 x 14 x 32
    'wc2': weight_variable([3, 3, 32, 64]),  # 6 x 6 x 64
    'wc3': weight_variable([2, 2, 64, 128]),  # 3 x 3 x 128 (= 1152 after flatten)
    'wf1': weight_variable([3 * 3 * 128, 1024]),  # 1152 -> 1024
    'wf2': weight_variable([1024, 512]),  # 1024 -> 512
    'out': weight_variable([512, 10])  # 512 -> 1
}

disc_biases = {
    'bc1': bias_variable([32]),
    'bc2': bias_variable([64]),
    'bc3': weight_variable([128]),
    'reshape': (3 * 3 * 128),
    'bf1': weight_variable([1024]),
    'bf2': weight_variable([512]),
    'bout': weight_variable([10])
}

gen_weights = {  # takes in 100
    'wc1': weight_variable([2, 2, 1, 64]),  # 14 x 14 x 32
    'wc2': weight_variable([3, 3, 64, 96]),  # 6 x 6 x 64
    'wc3': weight_variable([2, 2, 96, 128]),  # 3 x 3 x 128 (= 1152 after flatten)
    'wf1': weight_variable([1 * 1 * 128, 1024]),  # 1152 -> 1024
    'wf2': weight_variable([1024, 900]),  # 1024 -> 512
    'out': weight_variable([900, 784])  # 512 -> 1
}

gen_biases = {
    'bc1': bias_variable([64]),
    'bc2': bias_variable([96]),
    'bc3': weight_variable([128]),
    'reshape': (1 * 1 * 128),
    'bf1': weight_variable([1024]),
    'bf2': weight_variable([900]),
    'bout': weight_variable([784])
}


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# pooling 2x2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Note this does not add zero-padding (padding = 'VALID') (padding = 'SAME' will output same dimension as lastLayer)
def getConvLayer(lastLayer, weight, bias, stride=1):
    conv1 = tf.nn.conv2d(lastLayer, weight, strides=[1, stride, stride, 1], padding='VALID')
    return tf.nn.relu(conv1 + bias)


def getFullyConnectedLayer(lastLayer, inputOutputWeight, bias, useRelu=True):
    mult = (tf.matmul(lastLayer, inputOutputWeight) + bias)
    if (useRelu):
        return tf.nn.relu(mult)
    else:
        return mult


# Creates a convolutional layer with a stride of 1
def getHiddenLayer(lastLayer, weight, bias):
    # the conv layer uses 'SAME" padding to preserve the input dimensions (it's zero-padded)
    convLayer = getConvLayer(lastLayer, weight, bias)
    # pool 2x2, cut it in half (ex. 28 x 28 => 14 x 14 => 7 x7 ...)
    return max_pool_2x2(convLayer)


# used for printing accuracy, sets the dropout to 1 (no droput)
def printAccuracy(accuracy, step, inputPlaceholder, correctLabelPlaceholder, inputs, correctLabels, keep_prob):
    train_accuracy = accuracy.eval(
        feed_dict={inputPlaceholder: inputs, correctLabelPlaceholder: correctLabels, keep_prob: 1.0})
    print('step %d, training accuracy %g' % (step, train_accuracy))


def printShape(tensor):
    print(tensor.shape)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


# expects x to be of shape 28 x 28
def getCNN(x, keep_prob, weights, bias):
    # 3 hidden layers
    print("3 hidden")
    conv1 = getHiddenLayer(x, weights['wc1'], bias['bc1'])
    printShape(conv1)
    conv2 = getHiddenLayer(conv1, weights['wc2'], bias['bc2'])
    printShape(conv2)
    conv3 = getHiddenLayer(conv2, weights['wc3'], bias['bc3'])
    printShape(conv3)
    # flatten 3 to go into fully connected
    conv3_flattened = tf.reshape(conv3, [-1, bias['reshape']])
    printShape(conv3_flattened)
    print("fully connected")
    # fully connected 1 with dropout
    fullyConnected1 = getFullyConnectedLayer(conv3_flattened, weights['wf1'], bias['bf1'])
    fullyConnected1_dropout = tf.nn.dropout(fullyConnected1, keep_prob)
    printShape(fullyConnected1)
    # fully connected 2
    fullyConnected2 = getFullyConnectedLayer(fullyConnected1_dropout, weights['wf2'], bias['bf2'])
    printShape(fullyConnected2)
    # fully connected 3 no relu applied
    fullyConnected3_logit = getFullyConnectedLayer(fullyConnected2, weights['out'], bias['bout'], False)
    fullyConnected3_prob = tf.nn.sigmoid(fullyConnected3_logit)
    printShape(fullyConnected3_logit)
    return fullyConnected3_prob, fullyConnected3_logit


def main():
    # used for dropout later, hold a ref so we can remove it during testing
    disc_real_keep_prob = tf.placeholder(tf.float32)
    disc_fake_keep_prob = tf.placeholder(tf.float32)
    gen_keep_prob = tf.placeholder(tf.float32)

    # Load the data from the mnist
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # 28 x 28 mnist images = 784 row
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # reshape 784 back to 28 by 28
    # [? , width, height, # color channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # input to the generator of 100 numbers of noise
    Z = tf.placeholder(tf.float32, shape=[None, 100])
    Z_reshaped = tf.reshape(Z, [-1, 10, 10, 1])
    # generator will take in Z (random noise of 100) and output an image that's 28 x 28
    G_sample, _ = getCNN(Z_reshaped, gen_keep_prob, gen_weights, gen_biases)
    # this discriminator will take in the real images
    D_real, D_logit_real = getCNN(x_image, disc_real_keep_prob, disc_weights, disc_biases)
    # discriminator will take in the fake images the generator generates
    G_sample_reshaped = tf.reshape(G_sample, [-1, 28, 28, 1])
    D_fake, D_logit_fake = getCNN(G_sample_reshaped, disc_fake_keep_prob, disc_weights, disc_biases)

    D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    G_loss = -tf.reduce_mean(tf.log(D_fake))

    # Only update D(X)'s parameters, so var_list = theta_D
    theta_D = [disc_weights['wc1'], disc_weights['wc2'], disc_weights['wc3'],
               disc_weights['wf1'], disc_weights['wf2'], disc_weights['out'],
               disc_biases['bc1'], disc_biases['bc2'], disc_biases['bc3'],
               disc_biases['bf1'], disc_biases['bf2'], disc_biases['bout']]

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)

    # Only update G(X)'s parameters, so var_list = theta_G
    theta_G = [gen_weights['wc1'], gen_weights['wc2'], gen_weights['wc3'],
               gen_weights['wf1'], gen_weights['wf2'], gen_weights['out'],
               gen_biases['bc1'], gen_biases['bc2'], gen_biases['bc3'],
               gen_biases['bf1'], gen_biases['bf2'], gen_biases['bout']]

    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    mb_size = 128
    Z_dim = 100

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0

    for it in range(100):
        if it % 10 == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim), disc_fake_keep_prob: 1.0,
                                             disc_real_keep_prob: 1.0, gen_keep_prob: 1.0})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_mb, _ = mnist.train.next_batch(mb_size)

        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={x: X_mb, Z: sample_Z(mb_size, Z_dim), disc_fake_keep_prob: 1.0,
                                             disc_real_keep_prob: 1.0, gen_keep_prob: 1.0})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim), disc_fake_keep_prob: 1.0,
                                                                 disc_real_keep_prob: 1.0, gen_keep_prob: 1.0})

        if it % 10 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
    print()


main()
