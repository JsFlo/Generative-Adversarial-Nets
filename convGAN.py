import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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


# expects x to be of shape 28 x 28
def getCNN(x, keep_prob, weights, bias):
    # 3 hidden layers
    conv1 = getHiddenLayer(x, weights['wc1'], bias['bc1'])
    conv2 = getHiddenLayer(conv1, weights['wc2'], bias['bc2'])
    conv3 = getHiddenLayer(conv2, weights['wc3'], bias['bc3'])
    # flatten 3 to go into fully connected
    conv3_flattened = tf.reshape(conv3, [-1, bias['reshape']])
    # fully connected 1 with dropout
    fullyConnected1 = getFullyConnectedLayer(conv3_flattened, weights['wf1'], bias['bf1'])
    fullyConnected1_dropout = tf.nn.dropout(fullyConnected1, keep_prob)
    # fully connected 2
    fullyConnected2 = getFullyConnectedLayer(fullyConnected1_dropout, weights['wf2'], bias['bf2'])
    # fully connected 3 no relu applied
    return getFullyConnectedLayer(fullyConnected2, weights['out'], bias['bout'], False)


def main():
    # Load the data from the mnist
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # 28 x 28 mnist images = 784 row
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # reshape 784 back to 28 by 28
    # [? , width, height, # color channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 10 hot vectors (0 - 9)
    yCorrectLabels = tf.placeholder(tf.float32, shape=[None, 10])

    # used for dropout later, hold a ref so we can remove it during testing
    disc_keep_prob = tf.placeholder(tf.float32)
    yModel = getCNN(x_image, disc_keep_prob, disc_weights, disc_biases)
    # softmax and reduce mean
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yCorrectLabels, logits=yModel))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(yModel, 1), tf.argmax(yCorrectLabels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            batch = mnist.train.next_batch(50)
            if i % 25 == 0:
                printAccuracy(accuracy, i, x, yCorrectLabels, batch[0], batch[1], disc_keep_prob)

            train_step.run(feed_dict={x: batch[0], yCorrectLabels: batch[1], disc_keep_prob: 0.5})

        print(
            'test accuracy %g' % accuracy.eval(
                feed_dict={x: mnist.test.images, yCorrectLabels: mnist.test.labels, disc_keep_prob: 1.0}))


main()
