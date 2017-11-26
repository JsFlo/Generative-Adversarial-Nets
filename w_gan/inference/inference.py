import os
import numpy as np

import tensorflow as tf
import generator
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./models/1000',
                    help='Path to model checkpoint.')
parser.add_argument('--output_path', type=str, default='./out/images/',
                    help='Path to model checkpoint.')
parser.add_argument('--output_image_columns', type=int, default=4)
parser.add_argument('--output_num_images', type=int, default=16)
FLAGS = parser.parse_args()

GEN_INPUT_DIM = 100


def create_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_train_noise():
    return np.random.uniform(-1.0, 1.0, size=[FLAGS.output_num_images, GEN_INPUT_DIM]).astype(np.float32)


def get_counter():
    count = len([name for name in os.listdir(FLAGS.output_path)])
    print("count: {}".format(count))
    return count


def inf():
    create_dirs_if_not_exists(FLAGS.output_path)
    random_input = tf.placeholder(np.float32, shape=[None, GEN_INPUT_DIM], name='rand_input')
    is_train = tf.placeholder(tf.bool, name='is_train')

    # GENERATOR
    fake_image = generator.get_generator(random_input, GEN_INPUT_DIM, output_dim=3, is_train=is_train)

    # sess init
    sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess, FLAGS.model_path)

    sample_noise = get_train_noise()
    imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
    fileName = FLAGS.output_path + '/' + str(get_counter()) + '.jpg'
    utils.save_images(imgtest, fileName, FLAGS.output_image_columns)


inf()
