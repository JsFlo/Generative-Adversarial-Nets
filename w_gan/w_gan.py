import os
import numpy as np

import tensorflow as tf
import argparse

import discriminator
import generator
import trainer
import utils

VERSION = 'DEBUG'
DEFAULT_TRAIN_NUM_EPOCHS = 1000
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_PRINT_FREQ = 50
DEFAULT_SAVE_MODEL_FREQ = 500
DEFAULT_SAVE_IMAGES_FREQ = 100

parser = argparse.ArgumentParser()
# DIRECTORIES
parser.add_argument('--data_dir', type=str, required=True,
                    help='Path to images.')
parser.add_argument('--out_dir', type=str, default='w_gan_out/',
                    help='Directory to where images and models will be saved')

# MODEL PARAMS
parser.add_argument('--num_epochs', type=int, default=DEFAULT_TRAIN_NUM_EPOCHS)
parser.add_argument('--batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
parser.add_argument('--model_version', type=str, default=VERSION)
parser.add_argument('--print_freq', type=int, default=DEFAULT_PRINT_FREQ)
parser.add_argument('--save_model_freq', type=int, default=DEFAULT_SAVE_MODEL_FREQ)
parser.add_argument('--save_images_freq', type=int, default=DEFAULT_SAVE_IMAGES_FREQ)
FLAGS = parser.parse_args()

OUTPUT_MODEL_PATH = FLAGS.out_dir + 'model/' + FLAGS.model_version
OUTPUT_IMAGES_PATH = FLAGS.out_dir + 'images/' + FLAGS.model_version
OUTPUT_IMAGE_COLUMNS = 8

TRAIN_DISC_PER_BATCH = 5
TRAIN_GEN_PER_BATCH = 1

GEN_INPUT_DIM = 100
HEIGHT, WIDTH, CHANNEL = 128, 128, 3


def create_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_train_noise():
    return np.random.uniform(-1.0, 1.0, size=[FLAGS.batch_size, GEN_INPUT_DIM]).astype(np.float32)


def save_model(saver, sess, epoch_counter):
    create_dirs_if_not_exists(OUTPUT_MODEL_PATH)
    fileName = OUTPUT_MODEL_PATH + '/model' + str(epoch_counter) + '.chkpt'
    saver.save(sess, fileName)


def save_images(sess, fake_image, random_input, is_train, epoch_counter):
    create_dirs_if_not_exists(OUTPUT_IMAGES_PATH)
    sample_noise = get_train_noise()
    imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
    fileName = OUTPUT_IMAGES_PATH + '/epoch' + str(epoch_counter) + '.jpg'
    utils.save_images(imgtest, fileName, OUTPUT_IMAGE_COLUMNS)


def train():
    # PLACEHOLDERS
    real_image = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_image')
    random_input = tf.placeholder(tf.float32, shape=[None, GEN_INPUT_DIM], name='rand_input')
    is_train = tf.placeholder(tf.bool, name='is_train')

    # GENERATOR
    fake_image = generator.get_generator(random_input, GEN_INPUT_DIM, output_dim=3, is_train=is_train)

    # DISCRIMINATORS
    real_result = discriminator.get_discriminator(real_image, is_train)
    fake_result = discriminator.get_discriminator(fake_image, is_train, reuse=True)

    # LOSS
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.

    # GET VARIABLES INTO DIS_VARS AND GEN_VARS
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]

    # OPTIMIZER (RMSProp = some gradient decent)
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    image_batch, samples_num = trainer.get_training_image_batch(HEIGHT, WIDTH, FLAGS.batch_size,
                                                                data_dir=FLAGS.data_dir)
    batch_num = int(samples_num / FLAGS.batch_size)

    # sess init
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # quick save
    save_path = saver.save(sess, "/tmp/model.ckpt")
    ckpt = tf.train.latest_checkpoint('./model/' + VERSION)
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('Training samples:{}'.format(samples_num))
    for epoch_counter in range(FLAGS.num_epochs + 1):
        print("epoch: {}".format(epoch_counter))

        # BATCH TRAINING
        for batch_counter in range(batch_num):
            # print("batch: {}".format(batch_counter))

            # GET INPUT (100 RANDOM NOISE)
            train_noise = get_train_noise()

            # TRAIN DISCRIMINATOR
            for _ in range(TRAIN_DISC_PER_BATCH):
                train_image = sess.run(image_batch)
                sess.run(d_clip)

                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})
            # TRAIN GENERATOR
            for _ in range(TRAIN_GEN_PER_BATCH):
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})

            # print debug stuff every now and then
            if (epoch_counter % FLAGS.print_freq == 0):
                print('train:[%d],d_loss:%f,g_loss:%f' % (epoch_counter, dLoss, gLoss))

        # save images every now and then
        if epoch_counter % FLAGS.save_images_freq == 0:
            save_images(sess, fake_image, random_input, is_train, epoch_counter)

        # save model every now and then
        if epoch_counter % FLAGS.save_model_freq == 0:
            save_model(saver, sess, epoch_counter)

    coord.request_stop()
    coord.join(threads)


train()
