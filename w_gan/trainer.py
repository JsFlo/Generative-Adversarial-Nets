import os
import tensorflow as tf

DEFAULT_DIR_NAME = 'data'


def _get_default_dir():
    # get current working dir
    current_dir = os.getcwd()
    # get the path to the 'data' directory
    return os.path.join(current_dir, DEFAULT_DIR_NAME)


def get_training_image_batch(height, width, batch_size,
                             image_channels=3, data_dir=_get_default_dir()):
    # get an array of image paths
    images = []
    for each in os.listdir(data_dir):
        images.append(os.path.join(data_dir, each))
    # print images
    all_images = tf.convert_to_tensor(images, dtype=tf.string)

    images_queue = tf.train.slice_input_producer([all_images])

    content = tf.read_file(images_queue[0])
    # TODO: make this handle jpeg and png
    image = tf.image.decode_jpeg(content, channels=image_channels)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    size = [height, width]
    image = tf.image.resize_images(image, size)
    image.set_shape([height, width, image_channels])

    image = tf.cast(image, tf.float32)
    image = image / 255.0

    images_batch = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=200 + 3 * batch_size,
        min_after_dequeue=200)
    num_images = len(images)

    return images_batch, num_images
