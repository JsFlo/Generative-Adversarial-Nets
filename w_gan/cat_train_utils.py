import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf
import scipy.misc


def get_tensor_from_image(dir, filename, width=128, height=128, channels=3):
    current_file_path = os.path.join(dir, filename)
    converted_image = tf.convert_to_tensor(current_file_path, dtype=tf.string)
    # images_queue = tf.train.slice_input_producer([all_images])
    content = tf.read_file(converted_image)
    # TODO: make this handle jpeg and png
    image_tensor = tf.image.decode_jpeg(content, channels=3)
    image_tensor = tf.image.random_flip_left_right(image_tensor)
    image_tensor = tf.image.random_brightness(image_tensor, max_delta=0.1)
    image_tensor = tf.image.random_contrast(image_tensor, lower=0.9, upper=1.1)
    size = [width, height]
    image_tensor = tf.image.resize_images(image_tensor, size)
    image_tensor.set_shape([width, height, channels])

    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = image_tensor / 255.0
    return image_tensor


def get_data_rows(dir, dataFrame, idxStart=0, idxEnd=None):
    values = dataFrame.values[idxStart:idxEnd]
    filtered_values = []
    image_tensors = []
    for idx, row in enumerate(values):
        image_tensor = get_tensor_from_image(dir, str(row[1]))
        new_row = row[3:]
        filtered_values.append(new_row)
        image_tensors.append(image_tensor)
    return filtered_values, image_tensors