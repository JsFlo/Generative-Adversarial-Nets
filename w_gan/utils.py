import numpy as np
import scipy.misc


def print_shape(tensor):
    print(tensor.shape)


def save_images(images, stack_axis, image_path):
    inverse_images = (images + 1.) / 2.
    concat_output = concat_images(inverse_images, stack_axis)
    image = np.squeeze(concat_output)
    return scipy.misc.imsave(image_path, image, "JPEG")


def concat_images(images, stack_axis):
    for id, image in enumerate(images):
        if (id == 0):
            all_images = image
        else:
            all_images = np.concatenate((all_images, image), axis=stack_axis)
    return all_images
