import numpy as np
import scipy.misc

# 0 = stack images on top of each other, 1 = stack them side to side
AXIS_STACK_BOTTOM = 0
AXIS_STACK_RIGHT = 1


def save_images(images, image_path, row_limit=8):
    inverse_images = (images + 1.) / 2.
    concat_output = concat_images(inverse_images, row_limit=row_limit)
    image = np.squeeze(concat_output)
    return scipy.misc.imsave(image_path, image, "JPEG")


def concat_images(images, row_limit=8):
    for i, image in enumerate(images):
        if ((i % row_limit) == 0):
            if (i != 0):
                if (i == row_limit):
                    rows = row
                else:
                    rows = np.concatenate((rows, row), axis=AXIS_STACK_BOTTOM)

            # first time row is just single image
            row = image
        else:
            # stack them on the right
            row = np.concatenate((row, image), axis=AXIS_STACK_RIGHT)

    rows = np.concatenate((rows, row), axis=AXIS_STACK_BOTTOM)
    return rows
