import os
import cv2
from PIL import Image

input_src = "./data"
resized_dst = "./resized_data"
rgb_dst = "./resized_black/"


def preprocess_images():
    resize("ALL_CAT", "resized_data")
    convert_rgba_to_rgb("resized_data", "data_jpg")


def resize(src, dst, width=128, height=128):

    for listed_image in os.listdir(src):
        img = cv2.imread(os.path.join(src, listed_image))
        img = cv2.resize(img, (width, height))
        cv2.imwrite(os.path.join(dst, listed_image), img)


def convert_rgba_to_rgb(src, dst):
    for listed_image in os.listdir(src):
        png = Image.open(os.path.join(src, listed_image))

        if png.mode == 'RGBA':
            png.load()  # required for png.split()
            background = Image.new("RGB", png.size, (0, 0, 0))
            background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
            background.save(os.path.join(dst, listed_image.split('.')[0] + '.jpg'), 'JPEG')
        else:
            png.convert('RGB')
            png.save(os.path.join(dst, listed_image.split('.')[0] + '.jpg'), 'JPEG')

preprocess_images()