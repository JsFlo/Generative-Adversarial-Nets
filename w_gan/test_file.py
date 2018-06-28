import os
import numpy as np

import tensorflow as tf
import argparse

import discriminator
import generator
import trainer
import utils
import pandas as pd
import csv
import ast
import sys
from cat_trainer import Trainer
import itertools
from random import randint

parser = argparse.ArgumentParser()
# DIRECTORIES
parser.add_argument('--csv', type=str, required=True)
FLAGS = parser.parse_args()

def get_rows(filename, batch_size):
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        count = 0
        image_batch = []
        input_batch = []
        for row in datareader:
            input = np.array(row[2:20]).astype(np.float)
            image_label = np.array(ast.literal_eval(row[20]))

            image_batch.append(image_label)
            input_batch.append(input)
            count += 1

            if (count >= batch_size):
                yield input_batch, image_batch
                count = 0
                image_batch = []
                input_batch = []

counter = 0
for features, image_label in get_rows(FLAGS.csv, FLAGS.batch_size):
    counter +=1
    print(counter)
    print(len(features))