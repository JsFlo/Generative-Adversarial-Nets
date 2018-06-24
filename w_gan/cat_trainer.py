import numpy as np
import cat_train_utils as cat_utils
import random


class Trainer:
    def __init__(self, dir, dataFrame, totalBatch=9996):
        self.dir = dir
        self.dataFrame = dataFrame
        self.totalBatch = totalBatch

    def iterate_minibatches(self, batchsize):
        print("iterate mini")
        for start_idx in range(0, self.totalBatch - batchsize + 1, batchsize):
            print("start idx: %d, end: %d, steps: %d" % (start_idx, self.totalBatch - batchsize + 1, batchsize))
            rows, images = cat_utils.get_data_rows(self.dir, self.dataFrame, start_idx, start_idx + batchsize)
            yield rows, images

    def get_random(self, batchsize):
        last_val = self.totalBatch - batchsize + 1
        start_idx = random.randint(0, last_val)
        rows, images = cat_utils.get_data_rows(self.dir, self.dataFrame, start_idx, start_idx + batchsize)
        return
