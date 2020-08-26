"""
"""
import sys
import tensorflow as tf
import numpy as np
from math import ceil
from itertools import product as product


class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['MODEL']['PRIORS']['MIN_SIZES']
        self.steps = cfg['MODEL']['PRIORS']['STEPS']
        self.clip = cfg['MODEL']['PRIORS']['CLIP']
        image_size = cfg['INPUT']['IMAGE_SIZE']
        self.image_size = (image_size, image_size)
        self.features = cfg['MODEL']['PRIORS']['FEATURES']
        # self.features = [40, 20, 10, 5]

        for ii in range(4):
            if(self.steps[ii] != pow(2,(ii+3))):
                print("steps must be [8,16,32,64]")
                sys.exit()

    def forward(self):
        anchors = []
        for k, f in enumerate(self.features):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f), repeat=2):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]

                    cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                    cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                    anchors += [cx, cy, s_kx, s_ky]

        output = tf.reshape(tf.constant(anchors), (-1, 4))
        if self.clip:
            output = tf.clip_by_value(output, 0, 1)
        return output
