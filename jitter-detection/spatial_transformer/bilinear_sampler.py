"""
transformer.py
==============

Bilinear sampler layer.
"""

from tensorflow import keras
from .util import gen_sampling_grid, bilinear_sampler

import tensorflow as tf

class BilinearSampler(keras.layers.Layer):
    """
    Bilinear sampler.
    """
    def __init__(self, output_shape=None):
        super(BilinearSampler, self).__init__()
        self.out_shape = output_shape
        if self.out_shape is not None:
            self.output_height = self.out_shape[1]
            self.output_width = self.out_shape[2]

    def build(self, input_shape):
        image_shape = input_shape[1]
        batch_count = image_shape[0]
        self.input_height = image_shape[1]
        self.input_width = image_shape[2]
        if self.out_shape is None:
            self.output_height = image_shape[1]
            self.output_width = image_shape[2]

    def call(self, inputs):
        theta, image = inputs
        theta = tf.reshape(theta, (-1, 2, 3))
        batch_grid = gen_sampling_grid(self.input_height,
                                       self.input_width, theta)
        x_s = batch_grid[:,0,:,:]
        y_s = batch_grid[:,1,:,:]
        return bilinear_sampler(image, x_s, y_s)
