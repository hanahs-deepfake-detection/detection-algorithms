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
    def __init__(self, input_shape, output_shape):
        super(BilinearSampler, self).__init__()
        self.out_shape = output_shape
        self.output_height = self.out_shape[1]
        self.output_width = self.out_shape[2]
        self.in_shape = input_shape
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

    def call(self, inputs):
        theta = inputs[:,:6]
        image = tf.reshape(inputs[:,6:], self.in_shape)
        theta = tf.reshape(theta, (-1, 2, 3))
        batch_grid = gen_sampling_grid(self.input_height,
                                       self.input_width, theta)
        x_s = batch_grid[:,0,:,:]
        y_s = batch_grid[:,1,:,:]
        return bilinear_sampler(image, x_s, y_s)
