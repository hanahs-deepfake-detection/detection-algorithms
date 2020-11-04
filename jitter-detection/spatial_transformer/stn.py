"""
stn.py
======

Implementation of Spatial Transformer module[1] based on TFv1 version[2].

[1]: 'Spatial Transformer Networks', Jaderberg et. al,
     (https://arxiv.org/abs/1506.02025)
[2]: 'kevinzakka/spatial-transformer-network', Kevin Zakka,
     (https://github.com/kevinzakka/spatial-transformer-network)
"""

from tensorflow import keras
from .util import bilinear_sampler, gen_sampling_grid

import tensorflow as tf

class STN(keras.Model):
    """
    STN module implementation.
    """
    def __init__(self, output_shape=None):
        super(STN, self).__init__()
        self.out_shape = output_shape
        if self.out_shape is not None:
            self.output_height = self.out_shape[1]
            self.output_width = self.out_shape[2]

    def build(self, input_shape):
        batch_count = input_shape[0]
        if self.out_shape is None:
            self.output_height = input_shape[1]
            self.output_width = input_shape[2]
        self.conv1 = keras.layers.Conv2D(activation='relu', padding='same',
                                         filters=32, kernel_size=(5, 5))
        self.maxpool1 = keras.layers.MaxPool2D()
        self.conv2 = keras.layers.Conv2D(activation='relu', padding='same',
                                         filters=32, kernel_size=(5, 5))
        self.maxpool2 = keras.layers.MaxPool2D()
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(
            64, activation='tanh', kernel_initializer='zeros'
        )
        self.dropout = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(
            6, activation='tanh', kernel_initializer='zeros',
            bias_initializer=lambda shape, dtype=None: tf.constant(
                [1, 0, 0, 0, 1, 0], tf.float32
            )
        )

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0], self.output_height,
            self.output_width, input_shape[-1]
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)

        x = tf.reshape(x, (-1, 2, 3))
        batch_grid = gen_sampling_grid(self.output_height,
                                       self.output_width, x)
        x_s = batch_grid[:,0,:,:]
        y_s = batch_grid[:,1,:,:]
        return bilinear_sampler(inputs, x_s, y_s)
