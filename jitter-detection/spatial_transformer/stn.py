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
    def __init__(self):
        super(STN, self).__init__()

    def build(self, input_shape):
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer='he_uniform')
        self.dropout = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(6, activation='relu',
                                         kernel_initializer='he_uniform')
        self.theta = self.add_weight(
            shape=(input_shape[0], 6), initializer='random_normal'
        )

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        mat = self.theta * tf.reshape(x, (-1, 6))

        batch_count = inputs.shape[0]
        input_height = inputs.shape[1]
        input_width = inputs.shape[2]
        mat = tf.reshape(mat, (batch_count, 2, 3))
        batch_grid = gen_sampling_grid(input_height, input_width, mat)
        x_s = batch_grid[:,0,:,:]
        y_s = batch_grid[:,1,:,:]
        return bilinear_sampler(inputs, x_s, y_s)
