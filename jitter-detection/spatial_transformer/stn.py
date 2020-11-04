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
        batch_count = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        self.flatten = keras.layers.Flatten()
        self.dropout1 = keras.layers.Dropout(rate=0.5)
        self.dropout2 = keras.layers.Dropout(rate=0.5)
        self.w_fc1 = self.add_weight(shape=(height * width, 64))
        self.b_fc1 = self.add_weight(shape=(64,))
        self.w_fc2 = self.add_weight(shape=(64, 6))
        self.b_fc2 = self.add_weight(
            shape=(6,),
            initializer=lambda input_shape, dtype=None: tf.constant(
                [1, 0, 0, 0, 1, 0], tf.float32
            )
        )

    def call(self, inputs):
        x = self.flatten(inputs)
        x = tf.matmul(x, self.w_fc1) + self.b_fc1
        x = tf.tanh(x)
        x = self.dropout1(x)
        x = tf.matmul(x, self.w_fc2) + self.b_fc2
        x = tf.tanh(x)
        x = self.dropout2(x)
        x = tf.reshape(x, (-1, 2, 3))

        input_height = inputs.shape[1]
        input_width = inputs.shape[2]
        batch_grid = gen_sampling_grid(input_height, input_width, x)
        x_s = batch_grid[:,0,:,:]
        y_s = batch_grid[:,1,:,:]
        return bilinear_sampler(inputs, x_s, y_s)
