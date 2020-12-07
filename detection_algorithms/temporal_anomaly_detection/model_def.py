"""
Model Definition
"""

from tensorflow import keras
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, LSTM, MaxPool2D,
    TimeDistributed, Lambda
)

import tensorflow as tf
from . import spatial_transformer

def gen_model(batch_size, video_frames):
    inputs = keras.Input((video_frames, 384, 512, 3), batch_size=batch_size)
    x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu'))(inputs)
    x = TimeDistributed(MaxPool2D())(x)
    x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu'))(x)
    x = TimeDistributed(MaxPool2D())(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(64, activation='tanh', kernel_initializer='zeros'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(6, activation='tanh', kernel_initializer='zeros',
                        bias_initializer=lambda shape, dtype=None: tf.constant(
                            [1, 0, 0, 0, 1, 0], tf.float32
                        )))(x)
    x = Lambda(lambda ls: tf.concat([ls[0], tf.reshape(ls[1],
               (batch_size, video_frames, -1))], -1))([x, inputs])
    x = TimeDistributed(spatial_transformer.BilinearSampler(input_shape=(batch_size, 384, 512, 3),
                        output_shape=(batch_size, 224, 224, 3)))(x)
    resnet = ResNet101V2(include_top=False, weights=None)
    x = TimeDistributed(resnet)(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(32, return_sequences=True)(x)
    x = LSTM(32)(x)
    x = Dense(10, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model
