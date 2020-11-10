"""
Jitter Detection Model in Keras
"""

from random import sample
from tensorflow import keras
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, LSTM, MaxPool2D,
    TimeDistributed, Lambda
)
import cv2 as cv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

sys.path.append('../common/')
from data_pipeline import VideoPipeline
from spatial_transformer.bilinear_sampler import BilinearSampler
import dataset_util

dataset_dir = sys.argv[1]
batch_size = int(sys.argv[2])
video_frames = int(sys.argv[3]) 

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
x = TimeDistributed(BilinearSampler(input_shape=(batch_size, 384, 512, 3),
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
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

dataset = dataset_util.Dataset(dataset_dir, 'faceforensics')
dataset_df = dataset.get_metadata_dataframe()
filenames = dataset_df.index.to_list()
file_set = set(filenames)
train_files = sample(filenames, int(sys.argv[4]))
train_file_set = set(train_files)
valid_file_set = file_set - train_file_set
valid_files = list(valid_file_set)
train_data_gen = VideoPipeline(dataset_dir, train_files)
valid_data_gen = VideoPipeline(dataset_dir, valid_files)

checkpoint_callback = keras.callbacks.ModelCheckpoint('best_model',
                                                      monitor='loss',
                                                      save_weights_only=True,
                                                      mode='auto',
                                                      verbose=1)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs')
model.fit(train_data_gen, epochs=100, validation_data=valid_data_gen,
          callbacks=[checkpoint_callback, tensorboard_callback])
