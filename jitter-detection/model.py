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
from spatial_transformer.bilinear_sampler import BilinearSampler
import dataset_util

dataset_dir = '/run/media/curling_grad/LACIE/faceforensics-dataset'
batch_size = 1
video_frames = 10
class VideoSequence(keras.utils.Sequence):
    def __init__(self, filename_list):
        self.batch_size = batch_size 
        self.video_frames = video_frames
        self.video_height = 768
        self.video_width = 1024
        self.dataset = dataset_util.Dataset(dataset_dir, 'faceforensics')
        self.dataset_df = self.dataset.get_metadata_dataframe()
        self.filename_list = filename_list

    def __getitem__(self, index):
        filenames = self.filename_list[index * self.batch_size
                                      :(index + 1) * self.batch_size]
        batch = np.empty((self.batch_size, self.video_frames,
                          self.video_height, self.video_width, 3),
                         np.dtype('uint8'))
        labels = np.empty((self.batch_size,), np.dtype('float32'))
        for i, filename in enumerate(filenames):
            cap = cv.VideoCapture(filename)
            frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            assert frame_count >= self.video_frames
            video = np.empty(batch.shape[1:], np.dtype('uint8'))
            for frame in range(self.video_frames):
                _, current_frame = cap.read()
                current_frame = cv.resize(current_frame, (1024, 768))
                current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2RGB)
                current_frame = current_frame.astype(np.dtype('float32'))
                current_frame /= 255.0
                video[frame] = current_frame
            cap.release()
            batch[i] = video
            labels[i] = float(self.dataset_df.loc[filename]['label'] == 'FAKE')
        return batch, labels

    def __len__(self):
        return len(self.filename_list)

inputs = keras.Input((video_frames, 768, 1024, 3), batch_size=batch_size)
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
x = TimeDistributed(BilinearSampler(input_shape=(batch_size, 768, 1024, 3),
                    output_shape=(batch_size, 224, 224, 3)))(x)
resnet = ResNet101V2(include_top=False, weights=None)
x = TimeDistributed(resnet)(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(32, return_sequences=True)(x)
x = LSTM(32)(x)
x = Dense(10, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(1, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=x)
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

dataset = dataset_util.Dataset(dataset_dir, 'faceforensics')
dataset_df = dataset.get_metadata_dataframe()
filenames = dataset_df.index.to_list()
file_set = set(filenames)
train_files = sample(filenames, 140)
train_file_set = set(train_files)
valid_file_set = file_set - train_file_set
valid_files = list(valid_file_set)
train_data_gen = VideoSequence(train_files)
valid_data_gen = VideoSequence(valid_files)

checkpoint_callback = keras.callbacks.ModelCheckpoint('best_model.hdf5',
                                                      monitor='loss',
                                                      save_best_only=True,
                                                      mode='auto',
                                                      verbose=1)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs')
model.fit(train_data_gen, epochs=100, validation_data=valid_data_gen,
          callbacks=[checkpoint_callback, tensorboard_callback])
