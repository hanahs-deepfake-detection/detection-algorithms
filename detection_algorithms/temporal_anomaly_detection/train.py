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

from ..common import dataset_util
from .data_pipeline import VideoPipeline
from .model_def import gen_model
from .spatial_transformer.bilinear_sampler import BilinearSampler

def main():
    dataset_dir = sys.argv[1]
    batch_size = int(sys.argv[2])
    video_frames = int(sys.argv[3])

    with tf.device(sys.argv[5]):
        model = gen_model(batch_size, video_frames)
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        model.summary()

        dataset = dataset_util.Dataset(dataset_dir, 'faceforensics')
        dataset_df = dataset.get_metadata_dataframe()
        filenames = dataset_df.index.to_list()
        labels = dataset_df['label'].to_dict()
        file_set = set(filenames)
        train_files = sample(filenames, int(sys.argv[4]))
        train_file_set = set(train_files)
        valid_file_set = file_set - train_file_set
        valid_files = list(valid_file_set)
        train_data_gen = VideoPipeline(train_files, labels, batch_size,
                                       video_frames)
        valid_data_gen = VideoPipeline(valid_files, labels, batch_size,
                                       video_frames)

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            'best_model', monitor='loss', save_weights_only=True, mode='auto',
            verbose=1
        )
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs')
        model.fit(train_data_gen, epochs=100, validation_data=valid_data_gen,
                  callbacks=[checkpoint_callback, tensorboard_callback])

if __name__ == '__main__':
    main()
