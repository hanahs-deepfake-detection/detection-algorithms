from random import shuffle
from tensorflow import keras

import cv2 as cv
import numpy as np
import sys

class VideoPipeline(keras.utils.Sequence):
    def __init__(self, filenames, labels, batch_size, video_frames,
                 shuffle_data=True, video_height=384, video_width=512):
        self.batch_size = batch_size
        self.video_frames = video_frames
        self.video_height = video_height
        self.video_width = video_width
        self.labels = labels
        self.filenames = filenames
        self.chunkspec = []
        for filename in self.filenames:
            cap = cv.VideoCapture(filename)
            frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            chunk_count = frame_count // self.video_frames
            self.chunkspec.extend([(filename, x * self.video_frames)
                                   for x in range(chunk_count)])
            cap.release()
        self.total_batches = len(self.chunkspec) // \
                             self.batch_size * self.batch_size
        if shuffle_data:
            shuffle(self.chunkspec)

    def __getitem__(self, index):
        batch = np.empty((self.batch_size, self.video_frames,
                          self.video_height, self.video_width, 3),
                         np.dtype('float32'))
        labels = np.empty((self.batch_size,), np.dtype('float32'))
        for i, (filename, start_frame) in enumerate(
            self.chunkspec[index:index + self.batch_size]):
            cap = cv.VideoCapture(filename)
            cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
            video = np.empty(batch.shape[1:], 'uint8')
            for frame in range(self.video_frames):
                _, current_frame = cap.read()
                current_frame = cv.resize(
                    current_frame, (self.video_width, self.video_height)
                )
                current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2RGB)
                video[frame] = current_frame
            cap.release()
            batch[i] = video.astype('float32') / 255
            labels[i] = float(self.labels[filename] == 'REAL')
        return batch, labels

    def __len__(self):
        return self.total_batches
