import cv2 as cv
import numpy as np
import sys
sys.path.append('../common')
import dataset_util

class VideoPipeline(keras.utils.Sequence):
    def __init__(self, dataset_dir, filename_list):
        self.batch_size = batch_size 
        self.video_frames = video_frames
        self.video_height = 384
        self.video_width = 512
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
                current_frame = cv.resize(
                    current_frame,
                    (self.video_width, self.video_height)
                )
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
