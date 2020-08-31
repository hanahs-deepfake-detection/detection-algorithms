"""
dataset_util.py -- Deepfake dataset utility
"""

import os
import cv2 as cv
import dlib
import pandas as pd

class Dataset:
    """
    Dataset -- a class for accessing DFDC dataset
    """
    def __init__(self, dataset_root='../input/deepfake-detection-challenge'):
        """
        Initialize the Dataset class.

        Params
        ------
        dataset_root: str
            The root directory of DFDC dataset downloaded from Kaggle.
        """
        self.root = dataset_root
        self.train_data = os.path.join(self.root, 'train_sample_videos')
        self.test_data = os.path.join(self.root, 'test_videos')

    def get_metadata_dataframe(self):
        """
        Get pandas dataframe from dataset's `metadata.json`.
        """
        dataframe = pd.read_json(os.path.join(self.train_data, 'metadata.json'))
        return dataframe.T
    
    def get_video_path(self, filename: str, from_test_data=False):
        """
        Get path of a specified video file.

        Params
        ------
        filename: str
            The video file to get path.

        from_test_data: bool
            use video in test dataset if set. Default is False.
        """
        if from_test_data:
            return os.path.join(self.test_data, filename)
        else:
            return os.path.join(self.train_data, filename)

    def get_frame_from_video(self, filename: str, frame_no: int,
                             from_test_data=False):
        """
        Get specific frame from video file.

        Params
        ------
        filename: str
            The video file to get frame from.

        frame_no: int
            The specific frame index to get. IndexError is raised when frame_no
            contains invalid value.

        from_test_data: bool
            Use video in test dataset if set. Default is False.
        """
        cap = cv.VideoCapture(self.get_video_path(filename, from_test_data))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if frame_no < 0 or frame_no >= total_frames:
            raise IndexError(('Frame index %d out of range. '
                              + 'The video %s has %d frames.')
                             % (frame_no, filename, total_frames))
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_no)
        _, current_frame = cap.read()
        return current_frame

    def get_face_coords(self, filename: str, frame_no: int,
                        from_test_data=False,
                        landmark_dataset='./face_landmark_dataset.dat'):
        """
        Get face region from video file's specified frame.

        Params
        ------
        filename: str
            The video file to get frame from.

        frame_no: int
            The specific frame index to get.

        from_test_data: bool
            Use video in test dataset if set. Default is False.

        landmark_dataset: str
            The face landmark dataset used in `dlib.shape_predictor`. Default
            value assumes that the dataset file is in current directory.
        """
        frame = self.get_frame_from_video(filename, frame_no, from_test_data)
        dlib_detector = dlib.get_frontal_face_detector()
        # TODO: Use canonical path rather than current directory
        dlib_predictor = dlib.shape_predictor(landmark_dataset)
        # TODO: Support more than one face per frame
        face_rect = dlib_detector(frame, 1)[0]
        return face_rect
