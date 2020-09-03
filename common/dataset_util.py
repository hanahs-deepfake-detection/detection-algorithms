"""
dataset_util.py -- Deepfake dataset utility
"""

import os
from typing import Tuple

import cv2 as cv
import dlib
import numpy as np
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

    def get_metadata_dataframe(self) -> pd.DataFrame:
        """
        Get pandas dataframe from dataset's `metadata.json`.
        """
        dataframe = pd.read_json(os.path.join(self.train_data, 'metadata.json'))
        return dataframe.T

    def get_video_path(self, filename: str, from_test_data=False) -> str:
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
                             from_test_data=False) -> np.ndarray:
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
                        landmark_dataset='./face_landmark_dataset.dat') -> Tuple[
                            Tuple[int, int],
                            Tuple[int, int]
                        ]:
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

        Return Type
        -----------
        ((y1: int, x1: int), (y2: int, x2: int))
            The detected face region where (y1, x1) is the top-left corner of
            the rectangle, and (y2, x2) is the bottom-right one.

        Exceptions
        ----------
        Raises ValueError if face is not detected from the given frame.
        """
        frame = self.get_frame_from_video(filename, frame_no, from_test_data)
        dlib_detector = dlib.get_frontal_face_detector()
        # TODO: Use canonical path rather than current directory
        detection_result = dlib_detector(frame, 1)
        # TODO: Support more than one face per frame
        if not detection_result:
            raise ValueError('Cannot detect face from given frame')
        rect = detection_result[0]
        converted_rect = ((rect.top(), rect.left()), (rect.bottom(),rect.right()))
        return converted_rect

    def crop_frame(self, filename: str, frame_no: int,
                   region: Tuple[Tuple[int, int], Tuple[int, int]],
                   from_test_data=False) -> np.ndarray:
        """
        Crop specified region from video file's specified frame.

        Params
        ------
        filename: str
            The video file to get frame and crop from.

        frame_no: int
            The specific frame index to get.

        from_test_data: bool
            Use video in test dataset if set. Default is False.

        region: ((y1: int, x1: int), (y2: int, x2: int))
            The region to crop. (y1, x1) is the top-left corner of the
            rectangle, and (y2, x2) is the bottom-right one.
        """
        frame = self.get_frame_from_video(filename, frame, from_test_data)
        return frame[region[0][0]:region[1][0], region[0][1]:region[1][1]]
