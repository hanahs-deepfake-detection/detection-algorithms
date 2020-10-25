"""
face_extractor.py -- extract faces and save them to npz file
"""

import colorama
import cv2 as cv
import dlib
import glob
import numpy as np
import os
import pandas as pd

from sys import argv
from dataset_util import Dataset

FACE_SIZE = 224
INFO = f'{colorama.Fore.BLUE}INFO{colorama.Style.RESET_ALL}:'
WARN = f'{colorama.Fore.YELLOW}WARN{colorama.Style.RESET_ALL}:'
ERR = f'{colorama.Fore.RED}ERR{colorama.Style.RESET_ALL}:'

def extract_face_video(dataset, file_path):
    face_extracted_video = None
    skipped = 0
    last_frame = None
    success = False
    for i in range(20):
        try:
            face_coord = dataset.get_face_coords(file_path, i)
            frame = dataset.crop_frame(file_path, i, face_coord)
            last_frame = cv.resize(frame, (FACE_SIZE, FACE_SIZE))
            for _ in range(skipped + 1):
                if face_extracted_video is None:
                    face_extracted_video = np.expand_dims(last_frame, axis=0)
                else:
                    face_extracted_video = np.append(
                        face_extracted_video,
                        np.expand_dims(last_frame, axis=0),
                        axis=0
                    )
            skipped = 0
            success = True
        except ValueError:
            if last_frame is not None:
                face_extracted_video = np.append(
                    face_extracted_video,
                    np.expand_dims(last_frame, axis=0), axis=0)
            else:
                skipped += 1
        print('.', end='', flush=True)
    print()
    return success, face_extracted_video

if __name__ == '__main__':
    colorama.init()
    dataset = Dataset(argv[1], 'faceforensics')
    df = dataset.get_metadata_dataframe()
    inputs = None
    labels = np.array([])
    filenames = df.index.to_list()
    for i, filename in enumerate(filenames):
        print(f'{INFO} Processing video {filename}', end='')
        tempfile_path = os.path.join(argv[2], f'temp_{i}.npy')

        if os.path.exists(tempfile_path):
            print()
            print(f'{INFO} Found temp file for this video')
            extracted_face_frame = np.load(tempfile_path)
        else:
            success, extracted_face_frame = extract_face_video(dataset,
                                                               filename)
            if not success:
                print(f'{WARN} Could not detect face in first 20 frames.',
                      'Skipping this video...')
                continue
            np.save(tempfile_path, extracted_face_frame)

        if inputs is None:
            inputs = np.expand_dims(extracted_face_frame, axis=0)
        else:
            inputs = np.append(inputs,
                               np.expand_dims(extracted_face_frame, axis=0),
                               axis=0)
        labels = np.append(labels,
                           [float(df.loc[filename]['label'] == 'FAKE')])

    print(f'{INFO} Saving numpy arrays...')
    np.save(os.path.join(argv[2], 'labels.npy'), labels)
    np.savez_compressed(os.path.join(argv[2], 'inputs.npz'), inputs)

    print(f'{INFO} Removing temporary files...')
    file_list = glob.glob(os.path.join(argv[2], 'temp_*.npy'))
    for file_path in file_list:
        try:
            os.remove(file_path)
        except OSError:
            print(f'{ERR} Error while deleting file {file_path}')
