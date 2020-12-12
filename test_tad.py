"""
Test Temporal Anomaly Detection model.
"""

import argparse
import numpy as np

from detection_algorithms.common.dataset_util import Dataset
from detection_algorithms.temporal_anomaly_detection.model_def import gen_model
from detection_algorithms.temporal_anomaly_detection.predict import predict

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', action='store')
parser.add_argument('model_file', action='store')
parser.add_argument('-f', '--frame-count', action='store', default=10)
parser.add_argument('--gpu-for-face', action='store_true')
args = parser.parse_args()
print(args)

model_file = args.model_file
dataset_dir = args.dataset_dir
frame_count = int(args.frame_count)
use_gpu = args.gpu_for_face

dataset = Dataset(dataset_dir, 'faceforensics')
df = dataset.get_metadata_dataframe()
files = df.index.to_list()
file_count = len(files)
labels = df['label'].to_dict()

model = gen_model(1, frame_count)
model.load_weights(model_file).expect_partial()

model_runs = 0
confusion_matrix = np.zeros((2, 2), 'int32')
try:
    for file in files:
        print(f'[{model_runs}] Running model for file {file}')
        verdict, _ = predict(model, file, video_frames=frame_count,
                             use_gpu_for_face=use_gpu)
        ground_truth = int(labels[file] == 'FAKE')
        confusion_matrix[ground_truth][int(verdict)] += 1
except KeyboardInterrupt:
    print(confusion_matrix)

print(confusion_matrix)
