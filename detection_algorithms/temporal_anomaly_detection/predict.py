"""
Load weights & predict
"""

import colorama
import cv2 as cv
import cvlib
import dlib
import numpy as np

from .model_def import gen_model

def predict(model, input_video_path, video_frames=10,
            video_size=(384, 512), print_fn=print, threshold=0.5,
            use_gpu_for_face=False):
    cap = cv.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    model_runs = total_frames // video_frames
    actual_model_runs = 0
    output_total = 0
    for current_run in range(model_runs):
        print_fn(f'[{current_run}/{model_runs}] Running the model... ', end='')
        chunk = np.empty((video_frames, *video_size, 3), 'float32')
        faces_in_chunk = 0
        for i in range(video_frames):
            _, current_frame = cap.read()
            faces = cvlib.detect_face(current_frame,
                                      enable_gpu=use_gpu_for_face)
            if faces[0]:
                faces_in_chunk += 1
            current_frame = cv.resize(current_frame, video_size[::-1])
            current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2RGB)
            current_frame = current_frame.astype('float32') / 255
            chunk[i] = current_frame
        if (faces_in_chunk / video_frames) < 0.5:
            print_fn(colorama.Fore.YELLOW, end='')
            print_fn('skipped', end='')
            print_fn(colorama.Style.RESET_ALL)
            continue
        actual_model_runs += 1
        chunk = np.expand_dims(chunk, axis=0)
        current_output = model.predict(chunk)[0][0]
        print_fn(current_output)
        output_total += current_output
    cap.release()
    average_output = output_total / actual_model_runs
    print_fn(f'{model_runs} model runs. Average model output: {average_output}')
    if average_output > threshold:
        verdict = f'{colorama.Fore.RED}FAKE'
    else:
        verdict = f'{colorama.Fore.GREEN}REAL'
    print_fn(f'Verdict: {verdict}{colorama.Style.RESET_ALL}')
    return verdict[:-4] == 'FAKE', average_output
