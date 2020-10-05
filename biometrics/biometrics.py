import dataset_util
import os
import cv2
import dlib
from keras.models import Sequential
from keras.layers import Dense, LSTM


def make_sq(inx):
    res=[]
    prog=0
    for i in range(2): #3
        temp=[]
        for j in range(3): #100
            prog+=1
            print(prog)
            coord=dataset.get_face_coords(inx,(i*10)+j+1, from_test_data=True)
            temp.append(dataset.crop_frame(inx,(i*10)+j+1,coord, from_test_data=True))
        res.append(temp)
    return res


FILE_PATH="C:/Users/iowab/Documents/GitHub/detection-algorithms/input/deepfake-detection-challenge"

dataset=dataset_util.Dataset(FILE_PATH)

file_list = os.listdir(FILE_PATH+"/test_videos")

vid_data=make_sq(file_list[1])

for sq in vid_data:
    
