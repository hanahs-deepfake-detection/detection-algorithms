"""
Jitter Detection Model in Keras
"""

from sys import argv
from tensorflow import keras

import datetime
import numpy as np
import tensorflow as tf

FACE_SIZE = 224
FACE_SIZE = 224
resnet = keras.applications.ResNet50V2(include_top=False, weights=None,
                                       input_shape=(FACE_SIZE, FACE_SIZE, 3))
model = keras.Sequential([
    keras.Input(shape=(None, FACE_SIZE, FACE_SIZE, 3)),
    keras.layers.TimeDistributed(resnet),
    keras.layers.TimeDistributed(keras.layers.Flatten()),
    keras.layers.LSTM(64),
    keras.layers.Dense(1)
])
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())
model.summary()

input_file = np.load(argv[1])
inputs = input_file.f.arr_0
labels = np.load(argv[2])

epochs = 200
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)
tf.debugging.set_log_device_placement(True)

with tf.device('/GPU:0'):
    model.fit(inputs, labels, batch_size=5, epochs=epochs,
              validation_split=0.2, callbacks=[tensorboard_callback])
model.save('resnet_cnn_detector_model')
