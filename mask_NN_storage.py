
#*************************************************************************#
# this code is generated for generating the VGG neural network based on the
# classifed augmented datasets.
#*************************************************************************#

import json
import glob
from tqdm import tqdm
import numpy as np
import multiprocessing
import sys
import os
import math
import re
import time
import random
import natsort

import matplotlib.pyplot as plt

import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

#import tensorflow.keras as keras
#from tensorflow.keras.callbacks import TensorBoard

import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.regularizers import l1, l2
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, EarlyStopping

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
CROP_SAVE_DIR = os.path.join(ROOT_DIR, "croped_mask_from_img/aug_data")

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
#from distutils.version import LooseVersion
#assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
#assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

with open('row_col_msize.txt', 'r') as filehandle2:
    row_col_msize = json.load(filehandle2)

print(type(row_col_msize))
print(len(row_col_msize))
print(row_col_msize[0])
print(row_col_msize[1])

# generating batch data with different categories with image data generator keras
# type = 0, [1,0] = for std_up
# type = 1, [0,1] = for lay_down
CATEGORIES = ['std_up', 'lay_down']

img_generator = ImageDataGenerator(validation_split=0.3)
train_gen_data = img_generator.flow_from_directory(
    directory=str(CROP_SAVE_DIR), batch_size=28, 
    shuffle = False,
    target_size = (row_col_msize[0], row_col_msize[1]),
    classes = CATEGORIES,
    subset='training'
)

val_gen_data = img_generator.flow_from_directory(
    directory=str(CROP_SAVE_DIR), batch_size=28, 
    shuffle = False,
    target_size = (row_col_msize[0], row_col_msize[1]),
    classes = CATEGORIES,
    subset='validation'
)

model = KM.Sequential()

model.add(KL.Conv2D(filters = 64, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', input_shape = (row_col_msize[0], row_col_msize[1], 3)))
model.add(KL.Conv2D(filters = 64, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block0_conv2'))
model.add(KL.MaxPooling2D(pool_size=(2, 2)))

model.add(KL.Conv2D(filters = 128, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block1_conv1'))
model.add(KL.Conv2D(filters = 128, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block1_conv2'))
model.add(KL.MaxPooling2D(pool_size=(2, 2)))

model.add(KL.Conv2D(filters = 256, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block2_conv1'))
model.add(KL.Conv2D(filters = 256, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block2_conv2'))
model.add(KL.Conv2D(filters = 256, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block2_conv3'))
model.add(KL.MaxPooling2D(pool_size=(2, 2)))

model.add(KL.Conv2D(filters = 512, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block3_conv1'))
model.add(KL.Conv2D(filters = 512, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block3_conv2'))
model.add(KL.Conv2D(filters = 512, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block3_conv3'))
model.add(KL.MaxPooling2D(pool_size=(2, 2)))

model.add(KL.Conv2D(filters = 512, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block4_conv1'))
model.add(KL.Conv2D(filters = 512, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block4_conv2'))
model.add(KL.Conv2D(filters = 512, kernel_size = (3,3), strides = (1, 1),
padding='valid', activation='relu', name='block4_conv3'))
model.add(KL.MaxPooling2D( pool_size=(2, 2)))

model.add(KL.Flatten())
model.add(KL.Dense(4096, activation='relu'))
model.add(KL.Dense(4096, activation='relu'))
#model.add(KL.Dense(1, activation='softmax'))
model.add(KL.Dense(2, activation='sigmoid'))

#opt = SGD(lr=0.005)
opt = SGD(lr=0.0025)
model.compile(optimizer= opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_callback = EarlyStopping(monitor='val_accuracy', min_delta=1, patience=50)

print(model.summary())

NAME = "PIG_mask_CNN"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# running the model
history = model.fit_generator(train_gen_data, validation_data = val_gen_data,
epochs=100, shuffle=True, use_multiprocessing=True, callbacks = [tensorboard, early_callback])
model.save('VGGx2sig_Ver8.model')

# generate the matplotlib graph results

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'validation'], loc='upper left') 
plt.show()

