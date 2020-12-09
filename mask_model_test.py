
#*************************************************************************#
# this code is generated for testing the neural network model based on 
# the validation sets or test data sets and seek their accuracy
#*************************************************************************#


import tensorflow as tf
import json
import os
import sys
import numpy as np
import natsort

from PIL import Image
import matplotlib.pyplot as plt

import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.regularizers import l1, l2
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

CTG_UP_DIR = os.path.join(ROOT_DIR, "samples/pig/croped_mask_from_img/test_data/ctg_std_up")
CTG_DW_DIR = os.path.join(ROOT_DIR, "samples/pig/croped_mask_from_img/test_data/ctg_lay_down")
CTG_ERROR = os.path.join(ROOT_DIR, "samples/pig/croped_mask_from_img/test_data")

#OPEN_DIR = os.path.join(ROOT_DIR, "samples\pig\croped_mask_from_img\model_data\model_test_data")
#OPEN_DIR = os.path.join(ROOT_DIR, "samples\pig\croped_mask_from_img\model_val_data")
#OPEN_F_DIR = os.path.join(ROOT_DIR, "samples\pig\croped_mask_from_img\model_data\model_test_false_data")
#OPEN_F_DIR = os.path.join(ROOT_DIR, "samples\pig\croped_mask_from_img\model_val_false_data")

# function generating the data augmentation 
def data_aug(mask_img):
    exp_mask_img = np.expand_dims(mask_img, 0)
    gen_data = ImageDataGenerator(rotation_range = 90)
    iterator = gen_data.flow(exp_mask_img, batch_size = 1)
    return iterator
    
    for i in range(9):
        #plt.subplot(330+1+i)
        batch = iterator.next()
        image = batch[0].astype('uint8')
        #plt.imshow(image)
    #plt.show()

with open('row_col_msize.txt', 'r') as filehandle:
    row_col_msize = json.load(filehandle)

# type 0 = [1,0], standing up
# type 1 = [0,1], laying down
CATEGORIES = ['ctg_std_up', 'ctg_lay_down']

# Reading the output ground truth of test data
#output = np.loadtxt('model_test_output.txt')
#output = np.loadtxt('model_val_output.txt')

# Reading the max size of height and width
with open('row_col_msize.txt', 'r') as filehandle:
    row_col_msize = json.load(filehandle)
    
model = tf.keras.models.load_model("VGGx2sig.model")

# Reading the mask RGB image files
mask_RGB = np.array([])
mask_RGB = np.empty([1, row_col_msize[0], row_col_msize[1], 3])

output = np.array([])
output = np.empty([1, 2])
print(output)
print(output.shape)

file_dir = []

i=0

for folder in natsort.natsorted(os.listdir(CTG_UP_DIR)):
    if folder.endswith('.jpg'):
        file_dir.append(folder)
        mask_RGB_info = Image.open( os.path.join (CTG_UP_DIR, folder))
        RGB_temp = np.asarray(mask_RGB_info)
        RGB_temp = np.expand_dims(RGB_temp, axis=0)
        
        mask_RGB = np.append(mask_RGB, RGB_temp, axis = 0)
        output = np.append(output, [[1, 0]], axis = 0)
        
        prediction = model.predict(RGB_temp)
        print('Index : ', i+1 )
        print('File Name : ', folder)
        print('Prediction Result : ', prediction)
        print('Ground Truth Output : [1,0]')
        if prediction[0][0] < prediction[0][1]:
            print(CATEGORIES[0])
            
            label = str('Index : '+str(i+1) + '\n' + 'File Name : '+str(folder) + '\n' + 'Model Output :' +'\n' + str(CATEGORIES[0]).rjust(18) +'\n'+ '        '+str(prediction) +'\n'+ 'Expected Output : \n       ctg_lay_down' +'\n' + '        '+str(output[i]) +'\n\n')
            plt.figure(figsize=(8,8))
            plt.imshow(RGB_temp[0].astype(np.uint8))
            plt.axis('off')
            plt.text(0.5, 0.5, label, fontsize = 15)
            plt.tight_layout()
            plt.ion()
            plt.savefig(  os.path.join(CTG_ERROR, 'Error_ctg_up'+str(i+1)+'.png') )
            plt.show()
        i+=1

for folder in natsort.natsorted(os.listdir(CTG_DW_DIR)):
    if folder.endswith('.jpg'):
        file_dir.append(folder)
        mask_RGB_info = Image.open( os.path.join (CTG_DW_DIR, folder))
        RGB_temp = np.asarray(mask_RGB_info)
        RGB_temp = np.expand_dims(RGB_temp, axis=0)
        
        mask_RGB = np.append(mask_RGB, RGB_temp, axis = 0)
        output = np.append(output, [[0, 1]], axis = 0)
        
        prediction = model.predict(RGB_temp)
        print('Index : ', i+1 )
        print('File Name : ', folder)
        print('Prediction Result : ', prediction)
        print('Ground Truth Output : [0,1]')
        if prediction[0][0] > prediction[0][1]:
            print(CATEGORIES[1])
            
            label = str('Index : '+str(i+1) + '\n' + 'File Name : '+str(folder) + '\n' + 'Model Output :' +'\n' + str(CATEGORIES[0]).rjust(20) +'\n'+ '        '+str(prediction) +'\n'+ 'Expected Output : \n       ctg_lay_down' +'\n' + '        '+str(output[i]) +'\n\n')
            plt.figure(figsize=(8,8))
            plt.imshow(RGB_temp[0].astype(np.uint8))
            plt.axis('off')
            plt.text(0.5, 0.5, label, fontsize = 15)
            plt.tight_layout()
            plt.ion()
            plt.savefig(  os.path.join(CTG_ERROR, 'Error_ctg_down'+str(i+1)+'.png') )
            plt.show()        
        i+=1
 

mask_RGB = np.delete(mask_RGB, 0, axis = 0)
output = np.delete(output, 0, axis=0)
print(output)

'''
# running the predictions and the evaluation of the model
i = 0
for mask_RGB_index in mask_RGB:
    temp_mask_RGB = np.expand_dims(mask_RGB_index, axis=0)
    
    prediction = model.predict(temp_mask_RGB)
    print('Index : ', i+1 )
    print('File Name : ', file_dir[i])
    print('Prediction Result : ', prediction)
    print('Ground Truth Output : ', output[i])
    if prediction[0][0] > prediction[0][1]:
        print(CATEGORIES[0])
        if int(file_dir[i][::-1][4]) == 0:
            print('True')
        else:
            label = str('Index : '+str(i+1) + '\n' + 'File Name : '+str(file_dir[i]) + '\n' + 'Model Output :' +'\n' + str(CATEGORIES[0]).rjust(20) +'\n'+ '        '+str(prediction) +'\n'+ 'Expected Output : \n       ctg_lay_down' +'\n' + '        '+str(output[i]) +'\n\n')
            plt.imshow(mask_RGB[i].astype(np.uint8))
            plt.axis('off')
            plt.text(0.5, 0.5, label, fontsize = 15)
            plt.savefig('Error case'+str(mask_RGB_index))
            #plt.show()
            print('False')
    else:
        print(CATEGORIES[1])
        if int(file_dir[i][::-1][4]) == 1:
            print('True')
        else:
            label = str('Index : '+str(i+1) + '\n' + 'File Name : '+str(file_dir[i]) + '\n' + 'Model Output :' +'\n' + str(CATEGORIES[1]).rjust(20) +'\n'+ '        '+str(prediction) +'\n'+ 'Expected Output : \n        ctg_std_up' +'\n' + '        '+str(output[i]) +'\n\n')
            plt.imshow(mask_RGB[i].astype(np.uint8))
            #plt.set_xlabel('pixels')
            #plt.set_ylabel('pixels')
            plt.axis('off')
            plt.text(0.5, 0.5, label, fontsize = 15)
            plt.savefig('Error case'+str(mask_RGB_index))
            plt.show()
            print('False')
        
    print('')
    i+=1
'''

#score = model.evaluate_generator(generator=test_gen_data, steps = 28)
score = model.evaluate(mask_RGB, output, batch_size=28, verbose=True)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#plt.imshow(mask_RGB[0].astype(np.uint8))
#plt.show()