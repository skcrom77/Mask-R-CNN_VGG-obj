
#*************************************************************************#
# this code is generated for converting the masked information from the 
# mask R-CNN into an mask RGB image with, zero padding, masking augmentation
# and bluring effects on the image.
#*************************************************************************#

import json
import glob
import numpy as np
import sys
import os
import math
import copy

import cv2

import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("")
#ROT_DIR = os.path.join(ROOT_DIR, "samples\pig\croped_mask_from_img")

from PIL import Image, ImageOps
from numpy import savetxt

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from keras.preprocessing.image import ImageDataGenerator

from scipy.ndimage import gaussian_filter

# function generating the data augmentation by rotation
def data_rot_aug(mask_img):
    exp_mask_img = np.expand_dims(mask_img, 0)
    gen_data = ImageDataGenerator(rotation_range = 90)
    iterator = gen_data.flow(exp_mask_img, seed=1)
    return iterator

# function generating the data augmentation by horizontal  flip
def data_H_flip_aug(mask_img):
    exp_mask_img = np.expand_dims(mask_img, 0)
    gen_data = ImageDataGenerator(horizontal_flip = True)
    iterator = gen_data.flow(exp_mask_img, seed=1)
    return iterator

# function generating the data augmentation by  vertical flip
def data_V_flip_aug(mask_img):
    exp_mask_img = np.expand_dims(mask_img, 0)
    gen_data = ImageDataGenerator(vertical_flip = True)
    iterator = gen_data.flow(exp_mask_img, seed=1)
    return iterator

# function generating the data augmenta by rotation, horizontal and vertical flip
def data_rot_flip_aug(mask_img):
    exp_mask_img = np.expand_dims(mask_img, 0)
    gen_data = ImageDataGenerator(rotation_range = 90, horizontal_flip = True, vertical_flip = True)
    iterator = gen_data.flow(exp_mask_img, seed=1)
    return iterator

# function generating the augmentation of the mask
def aug_mask(mask_img, aug_num):

    for k in range(aug_num):
        for i in range(2, mask_img.shape[0]-1):
            for j in range(2, mask_img.shape[1]-1):
                if mask_img[i][j] == 1:
                    mask_img[i-1][j] = 1
                    mask_img[i][j-1] = 1
                    #mask_img[i+1][j] = 1
                    #mask_img[i][j+1] = 1
             
    return mask_img

# Function to convert a list of int to string
def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += str(ele)
    
    # return string   
    return str1 

# saving the rois informations

SAV_DIR = os.path.join(ROOT_DIR, 'croped_mask_from_img')
ROT_DIR = os.path.join(ROOT_DIR, 'croped_mask_from_img/aug_data/rot_data')
H_FLIP_DIR = os.path.join(ROOT_DIR, 'croped_mask_from_img/aug_data/H_flip_data')
V_FLIP_DIR = os.path.join(ROOT_DIR, 'croped_mask_from_img/aug_data/V_flip_data')
ROT_FLIP_DIR = os.path.join(ROOT_DIR, 'croped_mask_from_img/aug_data/rot_flip_data')

CROP_MASK_DIR = os.path.join(ROOT_DIR, "crop_mask_data/")

start_index = 1
file_index = 0

RGB_img = np.empty([1, 1024, 1024, 3])
mask_coord = np.empty([1, 1024, 1024])
rois_coord = []

max_H = 0
max_W = 0
file_name_zero_len = [0,0,0,0,0,0,0,0,0]

# gather the max size of Hight and Width based on the roi data
for folder in os.walk(CROP_MASK_DIR):
    obj_index = 0
    PATH = os.path.join(CROP_MASK_DIR, 'pig_data_train_' + str(start_index+file_index))
    DIR = PATH + '/*.json'
    for file in glob.glob(DIR):

        if '.json' in file:
            zero_len = file_name_zero_len[:-len(str(start_index+file_index))]
            try:
                with open(os.path.join(PATH, 'pig data_train_'+str(listToString(zero_len))+str(start_index+file_index)+'rois'+str(obj_index)+'.json' )) as rois_info:
                    temp1 = rois_info.readlines()
                    temp1[0] = temp1[0][ : -1]
                    temp1[1] = temp1[1][ : -1]
                    temp1[2] = temp1[2][ : -1]
                    temp1[3] = temp1[3][ : -1]
                    obj_index += 1
                    if max_W < int(temp1[2])-int(temp1[0]):
                        max_W = int(temp1[2])-int(temp1[0])
                    if max_H < int(temp1[3])-int(temp1[1]):
                        max_H = int(temp1[3])-int(temp1[1])
                
            except:
                pass         
                    
    file_index += 1

# Reading the input image, the rois info and mask info with the matching in order.
# Generate mask RGB image with the gathered information.

file_index = 1
TRAIN_DIR = os.path.join(ROOT_DIR, 'dataset/train')
#TEST_DIR = os.path.join(ROOT_DIR, 'dataset/test')

for folder in sorted(os.listdir(TRAIN_DIR)):
    
    if folder.endswith('.jpg'):
        print(folder)
        # Reading the input image and scaling into 1024 by 1024
        #----------------------------------------------------------------------#
        org_img_info = Image.open( os.path.join( TRAIN_DIR, folder) )
        org_img_info.load()
        temp = np.asarray(org_img_info, dtype = 'int32')
        image, _, _, _, _ = utils.resize_image(
            temp,
            min_dim = 800,
            max_dim = 1024,
            mode = 'square')
        image = np.expand_dims(image, axis = 0)
        RGB_img = image
        #----------------------------------------------------------------------#
        
        # Read the matching rois and mask info from the given directory
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        #PATH = os.path.join(CROP_MASK_DIR, 'pig_data_test_' + str(file_index))
        PATH = os.path.join(CROP_MASK_DIR, 'pig_data_train_' + str(file_index))
        #PATH = os.path.join(CROP_MASK_DIR, 'pig_data_act_out_' + str(file_index))
        DIR = PATH + '/*.json'
        
        obj_index = 0
        for file in glob.glob(DIR):
            print(file)
            if '.json' in file:
                zero_len = file_name_zero_len[:-len(str(file_index))]

                try:
                    with open(os.path.join(PATH, 'pig data_train_'+str(listToString(zero_len))+str(file_index)+'rois'+str(obj_index)+'.json' )) as rois_info:
                        #print(rois_info)
                        temp1 = rois_info.readlines()
                        temp1[0] = temp1[0][ : -1]
                        temp1[1] = temp1[1][ : -1]
                        temp1[2] = temp1[2][ : -1]
                        temp1[3] = temp1[3][ : -1]
                        rois_coord = temp1
                            
                        with open(os.path.join(PATH, 'pig data_train_'+str(listToString(zero_len))+str(file_index)+'un_corp_mask_coord'+str(obj_index)+'.json' )) as mask_info:
                            #print(mask_info)
                            #print(str(listToString(zero_len))+str(file_index))
                            temp2 = np.asarray([line.split() for line in mask_info], dtype = np.float32)
                            temp2 = np.expand_dims(temp2, axis = 0)
                            mask_coord = temp2
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

        
                            # Starting from this line cut the mask with the rois info and generate the mask RGB with zero padding
                            #------------------------------------------------------------------------------------------------------------------------------------------------#
                            # generating the mask RGB images with masking augmentation
                            CROP_rois_img = RGB_img[0][ int(rois_coord[0]) : int(rois_coord[2]), int(rois_coord[1]) : int(rois_coord[3]) ]
                            CROP_mask_img = mask_coord[0][ int(rois_coord[0]) : int(rois_coord[2]), int(rois_coord[1]) : int(rois_coord[3]) ]
                            temp_mask_img = copy.deepcopy(CROP_mask_img)
                            
                            # generate the augmentation to the mask
                            CROPaug_mask_img = aug_mask(temp_mask_img, 20)
                            
                            #plt.imshow(CROPaug_mask_img)
                            #plt.show()
                                
                            # generating the blured rois image
                            blur_rois_img = gaussian_filter(CROP_rois_img, sigma = 7)
                            
                            #plt.imshow(blur_rois_img)
                            #plt.show()
                                
                            # generating the bluring depending on the org - aug mask
                            aug_org_mask = CROPaug_mask_img - CROP_mask_img
                            R_temp = blur_rois_img[:,:,0] * aug_org_mask
                            G_temp = blur_rois_img[:,:,1] * aug_org_mask
                            B_temp = blur_rois_img[:,:,2] * aug_org_mask
                            aug_org_RGB_mask = np.stack((R_temp, G_temp, B_temp), axis = -1)
                            
                            #plt.imshow(aug_org_mask)
                            #plt.show()
                            
                            #plt.imshow(aug_org_RGB_mask.astype('uint8'))
                            #plt.show()
                               
                            # generating the masking RGB
                            R_temp = CROP_rois_img[:,:,0] * CROP_mask_img
                            G_temp = CROP_rois_img[:,:,1] * CROP_mask_img
                            B_temp = CROP_rois_img[:,:,2] * CROP_mask_img
                            CROP_RGB_mask = np.stack((R_temp, G_temp, B_temp), axis = -1)
                            
                            #plt.imshow(CROP_mask_img)
                            #plt.show()
                            
                            #plt.imshow(CROP_rois_img)
                            #plt.show()
                            
                            #plt.imshow(CROP_RGB_mask.astype('uint8'))
                            #plt.show()
                                
                            # adding the blurred area and GT mask RGB image
                            CROP_RGB_mask = CROP_RGB_mask + aug_org_RGB_mask
                            
                            #plt.imshow(CROP_RGB_mask.astype('uint8'))
                            #plt.show()
                            
                            # calculate the zero padding sizes
                            W_pad = int( (max_W - (int(rois_coord[2]) - int(rois_coord[0]))) / 2)
                            H_pad = int( (max_H - (int(rois_coord[3]) - int(rois_coord[1]))) / 2)
                            # generating the zero padding
                            CROP_RGB_mask = cv2.copyMakeBorder(CROP_RGB_mask, W_pad, W_pad, H_pad, H_pad, cv2.BORDER_CONSTANT )
                            if ((max_W - (int(rois_coord[2]) - int(rois_coord[0]))) % 2) != 0 :
                                CROP_RGB_mask = cv2.copyMakeBorder(CROP_RGB_mask, 1, 0, 0, 0, cv2.BORDER_CONSTANT )
                            if ((max_W - (int(rois_coord[3]) - int(rois_coord[1]))) % 2) != 0 :
                                CROP_RGB_mask = cv2.copyMakeBorder(CROP_RGB_mask, 0, 0, 1, 0, cv2.BORDER_CONSTANT )
                                
                            # generating the mask_RGB rotated and fliped result
                            rot_result = data_rot_aug(CROP_RGB_mask)
                            rot_flip_rslt = data_rot_flip_aug(CROP_RGB_mask)
                            h_flip_result = data_H_flip_aug(CROP_RGB_mask)
                            v_flip_result = data_V_flip_aug(CROP_RGB_mask)

                            rot_batch = rot_result.next()
                            rot_flip_batch = rot_flip_rslt.next()
                            h_flip_batch = h_flip_result.next()
                            v_flip_batch = v_flip_result.next()
                                    
                            rot_img = Image.fromarray(rot_batch[0].astype(np.uint8), 'RGB')
                            rot_flip_img = Image.fromarray(rot_flip_batch[0].astype(np.uint8), 'RGB')
                            h_flip_img = Image.fromarray(h_flip_batch[0].astype(np.uint8), 'RGB')
                            v_flip_img = Image.fromarray(v_flip_batch[0].astype(np.uint8), 'RGB')
                            
                            rot_img.save( os.path.join(ROT_DIR, 'mask_RGB_rot_img_'+str(file_index)+ '_'+ str(obj_index)+'_'+str(k)+'.jpg') )
                            rot_flip_img.save( os.path.join(ROT_FLIP_DIR, 'mask_RGB_rot_flip_img_'+str(file_index)+ '_'+ str(obj_index)+'_'+str(k)+'.jpg') )
                            h_flip_img.save( os.path.join(H_FLIP_DIR, 'mask_RGB_Hflip_img_'+str(file_index)+ '_'+ str(obj_index)+'_'+str(k)+'.jpg') )
                            v_flip_img.save( os.path.join(V_FLIP_DIR, 'mask_RGB_Vflip_img_'+str(file_index)+ '_'+ str(obj_index)+'_'+str(k)+'.jpg') )
                            
                            #rot_img.save( os.path.join(ROT_DIR, 'mask_RGB_rot_img_test_'+str(file_index)+ '_'+ str(obj_index)+'_'+str(k)+'.jpg') )
                            #rot_flip_img.save( os.path.join(ROT_FLIP_DIR, 'mask_RGB_rot_flip_img_test_'+str(file_index)+ '_'+ str(obj_index)+'_'+str(k)+'.jpg') )
                            #h_flip_img.save( os.path.join(H_FLIP_DIR, 'mask_RGB_Hflip_img_test_'+str(file_index)+ '_'+ str(obj_index)+'_'+str(k)+'.jpg') )
                            #v_flip_img.save( os.path.join(V_FLIP_DIR, 'mask_RGB_Vflip_img_test_'+str(file_index)+ '_'+ str(obj_index)+'_'+str(k)+'.jpg') )
                            
                            print(CROP_RGB_mask.shape)
                            # saving the RGB values to a specific folder
                            im = Image.fromarray(CROP_RGB_mask.astype(np.uint8), 'RGB')
                            im.save( os.path.join(SAV_DIR, 'mask_RGB_img_'+str(file_index)+ '_'+ str(obj_index)+'.jpg') )
                            #im.save( os.path.join(SAV_DIR, 'mask_RGB_img_test_'+str(file_index)+ '_'+ str(obj_index)+'.jpg') )
                            #------------------------------------------------------------------------------------------------------------------------------------------------#
                except:
                    pass
                    
                obj_index += 1
        file_index += 1

print(max_W)
print(max_H)

# saving the rois max width  and height
rois_msize = [max_W, max_H]
with open('row_col_msize.txt', 'w') as file2:
    json.dump(rois_msize, file2)

'''
print(len(rois_coord))
print(rois_coord[1][1])
print(ROOT_DIR)

plt.imshow(mask_coord[0])
plt.show()
'''