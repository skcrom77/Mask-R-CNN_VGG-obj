
#*************************************************************************#
# this code is generated for classifying the augmented data into the 
# classifyer of pig std_up or laying_down.
#*************************************************************************#

import json
import glob
import numpy as np
import sys
import os
import shutil
import math
import natsort

import matplotlib.pyplot as plt

# change the directory with the matching classification for classifying all the augmented data

def cls_augdata (save_loc, copied_from):

    for loc in save_loc:
        for folder in natsort.natsorted(os.listdir(loc)):
            if folder.endswith('.jpg'):
                for cmp_folder in natsort.natsorted(os.listdir(copied_from[0])):
                    if cmp_folder.endswith('.jpg'):
                        temp_folder = ''
                        temp_folder = cmp_folder[:-6]
                        temp_folder = temp_folder.replace('rot_','')
                        temp_folder += '.jpg'
                        if folder == temp_folder:
                            shutil.copy( os.path.join(copied_from[0], cmp_folder), loc)
                for cmp_folder in natsort.natsorted(os.listdir(copied_from[1])):
                    if cmp_folder.endswith('.jpg'):
                        temp_folder = ''
                        temp_folder = cmp_folder[:-6]
                        temp_folder = temp_folder.replace('Hflip_','')
                        temp_folder += '.jpg'
                        if folder == temp_folder:
                            shutil.copy( os.path.join(copied_from[1], cmp_folder), loc)
                for cmp_folder in natsort.natsorted(os.listdir(copied_from[2])):
                    if cmp_folder.endswith('.jpg'):
                        temp_folder = ''
                        temp_folder = cmp_folder[:-6]
                        temp_folder = temp_folder.replace('Vflip_','')
                        temp_folder += '.jpg'
                        if folder == temp_folder:
                            shutil.copy( os.path.join(copied_from[2], cmp_folder), loc)
                for cmp_folder in natsort.natsorted(os.listdir(copied_from[3])):
                    if cmp_folder.endswith('.jpg'):
                        temp_folder = ''
                        temp_folder = cmp_folder[:-6]
                        temp_folder = temp_folder.replace('rot_flip_','')
                        temp_folder += '.jpg'
                        if folder == temp_folder:
                            shutil.copy( os.path.join(copied_from[3], cmp_folder), loc)

# Root directory of the project
ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)

# directory which needs to be copied at
SAV_STD_UP_DIR = os.path.join(ROOT_DIR, "croped_mask_from_img/aug_data/std_up")
SAV_LY_DW_DIR = os.path.join(ROOT_DIR, "croped_mask_from_img/aug_data/lay_down")
SAV_LY_DW2_DIR = os.path.join(ROOT_DIR, "croped_mask_from_img/aug_data/lay_down2")
SAV_UNCONF_DIR = os.path.join(ROOT_DIR, "croped_mask_from_img/aug_data/unconf")
SAV_TEST_STD_UP = os.path.join(ROOT_DIR, "croped_mask_from_img/test_data/ctg_std_up")
SAV_TEST_LY_DW = os.path.join(ROOT_DIR, "croped_mask_from_img/test_data/ctg_lay_down")

list_save_at = [SAV_STD_UP_DIR, SAV_LY_DW_DIR, SAV_LY_DW2_DIR, SAV_UNCONF_DIR]

# directory files which needs to be copied from
ORG_IMG_DIR = os.path.join(ROOT_DIR, "croped_mask_from_img")
ROT_OPEN_DIR = os.path.join(ROOT_DIR, "croped_mask_from_img/aug_data/rot_data")
H_FLIP_DIR = os.path.join(ROOT_DIR, "croped_mask_from_img/aug_data/H_flip_data")
V_FLIP_DIR = os.path.join(ROOT_DIR, "croped_mask_from_img/aug_data/V_flip_data")
ROT_FLIP_DIR = os.path.join(ROOT_DIR, "croped_mask_from_img/aug_data/rot_flip_data")

list_cop_from = [ROT_OPEN_DIR, H_FLIP_DIR, V_FLIP_DIR, ROT_FLIP_DIR]

cls_augdata(list_save_at, list_cop_from)
