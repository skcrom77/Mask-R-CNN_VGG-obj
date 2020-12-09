
#*************************************************************************#
# this code saves the rois and uncorped masking results of Mask R CNN. 
#*************************************************************************#

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

#from samples.pig import pig
import pig

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

PIG_WEIGHTS_PATH = "/path/to/mask_rcnn_coco.h5"

config = pig.PigConfig()
PIG_DIR = os.path.join(ROOT_DIR, "dataset")

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Load validation dataset
dataset = pig.PigDataset()
dataset.load_pig(PIG_DIR,"train")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

DEVICE = "/cpu:0"
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Or, load the last model you trained
weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

print(dataset.image_ids)
print(type(dataset.image_ids))
print(len(dataset.image_ids))

# change the starting file index depending on the training or validation dataset
sav_dir_str_index = 0

for i in range(len(dataset.image_ids)):
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, dataset.image_ids[i],
                           use_mini_mask=False)
    
    info = dataset.image_info[dataset.image_ids[i]]
    print(info['id'])
    # Run object detection
    results = model.detect([image], verbose=1)

    ax = get_ax(1) 
    r = results[0]
    
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")

    mini_new_mask = utils.minimize_mask(r['rois'], r['masks'], config.MINI_MASK_SHAPE)

    #display_images([mini_new_mask[:,:,i] for i in range(min(mini_new_mask.shape[-1], 14))])

    print(len(gt_bbox))
    # saving test data information
    #SAV_DIR = os.path.join(ROOT_DIR, 'samples/pig/crop_mask_data/pig_data_test_'+str(sav_dir_str_index+i+1))
    # saving the training data informations
    SAV_DIR = os.path.join(ROOT_DIR, 'crop_mask_data/pig_data_train_'+str(sav_dir_str_index+i+1))
    # saving the actual output of mask rcnn informations
    #SAV_DIR = os.path.join(ROOT_DIR, 'samples/pig/crop_mask_data/pig_data_act_out_'+str(sav_dir_str_index+i+1))
    os.mkdir(SAV_DIR)
    print(SAV_DIR)
    for index in range(len(gt_bbox)):
        #np.savetxt( os.path.join( SAV_DIR, str(info["id"][:-4])+'un_corp_mask_coord'+str(index)+'.json'), r['masks'][:,:,index])
        np.savetxt( os.path.join( SAV_DIR, str(info["id"][:-4])+'un_corp_mask_coord'+str(index)+'.json'), gt_mask[:,:,index], fmt='%d')
        #np.savetxt( os.path.join( SAV_DIR, str(info["id"][:-4])+'rois'+str(index)+'.json'), r['rois'][index,:], fmt='%d')
        np.savetxt( os.path.join( SAV_DIR, str(info["id"][:-4])+'rois'+str(index)+'.json'), gt_bbox[index,:], fmt='%d')
       