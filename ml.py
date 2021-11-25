# import os
# os.chdir('./Mask_RCNN/mrcnn')
# os.sys.path.append("./Mask_RCNN/mrcnn")
# print(os.listdir())
# from Mask_RCNN import mrcnn
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import random
import math
import re
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from mrcnn import utils
from mrcnn import visualize
# from Mask_RCNN.mrcnn.visualize import display_images
# from Mask_RCNN.mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils

#import custom

# Root directory of the project
ROOT_DIR = "/home/umar/Desktop/Programmer Force/Projects/Mask_RCNN-with-flask"

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

WEIGHTS_PATH = "/home/umar/Desktop/Programmer Force/Projects/Mask_RCNN-with-flask/Mask_RCNN/mask_rcnn_coco.h5"   # change it

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 80  # Background + Car and truck

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9




config = CustomConfig()
#LOAD MODEL. Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)


# Load COCO weights Or, load the last model you trained
weights_path = WEIGHTS_PATH
# Load weights
print("Loading weights ", weights_path)
# print(os.listdir('../'))
model.load_weights(weights_path, by_name=True)
model.keras_model._make_predict_function()




class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def func(imgname):
#     # try:
#     # Load a random image from the images folder
#     #file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(f'/home/umar/Desktop/Programmer Force/Projects/Mask_RCNN-with-flask/uploads/{imgname}')
    
    
    # print("Func", image)
    # model._make_predict_function()
    # print(image)
    # Run detection
    results = model.detect([image], verbose=1)
    # print("result", results[0])

    # Visualize results
    r = results[0]
    # print(r)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,imgname, r['scores'])
    print(f"\n{imgname} stored in prediction folder\n")
    # except Exception as e:
    #     print(e)
    #     print("error here")
        
# func('img.jpg')